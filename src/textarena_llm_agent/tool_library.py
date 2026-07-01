"""File-backed library of verified synthesized tools (Voyager skill library).

Phase 5: tools now move through a closed 5-stage pipeline modeled on Voyager:

    tool_need -> tool_spec -> candidate_tool -> validated_tool -> active_tool

Plus two terminal failure states: ``demoted`` and ``disabled``.

Lifecycle is owned by ``ToolValidator``; this module just persists records and
enforces the legal transition graph in ``mark_status``.

Layout under <memory_dir>/tool_library/:
  tools.jsonl             — one SynthesizedToolRecord per line
  tool_src/<name>_v<n>.py — the executable source per version
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from .retrieval import BM25Retriever, ScoredItem, tokenize


# ---------------------------------------------------------------------------
# Lifecycle vocabulary
# ---------------------------------------------------------------------------


class ToolStatus(str, Enum):
    """Closed lifecycle vocabulary for synthesized tools.

    Stages 1..5 form a forward pipeline. From any stage the tool may also be
    demoted (recoverable: re-evaluate later) or disabled (terminal: never use).
    """
    TOOL_NEED = "tool_need"            # recurring sub-problem observed; no spec yet
    TOOL_SPEC = "tool_spec"            # LLM produced a spec/proposal, not yet compiled
    CANDIDATE_TOOL = "candidate_tool"  # AST-validated + smoke-executes
    VALIDATED_TOOL = "validated_tool"  # passed replay eval against historical frames
    ACTIVE_TOOL = "active_tool"        # passed A/B against active set; live for agent
    DEMOTED = "demoted"                # recoverable failure / superseded
    DISABLED = "disabled"              # terminal: unsafe or chronically failing


# Forward transitions only. ``demoted`` / ``disabled`` reachable from anywhere.
_FORWARD_TRANSITIONS: dict[str, set[str]] = {
    ToolStatus.TOOL_NEED.value: {ToolStatus.TOOL_SPEC.value},
    ToolStatus.TOOL_SPEC.value: {ToolStatus.CANDIDATE_TOOL.value},
    ToolStatus.CANDIDATE_TOOL.value: {ToolStatus.VALIDATED_TOOL.value},
    ToolStatus.VALIDATED_TOOL.value: {ToolStatus.ACTIVE_TOOL.value},
    ToolStatus.ACTIVE_TOOL.value: set(),  # terminal-success (besides demote/disable)
    ToolStatus.DEMOTED.value: {ToolStatus.CANDIDATE_TOOL.value, ToolStatus.VALIDATED_TOOL.value},
    ToolStatus.DISABLED.value: set(),
}

# Legacy values written before Phase 5 ("active") are remapped on read.
_LEGACY_STATUS_REMAP: dict[str, str] = {
    "active": ToolStatus.ACTIVE_TOOL.value,
}


def _is_legal_transition(old: str, new: str) -> bool:
    if new in (ToolStatus.DEMOTED.value, ToolStatus.DISABLED.value):
        return True
    return new in _FORWARD_TRANSITIONS.get(old, set())


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SynthesizedToolRecord:
    name: str
    description: str
    parameters: dict[str, Any]
    implementation: str
    game_id: str
    task_description: str
    version: int
    status: str = ToolStatus.TOOL_NEED.value
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    uses: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_used_at: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    # Phase 5 fields
    replay_score: float = 0.0
    ab_score: float = 0.0
    unit_tests_passed: int = 0
    policy_version: str = "v0"
    spec_json: dict[str, Any] = field(default_factory=dict)
    tool_id: str = ""                                    # stable id across versions
    status_history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class ToolLibrary:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "tool_src").mkdir(parents=True, exist_ok=True)
        self.tools_path = self.root / "tools.jsonl"
        self.failed_path = self.root / "failed_candidates.jsonl"
        if not self.tools_path.exists():
            self.tools_path.write_text("", encoding="utf-8")
        if not self.failed_path.exists():
            self.failed_path.write_text("", encoding="utf-8")
        self._bm25 = BM25Retriever()
        self._indexed_sig: tuple = ()

    # ------------------------------------------------------------------ retrieval
    def semantic_key(self, task_description: str) -> str:
        return " ".join(sorted(set(tokenize(task_description))))[:120]

    def has(self, *, task_description: str, game_id: str) -> bool:
        key = self.semantic_key(task_description)
        for rec in self._read():
            if rec.status != ToolStatus.ACTIVE_TOOL.value:
                continue
            if rec.game_id != game_id:
                continue
            if self.semantic_key(rec.task_description) == key:
                return True
        return False

    def retrieve(self, query: str, *, game_id: str, top_k: int = 3) -> list[SynthesizedToolRecord]:
        self._refresh_index()
        scored = self._bm25.retrieve(query=query, game_id=game_id, player=None, phase=None, top_k=top_k)
        out: list[SynthesizedToolRecord] = []
        for item in scored:
            rec = self._by_id(item.id)
            if rec and rec.status == ToolStatus.ACTIVE_TOOL.value:
                out.append(rec)
            if len(out) >= top_k:
                break
        return out

    def active_for(self, game_id: str) -> list[SynthesizedToolRecord]:
        return [r for r in self._read()
                if r.status == ToolStatus.ACTIVE_TOOL.value and r.game_id == game_id]

    def by_status(self, status: str, *, game_id: str | None = None) -> list[SynthesizedToolRecord]:
        return [r for r in self._read()
                if r.status == status and (game_id is None or r.game_id == game_id)]

    def get(self, record_id: str) -> SynthesizedToolRecord | None:
        return self._by_id(record_id)

    def get_by_tool_id(self, tool_id: str) -> SynthesizedToolRecord | None:
        if not tool_id:
            return None
        # Return the highest-version row sharing this tool_id (the head of the
        # version chain).
        rows = [r for r in self._read() if r.tool_id == tool_id]
        if not rows:
            return None
        rows.sort(key=lambda r: r.version, reverse=True)
        return rows[0]

    # ------------------------------------------------------------------ pipeline stage helpers
    def create_need(self, *, task_description: str, game_id: str,
                    policy_version: str = "v0") -> SynthesizedToolRecord:
        """Stage 1: record a recurring sub-problem with no spec yet."""
        records = self._read()
        rec = SynthesizedToolRecord(
            name="",
            description="",
            parameters={},
            implementation="",
            game_id=game_id,
            task_description=task_description,
            version=0,
            status=ToolStatus.TOOL_NEED.value,
            policy_version=policy_version,
        )
        rec.tool_id = rec.id
        rec.status_history = [_history_entry(None, ToolStatus.TOOL_NEED.value, reason="created")]
        records.append(rec)
        self._write(records)
        return rec

    def attach_spec(self, *, record_id: str, proposal) -> SynthesizedToolRecord | None:
        """Stage 2: attach an LLM-produced spec (ToolProposal) to an existing tool_need row.

        Mutates the record in place — does not bump version; the spec defines
        version 1 once it compiles."""
        records = self._read()
        for r in records:
            if r.id != record_id:
                continue
            if not _is_legal_transition(r.status, ToolStatus.TOOL_SPEC.value):
                return None
            r.name = proposal.name
            r.description = proposal.description
            r.parameters = proposal.parameters
            r.implementation = proposal.implementation
            r.spec_json = {
                "name": proposal.name,
                "description": proposal.description,
                "parameters": proposal.parameters,
                "test_cases": list(getattr(proposal, "test_cases", []) or []),
            }
            r.status_history = list(r.status_history) + [
                _history_entry(r.status, ToolStatus.TOOL_SPEC.value, reason="spec attached"),
            ]
            r.status = ToolStatus.TOOL_SPEC.value
            self._write(records)
            return r
        return None

    def record_ab_score(self, *, record_id: str, ab_score: float) -> SynthesizedToolRecord | None:
        """Persist an A/B score against an existing record without a state change."""
        records = self._read()
        for r in records:
            if r.id != record_id:
                continue
            r.ab_score = float(ab_score)
            r.status_history = list(r.status_history) + [
                _history_entry(r.status, r.status, reason=f"ab_score updated to {ab_score:.3f}",
                               scores={"ab_score": float(ab_score)}),
            ]
            self._write(records)
            return r
        return None

    def mark_status(self, *, record_id: str, new_status: str,
                    reason: str = "", scores: dict[str, Any] | None = None) -> SynthesizedToolRecord | None:
        """Transition a record's status. Returns the updated record or None if illegal."""
        records = self._read()
        for r in records:
            if r.id != record_id:
                continue
            if not _is_legal_transition(r.status, new_status):
                return None
            if scores:
                if "replay_score" in scores:
                    r.replay_score = float(scores["replay_score"])
                if "ab_score" in scores:
                    r.ab_score = float(scores["ab_score"])
                if "unit_tests_passed" in scores:
                    r.unit_tests_passed = int(scores["unit_tests_passed"])
                if "policy_version" in scores:
                    r.policy_version = str(scores["policy_version"])
            # Activation: bump version + demote earlier active versions sharing
            # this tool_id (or name) within the same game.
            if new_status == ToolStatus.ACTIVE_TOOL.value:
                family = [
                    x for x in records
                    if x.game_id == r.game_id
                    and (x.tool_id == r.tool_id or (r.name and x.name == r.name))
                    and x.id != r.id
                ]
                next_version = max((x.version for x in family), default=0) + 1
                r.version = max(next_version, 1)
                for x in family:
                    if x.status == ToolStatus.ACTIVE_TOOL.value:
                        x.status_history = list(x.status_history) + [
                            _history_entry(x.status, ToolStatus.DEMOTED.value,
                                           reason=f"superseded by v{r.version}"),
                        ]
                        x.status = ToolStatus.DEMOTED.value
                if r.name and r.implementation:
                    (self.root / "tool_src" / f"{r.name}_v{r.version}.py").write_text(
                        r.implementation, encoding="utf-8",
                    )
            r.status_history = list(r.status_history) + [
                _history_entry(r.status, new_status, reason=reason, scores=scores or {}),
            ]
            r.status = new_status
            self._write(records)
            return r
        return None

    # ------------------------------------------------------------------ legacy fast-path registration
    def register_verified(self, proposal) -> SynthesizedToolRecord:
        """Back-compat: one-shot register a fully-verified proposal as active.

        Phase 5 callers should drive a record through the 5 stages via
        ``ToolValidator``; this method is kept for legacy tests and as the
        last-resort fallback for ``ToolSynthesizer.synthesize_and_register``.
        """
        records = self._read()
        existing = [r for r in records if r.name == proposal.name and r.game_id == proposal.game_id]
        version = max((r.version for r in existing), default=0) + 1
        for r in records:
            if (r.name == proposal.name and r.game_id == proposal.game_id
                    and r.status == ToolStatus.ACTIVE_TOOL.value):
                r.status_history = list(r.status_history) + [
                    _history_entry(r.status, ToolStatus.DEMOTED.value,
                                   reason=f"superseded by v{version}"),
                ]
                r.status = ToolStatus.DEMOTED.value
        rec = SynthesizedToolRecord(
            name=proposal.name, description=proposal.description, parameters=proposal.parameters,
            implementation=proposal.implementation, game_id=proposal.game_id,
            task_description=proposal.task_description,
            version=version,
            status=ToolStatus.ACTIVE_TOOL.value,
            spec_json={
                "name": proposal.name,
                "description": proposal.description,
                "parameters": proposal.parameters,
                "test_cases": list(getattr(proposal, "test_cases", []) or []),
            },
            unit_tests_passed=len(list(getattr(proposal, "test_cases", []) or [])[:3]),
        )
        rec.tool_id = rec.id
        rec.status_history = [_history_entry(None, ToolStatus.ACTIVE_TOOL.value,
                                              reason="legacy register_verified")]
        records.append(rec)
        self._write(records)
        (self.root / "tool_src" / f"{proposal.name}_v{version}.py").write_text(
            proposal.implementation, encoding="utf-8",
        )
        return rec

    # ------------------------------------------------------------------ runtime telemetry
    def record_use(self, name: str, *, ok: bool) -> None:
        records = self._read()
        for r in records:
            if r.name == name and r.status == ToolStatus.ACTIVE_TOOL.value:
                r.uses += 1
                if ok:
                    r.successes += 1
                    r.consecutive_failures = 0
                else:
                    r.failures += 1
                    r.consecutive_failures += 1
                r.last_used_at = datetime.now(timezone.utc).isoformat()
                if r.consecutive_failures >= 3:
                    r.status_history = list(r.status_history) + [
                        _history_entry(r.status, ToolStatus.DEMOTED.value,
                                       reason="3 consecutive failures"),
                    ]
                    r.status = ToolStatus.DEMOTED.value
                break
        self._write(records)

    def record_failed_candidate(self, *, task_description: str, game_id: str) -> None:
        with self.failed_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "task_description": task_description, "game_id": game_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")

    def history(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self._read()]

    # ------------------------------------------------------------------ internal
    def _read(self) -> list[SynthesizedToolRecord]:
        if not self.tools_path.exists():
            return []
        out: list[SynthesizedToolRecord] = []
        for line in self.tools_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            raw_status = obj.get("status", ToolStatus.ACTIVE_TOOL.value)
            status = _LEGACY_STATUS_REMAP.get(raw_status, raw_status)
            rec_id = obj.get("id", "") or uuid4().hex[:12]
            tool_id = obj.get("tool_id") or rec_id
            spec_json = obj.get("spec_json")
            if not isinstance(spec_json, dict):
                spec_json = {}
            history = obj.get("status_history")
            if not isinstance(history, list):
                history = []
            out.append(SynthesizedToolRecord(
                name=obj.get("name", ""), description=obj.get("description", ""),
                parameters=obj.get("parameters") or {},
                implementation=obj.get("implementation", ""),
                game_id=obj.get("game_id", ""),
                task_description=obj.get("task_description", ""),
                version=int(obj["version"]) if "version" in obj and obj["version"] is not None else 1,
                status=status,
                created_at=obj.get("created_at", ""),
                uses=int(obj.get("uses") or 0),
                successes=int(obj.get("successes") or 0),
                failures=int(obj.get("failures") or 0),
                consecutive_failures=int(obj.get("consecutive_failures") or 0),
                last_used_at=obj.get("last_used_at"),
                id=rec_id,
                replay_score=float(obj.get("replay_score") or 0.0),
                ab_score=float(obj.get("ab_score") or 0.0),
                unit_tests_passed=int(obj.get("unit_tests_passed") or 0),
                policy_version=str(obj.get("policy_version") or "v0"),
                spec_json=spec_json,
                tool_id=tool_id,
                status_history=list(history),
            ))
        return out

    def _write(self, records: list[SynthesizedToolRecord]) -> None:
        self.tools_path.write_text(
            "".join(json.dumps(asdict(r), ensure_ascii=False, default=str) + "\n" for r in records),
            encoding="utf-8",
        )
        self._indexed_sig = ()  # force re-index

    def _by_id(self, rid: str) -> SynthesizedToolRecord | None:
        for r in self._read():
            if r.id == rid:
                return r
        return None

    def _refresh_index(self) -> None:
        sig = self._signature()
        if sig == self._indexed_sig:
            return
        items: list[ScoredItem] = []
        for r in self._read():
            text = f"{r.name} {r.description} {r.task_description} game:{r.game_id}"
            items.append(ScoredItem(source="tool", id=r.id, text=text, score=0.0,
                                    game_id=r.game_id, importance=2.0))
        self._bm25.indexer.index(items)
        self._indexed_sig = sig

    def _signature(self) -> tuple:
        try:
            st = self.tools_path.stat()
            return (st.st_mtime, st.st_size)
        except Exception:
            return ()


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------


def _history_entry(old: str | None, new: str, *, reason: str = "",
                   scores: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "from": old,
        "to": new,
        "reason": reason,
        "scores": dict(scores or {}),
        "at": datetime.now(timezone.utc).isoformat(),
    }
