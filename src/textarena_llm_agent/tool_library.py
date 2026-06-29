"""File-backed library of verified synthesized tools (Voyager skill library).

Layout under <memory_dir>/tool_library/:
  tools.jsonl             — one SynthesizedToolRecord per line
  tool_src/<name>_v<n>.py — the executable source per version
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .retrieval import BM25Retriever, ScoredItem, tokenize


@dataclass(slots=True)
class SynthesizedToolRecord:
    name: str
    description: str
    parameters: dict[str, Any]
    implementation: str
    game_id: str
    task_description: str
    version: int
    status: str = "active"            # active | demoted | disabled
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    uses: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_used_at: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex[:12])


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
            if rec.status != "active":
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
            if rec and rec.status == "active":
                out.append(rec)
            if len(out) >= top_k:
                break
        return out

    def active_for(self, game_id: str) -> list[SynthesizedToolRecord]:
        return [r for r in self._read() if r.status == "active" and r.game_id == game_id]

    # ------------------------------------------------------------------ registration
    def register_verified(self, proposal) -> SynthesizedToolRecord:  # ToolProposal
        records = self._read()
        # bump version if a tool with the same name exists
        existing = [r for r in records if r.name == proposal.name and r.game_id == proposal.game_id]
        version = max((r.version for r in existing), default=0) + 1
        # demote older versions of the same name for this game
        for r in records:
            if r.name == proposal.name and r.game_id == proposal.game_id and r.status == "active":
                r.status = "demoted"
        rec = SynthesizedToolRecord(
            name=proposal.name, description=proposal.description, parameters=proposal.parameters,
            implementation=proposal.implementation, game_id=proposal.game_id, task_description=proposal.task_description,
            version=version,
        )
        records.append(rec)
        self._write(records)
        (self.root / "tool_src" / f"{proposal.name}_v{version}.py").write_text(proposal.implementation, encoding="utf-8")
        return rec

    def record_use(self, name: str, *, ok: bool) -> None:
        records = self._read()
        for r in records:
            if r.name == name and r.status == "active":
                r.uses += 1
                if ok:
                    r.successes += 1
                    r.consecutive_failures = 0
                else:
                    r.failures += 1
                    r.consecutive_failures += 1
                r.last_used_at = datetime.now(timezone.utc).isoformat()
                if r.consecutive_failures >= 3:
                    r.status = "demoted"
                break
        self._write(records)

    def record_failed_candidate(self, *, task_description: str, game_id: str) -> None:
        with self.failed_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"task_description": task_description, "game_id": game_id, "created_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")

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
            out.append(SynthesizedToolRecord(
                name=obj.get("name", ""), description=obj.get("description", ""),
                parameters=obj.get("parameters") or {}, implementation=obj.get("implementation", ""),
                game_id=obj.get("game_id", ""), task_description=obj.get("task_description", ""),
                version=int(obj.get("version") or 1), status=obj.get("status", "active"),
                created_at=obj.get("created_at", ""), uses=int(obj.get("uses") or 0),
                successes=int(obj.get("successes") or 0), failures=int(obj.get("failures") or 0),
                consecutive_failures=int(obj.get("consecutive_failures") or 0),
                last_used_at=obj.get("last_used_at"), id=obj.get("id", ""),
            ))
        return out

    def _write(self, records: list[SynthesizedToolRecord]) -> None:
        self.tools_path.write_text("".join(json.dumps(asdict(r), ensure_ascii=False, default=str) + "\n" for r in records), encoding="utf-8")
        self._indexed_sig = ()  # force re-index

    def _by_id(self, tid: str) -> SynthesizedToolRecord | None:
        for r in self._read():
            if r.id == tid:
                return r
        return None

    def _refresh_index(self) -> None:
        sig = self._signature()
        if sig == self._indexed_sig:
            return
        items: list[ScoredItem] = []
        for r in self._read():
            text = f"{r.name} {r.description} {r.task_description} game:{r.game_id}"
            items.append(ScoredItem(source="tool", id=r.id, text=text, score=0.0, game_id=r.game_id, importance=2.0))
        self._bm25.indexer.index(items)
        self._indexed_sig = sig

    def _signature(self) -> tuple:
        try:
            st = self.tools_path.stat()
            return (st.st_mtime, st.st_size)
        except Exception:
            return ()
