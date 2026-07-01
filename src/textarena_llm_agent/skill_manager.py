"""SkillManager — lifecycle owner for evolving skills (Phase 3).

The previous ``EvolvingMemory.evolve_skills`` collapsed proposal, validation,
and activation into a single in-place mutation of ``skills.jsonl``. That made
it impossible for the Critic Agent (Phase 4) to gate skill activation on
evidence quality and prevented Phase 7's hypothesis harness from comparing
``with-skill`` vs ``without-skill`` policy versions.

This module separates those concerns:

* ``SkillStatus`` — the closed lifecycle: ``proposed → candidate → validated
  → active`` with two exit states (``deprecated``, ``rejected``).
* ``SkillVersion`` — one immutable version row per (skill, version) pair.
  Skills mutate by allocating a new version with a fresh status; old
  versions stay in the graph as historical evidence.
* ``SkillManager`` — the only writer to ``skill`` and ``skill_version`` nodes
  in the Evidence Graph. Validation gates (``replay_score >= 0.55`` and
  ``ab_score >= 0``) are enforced here; the Critic Agent's tools call into
  these methods rather than touching SQLite directly.

The legacy ``skills.jsonl`` file remains the human-readable view used by the
prompt builder; it is mirrored from active skill versions on every activate /
deprecate. This keeps the JSONL surface backward compatible while moving the
authoritative state into the graph.

Game agent contract: ``active_skills(game_id)`` returns ONLY ``status=active``
versions. Anything in ``proposed`` / ``candidate`` / ``validated`` is invisible
to the game agent — only the Critic Agent (via ``all_skills``) sees them. This
prevents premature exposure of low-evidence skills during ablation studies.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


# --- thresholds ------------------------------------------------------------
# Validation thresholds (per PROGRESS.md Phase 3 spec).
#   - replay_score >= 0.55 means the skill improved decisions in >= 55% of
#     replayed past states (a soft majority — chance is 0.5).
#   - ab_score >= 0 means the A/B match result against the previous policy
#     version is at least neutral (no regression).
PROMOTE_MIN_SUPPORTS: int = 3
VALIDATE_REPLAY_THRESHOLD: float = 0.55
VALIDATE_AB_THRESHOLD: float = 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return uuid4().hex[:12]


class SkillStatus(str, Enum):
    """Closed lifecycle vocabulary for ``skill_version.status``."""

    PROPOSED = "proposed"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


@dataclass(slots=True)
class SkillVersion:
    """One immutable row in ``skill_version`` (mirrored by SkillManager)."""

    skill_id: str
    version: int
    status: str
    guidance: str
    trigger: str
    evidence_ids: list[str] = field(default_factory=list)
    replay_score: float | None = None
    ab_score: float | None = None
    created_by: str = "agent_fallback"   # agent_fallback | critic | legacy | manual
    policy_version: str = "v0"
    name: str = ""
    game_id: str = ""
    id: str = ""                          # composite "{skill_id}@v{version}"
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"{self.skill_id}@v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillManager:
    """Single-writer owner of the skill lifecycle.

    Construction takes the Evidence Graph and (optionally) the EvolvingMemory
    so the JSONL mirror stays in sync. Reads are O(matching rows) thanks to
    indexes on ``skill_version(skill_id, status)``.
    """

    def __init__(self, graph: Any, memory: Any = None, *, policy_version: str = "v0") -> None:
        self.graph = graph
        self.memory = memory
        self.policy_version = policy_version

    # ------------------------------------------------------------ proposal
    def propose(
        self,
        *,
        name: str,
        guidance: str,
        trigger: str,
        evidence_ids: Iterable[str],
        game_id: str,
        created_by: str = "agent_fallback",
        skill_id: str | None = None,
    ) -> SkillVersion:
        """Create a new ``proposed`` skill version.

        If ``skill_id`` is None a fresh skill row is allocated. SUPPORTS edges
        are written from each evidence memory id → the new ``skill_version``
        node so downstream queries (``count_supporting``, Critic Agent's
        ``query_evidence_graph``) can audit the proposal's grounding.
        """
        evidence_list = [str(e) for e in evidence_ids if e]
        if skill_id is None:
            skill_id = _short_id()
            # First-time skill: also write the parent ``skill`` node.
            self.graph.add_node(
                "skill", skill_id,
                name=name[:120], game_id=game_id, attrs={"trigger": trigger},
            )
            version_num = 1
        else:
            # Allocate next version for an existing skill.
            existing = self.graph.skill_versions() if hasattr(self.graph, "skill_versions") else []
            same_skill = [v for v in existing if v.get("skill_id") == skill_id]
            version_num = (max((int(v.get("version") or 0) for v in same_skill), default=0)) + 1

        sv = SkillVersion(
            skill_id=skill_id, version=version_num, status=SkillStatus.PROPOSED.value,
            guidance=guidance[:900], trigger=trigger,
            evidence_ids=evidence_list, created_by=created_by,
            policy_version=self.policy_version, name=name, game_id=game_id,
        )
        self._upsert_version(sv)
        # SUPPORTS edges: each evidence memory id supports this skill version.
        for mid in evidence_list:
            try:
                self.graph.add_edge("memory", mid, "SUPPORTS", "skill_version", sv.id)
            except Exception:
                continue
        return sv

    # ---------------------------------------------------------- promotions
    def promote_to_candidate(self, skill_version_id: str) -> bool:
        """Move a ``proposed`` version to ``candidate`` if it has >=3 SUPPORTS edges."""
        sv = self._get_version(skill_version_id)
        if sv is None or sv.status != SkillStatus.PROPOSED.value:
            return False
        supports = self.graph.count_supporting("skill_version", skill_version_id, edge="SUPPORTS")
        if supports < PROMOTE_MIN_SUPPORTS:
            return False
        sv.status = SkillStatus.CANDIDATE.value
        self._upsert_version(sv)
        return True

    def validate(
        self,
        skill_version_id: str,
        *,
        replay_score: float,
        ab_score: float,
    ) -> bool:
        """Move a ``candidate`` to ``validated`` iff both gates pass."""
        sv = self._get_version(skill_version_id)
        if sv is None or sv.status != SkillStatus.CANDIDATE.value:
            return False
        sv.replay_score = float(replay_score)
        sv.ab_score = float(ab_score)
        if replay_score < VALIDATE_REPLAY_THRESHOLD or ab_score < VALIDATE_AB_THRESHOLD:
            # Fail the gate but persist the scores for audit; status stays candidate.
            self._upsert_version(sv)
            return False
        sv.status = SkillStatus.VALIDATED.value
        self._upsert_version(sv)
        return True

    def activate(self, skill_version_id: str) -> str | None:
        """Promote a ``validated`` version to ``active`` and bump policy_version."""
        sv = self._get_version(skill_version_id)
        if sv is None or sv.status != SkillStatus.VALIDATED.value:
            return None
        # Deprecate any other active version of the same skill so the agent
        # always sees a single active guidance per skill.
        for prev in self._versions_for_skill(sv.skill_id):
            if prev.id == sv.id:
                continue
            if prev.status == SkillStatus.ACTIVE.value:
                prev.status = SkillStatus.DEPRECATED.value
                self._upsert_version(prev)
        sv.status = SkillStatus.ACTIVE.value
        new_policy = self._bump_policy_version()
        sv.policy_version = new_policy
        self._upsert_version(sv)
        self._mirror_active_to_jsonl()
        return new_policy

    def deprecate(self, skill_version_id: str, *, reason: str = "") -> bool:
        sv = self._get_version(skill_version_id)
        if sv is None:
            return False
        sv.status = SkillStatus.DEPRECATED.value
        attrs = {"deprecate_reason": reason} if reason else None
        self._upsert_version(sv, extra_attrs=attrs)
        self._mirror_active_to_jsonl()
        return True

    def reject(self, skill_version_id: str, *, reason: str = "") -> bool:
        sv = self._get_version(skill_version_id)
        if sv is None or sv.status not in {SkillStatus.PROPOSED.value, SkillStatus.CANDIDATE.value}:
            return False
        sv.status = SkillStatus.REJECTED.value
        attrs = {"reject_reason": reason} if reason else None
        self._upsert_version(sv, extra_attrs=attrs)
        return True

    # -------------------------------------------------------------- queries
    def active_skills(self, game_id: str) -> list[dict[str, Any]]:
        """Game-agent view: only ``status=active`` versions for this game."""
        rows = self.graph.skill_versions(game_id=game_id, status=SkillStatus.ACTIVE.value) if hasattr(self.graph, "skill_versions") else []
        return [self._row_to_public(r) for r in rows]

    def all_skills(self, game_id: str | None = None) -> list[dict[str, Any]]:
        """Critic-agent view: all statuses, used to audit the lifecycle."""
        rows = self.graph.skill_versions(game_id=game_id) if hasattr(self.graph, "skill_versions") else []
        return [self._row_to_public(r) for r in rows]

    def get(self, skill_version_id: str) -> SkillVersion | None:
        return self._get_version(skill_version_id)

    # ----------------------------------------------------- evolution sweep
    def run_evolution_sweep(
        self,
        *,
        game_id: str,
        replay_fn: Any = None,
        ab_fn: Any = None,
        mutate_fn: Any = None,
        demote_loss_streak: int = 5,
    ) -> dict[str, int]:
        """Critic-driven sweep that replaces ``EvolvingMemory.evolve_skills``.

        ``replay_fn(skill_version) -> float`` and ``ab_fn(skill_version) -> float``
        are injected by the caller (typically the Critic Agent). When both
        functions are None the sweep is a no-op — proving that lifecycle
        transitions can ONLY happen via explicit Critic action, never as a
        side effect of running an episode.
        """
        counts = {"promoted": 0, "validated": 0, "activated": 0,
                  "rejected": 0, "deprecated": 0}
        for sv in self._versions_for_game(game_id):
            if sv.status == SkillStatus.PROPOSED.value:
                if self.promote_to_candidate(sv.id):
                    counts["promoted"] += 1
            elif sv.status == SkillStatus.CANDIDATE.value and replay_fn is not None and ab_fn is not None:
                r = float(replay_fn(sv))
                a = float(ab_fn(sv))
                if self.validate(sv.id, replay_score=r, ab_score=a):
                    counts["validated"] += 1
                elif r + a < 0:
                    if self.reject(sv.id, reason=f"replay={r:.2f} ab={a:.2f}"):
                        counts["rejected"] += 1
            elif sv.status == SkillStatus.VALIDATED.value:
                if self.activate(sv.id) is not None:
                    counts["activated"] += 1
        return counts

    # ---------------------------------------------------------- internals
    def _get_version(self, sv_id: str) -> SkillVersion | None:
        row = self.graph.get_node("skill_version", sv_id)
        if not row:
            return None
        return self._row_to_version(row)

    def _versions_for_skill(self, skill_id: str) -> list[SkillVersion]:
        rows = self.graph.skill_versions() if hasattr(self.graph, "skill_versions") else []
        return [self._row_to_version(r) for r in rows if r.get("skill_id") == skill_id]

    def _versions_for_game(self, game_id: str) -> list[SkillVersion]:
        rows = self.graph.skill_versions(game_id=game_id) if hasattr(self.graph, "skill_versions") else []
        return [self._row_to_version(r) for r in rows]

    def _row_to_version(self, row: dict[str, Any]) -> SkillVersion:
        attrs = row.get("attrs") or {}
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except Exception:
                attrs = {}
        return SkillVersion(
            skill_id=str(row.get("skill_id") or ""),
            version=int(row.get("version") or 1),
            status=str(row.get("status") or SkillStatus.PROPOSED.value),
            guidance=str(attrs.get("guidance") or ""),
            trigger=str(attrs.get("trigger") or ""),
            evidence_ids=list(attrs.get("evidence_ids") or []),
            replay_score=row.get("replay_score"),
            ab_score=row.get("ab_score"),
            created_by=str(row.get("created_by") or ""),
            policy_version=str(row.get("policy_version") or "v0"),
            name=str(attrs.get("name") or row.get("name") or ""),
            game_id=str(attrs.get("game_id") or row.get("game_id") or ""),
            id=str(row.get("id") or ""),
            created_at=str(row.get("created_at") or _now()),
        )

    def _row_to_public(self, row: dict[str, Any]) -> dict[str, Any]:
        sv = self._row_to_version(row)
        out = sv.to_dict()
        # Mirror useful columns at top level for prompt builders.
        out["status"] = sv.status
        return out

    def _upsert_version(self, sv: SkillVersion, *, extra_attrs: dict[str, Any] | None = None) -> None:
        attrs: dict[str, Any] = {
            "name": sv.name,
            "guidance": sv.guidance,
            "trigger": sv.trigger,
            "evidence_ids": sv.evidence_ids,
            "game_id": sv.game_id,
        }
        if extra_attrs:
            attrs.update(extra_attrs)
        self.graph.add_node(
            "skill_version", sv.id,
            skill_id=sv.skill_id, version=sv.version, status=sv.status,
            policy_version=sv.policy_version,
            replay_score=sv.replay_score, ab_score=sv.ab_score,
            created_by=sv.created_by, created_at=sv.created_at,
            attrs=attrs,
        )

    def _bump_policy_version(self) -> str:
        """Allocate a new ``policy_version`` row and return its name.

        Naming convention: ``v{N}`` where N is the highest existing integer
        suffix + 1. Skill set hash is left blank here — it is filled in by
        Phase 6's prompt compiler.
        """
        rows = self.graph.query(
            "SELECT name FROM policy_version ORDER BY created_at DESC LIMIT 200",
            (),
        )
        max_n = 0
        for r in rows:
            name = str(r.get("name") or "")
            if name.startswith("v") and name[1:].isdigit():
                max_n = max(max_n, int(name[1:]))
        new_name = f"v{max_n + 1}"
        pv_id = _short_id()
        self.graph.add_node(
            "policy_version", pv_id,
            name=new_name, parent=self.policy_version,
            skill_set_hash="", tool_set_hash="",
            attrs={"trigger": "skill_activate"},
        )
        self.policy_version = new_name
        return new_name

    def _mirror_active_to_jsonl(self) -> None:
        """Refresh ``skills.jsonl`` to contain only currently-active versions.

        The prompt builder (and any legacy reader) still reads this file; we
        regenerate it from the graph so the two views never diverge.
        """
        if self.memory is None or not hasattr(self.memory, "skills_path"):
            return
        try:
            rows = self.graph.skill_versions(status=SkillStatus.ACTIVE.value)
        except Exception:
            return
        lines: list[str] = []
        for r in rows:
            sv = self._row_to_version(r)
            payload = {
                "id": sv.skill_id,
                "skill_version_id": sv.id,
                "name": sv.name,
                "trigger": sv.trigger,
                "guidance": sv.guidance,
                "evidence": sv.evidence_ids,
                "status": "active",
                "version": sv.version,
                "policy_version": sv.policy_version,
                "created_by": sv.created_by,
                "tags": [sv.game_id] if sv.game_id else [],
                "created_at": sv.created_at,
                "updated_at": _now(),
            }
            lines.append(json.dumps(payload, ensure_ascii=False, default=str))
        path = Path(self.memory.skills_path)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


__all__ = [
    "SkillManager",
    "SkillStatus",
    "SkillVersion",
    "PROMOTE_MIN_SUPPORTS",
    "VALIDATE_REPLAY_THRESHOLD",
    "VALIDATE_AB_THRESHOLD",
]
