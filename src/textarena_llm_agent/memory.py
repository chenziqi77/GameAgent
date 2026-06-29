from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


@dataclass(slots=True)
class Experience:
    state_key: str
    game_id: str
    player: int
    action_text: str
    evaluator_score: float | None = None
    reward: float | None = None
    outcome: str | None = None
    critique: str | None = None
    lesson: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass(slots=True)
class SkillMemory:
    name: str
    trigger: str
    guidance: str
    evidence: list[str] = field(default_factory=list)
    uses: int = 0
    wins: int = 0
    losses: int = 0
    score_sum: float = 0.0
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    # Evolution fields:
    version: int = 1
    status: str = "active"           # active | mutated | demoted | promoted
    confidence: float = 1.0
    importance: float = 1.0
    last_used_at: str | None = None
    win_rate: float = 0.0
    consecutive_losses: int = 0


@dataclass(slots=True)
class Reflection:
    id: str
    game_id: str
    episode_seed: int
    outcome: str
    text: str
    state_keys: list[str] = field(default_factory=list)
    actionable_lesson: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class PatchRecord:
    patch_id: str
    game_id: str
    patch_text: str
    version: int
    evidence: str
    status: str = "active"           # active | reverted
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    win_rate_after: float | None = None


# Skill evolution thresholds (config-driven via EvolvingMemory).
DEFAULT_MUTATE_THRESHOLD: float = 0.35     # mutate a skill if win_rate < this over min_uses
DEFAULT_MUTATE_MIN_USES: int = 4
DEFAULT_DEMOTE_THRESHOLD: float = 0.20     # demote if win_rate < this over min_uses
DEFAULT_PROMOTE_THRESHOLD: float = 0.65    # promote if win_rate >= this over min_uses


class EvolvingMemory:
    """File-backed tiered memory: episodic experiences, evolving skills, reflections,
    and versioned prompt patches. BM25-backed recall.

    Patterns: Generative Agents (tiered memory + recency/importance/relevance retrieval),
    Reflexion (self-reflection + verbal reinforcement/skill mutation), ExpeL (insight
    consolidation), Voyager (skill library with usage stats + verification/demotion).
    """

    def __init__(self, root: str | Path = "workspace/textarena_memory") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiences_path = self.root / "experiences.jsonl"
        self.skills_path = self.root / "skills.jsonl"
        self.skill_history_path = self.root / "skill_updates.jsonl"
        self.retrieval_hits_path = self.root / "retrieval_hits.jsonl"
        self.reflections_path = self.root / "reflections.jsonl"
        self.patches_path = self.root / "prompt_patches.jsonl"
        self.rules_path = self.root / "rules.md"
        self.prompt_overrides_path = self.root / "prompt_overrides.md"
        for path in [self.experiences_path, self.skills_path, self.skill_history_path, self.retrieval_hits_path, self.reflections_path, self.patches_path]:
            if not path.exists():
                path.write_text("", encoding="utf-8")
        if not self.rules_path.exists():
            self.rules_path.write_text(
                "# TextArena Agent Tactical Memory\n\n"
                "- Select only legal bracketed actions from the current candidate set.\n"
                "- Prefer actions that improve expected payoff while preserving flexibility and information.\n",
                encoding="utf-8",
            )
        if not self.prompt_overrides_path.exists():
            self.prompt_overrides_path.write_text("# Prompt Overrides\n\n", encoding="utf-8")
        # Lazy BM25 retriever (avoids import cycle at construction).
        self._retriever = None

    # ------------------------------------------------------------------ recall
    def recall(self, query: str, *, max_items: int = 8, game_id: str = "", player: int | None = None, phase: str | None = None) -> list[dict[str, Any]]:
        if self._retriever is None:
            self._retriever = _make_retriever(self.root)
        # Re-index if memory grew since last index.
        self._retriever.refresh(game_id=game_id or self._infer_game_from_query(query))
        q = f"{query} {game_id} {phase or ''}".strip()
        scored = self._retriever.retrieve(query=q, game_id=game_id or self._infer_game_from_query(query), player=player, phase=phase, top_k=max_items)
        rows: list[dict[str, Any]] = []
        for item in scored:
            rows.append({"source": item.source, "score": item.score, "id": item.id, "text": item.text, "game_id": item.game_id, "player": item.player})
        self.record_retrieval(query=query, items=rows)
        return rows

    def format_for_prompt(self, query: str, *, max_items: int = 8, game_id: str = "", player: int | None = None, phase: str | None = None) -> str:
        parts: list[str] = []
        patches = self.active_prompt_patches(game_id) if game_id else []
        if patches:
            parts.append("\n".join(f"- {p.get('patch_text','')}" for p in patches if p.get("patch_text"))[:2000])
        items = self.recall(query, max_items=max_items, game_id=game_id, player=player, phase=phase)
        if items:
            parts.append("Relevant game memory:\n" + "\n\n".join(f"[{x['source']} score={x['score']:.2f}]\n{x['text']}" for x in items))
        return "\n\n".join(parts) if parts else "No relevant TextArena memory yet."

    def _infer_game_from_query(self, query: str) -> str:
        q = (query or "").lower()
        for fam in ["tictactoe", "kuhnpoker", "simplenegotiation", "stratego"]:
            if fam in q:
                return fam
        return ""

    # ------------------------------------------------------------------ experiences
    def record_experience(self, exp: Experience | dict[str, Any]) -> str:
        obj = asdict(exp) if isinstance(exp, Experience) else dict(exp)
        exp_id = str(obj.get("id") or uuid4().hex[:12])
        obj["id"] = exp_id
        obj.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with self.experiences_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        lesson = str(obj.get("lesson") or "").strip()
        if lesson:
            self.add_rule(lesson, evidence=f"experience:{exp_id}", tags=obj.get("tags") or [])
        return exp_id

    def recent_experiences(self, *, game_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = [r for r in self._read_jsonl(self.experiences_path) if str(r.get("game_id") or "") == game_id]
        return rows[-limit:]

    def unconsolidated_count(self, game_id: str) -> int:
        """Experiences since the last insight-consolidation marker for a game."""
        marker = self.root / f".last_insight_{game_id}"
        since: datetime | None = None
        if marker.exists():
            try:
                raw = marker.read_text(encoding="utf-8").strip()
                since = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None
            except Exception:
                since = None
        count = 0
        for r in self._read_jsonl(self.experiences_path):
            if str(r.get("game_id") or "") != game_id:
                continue
            ts = r.get("created_at")
            if since is None or not ts:
                count += 1
                continue
            try:
                created = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except Exception:
                count += 1
                continue
            if created > since:
                count += 1
        return count

    def mark_insight_consolidated(self, game_id: str) -> None:
        (self.root / f".last_insight_{game_id}").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    # ------------------------------------------------------------------ rules
    def add_rule(self, rule: str, *, evidence: str = "", tags: Iterable[str] = ()) -> bool:
        rule = _normalize_rule(rule)
        if not rule:
            return False
        existing = self.rules_path.read_text(encoding="utf-8", errors="replace")
        if rule.lower() in existing.lower():
            return False
        tag_text = ",".join(tags) if tags else "general"
        with self.rules_path.open("a", encoding="utf-8") as f:
            f.write(f"\n- {rule}  <!-- tags:{tag_text}; evidence:{evidence} -->\n")
        return True

    # ------------------------------------------------------------------ prompt patches (versioned)
    def add_prompt_patch(self, patch: str, *, game_id: str = "", evidence: str = "") -> PatchRecord:
        patch = patch.strip()
        if not patch:
            return PatchRecord(patch_id="", game_id=game_id, patch_text="", version=0, evidence=evidence)
        records = self._read_jsonl(self.patches_path)
        # dedup against active patches for this game
        for rec in records:
            if rec.get("status") == "active" and rec.get("game_id") == game_id and patch.lower() in str(rec.get("patch_text","")).lower():
                return PatchRecord(**{k: rec.get(k) for k in ["patch_id","game_id","patch_text","version","evidence","status","created_at","win_rate_after"]})
        version = len([r for r in records if r.get("game_id") == game_id]) + 1
        rec = PatchRecord(patch_id=uuid4().hex[:12], game_id=game_id, patch_text=patch, version=version, evidence=evidence, status="active")
        records.append(asdict(rec))
        self._write_jsonl(self.patches_path, records)
        # also mirror to legacy prompt_overrides.md for back-compat
        with self.prompt_overrides_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## Patch {datetime.now(timezone.utc).isoformat()} ({game_id})\n{patch}\nEvidence: {evidence}\n")
        return rec

    def active_prompt_patches(self, game_id: str) -> list[dict[str, Any]]:
        return [r for r in self._read_jsonl(self.patches_path) if r.get("status") == "active" and (not r.get("game_id") or r.get("game_id") == game_id)]

    def revert_patch(self, patch_id: str, *, reason: str = "") -> bool:
        records = self._read_jsonl(self.patches_path)
        changed = False
        for rec in records:
            if rec.get("patch_id") == patch_id and rec.get("status") == "active":
                rec["status"] = "reverted"
                changed = True
        if changed:
            self._write_jsonl(self.patches_path, records)
            self._append_skill_history({"event": "patch_reverted", "patch_id": patch_id, "reason": reason})
        return changed

    # ------------------------------------------------------------------ skills (evolving)
    def consolidate_skill_from_lesson(
        self,
        *,
        lesson: str,
        game_id: str,
        evidence: str,
        reward: float | None = None,
        evaluator_score: float | None = None,
        outcome: str | None = None,
        tags: Iterable[str] = (),
        importance: float = 1.0,
    ) -> str | None:
        guidance = _normalize_rule(lesson)
        if not guidance:
            return None
        tags_list = [str(t) for t in tags]
        trigger = f"game:{game_id} " + " ".join(tags_list)
        name = _skill_name(guidance, game_id)
        skills = self._read_jsonl(self.skills_path)
        existing_idx = next((i for i, s in enumerate(skills) if str(s.get("name", "")).lower() == name.lower()), None)
        score = float(evaluator_score) if evaluator_score is not None else 0.0
        win = 1 if (reward is not None and float(reward) > 0) or str(outcome).lower() in {"win", "terminal_win"} else 0
        loss = 1 if (reward is not None and float(reward) < 0) or str(outcome).lower() in {"loss", "terminal_loss"} else 0
        event = "created"
        if existing_idx is None:
            skill = asdict(SkillMemory(name=name, trigger=trigger, guidance=guidance, evidence=[evidence], uses=1, wins=win, losses=loss, score_sum=score, tags=tags_list, importance=importance))
            skills.append(skill)
        else:
            skill = dict(skills[existing_idx])
            skill["uses"] = int(skill.get("uses") or 0) + 1
            skill["wins"] = int(skill.get("wins") or 0) + win
            skill["losses"] = int(skill.get("losses") or 0) + loss
            skill["score_sum"] = float(skill.get("score_sum") or 0.0) + score
            skill["updated_at"] = datetime.now(timezone.utc).isoformat()
            skill["last_used_at"] = skill["updated_at"]
            skill["win_rate"] = round(skill["wins"] / max(1, skill["uses"]), 4)
            skill["consecutive_losses"] = (int(skill.get("consecutive_losses") or 0) + 1) if loss else 0
            ev = list(skill.get("evidence") or [])
            if evidence and evidence not in ev:
                ev.append(evidence)
            skill["evidence"] = ev[-30:]
            skill["guidance"] = _merge_guidance(str(skill.get("guidance") or ""), guidance)
            event = "updated"
            skills[existing_idx] = skill
        self._write_jsonl(self.skills_path, skills)
        skill_obj = skills[existing_idx] if existing_idx is not None else skills[-1]
        self._append_skill_history({"event": event, "skill_id": skill_obj.get("id"), "name": name, "game_id": game_id, "lesson": guidance, "evidence": evidence, "reward": reward, "outcome": outcome})
        return str(skill_obj.get("id"))

    def update_skill_usage(self, *, skill_id: str, win: bool, score: float = 0.0) -> None:
        """Increment a touched skill's win/loss stats from a terminal outcome."""
        skills = self._read_jsonl(self.skills_path)
        for i, s in enumerate(skills):
            if s.get("id") == skill_id:
                s["uses"] = int(s.get("uses") or 0) + 1
                if win:
                    s["wins"] = int(s.get("wins") or 0) + 1
                    s["consecutive_losses"] = 0
                else:
                    s["losses"] = int(s.get("losses") or 0) + 1
                    s["consecutive_losses"] = int(s.get("consecutive_losses") or 0) + 1
                s["score_sum"] = float(s.get("score_sum") or 0.0) + float(score)
                s["win_rate"] = round(s["wins"] / max(1, s["uses"]), 4)
                s["updated_at"] = datetime.now(timezone.utc).isoformat()
                s["last_used_at"] = s["updated_at"]
                skills[i] = s
                self._write_jsonl(self.skills_path, skills)
                self._append_skill_history({"event": "updated", "skill_id": skill_id, "name": s.get("name"), "game_id": _game_id_from_trigger(str(s.get("trigger") or "")), "win": win, "uses": s["uses"], "win_rate": s["win_rate"]})
                return

    def evolve_skills(self, *, game_id: str, llm=None,
                      mutate_threshold: float = DEFAULT_MUTATE_THRESHOLD,
                      demote_threshold: float = DEFAULT_DEMOTE_THRESHOLD,
                      promote_threshold: float = DEFAULT_PROMOTE_THRESHOLD,
                      min_uses: int = DEFAULT_MUTATE_MIN_USES) -> dict[str, int]:
        """Promotion / demotion / mutation sweep over skills touched by a game.

        Reflexion verbal reinforcement (mutation) + Voyager skill verification/demotion.
        """
        skills = self._read_jsonl(self.skills_path)
        counts = {"promoted": 0, "demoted": 0, "mutated": 0}
        for i, s in enumerate(skills):
            if game_id not in str(s.get("trigger","")) and game_id not in (s.get("tags") or []):
                continue
            if int(s.get("uses") or 0) < min_uses:
                continue
            wr = float(s.get("win_rate") or 0.0)
            if wr >= promote_threshold and s.get("status") != "promoted":
                s["status"] = "promoted"
                s["confidence"] = min(1.0, float(s.get("confidence") or 0.5) + 0.2)
                counts["promoted"] += 1
                self._append_skill_history({"event": "promoted", "skill_id": s.get("id"), "name": s.get("name"), "game_id": game_id, "win_rate": wr, "uses": s.get("uses")})
            elif wr < demote_threshold and s.get("status") != "demoted":
                s["status"] = "demoted"
                s["confidence"] = max(0.1, float(s.get("confidence") or 0.5) - 0.3)
                counts["demoted"] += 1
                self._append_skill_history({"event": "demoted", "skill_id": s.get("id"), "name": s.get("name"), "game_id": game_id, "win_rate": wr, "uses": s.get("uses")})
            elif wr < mutate_threshold and s.get("status") != "demoted" and llm is not None:
                new_guidance = _mutate_via_llm(llm, s, game_id)
                if new_guidance and new_guidance != s.get("guidance"):
                    s["version"] = int(s.get("version") or 1) + 1
                    s["guidance"] = new_guidance[:900]
                    s["status"] = "mutated"
                    s["consecutive_losses"] = 0
                    counts["mutated"] += 1
                    self._append_skill_history({"event": "mutated", "skill_id": s.get("id"), "name": s.get("name"), "game_id": game_id, "version": s["version"], "new_guidance": new_guidance[:300]})
            skills[i] = s
        self._write_jsonl(self.skills_path, skills)
        return counts

    def mutate_skill(self, skill_id: str, *, llm) -> str | None:
        skills = self._read_jsonl(self.skills_path)
        for s in skills:
            if s.get("id") == skill_id:
                new_guidance = _mutate_via_llm(llm, s, str(s.get("trigger","").split(":")[0] if ":" in str(s.get("trigger","")) else "unknown"))
                if new_guidance and new_guidance != s.get("guidance"):
                    s["version"] = int(s.get("version") or 1) + 1
                    s["guidance"] = new_guidance[:900]
                    s["status"] = "mutated"
                    self._write_jsonl(self.skills_path, skills)
                    self._append_skill_history({"event": "mutated", "skill_id": skill_id, "name": s.get("name"), "version": s["version"], "new_guidance": new_guidance[:300]})
                    return new_guidance
        return None

    def demote_skill(self, skill_id: str, *, reason: str = "") -> None:
        skills = self._read_jsonl(self.skills_path)
        for s in skills:
            if s.get("id") == skill_id:
                s["status"] = "demoted"
                s["confidence"] = max(0.1, float(s.get("confidence") or 0.5) - 0.3)
                self._write_jsonl(self.skills_path, skills)
                self._append_skill_history({"event": "demoted", "skill_id": skill_id, "reason": reason})
                return

    def promote_skill(self, skill_id: str) -> None:
        skills = self._read_jsonl(self.skills_path)
        for s in skills:
            if s.get("id") == skill_id:
                s["status"] = "promoted"
                s["confidence"] = min(1.0, float(s.get("confidence") or 0.5) + 0.2)
                self._write_jsonl(self.skills_path, skills)
                self._append_skill_history({"event": "promoted", "skill_id": skill_id})
                return

    def decay_confidence(self, *, half_life_days: float = 30.0) -> None:
        import math
        skills = self._read_jsonl(self.skills_path)
        now = datetime.now(timezone.utc)
        changed = False
        for s in skills:
            ts = s.get("updated_at") or s.get("created_at")
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                age = (now - dt).total_seconds() / 86400.0
                decay = math.exp(-age / half_life_days)
                new_conf = max(0.1, float(s.get("confidence") or 0.5) * (0.5 + 0.5 * decay))
                if abs(new_conf - float(s.get("confidence") or 0.5)) > 0.01:
                    s["confidence"] = round(new_conf, 3)
                    changed = True
            except Exception:
                continue
        if changed:
            self._write_jsonl(self.skills_path, skills)

    def active_skills(self, game_id: str) -> list[dict[str, Any]]:
        return [s for s in self._read_jsonl(self.skills_path) if s.get("status") in (None, "active", "promoted", "mutated") and (game_id in str(s.get("trigger","")) or game_id in (s.get("tags") or []))]

    # ------------------------------------------------------------------ reflections
    def record_reflection(self, r: Reflection | dict[str, Any]) -> str:
        obj = asdict(r) if isinstance(r, Reflection) else dict(r)
        rid = str(obj.get("id") or uuid4().hex[:12])
        obj["id"] = rid
        obj.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with self.reflections_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        if obj.get("actionable_lesson"):
            self.consolidate_skill_from_lesson(lesson=str(obj["actionable_lesson"]), game_id=str(obj.get("game_id") or "unknown"), evidence=f"reflection:{rid}", tags=[str(obj.get("game_id") or "unknown"), "reflection"])
        return rid

    def retrieve_reflections(self, *, query: str, game_id: str, top_k: int = 2) -> list[Reflection]:
        if self._retriever is None:
            self._retriever = _make_retriever(self.root)
        self._retriever.refresh(game_id=game_id)
        scored = self._retriever.retrieve(query=query, game_id=game_id, player=None, phase=None, top_k=top_k * 3)
        out: list[Reflection] = []
        seen = set()
        for item in scored:
            if item.source != "reflections":
                continue
            if item.id in seen:
                continue
            seen.add(item.id)
            # Read the full reflection record by id.
            for row in self._read_jsonl(self.reflections_path):
                if row.get("id") == item.id:
                    out.append(Reflection(
                        id=str(row.get("id")), game_id=str(row.get("game_id") or ""), episode_seed=int(row.get("episode_seed") or 0),
                        outcome=str(row.get("outcome") or ""), text=str(row.get("text") or ""),
                        state_keys=list(row.get("state_keys") or []), actionable_lesson=str(row.get("actionable_lesson") or ""),
                        created_at=str(row.get("created_at") or ""),
                    ))
                    break
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------------ insight consolidation (ExpeL)
    def consolidate_insights(self, *, game_id: str, llm) -> list[str]:
        from .insight import InsightExtractor
        extractor = InsightExtractor(llm=llm, memory=self)
        insights = extractor.consolidate(game_id=game_id, recent_experiences=self.recent_experiences(game_id=game_id, limit=300))
        self.mark_insight_consolidated(game_id)
        return insights

    # ------------------------------------------------------------------ retrieval bookkeeping
    def record_retrieval(self, *, query: str, items: list[dict[str, Any]]) -> str:
        hit_id = uuid4().hex[:12]
        self._append_jsonl(self.retrieval_hits_path, {
            "id": hit_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": query[:1000],
            "items": [{"source": x.get("source"), "score": x.get("score"), "id": x.get("id"), "text_preview": str(x.get("text") or "")[:240]} for x in items],
        })
        return hit_id

    # ------------------------------------------------------------------ stats / timeline
    def memory_stats(self) -> dict[str, Any]:
        skills = self._read_jsonl(self.skills_path)
        experiences = self._read_jsonl(self.experiences_path)
        retrievals = self._read_jsonl(self.retrieval_hits_path)
        reflections = self._read_jsonl(self.reflections_path)
        patches = [p for p in self._read_jsonl(self.patches_path) if p.get("status") == "active"]
        avg = 0.0
        if skills:
            avg = sum(float(s.get("score_sum") or 0.0) / max(1, int(s.get("uses") or 0)) for s in skills) / len(skills)
        active = [s for s in skills if s.get("status") in (None, "active", "promoted", "mutated")]
        return {
            "root": str(self.root),
            "experiences": len(experiences),
            "skills": len(skills),
            "active_skills": len(active),
            "reflections": len(reflections),
            "active_patches": len(patches),
            "skill_updates": len(self._read_jsonl(self.skill_history_path)),
            "retrievals": len(retrievals),
            "avg_skill_score": round(avg, 3),
            "rules_bytes": self.rules_path.stat().st_size if self.rules_path.exists() else 0,
            "prompt_overrides_bytes": self.prompt_overrides_path.stat().st_size if self.prompt_overrides_path.exists() else 0,
        }

    def skill_timeline(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.skill_history_path)

    # ------------------------------------------------------------------ low-level io
    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
        return rows

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in rows), encoding="utf-8")

    def _append_jsonl(self, path: Path, obj: dict[str, Any]) -> None:
        obj.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    def _append_skill_history(self, record: dict[str, Any]) -> None:
        record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with self.skill_history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def state_key_from_text(text: str) -> str:
    terms = _terms(text)
    return " ".join(terms[:32]) or "empty_state"


def _terms(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 1]


def _normalize_rule(rule: str) -> str:
    rule = re.sub(r"\s+", " ", (rule or "").strip(" -\n\t"))
    return rule[:600]


def _game_id_from_trigger(trigger: str) -> str:
    match = re.search(r"game:([^\s]+)", trigger or "")
    return match.group(1) if match else ""


def _skill_name(guidance: str, game_id: str) -> str:
    words = _terms(guidance)[:8]
    return f"{game_id}:{' '.join(words)[:80]}" if words else f"{game_id}:general"


def _merge_guidance(old: str, new: str) -> str:
    old = old.strip()
    new = new.strip()
    if not old:
        return new
    if new.lower() in old.lower():
        return old
    return (old + " / " + new)[:900]


def _mutate_via_llm(llm, skill: dict, game_id: str) -> str | None:
    """Ask an LLM to revise a skill's guidance using its failure record (Reflexion)."""
    try:
        system = (
            "You revise a failing gameplay skill into a sharper, more generalizable rule. "
            "Keep it under 600 chars, concrete, and game-appropriate. Return JSON: {\"guidance\": \"...\"}."
        )
        user = (
            f"Game: {game_id}\nSkill: {skill.get('name','')}\n"
            f"Current guidance: {skill.get('guidance','')}\n"
            f"Stats: uses={skill.get('uses',0)} wins={skill.get('wins',0)} losses={skill.get('losses',0)} win_rate={skill.get('win_rate',0)}\n"
            "Rewrite the guidance so the agent avoids the recurring failure mode while keeping what worked."
        )
        raw = llm.complete_json(system=system, user=user, temperature=0.3, max_tokens=400)
        g = str(raw.get("guidance") or "").strip()
        return _normalize_rule(g) if g else None
    except Exception:
        return None


class _RetrieverWrapper:
    """Wraps BM25Retriever with on-demand re-indexing keyed on corpus mtime."""

    def __init__(self, root: Path) -> None:
        from .retrieval import BM25Retriever, build_corpus_from_memory
        self._root = root
        self._bm25 = BM25Retriever()
        self._build = build_corpus_from_memory
        self._indexed_key: tuple = ()

    def refresh(self, *, game_id: str) -> None:
        key = self._mtime_key()
        if key != self._indexed_key:
            self._bm25.indexer.index(self._build(self._root, game_id=game_id))
            self._indexed_key = key

    def retrieve(self, *, query: str, game_id: str, player: int | None = None, phase: str | None = None, top_k: int = 8):
        self.refresh(game_id=game_id)
        return self._bm25.retrieve(query=query, game_id=game_id, player=player, phase=phase, top_k=top_k)

    def _mtime_key(self) -> tuple:
        mtimes = []
        for name in ["experiences.jsonl", "skills.jsonl", "reflections.jsonl", "rules.md"]:
            p = self._root / name
            mtimes.append((name, p.stat().st_mtime if p.exists() else 0.0, p.stat().st_size if p.exists() else 0))
        return tuple(mtimes)


def _make_retriever(root: Path) -> _RetrieverWrapper:
    return _RetrieverWrapper(root)
