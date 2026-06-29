"""ExpeL-style insight extraction across a batch of experiences."""
from __future__ import annotations

import json
from typing import Any

from .llm import DecisionLLM
from .memory import EvolvingMemory
from .prompts import INSIGHT_EXTRACTION_SYSTEM_PROMPT


class InsightExtractor:
    """Consolidate generalizable insights from a batch of experiences.

    Pattern: ExpeL (Zhao et al. 2024) — group experiences by success/failure, extract
    a stream of generalizable rules, dedup against existing skills, store as durable
    skills with evidence. Runs only when enough new experiences have accumulated.
    """

    def __init__(self, *, llm: DecisionLLM, memory: EvolvingMemory, min_episodes: int = 8) -> None:
        self.llm = llm
        self.memory = memory
        self.min_episodes = min_episodes

    def consolidate(self, *, game_id: str, recent_experiences: list[dict[str, Any]]) -> list[str]:
        if len(recent_experiences) < self.min_episodes:
            return []
        wins = [e for e in recent_experiences if str(e.get("outcome") or "").lower() in {"win", "terminal_win"} or (e.get("reward") is not None and float(e.get("reward") or 0) > 0)]
        losses = [e for e in recent_experiences if str(e.get("outcome") or "").lower() in {"loss", "terminal_loss"} or (e.get("reward") is not None and float(e.get("reward") or 0) < 0)]
        if not wins and not losses:
            return []
        evidence_w = [f"experience:{e.get('id')}" for e in wins if e.get("id")]
        evidence_l = [f"experience:{e.get('id')}" for e in losses if e.get("id")]
        user = (
            f"Game: {game_id}\n\n"
            f"=== WINNING experiences ({len(wins)}) ===\n{json.dumps(_skim(wins), ensure_ascii=False, default=str)}\n\n"
            f"=== LOSING experiences ({len(losses)}) ===\n{json.dumps(_skim(losses), ensure_ascii=False, default=str)}\n"
        )
        try:
            raw = self.llm.complete_json(system=INSIGHT_EXTRACTION_SYSTEM_PROMPT, user=user, temperature=0.2, max_tokens=900)
        except Exception:
            return []
        insights = raw.get("insights") if isinstance(raw, dict) else None
        if not isinstance(insights, list):
            return []
        written: list[str] = []
        for ins in insights:
            if not isinstance(ins, dict):
                continue
            guidance = str(ins.get("guidance") or "").strip()
            if not guidance:
                continue
            polarity = str(ins.get("polarity") or "positive")
            ev = evidence_w if polarity == "positive" else evidence_l
            tags = [game_id, f"player_{_common_player(ins, recent_experiences)}", polarity]
            sid = self.memory.consolidate_skill_from_lesson(
                lesson=guidance, game_id=game_id, evidence=",".join(ev[:5]) or f"insight:{game_id}",
                tags=tags, importance=3.0,
            )
            if sid:
                written.append(guidance)
        return written


def _skim(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows[:15]:
        out.append({
            "id": r.get("id"),
            "action_text": r.get("action_text"),
            "outcome": r.get("outcome"),
            "reward": r.get("reward"),
            "lesson": r.get("lesson"),
            "critique": r.get("critique"),
            "state_key": (r.get("state_key") or "")[:120],
        })
    return out


def _common_player(ins: dict, rows: list[dict[str, Any]]) -> int:
    for r in rows:
        try:
            return int(r.get("player") or 0)
        except Exception:
            continue
    return 0
