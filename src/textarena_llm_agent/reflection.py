"""Episode-level self-reflection + skill mutation (Reflexion)."""
from __future__ import annotations

import json
from typing import Any

from .llm import DecisionLLM
from .memory import EvolvingMemory, Reflection
from .prompts import REFLECTION_SYSTEM_PROMPT
from .evaluator import fallback_terminal_lesson


class Reflector:
    """Produce a concise, actionable reflection once per terminal episode.

    Pattern: Reflexion (Shinn et al. 2023) — verbal self-reflection stored across
    episodes and retrieved when a similar state arises; failing skills are revised.
    """

    def __init__(self, *, llm: DecisionLLM, memory: EvolvingMemory) -> None:
        self.llm = llm
        self.memory = memory

    def reflect_episode(self, *, game_id: str, seed: int, outcome: str,
                        transitions: list[dict[str, Any]]) -> Reflection:
        compact = _compact_transitions(transitions)
        user = (
            f"Game: {game_id}\nSeed: {seed}\nTerminal outcome: {outcome}\n\n"
            f"Transitions (state summary, action, evaluator critique, reward):\n{compact}"
        )
        try:
            raw = self.llm.complete_json(system=REFLECTION_SYSTEM_PROMPT, user=user, temperature=0.2, max_tokens=500)
        except Exception as exc:  # never let reflection break the episode loop
            raw = {"_error": str(exc)}
        text = str(raw.get("text") or f"Episode ended with outcome {outcome}; no detailed reflection available.")
        lesson = str(raw.get("actionable_lesson") or "")
        if not lesson:
            last_reward = None
            last_critique = ""
            for transition in reversed(transitions):
                if last_reward is None and transition.get("reward") is not None:
                    try:
                        last_reward = float(transition.get("reward"))
                    except Exception:
                        last_reward = None
                evaluation = transition.get("evaluation")
                if not last_critique and isinstance(evaluation, dict):
                    last_critique = str(evaluation.get("critique") or "")
                if last_reward is not None and last_critique:
                    break
            lesson = fallback_terminal_lesson(game_id=game_id, outcome=outcome, reward=last_reward, critique=last_critique)
        state_keys = list(raw.get("state_keys") or [])
        if not isinstance(state_keys, list):
            state_keys = []
        return Reflection(
            id="",  # memory assigns
            game_id=game_id,
            episode_seed=seed,
            outcome=outcome,
            text=text,
            state_keys=[str(s) for s in state_keys][:8],
            actionable_lesson=lesson,
        )


def _compact_transitions(transitions: list[dict[str, Any]]) -> str:
    lines = []
    for i, t in enumerate(transitions[-12:]):
        line = {
            "i": i,
            "turn": t.get("turn"),
            "action": t.get("action_text") or t.get("action"),
            "critique": (t.get("evaluation") or {}).get("critique") if isinstance(t.get("evaluation"), dict) else t.get("critique"),
            "reward": t.get("reward"),
            "outcome": t.get("outcome"),
        }
        lines.append(json.dumps(line, ensure_ascii=False, default=str))
    return "\n".join(lines)
