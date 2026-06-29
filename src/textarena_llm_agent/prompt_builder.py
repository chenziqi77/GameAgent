"""Per-game prompt builder (Reflexion-style dynamic injection + game-theoretic content)."""
from __future__ import annotations

from .game_specs import GAME_SPECS, GameSpec
from .prompts import DECISION_SYSTEM_PROMPT

# Maximum chars for the injected reflection and patch sections in the system prompt.
_REFLECTION_BUDGET = 1200
_PATCH_BUDGET = 1200


class GamePromptBuilder:
    """Builds a game-aware system prompt with rules, game-theoretic principles,
    active prompt patches, a reflection slot, and descriptions of available tools.

    Pattern: Reflexion (Shinn et al. 2023) — verbal self-reflection and per-task
    instruction specialization injected into the prompt context across episodes.
    """

    def __init__(self) -> None:
        pass

    def build_system(
        self,
        *,
        game_id: str,
        phase: str = "mid",
        reflection: str = "",
        active_patches: list[str] | None = None,
        tool_descriptions: list[str] | None = None,
    ) -> str:
        spec: GameSpec = GAME_SPECS.get(game_id)
        if spec is None:
            return DECISION_SYSTEM_PROMPT

        parts: list[str] = [
            f"You are an expert TextArena agent playing {spec.family} (phase: {phase}).",
            "",
            "## Rules",
            spec.rules.strip(),
            "",
            "## Action format",
            spec.action_format.strip() if spec.action_format else "Return exactly one legal bracketed action.",
            "",
            "## Game-theoretic principles",
            spec.game_theoretic_principles.strip() if spec.game_theoretic_principles else spec.strategic_notes.strip(),
            "",
            "## Strategic notes",
            spec.strategic_notes.strip(),
        ]

        patches = list(active_patches or [])
        if patches:
            patch_text = "\n".join(f"- {p.strip()}" for p in patches)[:_PATCH_BUDGET]
            parts.extend(["", "## Active prompt patches (learned)", patch_text])

        if reflection:
            parts.extend(["", "## Reflection from similar past games", reflection.strip()[:_REFLECTION_BUDGET]])

        tools = list(tool_descriptions or [])
        if tools:
            parts.extend(["", "## Available tools", "You may call these tools to inform your decision before returning your final action."])
            parts.extend(f"- {t}" for t in tools)

        parts.extend([
            "",
            "## Decision contract",
            (
                "Choose exactly one candidate_id from CANDIDATE_ACTIONS_JSON. "
                "Never invent an action outside the candidate set. Never rely on hidden information "
                "absent from the visible state. Return exactly one JSON object:"
            ),
            "{",
            '  "candidate_id": "C0",',
            '  "action": "[legal action text]",',
            '  "confidence": 0.0-1.0,',
            '  "rationale": "brief strategic explanation grounded in the principles above",',
            '  "plan": "what this action sets up next"',
            "}",
        ])
        return "\n".join(parts)


def patches_for(memory, game_id: str) -> list[str]:
    """Return the active prompt-patch texts for a game from an EvolvingMemory."""
    try:
        records = memory.active_prompt_patches(game_id)
    except Exception:
        return []
    out: list[str] = []
    for rec in records:
        if isinstance(rec, dict):
            out.append(str(rec.get("patch_text") or rec.get("patch") or ""))
        else:
            out.append(str(getattr(rec, "patch_text", "")))
    return [p for p in out if p]
