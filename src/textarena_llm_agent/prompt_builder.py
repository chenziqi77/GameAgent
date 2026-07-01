"""Per-game prompt builder — thin wrapper over the four-layer PromptCompiler.

Kept as a stable public surface for tests / external callers that import
``GamePromptBuilder``. New code should call ``PromptCompiler`` directly so the
caller can observe per-layer hashes / token counts.

Pattern: Reflexion (Shinn et al. 2023) — verbal self-reflection and per-task
instruction specialization injected into the prompt context across episodes.
The four-layer split (STATIC_PREFIX / GAME_STATIC / POLICY_STATIC / USER_DYNAMIC)
makes KV-cache prefixes stable within (game_id, policy_version) so the LLM
provider can reuse prefix-cache across decisions.
"""
from __future__ import annotations

from .prompt_compiler import PromptCompiler


class GamePromptBuilder:
    """Builds the system prompt for a game by composing the first three layers
    of the prompt compiler (STATIC_PREFIX + GAME_STATIC + POLICY_STATIC).

    The fourth layer (USER_DYNAMIC) is owned by ContextBudgeter and assembled
    in ``Agent.decide`` from state / memory / candidates / reflection.
    """

    def __init__(self) -> None:
        self._compiler = PromptCompiler()

    def build_system(
        self,
        *,
        game_id: str,
        phase: str = "mid",
        reflection: str = "",  # accepted for API back-compat; surfaced via USER_DYNAMIC
        active_patches: list[str] | None = None,
        tool_descriptions: list[str] | None = None,
        policy_version: str = "v0",
    ) -> str:
        L1 = self._compiler.static_prefix()
        L2 = self._compiler.game_static(game_id=game_id, phase=phase)
        L3 = self._compiler.policy_static(
            active_patches=active_patches,
            tool_descriptions=tool_descriptions,
            policy_version=policy_version,
        )
        system = L1 + "\n" + L2 + "\n" + L3
        if reflection.strip():
            # Reflection was historically part of the system prompt. Preserve
            # that surface so legacy callers that only inspect ``build_system``
            # output keep seeing it. New code routes reflection through
            # USER_DYNAMIC via PromptCompiler.compile().
            system += "\n## Reflection from similar past games\n" + reflection.strip()[:1200] + "\n"
        return system


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
