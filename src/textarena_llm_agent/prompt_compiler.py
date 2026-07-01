"""Four-layer prompt compiler for KV-cache friendliness.

Per 改进需求.pdf §3 "提示词四层"：将提示词拆为 4 段，使得 KV cache 在
游戏/策略/回合三个时间尺度上分别命中：

    STATIC_PREFIX  — agent identity + decision contract; never changes
    GAME_STATIC    — rules / action format / principles; changes only on game switch
    POLICY_STATIC  — active prompt patches + tool descriptions; changes on policy bump
    USER_DYNAMIC   — state / memory / candidates / reflection; changes every decision

The first three are concatenated into the system prompt (stable suffix-of-prefix
across calls within the same policy_version + game), and the fourth becomes the
user prompt.

Each layer has a stable hash (sha1 of the rendered text) so callers can verify
cache discipline: two consecutive decisions on the same (game_id, policy_version)
MUST share the same ``static_prefix_hash`` / ``game_static_hash`` /
``policy_static_hash``. A policy bump must keep STATIC_PREFIX + GAME_STATIC
identical while flipping POLICY_STATIC.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .context_packet import ContextBudgeter, DecisionContextPacket
from .game_specs import GAME_SPECS, GameSpec
from .prompts import DECISION_SYSTEM_PROMPT

# Budgets identical to legacy GamePromptBuilder so callers see no regression.
_REFLECTION_BUDGET = 1200
_PATCH_BUDGET = 1200


# ---------------------------------------------------------------------------
# Layer 1: STATIC_PREFIX — identity + decision contract, never changes
# ---------------------------------------------------------------------------

_STATIC_PREFIX = (
    "You are an expert TextArena agent. You play turn-based games against an opponent.\n"
    "\n"
    "## Decision contract\n"
    "Choose exactly one candidate_id from CANDIDATE_ACTIONS_JSON. "
    "Never invent an action outside the candidate set. Never rely on hidden information "
    "absent from the visible state. Return exactly one JSON object:\n"
    "{\n"
    '  "candidate_id": "C0",\n'
    '  "action": "[legal action text]",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "rationale": "brief strategic explanation grounded in the principles above",\n'
    '  "plan": "what this action sets up next"\n'
    "}\n"
)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (chars/4) — good enough for cache observation."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# CompiledPrompt — the four-layer artefact
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompiledPrompt:
    """Result of ``PromptCompiler.compile`` — four layers plus combined views.

    The system prompt is ``static_prefix + game_static + policy_static`` so the
    LLM provider's prefix cache can hit on (static_prefix), then again on
    (static_prefix + game_static), then again on (static_prefix + game_static +
    policy_static). The user prompt is ``user_dynamic``.
    """
    static_prefix: str
    game_static: str
    policy_static: str
    user_dynamic: str
    # cache observation
    static_prefix_hash: str
    game_static_hash: str
    policy_static_hash: str
    user_dynamic_hash: str
    # token estimates per layer
    static_prefix_tokens: int
    game_static_tokens: int
    policy_static_tokens: int
    user_dynamic_tokens: int
    # combined views (cached on construction)
    system: str = ""
    user: str = ""
    # context-packet sections (memory/state/candidates/reflection/patches) so
    # downstream emitters keep working.
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return (self.static_prefix_tokens + self.game_static_tokens
                + self.policy_static_tokens + self.user_dynamic_tokens)

    @property
    def stable_prefix_tokens(self) -> int:
        """Tokens that *should* hit the provider's prefix cache on repeat decisions
        within the same (game_id, policy_version)."""
        return (self.static_prefix_tokens + self.game_static_tokens
                + self.policy_static_tokens)

    def layer_hashes(self) -> dict[str, str]:
        return {
            "static_prefix": self.static_prefix_hash,
            "game_static": self.game_static_hash,
            "policy_static": self.policy_static_hash,
            "user_dynamic": self.user_dynamic_hash,
        }

    def to_packet(self) -> DecisionContextPacket:
        """Bridge back to the legacy DecisionContextPacket shape for the agent
        and tracing layers that already consume it."""
        return DecisionContextPacket(
            system=self.system,
            user=self.user,
            sections=dict(self.sections),
            budget_used=len(self.user),
            layer_hashes=self.layer_hashes(),
            layer_tokens={
                "static_prefix": self.static_prefix_tokens,
                "game_static": self.game_static_tokens,
                "policy_static": self.policy_static_tokens,
                "user_dynamic": self.user_dynamic_tokens,
            },
            stable_prefix_tokens=self.stable_prefix_tokens,
        )


# ---------------------------------------------------------------------------
# PromptCompiler
# ---------------------------------------------------------------------------


class PromptCompiler:
    """Compose the four prompt layers and emit a CompiledPrompt.

    Stateless by design — every call recomputes layers from inputs. Callers that
    want true caching at the application level can memoize on
    ``(game_id, phase, policy_version, patches_hash, tools_hash)`` for the
    system-prompt-side layers.
    """

    def __init__(self, *, context_budgeter: ContextBudgeter | None = None) -> None:
        self.budgeter = context_budgeter or ContextBudgeter()

    # -- layer 1 -------------------------------------------------------------
    def static_prefix(self) -> str:
        return _STATIC_PREFIX

    # -- layer 2 -------------------------------------------------------------
    def game_static(self, *, game_id: str, phase: str = "mid") -> str:
        spec: GameSpec | None = GAME_SPECS.get(game_id)
        if spec is None:
            return f"## Game\nUnknown game: {game_id}\n(phase: {phase})\n" + DECISION_SYSTEM_PROMPT
        parts: list[str] = [
            f"## Game: {spec.family}",
            f"(phase: {phase})",
            "",
            "## Rules",
            spec.rules.strip(),
            "",
            "## Action format",
            (spec.action_format.strip() if spec.action_format
             else "Return exactly one legal bracketed action."),
            "",
            "## Game-theoretic principles",
            (spec.game_theoretic_principles.strip()
             if spec.game_theoretic_principles else spec.strategic_notes.strip()),
            "",
            "## Strategic notes",
            spec.strategic_notes.strip(),
        ]
        return "\n".join(parts) + "\n"

    # -- layer 3 -------------------------------------------------------------
    def policy_static(self, *, active_patches: list[str] | None = None,
                      tool_descriptions: list[str] | None = None,
                      policy_version: str = "v0") -> str:
        parts: list[str] = [f"## Policy version: {policy_version}"]

        patches = [p.strip() for p in (active_patches or []) if p and p.strip()]
        if patches:
            patch_text = "\n".join(f"- {p}" for p in patches)[:_PATCH_BUDGET]
            parts.extend(["", "## Active prompt patches (learned)", patch_text])

        tools = [t for t in (tool_descriptions or []) if t]
        if tools:
            parts.extend([
                "",
                "## Available tools",
                "You may call these tools to inform your decision before returning your final action.",
            ])
            parts.extend(f"- {t}" for t in tools)

        return "\n".join(parts) + "\n"

    # -- layer 4 (delegated to ContextBudgeter for budget allocation) --------
    def user_dynamic(self, *, state_text: str, candidates: list[dict[str, Any]],
                     memory_excerpt: str, reflection: str,
                     patches: list[str] | None = None) -> tuple[str, dict[str, str]]:
        # We reuse ContextBudgeter so all the existing budget knobs / truncation
        # logic continues to work — but we discard its ``system`` field; that
        # role is now owned by the layered system prompt.
        packet = self.budgeter.build(
            spec_system="",  # not used; layered system is composed by compile()
            state_text=state_text,
            candidates=candidates,
            memory_excerpt=memory_excerpt,
            reflection=reflection,
            patches=list(patches or []),
        )
        return packet.user, dict(packet.sections)

    # -- compose -------------------------------------------------------------
    def compile(
        self,
        *,
        game_id: str,
        phase: str = "mid",
        policy_version: str = "v0",
        state_text: str,
        candidates: list[dict[str, Any]],
        memory_excerpt: str = "",
        reflection: str = "",
        active_patches: list[str] | None = None,
        tool_descriptions: list[str] | None = None,
    ) -> CompiledPrompt:
        L1 = self.static_prefix()
        L2 = self.game_static(game_id=game_id, phase=phase)
        L3 = self.policy_static(
            active_patches=active_patches,
            tool_descriptions=tool_descriptions,
            policy_version=policy_version,
        )
        L4, sections = self.user_dynamic(
            state_text=state_text, candidates=candidates,
            memory_excerpt=memory_excerpt, reflection=reflection,
            patches=list(active_patches or []),
        )
        system = L1 + "\n" + L2 + "\n" + L3
        compiled = CompiledPrompt(
            static_prefix=L1,
            game_static=L2,
            policy_static=L3,
            user_dynamic=L4,
            static_prefix_hash=_sha1(L1),
            game_static_hash=_sha1(L2),
            policy_static_hash=_sha1(L3),
            user_dynamic_hash=_sha1(L4),
            static_prefix_tokens=_estimate_tokens(L1),
            game_static_tokens=_estimate_tokens(L2),
            policy_static_tokens=_estimate_tokens(L3),
            user_dynamic_tokens=_estimate_tokens(L4),
            system=system,
            user=L4,
            sections=sections,
        )
        return compiled


__all__ = ["PromptCompiler", "CompiledPrompt"]
