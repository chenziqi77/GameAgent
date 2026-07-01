"""Trace schema dataclasses for Phase 1 logging upgrade.

These structures are the canonical on-disk record of every decision the agent
makes, the tool calls it issues, the prompt layers it compiles, and the
evaluation/critic activity that follows. They are emitted to `events.jsonl`
and a dedicated `decision_frames.jsonl` so downstream replay-eval, evidence
graph, and hypothesis harness can rebuild ground truth.

All dataclasses are JSON-serialisable via `asdict()` and use `slots=True` so
they do not silently accept unknown fields (catches typos at construction).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return uuid4().hex[:12]


@dataclass(slots=True)
class ToolTrace:
    """One tool call inside ToolLoop.run()."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    content_preview: str = ""
    latency_ms: float = 0.0
    round_index: int = 0
    error: str | None = None
    id: str = field(default_factory=_short_id)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromptTrace:
    """Per-decision record of the compiled prompt layers + cache hit data."""

    layer_hashes: dict[str, str] = field(default_factory=dict)
    cache_key: str = ""
    prompt_chars: int = 0
    system_chars: int = 0
    user_chars: int = 0
    cached_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_ratio: float = 0.0
    policy_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationTrace:
    """Critic / self-check evaluation result captured per-decision (optional)."""

    source: str = "none"  # "critic_agent" | "self_check_tool" | "none"
    accept: bool = True
    score: float | None = None
    suggested_candidate_id: str | None = None
    rationale: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DecisionFrame:
    """One decision = one (state, candidate_set, llm_call, tool_calls, outcome) tuple.

    This is the unit of replay-eval and the canonical row inserted into the
    `DecisionFrame` table of the Evidence Graph.
    """

    game_id: str
    episode_id: str
    turn: int
    step: int
    state_hash: str
    current_player: int
    candidate_id: str
    action_text: str
    action_index: int
    action_type: str
    legal_actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    plan: str = ""
    retrieved_memory_ids: list[str] = field(default_factory=list)
    used_skill_ids: list[str] = field(default_factory=list)
    used_tool_ids: list[str] = field(default_factory=list)
    policy_version: str = "v0"
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_hit_ratio: float = 0.0
    tool_traces: list[dict[str, Any]] = field(default_factory=list)
    prompt_trace: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    evaluator_overrode: bool = False
    id: str = field(default_factory=_short_id)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeTrace:
    """Aggregated record for one terminal episode."""

    episode_id: str
    game_id: str
    env_id: str
    seed: int | None
    player_id: int
    turns: int
    decision_frame_ids: list[str] = field(default_factory=list)
    rewards: dict[str, float] = field(default_factory=dict)
    outcome: str = ""  # "win" | "loss" | "draw" | "unknown"
    policy_version: str = "v0"
    total_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    critic_report_id: str | None = None
    started_at: str = field(default_factory=_now_iso)
    ended_at: str = field(default_factory=_now_iso)
    id: str = field(default_factory=_short_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ToolTrace",
    "PromptTrace",
    "EvaluationTrace",
    "DecisionFrame",
    "EpisodeTrace",
]
