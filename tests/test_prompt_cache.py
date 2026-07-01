"""Tests for the Phase 6 four-layer prompt compiler + KV-cache discipline.

The four layers are:

    STATIC_PREFIX  — agent identity / decision contract; never changes
    GAME_STATIC    — rules / action format / principles; changes only on game switch
    POLICY_STATIC  — active patches + tool descriptions; changes on policy bump
    USER_DYNAMIC   — state / memory / candidates / reflection; changes every decision

These tests pin the cache discipline so a regression that, say, accidentally
folds policy_version into STATIC_PREFIX would fail loudly. Token estimates are
asserted only with > 0 since the exact char/token ratio is implementation
detail.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from textarena_llm_agent.context_packet import ContextBudgeter
from textarena_llm_agent.prompt_builder import GamePromptBuilder
from textarena_llm_agent.prompt_compiler import CompiledPrompt, PromptCompiler


# ----------------------------------------------------------------- fixtures


def _state_text(turn: int) -> str:
    return json.dumps({
        "visible_state": "...board...",
        "to_move": 1,
        "turn": turn,
    }, ensure_ascii=False)


def _candidates() -> list[dict[str, Any]]:
    return [
        {"candidate_id": "C0", "action_text": "[1]", "action_type": "place",
         "score": 0.7, "reasons": ["center is strong"]},
        {"candidate_id": "C1", "action_text": "[2]", "action_type": "place",
         "score": 0.4, "reasons": ["corner backup"]},
    ]


@pytest.fixture()
def compiler() -> PromptCompiler:
    return PromptCompiler()


# ----------------------------------------------------------------- layer 1


def test_static_prefix_is_stable_across_games_and_policies(compiler: PromptCompiler) -> None:
    """STATIC_PREFIX must NOT depend on game / policy / patches / tools."""
    a = compiler.compile(
        game_id="TicTacToe", phase="opening", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="mem A", reflection="",
        active_patches=["patch one"], tool_descriptions=["tool_alpha"],
    )
    b = compiler.compile(
        game_id="KuhnPoker", phase="mid", policy_version="v7",
        state_text=_state_text(3), candidates=_candidates(),
        memory_excerpt="entirely different",
        reflection="reflection text",
        active_patches=["different patch"], tool_descriptions=["tool_beta"],
    )
    assert a.static_prefix == b.static_prefix
    assert a.static_prefix_hash == b.static_prefix_hash
    assert a.static_prefix_tokens > 0


# ----------------------------------------------------------------- layer 2


def test_game_static_stable_within_game_changes_across_games(compiler: PromptCompiler) -> None:
    """GAME_STATIC must be identical when game_id+phase are identical, and
    different across different games (rules differ)."""
    a = compiler.compile(
        game_id="TicTacToe", phase="opening", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="", reflection="",
        active_patches=["patch"], tool_descriptions=["tool"],
    )
    a2 = compiler.compile(
        game_id="TicTacToe", phase="opening", policy_version="v9",
        state_text=_state_text(7), candidates=_candidates(),
        memory_excerpt="totally different memory", reflection="",
        active_patches=["patch X"], tool_descriptions=["tool Y"],
    )
    b = compiler.compile(
        game_id="KuhnPoker", phase="opening", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="", reflection="",
        active_patches=["patch"], tool_descriptions=["tool"],
    )
    assert a.game_static_hash == a2.game_static_hash, "GAME_STATIC must not depend on policy/state"
    assert a.game_static_hash != b.game_static_hash, "TTT and KuhnPoker must differ"


def test_game_static_phase_changes_hash(compiler: PromptCompiler) -> None:
    a = compiler.compile(
        game_id="TicTacToe", phase="opening", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="", reflection="",
    )
    b = compiler.compile(
        game_id="TicTacToe", phase="endgame", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="", reflection="",
    )
    assert a.game_static_hash != b.game_static_hash


# ----------------------------------------------------------------- layer 3


def test_policy_static_changes_only_on_policy_bump(compiler: PromptCompiler) -> None:
    """A policy version bump must change POLICY_STATIC while leaving
    STATIC_PREFIX and GAME_STATIC identical (cache survives the bump)."""
    common = dict(
        game_id="TicTacToe", phase="mid",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="m", reflection="",
        active_patches=["pat1"], tool_descriptions=["t1"],
    )
    v0 = compiler.compile(policy_version="v0", **common)
    v1 = compiler.compile(policy_version="v1", **common)
    assert v0.static_prefix_hash == v1.static_prefix_hash
    assert v0.game_static_hash == v1.game_static_hash
    assert v0.policy_static_hash != v1.policy_static_hash


def test_policy_static_changes_when_patches_change(compiler: PromptCompiler) -> None:
    base = dict(
        game_id="TicTacToe", phase="mid", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="", reflection="",
        tool_descriptions=["tool_a"],
    )
    a = compiler.compile(active_patches=["patch A"], **base)
    b = compiler.compile(active_patches=["patch A", "patch B"], **base)
    assert a.policy_static_hash != b.policy_static_hash
    # but STATIC_PREFIX + GAME_STATIC are unchanged
    assert a.static_prefix_hash == b.static_prefix_hash
    assert a.game_static_hash == b.game_static_hash


def test_policy_static_changes_when_tools_change(compiler: PromptCompiler) -> None:
    base = dict(
        game_id="TicTacToe", phase="mid", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="", reflection="",
        active_patches=["pat"],
    )
    a = compiler.compile(tool_descriptions=["tool_alpha"], **base)
    b = compiler.compile(tool_descriptions=["tool_alpha", "tool_beta"], **base)
    assert a.policy_static_hash != b.policy_static_hash


# ----------------------------------------------------------------- layer 4


def test_user_dynamic_changes_each_decision(compiler: PromptCompiler) -> None:
    common = dict(
        game_id="TicTacToe", phase="mid", policy_version="v0",
        candidates=_candidates(),
        active_patches=["pat"], tool_descriptions=["tool"],
    )
    a = compiler.compile(state_text=_state_text(0), memory_excerpt="m1", reflection="r1", **common)
    b = compiler.compile(state_text=_state_text(1), memory_excerpt="m2", reflection="r2", **common)
    # STATIC + GAME + POLICY stable
    assert a.static_prefix_hash == b.static_prefix_hash
    assert a.game_static_hash == b.game_static_hash
    assert a.policy_static_hash == b.policy_static_hash
    # USER_DYNAMIC differs
    assert a.user_dynamic_hash != b.user_dynamic_hash
    assert a.user != b.user


# ----------------------------------------------------------------- compose / packet


def test_compile_produces_system_user_pair_compatible_with_legacy_packet(
        compiler: PromptCompiler) -> None:
    compiled = compiler.compile(
        game_id="TicTacToe", phase="mid", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="memory text", reflection="reflection text",
        active_patches=["patch1"], tool_descriptions=["tool1"],
    )
    # system = L1 + L2 + L3 (in that order, separated by blank lines)
    assert compiled.system.startswith(compiled.static_prefix)
    assert compiled.game_static in compiled.system
    assert compiled.policy_static in compiled.system
    assert compiled.user == compiled.user_dynamic
    # sections + user are populated by the budgeter
    assert "memory" in compiled.sections
    assert "candidates" in compiled.sections
    assert "VISIBLE_STATE_JSON" in compiled.user
    assert "CANDIDATE_ACTIONS_JSON" in compiled.user
    # packet bridge populates layer metadata
    pkt = compiled.to_packet()
    assert pkt.layer_hashes["static_prefix"] == compiled.static_prefix_hash
    assert pkt.layer_hashes["game_static"] == compiled.game_static_hash
    assert pkt.layer_hashes["policy_static"] == compiled.policy_static_hash
    assert pkt.layer_hashes["user_dynamic"] == compiled.user_dynamic_hash
    assert pkt.stable_prefix_tokens == (
        compiled.static_prefix_tokens + compiled.game_static_tokens
        + compiled.policy_static_tokens
    )
    assert pkt.layer_tokens["user_dynamic"] > 0


def test_total_tokens_and_stable_prefix_tokens(compiler: PromptCompiler) -> None:
    c = compiler.compile(
        game_id="TicTacToe", phase="mid", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="m", reflection="",
        active_patches=["p"], tool_descriptions=["t"],
    )
    assert c.total_tokens == (
        c.static_prefix_tokens + c.game_static_tokens
        + c.policy_static_tokens + c.user_dynamic_tokens
    )
    assert c.stable_prefix_tokens == (
        c.static_prefix_tokens + c.game_static_tokens + c.policy_static_tokens
    )
    assert c.stable_prefix_tokens < c.total_tokens


# ----------------------------------------------------------------- legacy compatibility


def test_legacy_game_prompt_builder_still_emits_system() -> None:
    """GamePromptBuilder is preserved as a thin wrapper so older callers that
    only need the system prompt keep working."""
    builder = GamePromptBuilder()
    out = builder.build_system(
        game_id="TicTacToe", phase="mid",
        reflection="some reflection",
        active_patches=["learned patch"],
        tool_descriptions=["count_open_lines: counts rows/cols/diags"],
        policy_version="v0",
    )
    assert "TicTacToe" in out or "Game" in out
    assert "Reflection" in out  # legacy callers expect reflection in system
    assert "learned patch" in out
    assert "count_open_lines" in out


# ----------------------------------------------------------------- cache simulation


def test_two_decisions_same_game_policy_share_full_stable_prefix(
        compiler: PromptCompiler) -> None:
    """Simulate two consecutive decisions in the same episode: the entire
    system prompt MUST be byte-identical, which is exactly what an LLM
    provider's prefix cache needs in order to bill ``cached_tokens > 0``."""
    base = dict(
        game_id="TicTacToe", phase="mid", policy_version="v0",
        candidates=_candidates(),
        active_patches=["pat_a"], tool_descriptions=["tool_x"],
    )
    decision_1 = compiler.compile(state_text=_state_text(0),
                                    memory_excerpt="m1", reflection="", **base)
    decision_2 = compiler.compile(state_text=_state_text(1),
                                    memory_excerpt="m2", reflection="", **base)
    assert decision_1.system == decision_2.system
    assert decision_1.user != decision_2.user


def test_layer_hashes_are_deterministic(compiler: PromptCompiler) -> None:
    args = dict(
        game_id="TicTacToe", phase="mid", policy_version="v0",
        state_text=_state_text(0), candidates=_candidates(),
        memory_excerpt="m", reflection="r",
        active_patches=["p"], tool_descriptions=["t"],
    )
    a = compiler.compile(**args)
    b = compiler.compile(**args)
    assert a.layer_hashes() == b.layer_hashes()
    assert a.system == b.system
    assert a.user == b.user
