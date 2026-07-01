"""Phase 1 wiring tests: DecisionFrame / EpisodeTrace are emitted with all
required fields, and Phase 2 mirrors them into the evidence graph.

This combines Phase 1's logging contract with Phase 2's evidence-graph mirror
so we keep the wiring tested as a single unit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from textarena_llm_agent import TextArenaAgentConfig, TextArenaDecisionAgent
from textarena_llm_agent.llm import HeuristicLLM


def _make_agent(tmp_path: Path) -> TextArenaDecisionAgent:
    cfg = TextArenaAgentConfig(
        memory_dir=str(tmp_path / "mem"),
        trace_dir=str(tmp_path / "trace"),
        enable_tool_synthesis=False,
        reflection_enabled=False,
        use_llm=False,
        policy_version="v0-test",
    )
    return TextArenaDecisionAgent(cfg, llm=HeuristicLLM(), evaluator_llm=HeuristicLLM())


def test_decision_frame_emits_required_fields(tmp_path: Path) -> None:
    textarena = pytest.importorskip("textarena")
    env = textarena.make("TicTacToe-v0")
    env.reset(num_players=2, seed=0)

    agent = _make_agent(tmp_path)
    out = agent.decide(env)
    assert out is not None
    # Phase 1 contract on the returned Decision object:
    assert out.episode_id and len(out.episode_id) == 12
    assert out.state_hash and len(out.state_hash) >= 8
    assert out.policy_version == "v0-test"
    assert out.latency_ms > 0
    assert isinstance(out.legal_actions, list) and len(out.legal_actions) == 9  # empty 3x3

    frames_path = Path(agent.tracer.decision_frames_path)
    lines = [json.loads(l) for l in frames_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    frame = lines[0]
    for k in ("id", "episode_id", "game_id", "state_hash", "candidate_id",
              "action_text", "legal_actions", "policy_version", "latency_ms",
              "prompt_tokens", "completion_tokens", "cached_tokens"):
        assert k in frame, f"missing key in DecisionFrame: {k}"


def test_decision_frame_is_mirrored_into_evidence_graph(tmp_path: Path) -> None:
    textarena = pytest.importorskip("textarena")
    env = textarena.make("TicTacToe-v0")
    env.reset(num_players=2, seed=0)

    agent = _make_agent(tmp_path)
    out = agent.decide(env)
    assert out is not None
    g = agent.evidence_graph
    assert g is not None
    # DecisionFrame is in the graph and the parent episode was auto-created.
    # The canonical game_id (used inside the agent) strips the textarena -vN suffix.
    df_rows = g.replay_targets(out.game_id, limit=10)
    assert any(r["episode_id"] == out.episode_id for r in df_rows)
    ep_row = g.get_node("episode", out.episode_id)
    assert ep_row is not None
    g.close()
