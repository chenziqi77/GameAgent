"""Tests for the Phase 4 EpisodeCriticAgent.

The critic owns post-episode evolution. Game agent never overrides its own
action mid-episode anymore — instead, after each terminal episode the critic
runs a bounded tool loop and writes (a) skill proposals through SkillManager,
(b) a CriticReport row to the evidence graph, (c) SUPPORTS edges from the
report to each proposed skill_version.

These tests use HeuristicLLM so we exercise the deterministic ``_run_fallback``
path, which guarantees:
  * at least one ``critic_tool_call`` event per episode
  * at least one CriticReport persisted via graph.ingest_critic_report
  * one SUPPORTS edge from the report to a freshly-proposed skill version
    when there are >= 2 evidence memory ids and no agent-side proposal yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from textarena_llm_agent.critic_agent import (
    CRITIC_TOOL_SCHEMA,
    CriticReport,
    EpisodeCriticAgent,
)
from textarena_llm_agent.evidence_graph import EvidenceGraph
from textarena_llm_agent.llm import HeuristicLLM
from textarena_llm_agent.skill_manager import SkillManager, SkillStatus


# ----------------------------------------------------------------- fixtures

def _make_graph(tmp_path: Path) -> EvidenceGraph:
    return EvidenceGraph(tmp_path / "g.sqlite")


def _seed_episode_with_frames_and_memories(
    graph: EvidenceGraph, *, episode_id: str, game_id: str = "TTT",
    n_frames: int = 3,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Seed an episode + N decision_frames + N memory rows + PRODUCED edges.

    Returns (frame_ids, memory_ids, transitions) so tests can feed transitions
    straight into ``critic.run`` and assert on the resulting state of the graph.
    """
    graph.add_node("episode", episode_id, game_id=game_id, env_id=game_id, seed=0,
                   outcome="terminal_loss", policy_version="v0", attrs={})
    frame_ids: list[str] = []
    memory_ids: list[str] = []
    transitions: list[dict[str, Any]] = []
    for i in range(n_frames):
        fid = f"frame_{episode_id}_{i}"
        mid = f"mem_{episode_id}_{i}"
        graph.add_node("decision_frame", fid, episode_id=episode_id, game_id=game_id,
                       state_hash=f"hash{i}", turn=i, step=i, candidate_id=f"c{i}",
                       action_text=f"play {i}", policy_version="v0", attrs={})
        graph.add_edge("episode", episode_id, "CONTAINS", "decision_frame", fid)
        graph.add_node("memory", mid, kind="experience", game_id=game_id, player=0,
                       do_not_learn=0, attrs={"lesson": f"observed pattern {i}"})
        graph.add_edge("decision_frame", fid, "PRODUCED", "memory", mid)
        frame_ids.append(fid)
        memory_ids.append(mid)
        transitions.append({
            "turn": i, "action_text": f"play {i}",
            "decision_frame_id": fid,
            "reward": -1.0 if i == n_frames - 1 else None,
            "outcome": "terminal_loss" if i == n_frames - 1 else None,
            "evaluation": {"critique": "", "lesson": ""},
        })
    return frame_ids, memory_ids, transitions


# ----------------------------------------------------------------- core tests


def test_critic_run_emits_at_least_one_tool_call_event(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    episode_id = "ep_emit"
    _, _, transitions = _seed_episode_with_frames_and_memories(graph, episode_id=episode_id)
    mgr = SkillManager(graph, policy_version="v0")
    events: list[tuple[str, dict[str, Any]]] = []
    critic = EpisodeCriticAgent(
        llm=HeuristicLLM(), graph=graph, skill_manager=mgr,
        emit=lambda ev, payload: events.append((ev, payload)),
    )
    report = critic.run(
        episode_id=episode_id, game_id="TTT", outcome="terminal_loss",
        transitions=transitions, policy_version="v0", player=0,
        rewards={"0": -1.0, "1": 1.0},
    )
    assert isinstance(report, CriticReport)
    # The fallback path must produce >= 1 critic_tool_call (analyze_episode_trace).
    tool_call_events = [e for e in events if e[0] == "critic_tool_call"]
    assert len(tool_call_events) >= 1, f"events: {events}"
    # And the report itself must contain that call.
    assert len(report.tool_calls) >= 1
    graph.close()


def test_critic_persists_report_with_summarized_as_edge(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    episode_id = "ep_report"
    _, _, transitions = _seed_episode_with_frames_and_memories(graph, episode_id=episode_id)
    mgr = SkillManager(graph, policy_version="v0")
    critic = EpisodeCriticAgent(llm=HeuristicLLM(), graph=graph, skill_manager=mgr)
    report = critic.run(
        episode_id=episode_id, game_id="TTT", outcome="terminal_loss",
        transitions=transitions, policy_version="v0", player=0,
    )
    # Exactly one critic_report node for this episode.
    rows = graph.query(
        "SELECT id, episode_id FROM critic_report WHERE episode_id = ?",
        (episode_id,),
    )
    assert len(rows) == 1
    assert rows[0]["id"] == report.id
    # SUMMARIZED_AS edge from episode -> critic_report.
    edges = graph.query(
        "SELECT * FROM evidence_edges WHERE src_type='episode' AND src_id=? "
        "AND edge='SUMMARIZED_AS' AND dst_type='critic_report' AND dst_id=?",
        (episode_id, report.id),
    )
    assert len(edges) == 1
    graph.close()


def test_critic_proposes_skill_with_supports_edges(tmp_path: Path) -> None:
    """When the critic proposes a skill, the lifecycle invariants must hold:

    - SkillManager records a new proposed skill_version
    - The proposed version has SUPPORTS edges from each evidence memory
    - The CriticReport node has a SUPPORTS edge to the proposed skill_version
    """
    graph = _make_graph(tmp_path)
    episode_id = "ep_propose"
    _, memory_ids, transitions = _seed_episode_with_frames_and_memories(
        graph, episode_id=episode_id, n_frames=3,
    )
    mgr = SkillManager(graph, policy_version="v0")
    critic = EpisodeCriticAgent(llm=HeuristicLLM(), graph=graph, skill_manager=mgr)
    report = critic.run(
        episode_id=episode_id, game_id="TTT", outcome="terminal_loss",
        transitions=transitions, policy_version="v0", player=0,
    )
    # Fallback path should have proposed exactly one skill from the available evidence.
    assert len(report.skill_proposals) == 1, f"got {report.skill_proposals}"
    prop = report.skill_proposals[0]
    sv_id = prop["skill_version_id"]
    # The proposal must have status=proposed.
    sv = mgr.get(sv_id)
    assert sv is not None and sv.status == SkillStatus.PROPOSED.value
    # Memory -> skill_version SUPPORTS edges must equal evidence count.
    assert graph.count_supporting("skill_version", sv_id) >= 2
    # CriticReport -> skill_version SUPPORTS edge must exist.
    sup_edges = graph.query(
        "SELECT * FROM evidence_edges WHERE src_type='critic_report' AND src_id=? "
        "AND edge='SUPPORTS' AND dst_type='skill_version' AND dst_id=?",
        (report.id, sv_id),
    )
    assert len(sup_edges) == 1
    graph.close()


def test_critic_does_not_double_propose_when_agent_already_proposed(tmp_path: Path) -> None:
    """If learn_from_outcome already wrote an ``agent_fallback`` proposal for the
    same skill_id, the critic's fallback path must not duplicate it."""
    graph = _make_graph(tmp_path)
    episode_id = "ep_no_dup"
    _, memory_ids, transitions = _seed_episode_with_frames_and_memories(
        graph, episode_id=episode_id, n_frames=3,
    )
    mgr = SkillManager(graph, policy_version="v0")
    # Simulate the agent_fallback proposal that learn_from_outcome would have made.
    mgr.propose(
        name="TTT:agent_fallback_seed", guidance="agent-side proposal",
        trigger="game:TTT player_0", evidence_ids=memory_ids[:2],
        game_id="TTT", created_by="agent_fallback",
    )
    critic = EpisodeCriticAgent(llm=HeuristicLLM(), graph=graph, skill_manager=mgr)
    report = critic.run(
        episode_id=episode_id, game_id="TTT", outcome="terminal_loss",
        transitions=transitions, policy_version="v0", player=0,
    )
    # Critic should not have added a SECOND proposal — the agent already did.
    assert len(report.skill_proposals) == 0
    # The single existing proposal must still be visible to the critic view.
    all_skills = mgr.all_skills("TTT")
    assert len([s for s in all_skills if s.get("status") == "proposed"]) == 1
    graph.close()


def test_critic_tool_schema_is_valid_openai_format() -> None:
    """Sanity-check the tool schema so OpenAI's tools= API doesn't 400."""
    assert isinstance(CRITIC_TOOL_SCHEMA, list)
    names = set()
    for entry in CRITIC_TOOL_SCHEMA:
        assert entry["type"] == "function"
        fn = entry["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        assert fn["parameters"]["type"] == "object"
        names.add(fn["name"])
    # All seven advertised critic tools must be present.
    assert names == {
        "analyze_episode_trace", "query_evidence_graph", "propose_skill",
        "mark_do_not_learn", "propose_tool", "design_experiment",
        "write_critic_report",
    }


def test_critic_propose_skill_rejects_thin_evidence(tmp_path: Path) -> None:
    """Direct registry call with < 2 evidence ids must return an error,
    not create a skill version (Phase 3 invariant)."""
    graph = _make_graph(tmp_path)
    episode_id = "ep_thin"
    _, memory_ids, transitions = _seed_episode_with_frames_and_memories(
        graph, episode_id=episode_id, n_frames=3,
    )
    mgr = SkillManager(graph, policy_version="v0")
    critic = EpisodeCriticAgent(llm=HeuristicLLM(), graph=graph, skill_manager=mgr)
    # Inspect the internal registry by running a fake report payload.
    fake_report = CriticReport(episode_id=episode_id, game_id="TTT",
                                policy_version="v0", outcome="terminal_loss")
    registry = critic._build_registry(  # noqa: SLF001 — internal API test
        report=fake_report, episode_id=episode_id, game_id="TTT",
        player=0, transitions=transitions,
    )
    out = registry["propose_skill"](
        name="thin", trigger="game:TTT", guidance="too thin",
        evidence_ids=[memory_ids[0]],
    )
    assert out["ok"] is False
    assert "evidence" in out.get("error", "").lower()
    # No skill_version was created.
    assert graph.query("SELECT id FROM skill_version", ()) == []
    graph.close()


def test_critic_report_marks_fallback_in_heuristic_mode(tmp_path: Path) -> None:
    """Under HeuristicLLM the report.fallback flag must be True so downstream
    consumers can tell apart real LLM critics from the deterministic fallback."""
    graph = _make_graph(tmp_path)
    episode_id = "ep_fallback"
    _, _, transitions = _seed_episode_with_frames_and_memories(graph, episode_id=episode_id)
    mgr = SkillManager(graph, policy_version="v0")
    critic = EpisodeCriticAgent(llm=HeuristicLLM(), graph=graph, skill_manager=mgr)
    report = critic.run(
        episode_id=episode_id, game_id="TTT", outcome="terminal_loss",
        transitions=transitions, policy_version="v0", player=0,
    )
    assert report.fallback is True
    graph.close()
