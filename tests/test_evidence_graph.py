"""Tests for the SQLite-backed Evidence Graph (Phase 2).

The graph indexes the JSONL evidence layer: nodes are typed (episode,
decision_frame, memory, ...) and edges use a closed vocabulary. Reads must
remain cheap, writes must be idempotent (re-ingesting the same id is a no-op),
and the legacy JSONL bootstrap must survive being run twice.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from textarena_llm_agent.evidence_graph import (
    EDGE_PREDICATES,
    NODE_TABLES,
    EdgeRow,
    EvidenceGraph,
)


def test_add_node_and_get_node_roundtrip(tmp_path: Path) -> None:
    g = EvidenceGraph(tmp_path / "g.sqlite")
    g.add_node("episode", "E1", game_id="TicTacToe-v0", policy_version="v0", outcome="win",
               attrs={"seed": 42, "note": "hello"})
    row = g.get_node("episode", "E1")
    assert row is not None
    assert row["id"] == "E1"
    assert row["game_id"] == "TicTacToe-v0"
    assert row["attrs"]["seed"] == 42
    assert g.count_nodes("episode") == 1
    g.close()


def test_add_node_is_idempotent(tmp_path: Path) -> None:
    g = EvidenceGraph(tmp_path / "g.sqlite")
    g.add_node("episode", "E1", game_id="TTT")
    g.add_node("episode", "E1", game_id="TTT", outcome="loss")  # second insert -> update
    assert g.count_nodes("episode") == 1
    row = g.get_node("episode", "E1")
    assert row is not None and row["outcome"] == "loss"
    g.close()


def test_unknown_node_or_edge_raises(tmp_path: Path) -> None:
    g = EvidenceGraph(tmp_path / "g.sqlite")
    with pytest.raises(ValueError):
        g.add_node("not_a_table", "x")
    with pytest.raises(ValueError):
        g.add_edge("episode", "E1", "BOGUS", "memory", "M1")
    g.close()


def test_edge_predicates_closed_vocab() -> None:
    # Sanity: edge vocabulary is a closed frozenset so external callers cannot
    # extend it silently. The eight predicates in the spec must all be present.
    expected = {"CONTAINS", "PRODUCED", "SUPPORTS", "CONTRADICTS",
                "SUMMARIZED_AS", "DERIVED_FROM", "SUPPORTED_BY", "VALIDATED_BY"}
    assert expected <= EDGE_PREDICATES
    assert "skill" in NODE_TABLES and "decision_frame" in NODE_TABLES


def test_supports_edge_query(tmp_path: Path) -> None:
    g = EvidenceGraph(tmp_path / "g.sqlite")
    g.add_node("skill", "S1", name="OpenCenter", game_id="TTT")
    g.add_node("skill_version", "S1@v1", skill_id="S1", version=1, status="proposed",
               policy_version="v0", created_by="critic")
    g.add_node("memory", "M1", kind="experience", game_id="TTT")
    g.add_node("memory", "M2", kind="experience", game_id="TTT")
    g.add_edge("memory", "M1", "SUPPORTS", "skill_version", "S1@v1")
    g.add_edge("memory", "M2", "SUPPORTS", "skill_version", "S1@v1")
    # idempotent on duplicate edges
    g.add_edge("memory", "M2", "SUPPORTS", "skill_version", "S1@v1")
    rows = g.nodes_supporting("skill_version", "S1@v1")
    assert {r["src_id"] for r in rows} == {"M1", "M2"}
    assert g.count_supporting("skill_version", "S1@v1") == 2
    g.close()


def test_ingest_decision_frame_creates_episode_and_edge(tmp_path: Path) -> None:
    g = EvidenceGraph(tmp_path / "g.sqlite")
    frame = {
        "id": "DF1", "episode_id": "EP1", "game_id": "TTT",
        "state_hash": "abc", "turn": 0, "step": 0,
        "candidate_id": "C0", "action_text": "[0 0]", "policy_version": "v0",
    }
    g.ingest_decision_frame(frame)
    # parent episode auto-created via CONTAINS edge
    assert g.get_node("episode", "EP1") is not None
    assert g.get_node("decision_frame", "DF1") is not None
    targets = g.nodes_produced_by("episode", "EP1", edge="CONTAINS")
    assert {t["dst_id"] for t in targets} == {"DF1"}
    # replay_targets returns the frame
    rows = g.replay_targets("TTT", limit=5)
    assert any(r["id"] == "DF1" for r in rows)
    g.close()


def test_query_only_allows_select(tmp_path: Path) -> None:
    g = EvidenceGraph(tmp_path / "g.sqlite")
    g.add_node("episode", "E1", game_id="TTT")
    rows = g.query("SELECT id FROM episode WHERE id = ?", ("E1",))
    assert rows == [{"id": "E1"}]
    with pytest.raises(ValueError):
        g.query("DELETE FROM episode")
    g.close()


def test_bootstrap_from_jsonl_is_idempotent(tmp_path: Path) -> None:
    # Build a minimal legacy memory dir
    mem = tmp_path / "mem"
    mem.mkdir()
    (mem / "experiences.jsonl").write_text(
        json.dumps({"id": "X1", "game_id": "TTT", "player": 0, "lesson": "open center"}) + "\n",
        encoding="utf-8",
    )
    (mem / "reflections.jsonl").write_text(
        json.dumps({"id": "R1", "game_id": "TTT", "text": "lost a corner trap"}) + "\n",
        encoding="utf-8",
    )
    (mem / "skills.jsonl").write_text(
        json.dumps({"id": "SK1", "name": "OpenCenter", "game_id": "TTT"}) + "\n",
        encoding="utf-8",
    )
    g = EvidenceGraph(tmp_path / "g.sqlite")
    c1 = g.bootstrap_from_jsonl(mem)
    c2 = g.bootstrap_from_jsonl(mem)  # second pass must not duplicate
    assert c1 == c2  # counts equal between runs
    assert g.count_nodes("memory") == 2
    assert g.count_nodes("skill") == 1
    # legacy skill marked active@v1 with evidence_count_low attr
    sv = g.get_node("skill_version", "SK1@v1")
    assert sv is not None and sv["status"] == "active"
    assert sv["attrs"].get("evidence_count_low") is True
    g.close()


def test_agent_constructor_opens_graph(tmp_path: Path) -> None:
    # Integration: TextArenaDecisionAgent must open the graph and inject it
    # into memory.graph, so record_experience mirrors to SQLite.
    from textarena_llm_agent import TextArenaAgentConfig, TextArenaDecisionAgent
    from textarena_llm_agent.llm import HeuristicLLM

    cfg = TextArenaAgentConfig(
        memory_dir=str(tmp_path / "mem"),
        trace_dir=str(tmp_path / "trace"),
        enable_tool_synthesis=False,
        reflection_enabled=False,
        use_llm=False,
    )
    agent = TextArenaDecisionAgent(cfg, llm=HeuristicLLM(), evaluator_llm=HeuristicLLM())
    assert agent.evidence_graph is not None
    assert agent.memory.graph is agent.evidence_graph
    # Recording an experience should now appear in the memory table.
    from textarena_llm_agent.memory import Experience
    eid = agent.memory.record_experience(Experience(
        state_key="s0", game_id="TTT", player=0, action_text="[0 0]",
        reward=1.0, outcome="win",
    ))
    row = agent.evidence_graph.get_node("memory", eid)
    assert row is not None
    assert row["game_id"] == "TTT"
    assert row["kind"] == "experience"
    agent.evidence_graph.close()
