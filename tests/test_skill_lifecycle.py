"""Tests for the Phase 3 SkillManager lifecycle.

The previous ``EvolvingMemory.evolve_skills`` collapsed proposal, validation,
and activation into a single in-place mutation of ``skills.jsonl``. This suite
locks in the new closed lifecycle:

    proposed → candidate → validated → active
                 ↘                ↘
                  rejected         deprecated

Each transition must be gated by an explicit Critic-Agent action (proposing,
promoting, validating, activating) and the Evidence Graph must record both
the SUPPORTS edges that justified each promotion and a new ``policy_version``
row whenever a skill is activated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from textarena_llm_agent.evidence_graph import EvidenceGraph
from textarena_llm_agent.skill_manager import (
    PROMOTE_MIN_SUPPORTS,
    SkillManager,
    SkillStatus,
    VALIDATE_AB_THRESHOLD,
    VALIDATE_REPLAY_THRESHOLD,
)


# ----------------------------------------------------------------- helpers


def _graph(tmp_path: Path) -> EvidenceGraph:
    return EvidenceGraph(tmp_path / "g.sqlite")


def _seed_memory_nodes(graph: EvidenceGraph, ids: list[str], game_id: str = "TTT") -> None:
    for mid in ids:
        graph.add_node("memory", mid, kind="experience", game_id=game_id, player=0,
                       do_not_learn=0, attrs={"lesson": f"lesson for {mid}"})


# --------------------------------------------------------------- lifecycle


def test_propose_writes_supports_edges_and_keeps_proposed_status(tmp_path: Path) -> None:
    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1", "M2"])
    mgr = SkillManager(graph)

    sv = mgr.propose(
        name="open_center", guidance="Take centre when free",
        trigger="game:TTT", evidence_ids=["M1", "M2"], game_id="TTT",
    )

    assert sv.status == SkillStatus.PROPOSED.value
    assert sv.version == 1
    # Both evidence ids should now have a SUPPORTS edge to this version.
    assert graph.count_supporting("skill_version", sv.id) == 2
    # Skill node was auto-created.
    assert graph.get_node("skill", sv.skill_id) is not None
    graph.close()


def test_propose_to_reject_path(tmp_path: Path) -> None:
    """A proposal with only one piece of evidence cannot be promoted and can be rejected."""
    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1"])
    mgr = SkillManager(graph)

    sv = mgr.propose(
        name="thin_skill", guidance="One-shot lesson",
        trigger="game:TTT", evidence_ids=["M1"], game_id="TTT",
    )
    # Should fail to promote (only 1 SUPPORTS edge, need 3).
    assert mgr.promote_to_candidate(sv.id) is False
    assert mgr.reject(sv.id, reason="thin evidence") is True
    after = mgr.get(sv.id)
    assert after is not None and after.status == SkillStatus.REJECTED.value
    graph.close()


def test_promote_requires_3_supports_edges(tmp_path: Path) -> None:
    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1", "M2"])
    mgr = SkillManager(graph)

    sv = mgr.propose(
        name="needs_more", guidance="Centre control",
        trigger="game:TTT", evidence_ids=["M1", "M2"], game_id="TTT",
    )
    # 2 < PROMOTE_MIN_SUPPORTS — should not promote.
    assert PROMOTE_MIN_SUPPORTS == 3
    assert mgr.promote_to_candidate(sv.id) is False
    # Add one more evidence id and edge to clear the threshold.
    graph.add_node("memory", "M3", kind="experience", game_id="TTT")
    graph.add_edge("memory", "M3", "SUPPORTS", "skill_version", sv.id)
    assert mgr.promote_to_candidate(sv.id) is True
    after = mgr.get(sv.id)
    assert after is not None and after.status == SkillStatus.CANDIDATE.value
    graph.close()


def test_validate_thresholds(tmp_path: Path) -> None:
    """Both replay_score >= 0.55 AND ab_score >= 0 must hold to validate."""
    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1", "M2", "M3"])
    mgr = SkillManager(graph)
    sv = mgr.propose(
        name="gated", guidance="Pin corner threats",
        trigger="game:TTT", evidence_ids=["M1", "M2", "M3"], game_id="TTT",
    )
    assert mgr.promote_to_candidate(sv.id) is True

    # Below threshold on replay should fail.
    assert mgr.validate(sv.id, replay_score=0.50, ab_score=0.5) is False
    assert mgr.get(sv.id).status == SkillStatus.CANDIDATE.value
    # Negative A/B score should fail even with good replay.
    assert mgr.validate(sv.id, replay_score=0.80, ab_score=-0.1) is False
    assert mgr.get(sv.id).status == SkillStatus.CANDIDATE.value
    # Both gates pass → validated.
    assert mgr.validate(
        sv.id,
        replay_score=VALIDATE_REPLAY_THRESHOLD,
        ab_score=VALIDATE_AB_THRESHOLD,
    ) is True
    assert mgr.get(sv.id).status == SkillStatus.VALIDATED.value
    graph.close()


def test_propose_to_validate_to_activate_path(tmp_path: Path) -> None:
    """End-to-end: proposed → candidate → validated → active bumps policy version."""
    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1", "M2", "M3"])
    mgr = SkillManager(graph)
    sv = mgr.propose(
        name="full_path", guidance="Block forks",
        trigger="game:TTT", evidence_ids=["M1", "M2", "M3"], game_id="TTT",
    )
    assert mgr.promote_to_candidate(sv.id) is True
    assert mgr.validate(sv.id, replay_score=0.7, ab_score=0.2) is True

    pv_count_before = graph.count_nodes("policy_version")
    new_pv = mgr.activate(sv.id)
    assert new_pv is not None and new_pv.startswith("v")
    # A new policy_version row was allocated.
    assert graph.count_nodes("policy_version") == pv_count_before + 1
    after = mgr.get(sv.id)
    assert after is not None and after.status == SkillStatus.ACTIVE.value
    assert after.policy_version == new_pv
    graph.close()


def test_active_skills_filters_status(tmp_path: Path) -> None:
    """Game agent must only see ``status=active`` versions; critic sees all."""
    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1", "M2", "M3", "M4", "M5", "M6"])
    mgr = SkillManager(graph)

    # Skill A — fully activated.
    sa = mgr.propose(
        name="A", guidance="Block forks", trigger="game:TTT",
        evidence_ids=["M1", "M2", "M3"], game_id="TTT",
    )
    mgr.promote_to_candidate(sa.id)
    mgr.validate(sa.id, replay_score=0.7, ab_score=0.5)
    mgr.activate(sa.id)

    # Skill B — stuck at candidate.
    sb = mgr.propose(
        name="B", guidance="Mirror opponent", trigger="game:TTT",
        evidence_ids=["M4", "M5", "M6"], game_id="TTT",
    )
    mgr.promote_to_candidate(sb.id)

    active = mgr.active_skills("TTT")
    assert {s["name"] for s in active} == {"A"}
    all_skills = mgr.all_skills("TTT")
    assert {s["name"] for s in all_skills} >= {"A", "B"}
    # Critic must be able to see the candidate explicitly.
    assert any(s["status"] == SkillStatus.CANDIDATE.value for s in all_skills)
    graph.close()


def test_deprecate_active_removes_from_active_view(tmp_path: Path) -> None:
    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1", "M2", "M3"])
    mgr = SkillManager(graph)
    sv = mgr.propose(
        name="will_deprecate", guidance="Outdated tactic",
        trigger="game:TTT", evidence_ids=["M1", "M2", "M3"], game_id="TTT",
    )
    mgr.promote_to_candidate(sv.id)
    mgr.validate(sv.id, replay_score=0.7, ab_score=0.4)
    mgr.activate(sv.id)
    assert len(mgr.active_skills("TTT")) == 1

    assert mgr.deprecate(sv.id, reason="new dominant strategy") is True
    assert mgr.active_skills("TTT") == []
    assert mgr.get(sv.id).status == SkillStatus.DEPRECATED.value
    graph.close()


def test_run_evolution_sweep_promotes_proposals_only_when_supports_threshold_met(tmp_path: Path) -> None:
    """run_evolution_sweep is the Critic-Agent entrypoint replacing evolve_skills.

    Without replay_fn/ab_fn it must NOT auto-validate or auto-activate — those
    transitions belong to the Critic Agent's explicit tool calls in Phase 4.
    """
    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1", "M2", "M3"])
    mgr = SkillManager(graph)
    sv = mgr.propose(
        name="sweep_target", guidance="generic",
        trigger="game:TTT", evidence_ids=["M1", "M2", "M3"], game_id="TTT",
    )
    counts = mgr.run_evolution_sweep(game_id="TTT")
    # The 3 SUPPORTS edges should clear the candidate threshold.
    assert counts["promoted"] == 1
    # No replay/ab functions were supplied; the version must remain at candidate.
    assert mgr.get(sv.id).status == SkillStatus.CANDIDATE.value
    assert counts["activated"] == 0
    graph.close()


def test_activate_bumps_policy_version_and_mirrors_to_skills_jsonl(tmp_path: Path) -> None:
    """Active skills must appear in skills.jsonl so the prompt builder picks them up."""
    from textarena_llm_agent.memory import EvolvingMemory

    graph = _graph(tmp_path)
    _seed_memory_nodes(graph, ["M1", "M2", "M3"])
    mem = EvolvingMemory(tmp_path / "mem", graph=graph)
    mgr = SkillManager(graph, mem, policy_version="v0")
    sv = mgr.propose(
        name="mirror_test", guidance="Always block immediate wins",
        trigger="game:TTT", evidence_ids=["M1", "M2", "M3"], game_id="TTT",
    )
    mgr.promote_to_candidate(sv.id)
    mgr.validate(sv.id, replay_score=0.6, ab_score=0.1)
    new_pv = mgr.activate(sv.id)
    assert new_pv == "v1"

    # skills.jsonl must contain exactly the one active skill version.
    rows = mem._read_jsonl(mem.skills_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "active"
    assert rows[0]["policy_version"] == "v1"
    assert "Always block immediate wins" in rows[0]["guidance"]
    graph.close()
