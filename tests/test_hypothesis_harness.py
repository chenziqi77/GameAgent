"""Phase 7 — hypothesis-driven evaluation tests.

Covers the public surface of ``hypothesis.py``:

* ``baseline_templates`` returns the 9 ablation arms with correct flag matrix.
* ``HypothesisHarness`` runs (arm × game × episode) deterministically and the
  aggregated ``ArmMetrics`` carries all 5 metric classes (棋力 / 学习效率 /
  泛化 / 技能质量 / 系统效率).
* ``Hypothesis.judge`` honours direction + margin.
* ``replay_eval`` handles missing cohorts via the tolerant ``episodes_for_policy``
  path.
* ``opponent_pool_snapshot`` copies the memory dir and writes a manifest.
* ``render_hypothesis_report`` emits the expected sections.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from textarena_llm_agent.hypothesis import (
    ArmMetrics,
    BaselineSpec,
    DEFAULT_HYPOTHESES,
    Hypothesis,
    HypothesisHarness,
    MatchSample,
    baseline_by_id,
    baseline_templates,
    opponent_pool_snapshot,
    render_hypothesis_report,
    replay_eval,
)


# ----------------------------------------------------------------- baselines


def test_baseline_templates_has_nine_entries() -> None:
    arms = baseline_templates()
    assert len(arms) == 9
    ids = [a.id for a in arms]
    assert ids == [
        "random", "heuristic", "raw_llm", "llm_rag", "llm_reflection",
        "llm_skill_no_provenance", "llm_skill_provenance", "llm_skill_tools",
        "full",
    ]
    full = baseline_by_id("full")
    assert all(full.flags[k] for k in ("llm", "rag", "reflection", "skill",
                                        "provenance", "tools"))
    raw = baseline_by_id("raw_llm")
    assert raw.flags["llm"] is True
    assert raw.flags["skill"] is False


def test_baseline_by_id_raises_on_unknown() -> None:
    with pytest.raises(KeyError):
        baseline_by_id("does-not-exist")


# ----------------------------------------------------------------- hypothesis judging


def test_hypothesis_judge_higher_is_better() -> None:
    h = Hypothesis(id="H_test", statement="x", arm_a="a", arm_b="b",
                   metric="win_rate", direction="higher_is_better", margin=0.05)
    passed, note = h.judge(0.70, 0.60)
    assert passed
    assert "0.7000" in note and "0.6000" in note

    passed2, _ = h.judge(0.62, 0.60)
    assert not passed2  # below margin


def test_hypothesis_judge_lower_is_better() -> None:
    h = Hypothesis(id="H_test", statement="x", arm_a="a", arm_b="b",
                   metric="invalid_move_rate", direction="lower_is_better",
                   margin=0.05)
    passed, _ = h.judge(0.02, 0.10)  # arm_a lower by 0.08 → pass
    assert passed
    passed2, _ = h.judge(0.20, 0.10)  # arm_a higher → fail
    assert not passed2


# ----------------------------------------------------------------- harness


def test_harness_runs_with_synthetic_runner() -> None:
    arms = baseline_templates()
    harness = HypothesisHarness(baselines=arms, games=["TicTacToe"],
                                n_episodes=2, seed=0, opponent="random")
    result = harness.run()
    # 9 arms × 1 game × 2 episodes = 18 matches
    assert len(result.matches) == 18
    assert set(result.arm_metrics.keys()) == {a.id for a in arms}
    # all 5 metric classes present on each ArmMetrics
    for m in result.arm_metrics.values():
        assert isinstance(m, ArmMetrics)
        # 棋力
        assert 0.0 <= m.win_rate <= 1.0
        assert 0.0 <= m.invalid_move_rate <= 1.0
        # 学习效率
        assert m.win_rate_slope >= -1.0
        # 泛化
        assert isinstance(m.cross_game_win_rate, dict)
        # 技能质量
        assert m.skill_count >= 0
        # 系统效率
        assert m.avg_prompt_tokens >= 0


def test_harness_is_deterministic() -> None:
    arms = baseline_templates()
    a = HypothesisHarness(baselines=arms, games=["TicTacToe"],
                          n_episodes=3, seed=42, opponent="random").run()
    b = HypothesisHarness(baselines=arms, games=["TicTacToe"],
                          n_episodes=3, seed=42, opponent="random").run()
    # Same seed → identical rewards in same order
    assert [m.reward for m in a.matches] == [m.reward for m in b.matches]


def test_harness_judges_all_default_hypotheses() -> None:
    arms = baseline_templates()
    harness = HypothesisHarness(baselines=arms, games=["TicTacToe"],
                                n_episodes=4, seed=0, opponent="random")
    result = harness.run()
    judged = {h.hypothesis_id for h in result.hypotheses}
    assert judged == {h.id for h in DEFAULT_HYPOTHESES}


def test_harness_accepts_injected_match_runner() -> None:
    """A stub runner lets tests pin exact outcomes."""
    def stub(arm: BaselineSpec, opp: BaselineSpec, game: str, seed: int) -> MatchSample:
        # arm_a always wins, arm_b always loses
        reward = 1.0 if arm.id == "raw_llm" else -1.0
        return MatchSample(arm=arm.id, opponent=opp.id, game_id=game,
                            reward=reward, invalid=False, turns=1, seed=seed)

    arms = [baseline_by_id("raw_llm"), baseline_by_id("random")]
    harness = HypothesisHarness(baselines=arms, hypotheses=[], games=["G"],
                                n_episodes=2, seed=0, opponent="random",
                                match_runner=stub)
    result = harness.run()
    assert result.arm_metrics["raw_llm"].win_rate == 1.0
    assert result.arm_metrics["random"].win_rate == 0.0


# ----------------------------------------------------------------- persistence


def test_harness_writes_report_when_output_dir_given(tmp_path: Path) -> None:
    arms = baseline_templates()
    out_dir = tmp_path / "hyp_out"
    harness = HypothesisHarness(baselines=arms, games=["TicTacToe"],
                                n_episodes=2, seed=0, opponent="random",
                                output_dir=out_dir)
    harness.run()
    assert (out_dir / "hypothesis_report.md").exists()
    assert (out_dir / "hypothesis_result.json").exists()
    assert (out_dir / "matches.jsonl").exists()
    payload = json.loads((out_dir / "hypothesis_result.json").read_text(encoding="utf-8"))
    assert "arms" in payload and "hypotheses" in payload and "arm_metrics" in payload


def test_render_hypothesis_report_has_expected_sections() -> None:
    arms = baseline_templates()
    result = HypothesisHarness(baselines=arms, games=["TicTacToe"],
                               n_episodes=2, seed=0, opponent="random").run()
    md = render_hypothesis_report(result)
    assert "# Hypothesis-Driven Evaluation Report" in md
    assert "## Hypotheses" in md
    assert "## Arm Metrics" in md
    # one row per arm in the metrics table
    for arm in result.arms:
        assert f"| {arm} |" in md


# ----------------------------------------------------------------- replay_eval


class _StubGraph:
    """Minimal stand-in for EvidenceGraph in replay_eval tests."""

    def __init__(self, episodes_by_policy: dict[str, list[dict[str, Any]]]):
        self._eps = episodes_by_policy

    def episodes_for_policy(self, policy_version: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._eps.get(policy_version, []))[:limit]


def test_replay_eval_aggregates_top_level_outcome() -> None:
    graph = _StubGraph({
        "v0": [{"outcome": "win", "reward": 1.0}, {"outcome": "loss", "reward": -1.0}],
        "v1": [{"outcome": "win", "reward": 1.0}, {"outcome": "win", "reward": 1.0}],
    })
    result = replay_eval(graph, policy_a="v1", policy_b="v0")
    assert result.episodes_a == 2 and result.episodes_b == 2
    assert result.win_rate_a == 1.0
    assert result.win_rate_b == 0.5
    assert pytest.approx(result.diff, abs=1e-6) == 0.5


def test_replay_eval_handles_missing_cohorts() -> None:
    graph = _StubGraph({})
    result = replay_eval(graph, policy_a="v0", policy_b="v1")
    assert result.episodes_a == 0 and result.episodes_b == 0
    assert result.win_rate_a == 0.0 and result.win_rate_b == 0.0
    assert "Missing cohort" in result.note


def test_replay_eval_parses_attrs_json_fallback() -> None:
    graph = _StubGraph({
        "v0": [{"id": "e1", "attrs_json": json.dumps({"outcome": "win", "reward": 1.0})}],
        "v1": [{"id": "e2", "attrs_json": json.dumps({"outcome": "loss", "reward": -1.0})}],
    })
    result = replay_eval(graph, policy_a="v0", policy_b="v1")
    assert result.win_rate_a == 1.0
    assert result.win_rate_b == 0.0


# ----------------------------------------------------------------- opponent pool snapshot


def test_opponent_pool_snapshot_copies_files_and_writes_manifest(tmp_path: Path) -> None:
    src = tmp_path / "mem"
    src.mkdir()
    (src / "skills.jsonl").write_text('{"id":"s1"}\n', encoding="utf-8")
    (src / "rules.md").write_text("# rules\n", encoding="utf-8")
    sub = src / "patches"
    sub.mkdir()
    (sub / "p1.json").write_text("{}", encoding="utf-8")

    dest_root = tmp_path / "pool"
    snap = opponent_pool_snapshot(src, dest_root, policy_version="v3")
    assert snap.policy_version == "v3"
    assert snap.files_copied >= 3  # skills.jsonl + rules.md + patches/p1.json
    manifest = json.loads(Path(snap.manifest_path).read_text(encoding="utf-8"))
    assert manifest["policy_version"] == "v3"
    assert manifest["files_copied"] == snap.files_copied
    assert (dest_root / "v3" / "skills.jsonl").exists()
    assert (dest_root / "v3" / "patches" / "p1.json").exists()


def test_opponent_pool_snapshot_tolerates_missing_source(tmp_path: Path) -> None:
    snap = opponent_pool_snapshot(tmp_path / "does-not-exist",
                                  tmp_path / "pool", policy_version="v0")
    assert snap.files_copied == 0
    assert Path(snap.manifest_path).exists()
