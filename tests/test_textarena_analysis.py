from __future__ import annotations

import json
from pathlib import Path

from textarena_llm_agent.analysis import AnalysisPaths, analyze, wilson_interval, write_outputs


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_wilson_interval_is_bounded():
    lo, hi = wilson_interval(3, 5)
    assert 0.0 <= lo <= hi <= 1.0


def test_analyze_builds_ablation_and_memory_metrics(tmp_path: Path):
    eval_root = tmp_path / "eval"
    memory_root = tmp_path / "memory"
    game_dir = eval_root / "TicTacToe"
    _write_json(game_dir / "trend.json", [
        {"bin": "early", "episodes": 4, "win_rate": 0.25, "avg_reward": 0.0, "invalid_move_rate": 0.25, "skill_count": 1},
        {"bin": "mid", "episodes": 4, "win_rate": 0.5, "avg_reward": 0.25, "invalid_move_rate": 0.0, "skill_count": 2},
        {"bin": "late", "episodes": 4, "win_rate": 0.75, "avg_reward": 0.75, "invalid_move_rate": 0.0, "skill_count": 3},
    ])
    _write_json(game_dir / "elo.json", {"ratings": {"llm_evolved": 1030, "llm_no_memory": 1000, "random": 960}, "rounds": 2, "history": []})
    _write_json(game_dir / "exploitability.json", {"method": "loss_rate_vs_optimal", "p_loss": 0.0})
    _write_jsonl(game_dir / "match_results.jsonl", [
        {"reward_a": 1, "invalid": False, "turns": 5, "first_player": 0},
        {"reward_a": 0, "invalid": False, "turns": 7, "first_player": 1},
        {"reward_a": -1, "invalid": True, "turns": 9, "first_player": 0},
    ])
    mem_dir = memory_root / "TicTacToe" / "evolved"
    _write_jsonl(mem_dir / "experiences.jsonl", [{"id": "e1"}])
    _write_jsonl(mem_dir / "reflections.jsonl", [{"id": "r1"}])
    _write_jsonl(mem_dir / "retrieval_hits.jsonl", [{"items": [{"id": "e1"}]}])
    _write_jsonl(mem_dir / "skills.jsonl", [{"id": "s1", "status": "promoted", "win_rate": 1.0}])
    _write_jsonl(mem_dir / "skill_updates.jsonl", [{"event": "promoted"}])

    result = analyze(AnalysisPaths(eval_root=eval_root, memory_root=memory_root, output_dir=tmp_path / "out"))
    game = result["games"][0]
    assert game["elo"]["evolved_minus_no_memory"] == 30
    assert game["trend_delta"]["avg_reward_late_minus_early"] == 0.75
    assert game["memory"]["skill_learning_present"] is True
    assert result["aggregate"]["positive_ablation_games"] == 1

    outputs = write_outputs(result, tmp_path / "out")
    assert Path(outputs["html"]).exists()
    assert Path(outputs["markdown"]).exists()
    assert Path(outputs["html"]).name.startswith("report_20")
    assert Path(outputs["json"]).name.startswith("metrics_analysis_20")
    assert not (tmp_path / "out" / "report.html").exists()
    assert not (tmp_path / "out" / "metrics_analysis.json").exists()

    legacy = write_outputs(result, tmp_path / "legacy", timestamped=False)
    assert Path(legacy["html"]).name == "report.html"
    assert Path(legacy["json"]).name == "metrics_analysis.json"


def test_analysis_does_not_invent_no_memory_ablation(tmp_path: Path):
    eval_root = tmp_path / "eval"
    memory_root = tmp_path / "memory"
    game_dir = eval_root / "TicTacToe"
    _write_json(game_dir / "trend.json", [
        {"bin": "early", "episodes": 1, "win_rate": 0.0, "avg_reward": 0.0, "invalid_move_rate": 0.0, "skill_count": 1},
        {"bin": "late", "episodes": 1, "win_rate": 0.0, "avg_reward": 0.0, "invalid_move_rate": 0.0, "skill_count": 2},
    ])
    _write_json(game_dir / "elo.json", {"ratings": {"llm_evolved": 1000, "random": 1000}, "rounds": 0, "method": "sampled_from_trend_matches"})
    _write_json(game_dir / "exploitability.json", {"method": "sampled_tactical_proxy", "best_candidate_rate": 1.0})
    _write_jsonl(game_dir / "match_results.jsonl", [
        {"reward_a": 0, "invalid": False, "turns": 4, "first_player": 0},
        {"reward_a": 0, "invalid": False, "turns": 4, "first_player": 1},
    ])
    mem_dir = memory_root / "TicTacToe" / "evolved"
    _write_jsonl(mem_dir / "skills.jsonl", [{"id": "s1"}])
    _write_jsonl(mem_dir / "skill_updates.jsonl", [{"event": "created"}])

    result = analyze(AnalysisPaths(eval_root=eval_root, memory_root=memory_root, output_dir=tmp_path / "out"))
    game = result["games"][0]
    assert game["elo"]["evolved_minus_no_memory"] is None
    assert game["elo"]["ablation_available"] is False
    assert result["aggregate"]["ablation_available_games"] == 0
    assert result["aggregate"]["mean_evolved_minus_no_memory"] is None
