from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .agent import TextArenaAgentConfig, TextArenaDecisionAgent
from .cli import build_env
from .llm import HeuristicLLM


@dataclass(slots=True)
class EpisodeMetrics:
    game: str
    seed: int
    reward: float
    outcome: str
    turns: int
    invalid_move: bool
    avg_evaluator_score: float
    final_skills: int


def run_evaluation(*, game: str, episodes: int, max_steps: int, output_dir: str | Path, seed: int = 0) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[EpisodeMetrics] = []
    for ep in range(episodes):
        ep_seed = seed + ep
        trace_dir = out_dir / "traces" / f"{game}_{ep_seed}"
        mem_dir = out_dir / "memory"
        env = build_env(game, seed=ep_seed)
        agent = TextArenaDecisionAgent(TextArenaAgentConfig(memory_dir=str(mem_dir), trace_dir=str(trace_dir)), llm=HeuristicLLM(), evaluator_llm=HeuristicLLM())
        eval_scores: list[float] = []
        turns = 0
        while turns < max_steps and not bool(getattr(env.state, "done", False)):
            decision = agent.act(env)
            if decision.evaluation.get("score") is not None:
                eval_scores.append(float(decision.evaluation["score"]))
            turns += 1
        rewards = getattr(env.state, "rewards", None) or {}
        reward = float(rewards.get(0, 0.0)) if isinstance(rewards, dict) else 0.0
        invalid = bool(getattr(env.state, "game_info", {}).get(0, {}).get("invalid_move", False))
        outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
        metrics.append(EpisodeMetrics(game=game, seed=ep_seed, reward=reward, outcome=outcome, turns=turns, invalid_move=invalid, avg_evaluator_score=mean(eval_scores) if eval_scores else 0.0, final_skills=agent.memory.memory_stats()["skills"]))
    summary = summarize_metrics(metrics)
    (out_dir / "episode_metrics.jsonl").write_text("".join(json.dumps(asdict(m), ensure_ascii=False) + "\n" for m in metrics), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize_metrics(metrics: list[EpisodeMetrics]) -> dict[str, Any]:
    n = max(1, len(metrics))
    return {
        "episodes": len(metrics),
        "win_rate": sum(1 for m in metrics if m.outcome == "win") / n,
        "draw_rate": sum(1 for m in metrics if m.outcome == "draw") / n,
        "loss_rate": sum(1 for m in metrics if m.outcome == "loss") / n,
        "invalid_move_rate": sum(1 for m in metrics if m.invalid_move) / n,
        "avg_reward": mean([m.reward for m in metrics]) if metrics else 0.0,
        "avg_turns": mean([m.turns for m in metrics]) if metrics else 0.0,
        "avg_evaluator_score": mean([m.avg_evaluator_score for m in metrics]) if metrics else 0.0,
        "final_skills": metrics[-1].final_skills if metrics else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the TextArena LLM/heuristic agent and write metrics.")
    parser.add_argument("--game", default="TicTacToe")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="workspace/textarena_experiments/latest")
    args = parser.parse_args(argv)
    print(json.dumps(run_evaluation(game=args.game, episodes=args.episodes, max_steps=args.steps, output_dir=args.output_dir, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
