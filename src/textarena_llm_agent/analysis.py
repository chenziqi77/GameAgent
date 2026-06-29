from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


DEFAULT_CRITIC_MODEL = "gpt-5.5"


def display_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


SOFT_SKILLS: dict[str, list[str]] = {
    "TicTacToe": ["Strategic Planning", "Logical Reasoning"],
    "KuhnPoker": ["Strategic Planning", "Theory of Mind", "Bluffing", "Uncertainty Estimation"],
    "SimpleNegotiation": ["Strategic Planning", "Theory of Mind", "Bluffing", "Adaptability"],
    "Stratego": ["Strategic Planning", "Pattern Recognition", "Theory of Mind", "Uncertainty Estimation", "Adaptability"],
}


METRIC_CONTRACT: list[dict[str, str]] = [
    {
        "metric": "Win/draw/loss rate vs fixed opponent",
        "why": "Direct task success signal. Use a fixed random/heuristic/optimal pool so a rising curve means policy improvement rather than a changing opponent.",
        "source": "TextArena supports model-vs-model evaluation; RL benchmarks commonly report success/reward by episode.",
        "local_field": "trend.json + match_results.jsonl",
    },
    {
        "metric": "Average reward and reward slope",
        "why": "Captures margin and partial progress in games where win rate is sparse or draw-heavy.",
        "source": "Standard RL objective and ExpeL/WebShop-style average reward reporting.",
        "local_field": "trend.avg_reward, match reward_a",
    },
    {
        "metric": "Elo / TrueSkill-style relative rating",
        "why": "Aggregates pairwise tournament results against heterogeneous opponents. TextArena uses TrueSkill online; Elo is a lightweight local substitute.",
        "source": "TextArena leaderboard uses TrueSkill; Elo/TrueSkill are relative skill ratings.",
        "local_field": "elo.json",
    },
    {
        "metric": "Reward confidence / standard error",
        "why": "Separates large, reliable gains from noisy small-sample swings; report mean reward with standard error and Wilson intervals for win rate.",
        "source": "Standard experimental reporting for stochastic RL evaluations.",
        "local_field": "match_summary.reward_se, trend_delta.*_wilson_95",
    },
    {
        "metric": "First-player / second-player side bias",
        "why": "Many games are asymmetric by turn order; a strong agent should not only win from the easier side.",
        "source": "Game evaluation protocol control for seat/turn-order advantage.",
        "local_field": "match_summary.side_breakdown",
    },
    {
        "metric": "Evolved minus no-memory ablation",
        "why": "Primary evidence that memory, reflections, and skills are useful rather than the base policy alone.",
        "source": "Reflexion, ExpeL, and Voyager all rely on ablations to isolate memory/reflection/skill-library gains.",
        "local_field": "elo.ratings.llm_evolved - elo.ratings.llm_no_memory",
    },
    {
        "metric": "Past-self / checkpoint tournament",
        "why": "Compares the latest policy against earlier memory snapshots, giving a direct learning-over-time test independent of a single final rating.",
        "source": "Self-play and policy checkpoint evaluation in RL / multi-agent learning.",
        "local_field": "future: snapshot pool match_results.jsonl",
    },
    {
        "metric": "Exploitability / best-response value",
        "why": "Game-theoretic robustness metric: lower exploitability means the policy is harder to punish by an oracle best response.",
        "source": "OpenSpiel exposes exploitability / best-response and related multi-agent evaluation tools.",
        "local_field": "exploitability.json",
    },
    {
        "metric": "Invalid action rate",
        "why": "LLM game agents can lose by format or legality errors; decreasing invalidity is real capability improvement.",
        "source": "Agent benchmark diagnostics and ExpeL additional invalid-action statistics.",
        "local_field": "trend.invalid_move_rate, match invalid",
    },
    {
        "metric": "Critic failure-mode counts",
        "why": "Tracks whether repeated strategic mistakes identified by the critic decline after reflections and skill mutations.",
        "source": "Reflexion-style verbal reinforcement and error analysis.",
        "local_field": "skill_updates.jsonl, reflections.jsonl, evaluator critiques",
    },
    {
        "metric": "Sample efficiency / turn efficiency",
        "why": "Fewer turns for equal or better reward can indicate more decisive play; more turns may indicate robust drawing against strong opponents.",
        "source": "Trajectory statistics used in agent-learning evaluations.",
        "local_field": "trend.avg_turns, match turns",
    },
    {
        "metric": "Skill/reflection/retrieval growth",
        "why": "Mechanistic evidence that the agent is actually accumulating reusable experience, not just replaying a static prompt.",
        "source": "Reflexion episodic memory, ExpeL experience-to-insight learning, Voyager skill library.",
        "local_field": "memory/*.jsonl, skill_timeline.jsonl",
    },
]


@dataclass(slots=True)
class AnalysisPaths:
    eval_root: Path
    memory_root: Path
    output_dir: Path
    references_dir: Path | None = None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    avg = statistics.fmean(values)
    if len(values) < 2:
        return avg, 0.0
    return avg, statistics.stdev(values) / math.sqrt(len(values))


def slope_from_bins(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key) or 0.0) for r in rows if int(r.get("episodes") or 0) > 0]
    if len(vals) < 2:
        return 0.0
    return round(vals[-1] - vals[0], 4)


def nonempty_trend_edges(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    nonempty = [r for r in rows if int(r.get("episodes") or 0) > 0]
    if not nonempty:
        return {}, {}
    return nonempty[0], nonempty[-1]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def load_memory_stats(memory_root: Path, game: str, eval_timeline: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        memory_root / game / "evolved",
        memory_root / game,
        memory_root,
    ]
    mem_dir = next((p for p in candidates if (p / "experiences.jsonl").exists() or (p / "skills.jsonl").exists()), candidates[0])
    skills = read_jsonl(mem_dir / "skills.jsonl")
    skill_updates = read_jsonl(mem_dir / "skill_updates.jsonl")
    if not skill_updates and eval_timeline:
        skill_updates = eval_timeline
    reflections = read_jsonl(mem_dir / "reflections.jsonl")
    retrievals = read_jsonl(mem_dir / "retrieval_hits.jsonl")
    experiences = read_jsonl(mem_dir / "experiences.jsonl")
    patches = [p for p in read_jsonl(mem_dir / "prompt_patches.jsonl") if p.get("status") in (None, "active")]
    active_skills = [s for s in skills if s.get("status") in (None, "active", "promoted", "mutated")]
    event_counts = Counter(str(e.get("event") or "unknown") for e in skill_updates)
    avg_skill_wr = statistics.fmean([float(s.get("win_rate") or 0.0) for s in skills]) if skills else 0.0
    avg_retrieval_items = statistics.fmean([len(r.get("items") or []) for r in retrievals]) if retrievals else 0.0
    return {
        "memory_dir": str(mem_dir),
        "experiences": len(experiences),
        "reflections": len(reflections),
        "retrievals": len(retrievals),
        "avg_retrieval_items": round(avg_retrieval_items, 3),
        "skills": len(skills),
        "active_skills": len(active_skills),
        "promoted_skills": sum(1 for s in skills if s.get("status") == "promoted"),
        "mutated_skills": sum(1 for s in skills if s.get("status") == "mutated"),
        "demoted_skills": sum(1 for s in skills if s.get("status") == "demoted"),
        "avg_skill_win_rate": round(avg_skill_wr, 4),
        "skill_updates": len(skill_updates),
        "skill_event_counts": dict(sorted(event_counts.items())),
        "active_prompt_patches": len(patches),
        "rules_lines": count_lines(mem_dir / "rules.md"),
        "mechanism_present": bool(experiences or reflections or retrievals or skills or skill_updates or patches),
        "skill_learning_present": bool(skills or skill_updates or patches),
    }


def side_breakdown(matches: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in (0, 1):
        rows = [m for m in matches if int(m.get("first_player") or 0) == side]
        rewards = [float(m.get("reward_a") or 0.0) for m in rows]
        avg, se = mean_se(rewards)
        out[f"agent_side_{side}"] = {
            "games": len(rows),
            "win_rate": round(sum(1 for r in rewards if r > 0) / max(1, len(rows)), 4),
            "draw_rate": round(sum(1 for r in rewards if r == 0) / max(1, len(rows)), 4),
            "loss_rate": round(sum(1 for r in rewards if r < 0) / max(1, len(rows)), 4),
            "avg_reward": round(avg, 4),
            "reward_se": round(se, 4),
        }
    return out


def analyze_game(game_dir: Path, memory_root: Path) -> dict[str, Any]:
    game = game_dir.name
    trend = read_json(game_dir / "trend.json") or []
    elo = read_json(game_dir / "elo.json") or {}
    exploitability = read_json(game_dir / "exploitability.json") or {}
    matches = read_jsonl(game_dir / "match_results.jsonl")
    skill_timeline = read_jsonl(game_dir / "skill_timeline.jsonl")
    ratings = elo.get("ratings") if isinstance(elo, dict) else {}
    ratings = ratings if isinstance(ratings, dict) else {}
    evolved_rating = float(ratings.get("llm_evolved") or 0.0)
    no_memory_rating = float(ratings.get("llm_no_memory") or 0.0) if "llm_no_memory" in ratings else None
    random_rating = float(ratings.get("random") or 0.0) if "random" in ratings else None
    optimal_names = [n for n in ratings if n.startswith("optimal")]
    best_optimal = max([float(ratings[n]) for n in optimal_names], default=0.0)
    total_episodes = sum(int(r.get("episodes") or 0) for r in trend)
    wins = sum(1 for m in matches if float(m.get("reward_a") or 0.0) > 0)
    losses = sum(1 for m in matches if float(m.get("reward_a") or 0.0) < 0)
    draws = sum(1 for m in matches if float(m.get("reward_a") or 0.0) == 0)
    reward_values = [float(m.get("reward_a") or 0.0) for m in matches]
    avg_reward, reward_se = mean_se(reward_values)
    early, late = nonempty_trend_edges(trend)
    late_wins = round(float(late.get("win_rate") or 0.0) * int(late.get("episodes") or 0))
    late_ci = wilson_interval(late_wins, int(late.get("episodes") or 0))
    memory = load_memory_stats(memory_root, game, skill_timeline)
    sample_warning = total_episodes < 30
    ablation_delta = round(evolved_rating - no_memory_rating, 2) if ratings and no_memory_rating is not None else None
    random_delta = round(evolved_rating - random_rating, 2) if ratings and random_rating is not None else None
    optimal_gap = round(evolved_rating - best_optimal, 2) if best_optimal else None
    improvement_score = 0
    improvement_score += 1 if slope_from_bins(trend, "win_rate") > 0 else 0
    improvement_score += 1 if slope_from_bins(trend, "avg_reward") > 0 else 0
    improvement_score += 1 if ablation_delta is not None and ablation_delta > 0 else 0
    improvement_score += 1 if random_delta is not None and random_delta > 0 else 0
    improvement_score += 1 if memory["skill_learning_present"] else 0
    if sample_warning:
        evidence_grade = "pilot_only"
    elif improvement_score >= 4:
        evidence_grade = "strong_positive"
    elif improvement_score >= 2:
        evidence_grade = "mixed_positive"
    else:
        evidence_grade = "weak_or_negative"
    return {
        "game": game,
        "soft_skills": SOFT_SKILLS.get(game, []),
        "trend": trend,
        "trend_delta": {
            "win_rate_late_minus_early": slope_from_bins(trend, "win_rate"),
            "avg_reward_late_minus_early": slope_from_bins(trend, "avg_reward"),
            "invalid_rate_late_minus_early": slope_from_bins(trend, "invalid_move_rate"),
            "skill_count_late_minus_early": slope_from_bins(trend, "skill_count"),
            "early_win_rate": round(float(early.get("win_rate") or 0.0), 4),
            "late_win_rate": round(float(late.get("win_rate") or 0.0), 4),
            "late_win_rate_wilson_95": [round(late_ci[0], 4), round(late_ci[1], 4)],
        },
        "match_summary": {
            "games": len(matches),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": round(wins / max(1, len(matches)), 4),
            "draw_rate": round(draws / max(1, len(matches)), 4),
            "loss_rate": round(losses / max(1, len(matches)), 4),
            "avg_reward": round(avg_reward, 4),
            "reward_se": round(reward_se, 4),
            "invalid_rate": round(sum(1 for m in matches if m.get("invalid")) / max(1, len(matches)), 4),
            "avg_turns": round(statistics.fmean([float(m.get("turns") or 0.0) for m in matches]), 4) if matches else 0.0,
            "side_breakdown": side_breakdown(matches),
        },
        "elo": {
            "ratings": ratings,
            "rounds": elo.get("rounds", 0) if isinstance(elo, dict) else 0,
            "evolved_minus_no_memory": ablation_delta,
            "evolved_minus_random": random_delta,
            "ablation_available": ablation_delta is not None,
            "evolved_minus_best_optimal": optimal_gap,
            "rank": _rank("llm_evolved", ratings),
        },
        "exploitability": exploitability,
        "memory": memory,
        "sample_warning": sample_warning,
        "evidence_grade": evidence_grade,
        "interpretation": interpret_game(evidence_grade, ablation_delta, trend, exploitability, memory, total_episodes),
    }


def _rank(name: str, ratings: dict[str, Any]) -> int | None:
    if name not in ratings:
        return None
    ordered = sorted(ratings.items(), key=lambda kv: float(kv[1]), reverse=True)
    for idx, (n, _v) in enumerate(ordered, start=1):
        if n == name:
            return idx
    return None


def interpret_game(
    evidence_grade: str,
    ablation_delta: float | None,
    trend: list[dict[str, Any]],
    exploitability: dict[str, Any],
    memory: dict[str, Any],
    total_episodes: int,
) -> str:
    parts: list[str] = []
    if evidence_grade == "pilot_only":
        parts.append(f"Pilot evidence only: trend uses {total_episodes} episodes, below the 30+ episode minimum recommended for stable claims.")
    if slope_from_bins(trend, "avg_reward") > 0 or slope_from_bins(trend, "win_rate") > 0:
        parts.append("The early-to-late curve improves on at least one outcome metric.")
    else:
        parts.append("The early-to-late curve does not yet show a stable positive slope.")
    if ablation_delta is None:
        parts.append("No-memory ablation is unavailable for this run; sampled Elo only covers observed opponents.")
    elif ablation_delta > 0:
        parts.append(f"Evolved memory beats the no-memory ablation by {ablation_delta} Elo points.")
    elif ablation_delta < 0:
        parts.append(f"No-memory is ahead by {abs(ablation_delta)} Elo points; this weakens the skill/memory claim.")
    else:
        parts.append("No-memory ablation is tied or unavailable.")
    if exploitability.get("method") == "best_response_value":
        parts.append(f"Exploitability proxy is {exploitability.get('exploitability')}; lower is better and should decrease across stronger runs.")
    elif exploitability.get("method") == "loss_rate_vs_optimal":
        parts.append(f"Optimal-opponent loss rate is {exploitability.get('p_loss')}; this is the main robustness proxy for this game.")
    if memory.get("skill_learning_present"):
        parts.append("Skill/prompt evolution artifacts are present, so mechanism-level analysis is possible.")
    else:
        parts.append("Skill update artifacts are absent; current evidence mainly shows episodic memory/retrieval use, not skill evolution.")
    return " ".join(parts)


def analyze(paths: AnalysisPaths, *, critic_llm: bool = False, critic_model: str | None = None) -> dict[str, Any]:
    games = [
        analyze_game(p, paths.memory_root)
        for p in sorted(paths.eval_root.iterdir())
        if p.is_dir() and (p / "trend.json").exists()
    ] if paths.eval_root.exists() else []
    aggregate = aggregate_games(games)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_root": str(paths.eval_root),
        "memory_root": str(paths.memory_root),
        "metric_contract": METRIC_CONTRACT,
        "games": games,
        "aggregate": aggregate,
        "recommended_next_experiments": recommended_next_experiments(games),
        "references": reference_inventory(paths.references_dir),
        "critic_review": None,
    }
    if critic_llm:
        result["critic_review"] = run_critic_review(result, model=critic_model)
    return result


def aggregate_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    if not games:
        return {}
    ablations = [float(g.get("elo", {}).get("evolved_minus_no_memory")) for g in games if g.get("elo", {}).get("evolved_minus_no_memory") is not None]
    return {
        "games": len(games),
        "pilot_games": sum(1 for g in games if g.get("sample_warning")),
        "positive_ablation_games": sum(1 for v in ablations if v > 0),
        "ablation_available_games": len(ablations),
        "positive_reward_slope_games": sum(1 for g in games if float(g.get("trend_delta", {}).get("avg_reward_late_minus_early") or 0.0) > 0),
        "games_with_skill_learning_artifacts": sum(1 for g in games if g.get("memory", {}).get("skill_learning_present")),
        "total_matches": sum(int(g.get("match_summary", {}).get("games") or 0) for g in games),
        "mean_evolved_minus_no_memory": round(statistics.fmean(ablations), 3) if ablations else None,
        "mean_late_minus_early_reward": round(statistics.fmean([float(g.get("trend_delta", {}).get("avg_reward_late_minus_early") or 0.0) for g in games]), 3),
        "mean_reward_se": round(statistics.fmean([float(g.get("match_summary", {}).get("reward_se") or 0.0) for g in games]), 3),
        "mean_invalid_rate": round(statistics.fmean([float(g.get("match_summary", {}).get("invalid_rate") or 0.0) for g in games]), 3),
    }


def recommended_next_experiments(games: list[dict[str, Any]]) -> list[dict[str, str]]:
    recs = [
        {
            "priority": "P0",
            "experiment": "Run >=30 episodes per game with 3 seeds",
            "purpose": "Reduce pilot noise and provide confidence intervals for win rate, reward, invalid action rate, and side bias.",
        },
        {
            "priority": "P0",
            "experiment": "Memory ablation: evolved vs no_memory vs reflection_only vs retrieval_only",
            "purpose": "Isolate whether skill/memory mechanisms cause gains beyond base LLM policy quality.",
        },
        {
            "priority": "P1",
            "experiment": "Past-self tournament snapshots",
            "purpose": "Show monotonic improvement by matching late policies against earlier memory snapshots.",
        },
        {
            "priority": "P1",
            "experiment": "Critic LLM skill audit on terminal losses",
            "purpose": "Use a separate model to label failure cause, propose skill mutation, and track whether repeated failure modes decline.",
        },
        {
            "priority": "P2",
            "experiment": "Exploitability over checkpoints for KuhnPoker/TicTacToe",
            "purpose": "Turn game-theoretic robustness into a curve, not only one endpoint.",
        },
    ]
    if any(not g.get("memory", {}).get("skill_learning_present") for g in games):
        recs.insert(1, {
            "priority": "P0",
            "experiment": "Enable actionable lesson extraction so skill_updates.jsonl is non-empty",
            "purpose": "The current mechanism claim is weak when experiences/reflections exist but no skills are created, promoted, demoted, or mutated.",
        })
    return recs


def reference_inventory(ref_dir: Path | None) -> list[dict[str, str]]:
    if ref_dir is None:
        return []
    refs = [
        ("TextArena", "textarena_2504.11442.pdf", "https://arxiv.org/abs/2504.11442"),
        ("Reflexion", "reflexion_2303.11366.pdf", "https://arxiv.org/abs/2303.11366"),
        ("ExpeL", "expel_2308.10144.pdf", "https://arxiv.org/abs/2308.10144"),
        ("Voyager", "voyager_2305.16291.pdf", "https://arxiv.org/abs/2305.16291"),
        ("OpenSpiel", "openspiel_1908.09453.pdf", "https://arxiv.org/abs/1908.09453"),
    ]
    out = []
    for title, filename, url in refs:
        path = ref_dir / filename
        out.append({"title": title, "local_path": str(path), "url": url, "downloaded": str(path.exists())})
    return out


def run_critic_review(analysis_result: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
    try:
        from .llm import OpenAIChatLLM
        llm = OpenAIChatLLM.from_env()
        llm.model = model or DEFAULT_CRITIC_MODEL
        compact = {
            "aggregate": analysis_result.get("aggregate"),
            "games": [
                {
                    "game": g.get("game"),
                    "trend_delta": g.get("trend_delta"),
                    "elo": g.get("elo"),
                    "exploitability": g.get("exploitability"),
                    "memory": g.get("memory"),
                    "evidence_grade": g.get("evidence_grade"),
                }
                for g in analysis_result.get("games", [])
            ],
        }
        system = (
            "You are a skeptical multi-agent RL and game-theory experiment reviewer. "
            "Return JSON with keys verdict, main_weaknesses, strongest_evidence, required_next_experiments. "
            "Do not overclaim beyond the metrics."
        )
        user = "Review whether these TextArena results support agent improvement through memory/skill evolution:\n" + json.dumps(compact, ensure_ascii=False, indent=2)
        obj = llm.complete_json(system=system, user=user, temperature=0.0, max_tokens=900)
        obj.setdefault("model", llm.model)
        obj.setdefault("status", "ok")
        return obj
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def artifact_timestamp(created_at: str | None = None) -> str:
    raw = created_at or datetime.now(timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_outputs(result: dict[str, Any], output_dir: Path, *, timestamped: bool = True) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = artifact_timestamp(str(result.get("created_at") or ""))
    if timestamped:
        json_path = output_dir / f"metrics_analysis_{stamp}.json"
        md_path = output_dir / f"metrics_analysis_{stamp}.md"
        html_path = output_dir / f"report_{stamp}.html"
    else:
        json_path = output_dir / "metrics_analysis.json"
        md_path = output_dir / "metrics_analysis.md"
        html_path = output_dir / "report.html"
    payload_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    payload_md = render_markdown(result)
    payload_html = render_html(result)
    json_path.write_text(payload_json, encoding="utf-8")
    md_path.write_text(payload_md, encoding="utf-8")
    html_path.write_text(payload_html, encoding="utf-8")
    outputs = {"json": str(json_path), "markdown": str(md_path), "html": str(html_path)}
    return outputs


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TextArena Agent Metrics Analysis",
        "",
        f"Created: `{result.get('created_at')}`",
        "",
        "## Metric Contract",
        "",
        "| Metric | Why it matters | Local field |",
        "|---|---|---|",
    ]
    for row in result.get("metric_contract", []):
        lines.append(f"| {row['metric']} | {row['why']} | `{row['local_field']}` |")
    lines += ["", "## Game Summary", "", "| Game | Evidence | Episodes | Win Rate | Avg Reward ± SE | Late-Early Reward | Elo Evolved-NoMem | Invalid | Skill Artifacts | Interpretation |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for g in result.get("games", []):
        episodes = sum(int(r.get("episodes") or 0) for r in g.get("trend", []))
        lines.append(
            f"| {g['game']} | {g['evidence_grade']} | {episodes} | "
            f"{g['match_summary']['win_rate']} | {g['match_summary']['avg_reward']} ± {g['match_summary']['reward_se']} | "
            f"{g['trend_delta']['avg_reward_late_minus_early']} | {display_metric(g['elo']['evolved_minus_no_memory'])} | "
            f"{g['match_summary']['invalid_rate']} | {g['memory']['skill_updates']} | {g['interpretation']} |"
        )
    lines += ["", "## Recommended Next Experiments", ""]
    for rec in result.get("recommended_next_experiments", []):
        lines.append(f"- **{rec['priority']} {rec['experiment']}**: {rec['purpose']}")
    if result.get("critic_review"):
        lines += ["", "## Critic LLM Review", "", "```json", json.dumps(result["critic_review"], ensure_ascii=False, indent=2), "```"]
    lines += ["", "## Local References", ""]
    for ref in result.get("references", []):
        lines.append(f"- {ref['title']}: `{ref['local_path']}` ({ref['url']})")
    return "\n".join(lines) + "\n"


def render_html(result: dict[str, Any]) -> str:
    data = json.dumps(result, ensure_ascii=False, default=str)
    rows = []
    for g in result.get("games", []):
        episodes = sum(int(r.get("episodes") or 0) for r in g.get("trend", []))
        rows.append(
            "<tr>"
            f"<td>{escape(g['game'])}</td>"
            f"<td>{escape(', '.join(g.get('soft_skills') or []))}</td>"
            f"<td>{escape(g['evidence_grade'])}</td>"
            f"<td>{episodes}</td>"
            f"<td>{g['trend_delta']['win_rate_late_minus_early']}</td>"
            f"<td>{g['trend_delta']['avg_reward_late_minus_early']}</td>"
            f"<td>{g['match_summary']['avg_reward']} ± {g['match_summary']['reward_se']}</td>"
            f"<td>{escape(display_metric(g['elo']['evolved_minus_no_memory']))}</td>"
            f"<td>{escape(display_metric(g['elo']['evolved_minus_random']))}</td>"
            f"<td>{g['match_summary']['invalid_rate']}</td>"
            f"<td>{g['memory']['experiences']}/{g['memory']['reflections']}/{g['memory']['skill_updates']}</td>"
            f"<td>{escape(g['interpretation'])}</td>"
            "</tr>"
        )
    contract_rows = "".join(
        f"<tr><td>{escape(r['metric'])}</td><td>{escape(r['why'])}</td><td><code>{escape(r['local_field'])}</code></td></tr>"
        for r in result.get("metric_contract", [])
    )
    refs = "".join(
        f"<li><b>{escape(r['title'])}</b>: <code>{escape(r['local_path'])}</code> <a href='{escape(r['url'])}'>{escape(r['url'])}</a></li>"
        for r in result.get("references", [])
    )
    recs = "".join(
        f"<li><b>{escape(r['priority'])} {escape(r['experiment'])}</b>: {escape(r['purpose'])}</li>"
        for r in result.get("recommended_next_experiments", [])
    )
    critic = escape(json.dumps(result.get("critic_review"), ensure_ascii=False, indent=2, default=str)) if result.get("critic_review") else "Critic LLM not requested or unavailable."
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TextArena Agent Metrics Analysis</title>
  <style>
    :root {{ --bg:#f6f7f5; --panel:#fff; --ink:#202124; --muted:#62666d; --line:#dadbd6; --accent:#136f63; --warn:#a15c18; --bad:#9f2f2f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, Arial, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:20px 28px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0; font-size:25px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    main {{ padding:16px; display:grid; gap:16px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .muted {{ color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }}
    .card {{ border:1px solid var(--line); border-radius:7px; padding:12px; background:#fff; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid #e5e5e1; padding:8px; text-align:left; vertical-align:top; }}
    th {{ background:#fafaf8; }}
    code,pre {{ font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    pre {{ background:#f8f9fa; border-radius:6px; padding:10px; white-space:pre-wrap; max-height:420px; overflow:auto; }}
    canvas {{ width:100%; height:230px; border:1px solid var(--line); border-radius:7px; background:#fff; }}
    .warn {{ color:var(--warn); font-weight:700; }}
    .bad {{ color:var(--bad); font-weight:700; }}
    @media (max-width: 900px) {{ header {{ padding:16px; }} main {{ padding:8px; }} }}
  </style>
</head>
<body>
<header>
  <h1>TextArena Agent 指标分析与能力提升证据</h1>
  <div class="muted">输入: <code>{escape(str(result.get('eval_root')))}</code> + <code>{escape(str(result.get('memory_root')))}</code> | 生成时间: {escape(str(result.get('created_at')))}</div>
</header>
<main>
  <section>
    <h2>结论摘要</h2>
    <div class="grid">
      <div class="card"><b>整体统计</b><pre>{escape(json.dumps(result.get('aggregate'), ensure_ascii=False, indent=2))}</pre></div>
      <div class="card"><b>证据解释</b><p>当前报告把小样本运行标记为 <code>pilot_only</code>。要证明“agent 在博弈环境中因 memory/skill 进化而提升”，至少需要趋势曲线、no-memory ablation、固定对手池 Elo/TrueSkill、oracle exploitability，以及 skill/reflection/retrieval 事件同步增长。</p></div>
    </div>
  </section>
  <section>
    <h2>关键指标表</h2>
    <table><tr><th>Game</th><th>Soft skills</th><th>Evidence</th><th>Episodes</th><th>Δ win</th><th>Δ reward</th><th>Avg reward ± SE</th><th>Δ Elo no-mem</th><th>Δ Elo random</th><th>Invalid</th><th>Exp/Refl/SkillEvents</th><th>Interpretation</th></tr>{''.join(rows)}</table>
  </section>
  <section>
    <h2>可视化曲线</h2>
    <div class="grid">
      <div class="card"><b>Early/Mid/Late Reward</b><canvas id="rewardChart" width="720" height="260"></canvas></div>
      <div class="card"><b>Elo Ablation Delta</b><canvas id="eloChart" width="720" height="260"></canvas></div>
      <div class="card"><b>Memory / Skill Mechanism</b><canvas id="memoryChart" width="720" height="260"></canvas></div>
    </div>
  </section>
  <section>
    <h2>指标合同</h2>
    <table><tr><th>Metric</th><th>Why</th><th>Local field</th></tr>{contract_rows}</table>
  </section>
  <section>
    <h2>下一步实验</h2>
    <ul>{recs}</ul>
  </section>
  <section>
    <h2>Critic LLM Review</h2>
    <pre>{critic}</pre>
  </section>
  <section>
    <h2>参考资料与本地下载</h2>
    <ul>{refs}</ul>
  </section>
</main>
<script id="analysis-data" type="application/json">{escape(data)}</script>
<script>
const data = JSON.parse(document.getElementById('analysis-data').textContent);
function drawLine(canvas, series, labels) {{
  const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
  const pad=38, w=canvas.width-pad-16, h=canvas.height-pad-18;
  const vals = series.flatMap(s => s.values);
  const min = Math.min(0, ...vals), max = Math.max(1, ...vals);
  const x = i => pad + i * w / Math.max(1, labels.length - 1);
  const y = v => pad + h - (v - min) * h / Math.max(0.001, max - min);
  ctx.strokeStyle='#a9aaa5'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,pad+h); ctx.lineTo(pad+w,pad+h); ctx.stroke();
  ctx.fillStyle='#62666d'; ctx.font='12px Arial'; labels.forEach((l,i)=>ctx.fillText(l, x(i)-12, pad+h+16));
  const colors=['#136f63','#a15c18','#516b91','#7a6f45'];
  series.forEach((s,si)=>{{ ctx.strokeStyle=colors[si%colors.length]; ctx.lineWidth=2; ctx.beginPath(); s.values.forEach((v,i)=> i ? ctx.lineTo(x(i),y(v)) : ctx.moveTo(x(i),y(v))); ctx.stroke(); ctx.fillStyle=ctx.strokeStyle; ctx.fillText(s.label, pad+8, pad+14+si*16); }});
}}
function drawBar(canvas, labels, values) {{
  const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
  const pad=42, w=canvas.width-pad-16, h=canvas.height-pad-22, max=Math.max(1, ...values.map(v=>Math.abs(v)));
  ctx.strokeStyle='#a9aaa5'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,pad+h); ctx.lineTo(pad+w,pad+h); ctx.stroke();
  const bw=w/Math.max(1,labels.length);
  labels.forEach((l,i)=>{{ const v=values[i]; const bh=Math.abs(v)/max*h; ctx.fillStyle=v>=0?'#136f63':'#9f2f2f'; ctx.fillRect(pad+i*bw+8, pad+h-bh, Math.max(10,bw-16), bh); ctx.fillStyle='#62666d'; ctx.font='12px Arial'; ctx.fillText(l.slice(0,16), pad+i*bw+8, pad+h+16); ctx.fillText(String(v), pad+i*bw+8, pad+h-bh-4); }});
}}
const games = data.games || [];
drawLine(document.getElementById('rewardChart'), games.map(g => ({{label:g.game, values:(g.trend||[]).map(t => Number(t.avg_reward||0))}})), ['early','mid','late']);
drawBar(document.getElementById('eloChart'), games.map(g=>g.game), games.map(g=>Number(g.elo.evolved_minus_no_memory||0)));
drawBar(document.getElementById('memoryChart'), games.map(g=>g.game), games.map(g=>Number(g.memory.experiences||0)+Number(g.memory.reflections||0)+Number(g.memory.skill_updates||0)));
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze TextArena improvement, memory/skill evolution, and game-theoretic metrics.")
    parser.add_argument("--eval-root", default="workspace/analysis_current/eval_runs")
    parser.add_argument("--memory-root", default="workspace/analysis_current/memory")
    parser.add_argument("--output-dir", default="workspace/analysis_reports/textarena_metrics")
    parser.add_argument("--references-dir", default="workspace/reference_materials/textarena_metrics")
    parser.add_argument("--critic-llm", action="store_true", help="Ask an OpenAI-compatible critic LLM to review the evidence.")
    parser.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL)
    parser.add_argument("--no-timestamp", action="store_true", help="write legacy latest filenames instead of timestamped report filenames")
    args = parser.parse_args(argv)
    paths = AnalysisPaths(
        eval_root=Path(args.eval_root),
        memory_root=Path(args.memory_root),
        output_dir=Path(args.output_dir),
        references_dir=Path(args.references_dir),
    )
    result = analyze(paths, critic_llm=args.critic_llm, critic_model=args.critic_model)
    outputs = write_outputs(result, paths.output_dir, timestamped=not args.no_timestamp)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
