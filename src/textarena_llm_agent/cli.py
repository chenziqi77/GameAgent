from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import textarena as ta

from .agent import TextArenaAgentConfig, TextArenaDecisionAgent
from .game_specs import default_env_id
from .llm import HeuristicLLM, OpenAIChatLLM
from .tracing import TextArenaRunTracer
from .visualization import TextArenaVisualizationServer


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _snapshot_memory(memory_dir: Path, snapshot_root: Path, *, stage: str, stamp: str) -> str:
    target = snapshot_root / f"{stage}_{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    if memory_dir.exists():
        # Checkpoint any live sqlite so the .db file is self-contained, then skip
        # the -wal / -shm sidecars during copy (they hold live locks on Windows).
        for db_path in memory_dir.glob("*.sqlite"):
            try:
                import sqlite3
                with sqlite3.connect(str(db_path)) as _c:
                    _c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
        shutil.copytree(
            memory_dir,
            target,
            ignore=shutil.ignore_patterns("*.sqlite-wal", "*.sqlite-shm"),
        )
    else:
        target.mkdir(parents=True, exist_ok=True)
        for name in ["experiences.jsonl", "skills.jsonl", "skill_updates.jsonl", "reflections.jsonl", "retrieval_hits.jsonl", "prompt_patches.jsonl"]:
            (target / name).write_text("", encoding="utf-8")
        (target / "rules.md").write_text("# Empty initial memory snapshot\n", encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps({"stage": stage, "source": str(memory_dir), "snapshot": str(target), "created_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def _llm_manifest(llm) -> dict[str, object]:
    return {
        "class": llm.__class__.__name__,
        "model": getattr(llm, "model", None),
        "base_url": getattr(llm, "base_url", None),
        "api_key_env_present": {
            "MCP_API_KEY": bool(os.getenv("MCP_API_KEY")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "SCS_LLM_API_KEY": bool(os.getenv("SCS_LLM_API_KEY")),
            "CRITIC_API_KEY": bool(os.getenv("CRITIC_API_KEY")),
        },
    }


def build_env(game: str, *, seed: int | None = None, num_players: int = 2):
    env_id = game if "-v0" in game else default_env_id(game)
    env = ta.make(env_id)
    env.reset(num_players=num_players, seed=seed)
    return env


# ---------------------------------------------------------------------------
# Subcommand: run (legacy default)
# ---------------------------------------------------------------------------


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--game", default="TicTacToe", help="Game family or TextArena env id")
    parser.add_argument("--steps", type=int, default=40, help="max decision steps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--memory-dir", default="workspace/textarena_memory")
    parser.add_argument("--llm", choices=["heuristic", "openai"], default="openai")
    parser.add_argument("--model", default="", help="override OpenAI-compatible model name")
    parser.add_argument("--critic-model", default="gpt-5.5", help="OpenAI-compatible evaluator/critic model name")
    parser.add_argument("--critic-prefix", default="CRITIC", help="environment prefix for critic API variables, e.g. CRITIC_MODEL")
    parser.add_argument("--llm-max-tokens", type=int, default=900, help="max output tokens for LLM decisions/evaluation")
    parser.add_argument("--disable-tools", action="store_true", help="skip the LLM tool-calling loop for cheaper direct JSON decisions")
    parser.add_argument("--heuristic-evaluator", action="store_true", help="use local heuristic evaluator while keeping LLM decisions")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--no-evaluator-override", action="store_true")
    parser.add_argument("--jsonl", default="", help="optional path to save decision trace jsonl")
    parser.add_argument("--trace-dir", default="workspace/textarena_runs/latest")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--pause-at-start", action="store_true")
    parser.add_argument("--step-delay", type=float, default=0.0)


def _cmd_run(args: argparse.Namespace) -> int:
    stamp = _timestamp()
    llm = OpenAIChatLLM.from_env() if args.llm == "openai" else HeuristicLLM()
    if args.model and isinstance(llm, OpenAIChatLLM):
        llm.model = args.model
    if args.heuristic_evaluator:
        evaluator_llm = HeuristicLLM()
    elif args.llm == "openai" and (args.critic_model or args.critic_prefix):
        evaluator_llm = OpenAIChatLLM.from_env(prefix=args.critic_prefix)
        if args.critic_model:
            evaluator_llm.model = args.critic_model
    else:
        evaluator_llm = llm
    tracer = TextArenaRunTracer(args.trace_dir)
    tracer.write_control({"paused": bool(args.pause_at_start), "step_requested": False, "stop_requested": False})
    memory_dir = Path(args.memory_dir)
    snapshot_root = Path(args.trace_dir) / "memory_snapshots"
    initial_snapshot = _snapshot_memory(memory_dir, snapshot_root, stage="initial", stamp=stamp)
    cfg = TextArenaAgentConfig(
        memory_dir=args.memory_dir,
        top_k_actions=args.top_k,
        use_llm=args.llm == "openai",
        allow_evaluator_override=not args.no_evaluator_override,
        decision_max_tokens=args.llm_max_tokens,
        evaluator_max_tokens=args.llm_max_tokens,
        enable_tools_in_loop=not args.disable_tools,
        enable_tool_synthesis=not args.disable_tools,
        trace_dir=args.trace_dir,
        enable_tracing=True,
    )
    env = build_env(args.game, seed=args.seed)
    agent = TextArenaDecisionAgent(cfg, llm=llm, evaluator_llm=evaluator_llm, tracer=tracer)
    server = None
    if args.visualize:
        server = TextArenaVisualizationServer(tracer, host=args.dashboard_host, port=args.dashboard_port)
        print(f"dashboard={server.start(open_browser=args.open_browser)}")
    out_path = Path(args.jsonl) if args.jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
    run_status = "completed"
    run_error: str | None = None
    try:
        for step in range(args.steps):
            if tracer.read_control().get("stop_requested"):
                print(f"stopped by dashboard before step {step}")
                break
            if bool(getattr(env.state, "done", False)):
                print(f"terminal after {step} steps; rewards={getattr(env.state, 'rewards', None)}")
                break
            decision = agent.act(env)
            print(f"{step:03d} action={decision.candidate_id} {decision.action_text}")
            print(f"     rationale={decision.rationale}")
            print(f"     eval={decision.evaluation.get('score')} accept={decision.evaluation.get('accept')} critique={decision.evaluation.get('critique')}")
            if out_path:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(decision.to_json().replace("\n", " ") + "\n")
            if args.step_delay > 0:
                time.sleep(max(0.0, args.step_delay))
    except RuntimeError as exc:
        if "Visualization requested stop" not in str(exc):
            run_status = "failed"
            run_error = str(exc)
            raise
        run_status = "stopped"
        print("stopped by dashboard request")
    except Exception as exc:
        run_status = "failed"
        run_error = str(exc)
        raise
    finally:
        evolved_snapshot = _snapshot_memory(memory_dir, snapshot_root, stage="evolved", stamp=stamp)
        run_manifest = {
            "run_id": f"textarena_agent_{stamp}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "game": args.game,
            "steps": args.steps,
            "seed": args.seed,
            "memory_dir": str(memory_dir),
            "trace_dir": args.trace_dir,
            "jsonl": args.jsonl or None,
            "actor": _llm_manifest(llm),
            "critic": _llm_manifest(evaluator_llm),
            "uses_heuristic_actor": isinstance(llm, HeuristicLLM),
            "uses_heuristic_critic": isinstance(evaluator_llm, HeuristicLLM),
            "critic_updates_memory_during_run": True,
            "initial_memory_snapshot": initial_snapshot,
            "evolved_memory_snapshot": evolved_snapshot,
            "status": run_status,
            "error": run_error,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_text = json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str)
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.trace_dir) / f"run_manifest_{stamp}.json").write_text(manifest_text, encoding="utf-8")
        (Path(args.trace_dir) / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
        if server is not None:
            server.stop()
    return 0


# ---------------------------------------------------------------------------
# Subcommand: replay-eval
# ---------------------------------------------------------------------------


def _cmd_replay_eval(args: argparse.Namespace) -> int:
    """Offline A/B over the evidence graph — no env, no LLM."""
    from .evidence_graph import EvidenceGraph
    from .hypothesis import replay_eval

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(json.dumps({"error": f"evidence graph not found: {graph_path}"}))
        return 2
    graph = EvidenceGraph(str(graph_path))
    try:
        result = replay_eval(graph, policy_a=args.policy_a, policy_b=args.policy_b,
                             limit=args.limit)
    finally:
        try:
            graph.close()
        except Exception:
            pass
    payload = result.to_dict()
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: tournament — Elo tournament across the existing harness
# ---------------------------------------------------------------------------


def _cmd_tournament(args: argparse.Namespace) -> int:
    from .evaluation import EvaluationHarness
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    if args.llm == "openai":
        llm = OpenAIChatLLM.from_env()
        if args.model:
            llm.model = args.model
        evaluator_llm = OpenAIChatLLM.from_env(prefix=args.critic_prefix)
        if args.critic_model:
            evaluator_llm.model = args.critic_model
    else:
        llm = HeuristicLLM()
        evaluator_llm = HeuristicLLM()
    harness = EvaluationHarness(
        games=games,
        episodes=args.episodes,
        max_steps=args.steps,
        memory_root=Path(args.memory_root),
        output_root=Path(args.output_dir),
        seed=args.seed,
        llm=llm,
        evaluator_llm=evaluator_llm,
        elo_rounds=args.elo_rounds,
    )
    out: dict[str, object] = {}
    for game in games:
        out_dir = Path(args.output_dir) / game
        out_dir.mkdir(parents=True, exist_ok=True)
        elo = harness.elo_tournament(game, out_dir, rounds=args.elo_rounds)
        (out_dir / "elo.json").write_text(json.dumps(elo, ensure_ascii=False, indent=2), encoding="utf-8")
        out[game] = elo
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: hypothesis-report — run the harness and emit Markdown
# ---------------------------------------------------------------------------


def _cmd_hypothesis_report(args: argparse.Namespace) -> int:
    from .hypothesis import (
        DEFAULT_HYPOTHESES,
        HypothesisHarness,
        baseline_templates,
    )

    hypotheses = list(DEFAULT_HYPOTHESES)
    if args.hypothesis:
        wanted = set(args.hypothesis)
        hypotheses = [h for h in hypotheses if h.id in wanted]
        if not hypotheses:
            print(json.dumps({"error": "no matching hypothesis IDs",
                              "requested": list(wanted)}))
            return 2

    baselines = baseline_templates()
    if args.arms:
        wanted_arms = set(args.arms)
        baselines = [b for b in baselines if b.id in wanted_arms]
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    out_path = Path(args.out) if args.out else None
    output_dir = out_path.parent if out_path else None
    harness = HypothesisHarness(
        baselines=baselines,
        hypotheses=hypotheses,
        games=games,
        n_episodes=args.episodes,
        seed=args.seed,
        opponent=args.opponent,
        output_dir=output_dir,
    )
    result = harness.run()
    payload = result.to_dict()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from .hypothesis import render_hypothesis_report
        out_path.write_text(render_hypothesis_report(result), encoding="utf-8")
        (out_path.parent / "hypothesis_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    summary = {
        "hypotheses": [
            {"id": h.hypothesis_id, "passed": h.passed,
             "metric": h.metric, "value_a": h.value_a, "value_b": h.value_b}
            for h in result.hypotheses
        ],
        "arms": result.arms,
        "n_matches": len(result.matches),
        "report_path": str(out_path) if out_path else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# Main / argument parser
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an evolvable LLM/MINIAgent-style agent on TextArena games.",
        prog="textarena_llm_agent",
    )
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="run one or more decision steps in a live environment (default)")
    _add_run_args(run_p)
    run_p.set_defaults(func=_cmd_run)

    rep = sub.add_parser("replay-eval", help="offline A/B from the evidence graph")
    rep.add_argument("--graph", default="workspace/textarena_memory/evidence_graph.sqlite",
                     help="path to the evidence graph sqlite file")
    rep.add_argument("--policy-a", required=True, dest="policy_a")
    rep.add_argument("--policy-b", required=True, dest="policy_b")
    rep.add_argument("--limit", type=int, default=100,
                     help="max episodes per cohort")
    rep.add_argument("--out", default="", help="optional output JSON path")
    rep.set_defaults(func=_cmd_replay_eval)

    tour = sub.add_parser("tournament", help="run Elo tournament across the standard pool")
    tour.add_argument("--games", default="TicTacToe,KuhnPoker")
    tour.add_argument("--episodes", type=int, default=20)
    tour.add_argument("--steps", type=int, default=60)
    tour.add_argument("--seed", type=int, default=0)
    tour.add_argument("--memory-root", default="workspace/textarena_memory")
    tour.add_argument("--output-dir", default="workspace/eval_runs")
    tour.add_argument("--llm", choices=["heuristic", "openai"], default="heuristic")
    tour.add_argument("--model", default="")
    tour.add_argument("--critic-model", default="gpt-5.5")
    tour.add_argument("--critic-prefix", default="CRITIC")
    tour.add_argument("--elo-rounds", type=int, default=4)
    tour.set_defaults(func=_cmd_tournament)

    hyp = sub.add_parser("hypothesis-report",
                         help="evaluate hypotheses across baselines and emit a Markdown report")
    hyp.add_argument("--hypothesis", "-H", action="append", default=[],
                     help="hypothesis ID(s) to evaluate; default is all (H1..H4)")
    hyp.add_argument("--arms", action="append", default=[],
                     help="restrict to the named baseline ids (default: all 9)")
    hyp.add_argument("--games", default="TicTacToe,KuhnPoker")
    hyp.add_argument("--episodes", type=int, default=20)
    hyp.add_argument("--seed", type=int, default=0)
    hyp.add_argument("--opponent", default="random",
                     help="baseline id used as the common opponent")
    hyp.add_argument("--out", default="workspace/eval_runs/hypothesis_report.md",
                     help="Markdown report output path")
    hyp.set_defaults(func=_cmd_hypothesis_report)

    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # backwards-compat: invocations without a subcommand fall back to `run`
        _add_run_args(parser)
        args = parser.parse_args(argv)
        return _cmd_run(args)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
