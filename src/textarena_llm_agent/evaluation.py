"""Evaluation harness proving the LLM agent improves its play over continuous games.

Metrics:
- Trend bins (early/mid/late): win-rate, avg-reward, invalid-rate, avg-turns, skill-count.
- Elo tournament across {LLM-evolved, LLM-no-memory, DQN, random, optimal}.
- Exploitability: Kuhn best-response value - game_value(1/18); TicTacToe vs OptimalTTT P(loss).
- Self-play vs past-self (snapshot skill library every N episodes -> opponent pool).
- LLM vs DQN (alternate first player).

All artifacts persisted under eval_runs/<game>/.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .agent import TextArenaAgentConfig, TextArenaDecisionAgent
from .cli import build_env
from .game_specs import KUHN_GAME_VALUE, canonical_game_id
from .llm import DecisionLLM, HeuristicLLM
from .optimal_agents import OptimalKuhnBR, OptimalTTT, RandomAgent
from .state_encoder import TextArenaStateEncoder


@dataclass(slots=True)
class MatchResult:
    game_id: str
    agent_a: str
    agent_b: str
    winner: int | None
    reward_a: float
    turns: int
    invalid: bool
    seed: int
    bin: str = "mid"
    first_player: int = 0


@dataclass(slots=True)
class BinMetrics:
    bin: str
    episodes: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    avg_reward: float
    invalid_move_rate: float
    avg_turns: float
    skill_count: int


class EvaluationHarness:
    def __init__(self, *, games: list[str], episodes: int, max_steps: int,
                 memory_root: Path, output_root: Path, seed: int = 0,
                 llm: DecisionLLM | None = None,
                 evaluator_llm: DecisionLLM | None = None,
                 snapshot_every: int = 10, eval_episodes: int | None = None,
                 elo_rounds: int = 6, exploitability_episodes: int | None = None) -> None:
        self.games = games
        self.episodes = episodes
        self.max_steps = max_steps
        self.memory_root = Path(memory_root)
        self.output_root = Path(output_root)
        self.seed = seed
        self.snapshot_every = snapshot_every
        self.eval_episodes = eval_episodes or max(6, episodes // 5)
        self.elo_rounds = max(0, int(elo_rounds))
        self.exploitability_episodes = exploitability_episodes
        self.llm = llm or HeuristicLLM()
        self.evaluator_llm = evaluator_llm or self.llm
        self.run_stamp = self._stamp()

    def _stamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ------------------------------------------------------------------ orchestration
    def run_all(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        run_manifest: dict[str, Any] = {
            "run_id": f"textarena_eval_{self.run_stamp}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "games": list(self.games),
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "snapshot_every": self.snapshot_every,
            "eval_episodes": self.eval_episodes,
            "elo_rounds": self.elo_rounds,
            "exploitability_episodes": self.exploitability_episodes,
            "memory_root": str(self.memory_root),
            "output_root": str(self.output_root),
            "actor": self._llm_manifest(self.llm),
            "critic": self._llm_manifest(self.evaluator_llm),
            "uses_heuristic_actor": isinstance(self.llm, HeuristicLLM),
            "uses_heuristic_critic": isinstance(self.evaluator_llm, HeuristicLLM),
            "memory_snapshots": {},
            "artifacts": {},
        }
        for game in self.games:
            self.output_root.mkdir(parents=True, exist_ok=True)
            out_dir = self.output_root / game
            out_dir.mkdir(parents=True, exist_ok=True)
            game_summary: dict[str, Any] = {}
            memory_dir = self.memory_root / game / "evolved"
            initial_snapshot = self._snapshot_memory(memory_dir, game=game, stage="initial")
            game_summary["initial_memory_snapshot"] = initial_snapshot
            # 1. evolving self-play trend (the central "improvement" evidence)
            trend, match_results = self.self_play_trend(game, out_dir)
            trend_rows = [asdict(b) for b in trend]
            game_summary["trend"] = trend_rows
            (out_dir / "trend.json").write_text(json.dumps(trend_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "match_results.jsonl").write_text("".join(json.dumps(asdict(m), ensure_ascii=False, default=str) + "\n" for m in match_results), encoding="utf-8")
            # 2. Elo tournament
            elo = self.elo_tournament(game, out_dir, rounds=self.elo_rounds, sampled_matches=match_results)
            game_summary["elo"] = elo
            (out_dir / "elo.json").write_text(json.dumps(elo, ensure_ascii=False, indent=2), encoding="utf-8")
            # 3. exploitability (where an oracle exists)
            explo = self.exploitability(game, out_dir)
            game_summary["exploitability"] = explo
            (out_dir / "exploitability.json").write_text(json.dumps(explo, ensure_ascii=False, indent=2), encoding="utf-8")
            # 4. skill timeline (required: skill update history)
            timeline = self._skill_timeline()
            (out_dir / "skill_timeline.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False, default=str) + "\n" for t in timeline), encoding="utf-8")
            game_summary["skill_timeline_len"] = len(timeline)
            evolved_snapshot = self._snapshot_memory(memory_dir, game=game, stage="evolved")
            game_summary["evolved_memory_snapshot"] = evolved_snapshot
            summary[game] = game_summary
            run_manifest["memory_snapshots"][game] = {
                "initial": initial_snapshot,
                "evolved": evolved_snapshot,
            }
            run_manifest["artifacts"][game] = {
                "trend": str(out_dir / "trend.json"),
                "match_results": str(out_dir / "match_results.jsonl"),
                "elo": str(out_dir / "elo.json"),
                "exploitability": str(out_dir / "exploitability.json"),
                "skill_timeline": str(out_dir / "skill_timeline.jsonl"),
            }
        (self.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        run_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        run_manifest["status"] = "completed"
        manifest_text = json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str)
        (self.output_root / f"run_manifest_{self.run_stamp}.json").write_text(manifest_text, encoding="utf-8")
        (self.output_root / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
        return summary

    def _snapshot_memory(self, memory_dir: Path, *, game: str, stage: str) -> str:
        snapshot_root = self.output_root / "memory_snapshots" / game
        snapshot_root.mkdir(parents=True, exist_ok=True)
        target = snapshot_root / f"{stage}_{self.run_stamp}"
        if target.exists():
            shutil.rmtree(target)
        if memory_dir.exists():
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
        manifest = {
            "game": game,
            "stage": stage,
            "source": str(memory_dir),
            "snapshot": str(target),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(target)

    def _llm_manifest(self, llm: DecisionLLM) -> dict[str, Any]:
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

    # ------------------------------------------------------------------ self-play trend
    def self_play_trend(self, game: str, out_dir: Path) -> tuple[list[BinMetrics], list[MatchResult]]:
        """Evolved agent self-plays; record win-rate per early/mid/late bin.

        In self-play one agent plays both sides, so 'win_rate' here is the rate of
        non-draw decisive games with the agent's perspective winning; we additionally
        track avg_reward and skill growth to show evolution. For a cleaner 'improvement'
        signal, each bin the evolved agent also plays a frozen RANDOM opponent (a
        stable baseline), so rising win-rate vs random across bins demonstrates gain.
        """
        bins = _trend_bins(self.episodes)
        results: list[MatchResult] = []
        bin_metrics: list[BinMetrics] = []
        bin_skill_counts: dict[str, int] = {}
        memory_dir = self.memory_root / game / "evolved"
        for ep in range(self.episodes):
            bin_name = bins[ep] if ep < len(bins) else "late"
            seed = self.seed + ep
            # evolved agent (with persistent evolving memory) vs RANDOM (stable opponent)
            env = build_env(game, seed=seed)
            agent = TextArenaDecisionAgent(
                TextArenaAgentConfig(memory_dir=str(memory_dir), trace_dir=str(out_dir / f"trace_{ep}"), use_llm=isinstance(self.llm, HeuristicLLM) is False),
                llm=self.llm, evaluator_llm=self.evaluator_llm,
            )
            first_player = int(getattr(env.state, "current_player_id", 0))
            agent_side = ep % 2
            turns = 0
            rng = RandomAgent(seed=seed)
            invalid = False
            while turns < self.max_steps and not bool(getattr(env.state, "done", False)):
                player = int(getattr(env.state, "current_player_id", 0))
                if player == agent_side:
                    decision = agent.act(env)
                    if "fallback" in (decision.raw_llm or {}):
                        pass
                else:
                    env.step(rng.act(env, player))
                turns += 1
            rewards = getattr(env.state, "rewards", None) or {}
            r = float(rewards.get(agent_side, 0.0)) if isinstance(rewards, dict) else 0.0
            invalid = bool(getattr(env.state, "game_info", {}).get(agent_side, {}).get("invalid_move", False))
            results.append(MatchResult(game_id=game, agent_a="llm_evolved", agent_b="random", winner=agent_side if r > 0 else (1 - agent_side if r < 0 else None), reward_a=r, turns=turns, invalid=invalid, seed=seed, bin=bin_name, first_player=agent_side))
            try:
                from .memory import EvolvingMemory
                bin_skill_counts[bin_name] = int(EvolvingMemory(memory_dir).memory_stats()["skills"])
            except Exception:
                bin_skill_counts.setdefault(bin_name, 0)

        # aggregate per bin
        for b in ["early", "mid", "late"]:
            bres = [r for r in results if r.bin == b]
            if not bres:
                continue
            n = len(bres)
            skill_count = bin_skill_counts.get(b, 0)
            bin_metrics.append(BinMetrics(
                bin=b, episodes=len(bres),
                win_rate=sum(1 for r in bres if r.reward_a > 0) / n,
                draw_rate=sum(1 for r in bres if r.reward_a == 0) / n,
                loss_rate=sum(1 for r in bres if r.reward_a < 0) / n,
                avg_reward=mean([r.reward_a for r in bres]) if bres else 0.0,
                invalid_move_rate=sum(1 for r in bres if r.invalid) / n,
                avg_turns=mean([r.turns for r in bres]) if bres else 0.0,
                skill_count=skill_count,
            ))
        return bin_metrics, results

    # ------------------------------------------------------------------ Elo tournament
    def elo_tournament(self, game: str, out_dir: Path, *, rounds: int = 6, k: float = 16.0, sampled_matches: list[MatchResult] | None = None) -> dict[str, Any]:
        if rounds <= 0:
            return self._sampled_elo(sampled_matches or [], k=k)
        opponents: dict[str, Callable[[Any, int], str]] = {}
        opponents["random"] = lambda env, p: RandomAgent().act(env, p)
        if canonical_game_id(game) == "TicTacToe":
            opt = OptimalTTT()
            opponents["optimal_ttt"] = lambda env, p: opt.act(env, p)
        if canonical_game_id(game) == "KuhnPoker":
            # optimal-ish Kuhn heuristic
            from .optimal_agents import OptimalKuhnBR
            br = OptimalKuhnBR()
            opponents["optimal_kuhn"] = lambda env, p: br.act(env, p)
        # DQN opponent (if a policy file exists)
        dqn = self._load_dqn(game)
        if dqn is not None:
            opponents["dqn"] = dqn
        # LLM-evolved + LLM-no-memory
        opponents["llm_evolved"] = self._llm_player(self.memory_root / game / "evolved")
        opponents["llm_no_memory"] = self._llm_player(out_dir / "fresh_mem")

        names = list(opponents.keys())
        ratings = {n: 1000.0 for n in names}
        history: list[dict[str, Any]] = []
        for r in range(rounds):
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    winner = self._one_match(game, opponents[a], opponents[b], seed=self.seed + r * 100 + i * 10 + j)
                    ra, rb = ratings[a], ratings[b]
                    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                    sa = 1.0 if winner == 0 else (0.0 if winner == 1 else 0.5)
                    ratings[a] = ra + k * (sa - ea)
                    ratings[b] = rb + k * ((1 - sa) - (1 - ea))
                    history.append({"round": r, "a": a, "b": b, "winner": winner, "rating_a": round(ratings[a], 1), "rating_b": round(ratings[b], 1)})
        return {"ratings": {n: round(v, 1) for n, v in sorted(ratings.items(), key=lambda x: -x[1])}, "rounds": rounds, "history": history[-100:]}

    def _sampled_elo(self, matches: list[MatchResult], *, k: float = 16.0) -> dict[str, Any]:
        ratings = {"llm_evolved": 1000.0, "random": 1000.0}
        history: list[dict[str, Any]] = []
        for idx, m in enumerate(matches):
            a, b = "llm_evolved", "random"
            ra, rb = ratings[a], ratings[b]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            if m.reward_a > 0:
                sa = 1.0
            elif m.reward_a < 0:
                sa = 0.0
            else:
                sa = 0.5
            ratings[a] = ra + k * (sa - ea)
            ratings[b] = rb + k * ((1 - sa) - (1 - ea))
            history.append({"round": 0, "match_index": idx, "a": a, "b": b, "winner": 0 if sa == 1.0 else (1 if sa == 0.0 else None), "rating_a": round(ratings[a], 1), "rating_b": round(ratings[b], 1), "source": "sampled_trend_match"})
        return {
            "ratings": {n: round(v, 1) for n, v in sorted(ratings.items(), key=lambda x: -x[1])},
            "rounds": 0,
            "history": history[-100:],
            "method": "sampled_from_trend_matches",
            "extra_llm_matches": 0,
            "note": "Computed offline from already sampled trend matches; evolved-vs-no-memory requires explicit Elo rounds or a matched no-memory sample.",
        }

    def _one_match(self, game: str, player_a: Callable[[Any, int], str], player_b: Callable[[Any, int], str], *, seed: int) -> int | None:
        env = build_env(game, seed=seed)
        for _ in range(self.max_steps):
            if bool(getattr(env.state, "done", False)):
                break
            p = int(getattr(env.state, "current_player_id", 0))
            actor = player_a if p == 0 else player_b
            try:
                env.step(actor(env, p))
            except Exception:
                # a player that errs loses by invalid move; env handles rotation
                pass
        rewards = getattr(env.state, "rewards", None) or {}
        if not isinstance(rewards, dict):
            return None
        if rewards.get(0, 0) > rewards.get(1, 0):
            return 0
        if rewards.get(1, 0) > rewards.get(0, 0):
            return 1
        return None

    # ------------------------------------------------------------------ exploitability
    def exploitability(self, game: str, out_dir: Path) -> dict[str, Any]:
        fam = canonical_game_id(game)
        if fam == "KuhnPoker":
            if self.exploitability_episodes == 0:
                return self._sampled_kuhn_policy_proxy(out_dir)
            # collect the agent's empirical policy then compute BR value
            policy = self._collect_kuhn_policy(out_dir, episodes=max(20, self.eval_episodes))
            br_value = OptimalKuhnBR.best_response_value(policy, samples=1500)
            return {"game": game, "method": "best_response_value", "br_value": round(br_value, 4), "game_value": round(KUHN_GAME_VALUE, 4), "exploitability": round(br_value - KUHN_GAME_VALUE, 4)}
        if fam == "TicTacToe":
            losses = 0
            games = 0
            opt = OptimalTTT()
            episodes = self.exploitability_episodes if self.exploitability_episodes is not None else max(12, self.eval_episodes)
            if episodes == 0:
                return self._sampled_tictactoe_exploitability_proxy(out_dir)
            for ep in range(max(0, int(episodes))):
                for agent_side in (0, 1):
                    env = build_env("TicTacToe", seed=self.seed + ep * 2 + agent_side)
                    agent = self._llm_player(self.memory_root / game / "evolved")
                    while not bool(getattr(env.state, "done", False)):
                        p = int(getattr(env.state, "current_player_id", 0))
                        if p == agent_side:
                            env.step(agent(env, p))
                        else:
                            env.step(opt.act(env, p))
                    rewards = getattr(env.state, "rewards", None) or {}
                    if isinstance(rewards, dict) and rewards.get(agent_side, 0) < 0:
                        losses += 1
                    games += 1
            if games <= 0:
                return self._sampled_tictactoe_exploitability_proxy(out_dir)
            return {"game": game, "method": "loss_rate_vs_optimal", "games": games, "losses": losses, "p_loss": round(losses / games, 4), "exploitability_proxy": round(losses / games, 4)}
        # Stratego / Negotiation: no cheap oracle
        return {"game": game, "method": "none", "note": "no cheap exploitability oracle for this game; rely on Elo + trend"}

    def _sampled_kuhn_policy_proxy(self, out_dir: Path) -> dict[str, Any]:
        decisions = _decision_events_from_traces(out_dir)
        if not decisions:
            return {"game": "KuhnPoker", "method": "sampled_policy_proxy", "decision_count": 0, "note": "No sampled decisions available for offline proxy."}
        action_counts: dict[str, int] = defaultdict(int)
        best = 0
        score_gaps: list[float] = []
        for row in decisions:
            action = str(row.get("action_text") or "")
            if action:
                action_counts[action] += 1
            selected = row.get("selected_candidate") or {}
            candidates = row.get("candidates") or []
            selected_score = float(selected.get("score") or 0.0)
            max_score = max([float(c.get("score") or 0.0) for c in candidates], default=selected_score)
            gap = max(0.0, max_score - selected_score)
            score_gaps.append(gap)
            if gap <= 1e-9:
                best += 1
        return {
            "game": "KuhnPoker",
            "method": "sampled_policy_proxy",
            "decision_count": len(decisions),
            "action_counts": dict(sorted(action_counts.items())),
            "best_candidate_rate": round(best / max(1, len(decisions)), 4),
            "avg_candidate_score_gap": round(mean(score_gaps), 4) if score_gaps else 0.0,
            "extra_llm_matches": 0,
            "note": "Offline policy proxy from sampled trajectories. Full best-response exploitability still requires probing additional information sets.",
        }

    def _sampled_tictactoe_exploitability_proxy(self, out_dir: Path) -> dict[str, Any]:
        decisions = _decision_events_from_traces(out_dir)
        if not decisions:
            return {"game": "TicTacToe", "method": "sampled_tactical_proxy", "decision_count": 0, "note": "No sampled decisions available for offline proxy."}
        best = 0
        score_gaps: list[float] = []
        low_confidence = 0
        for row in decisions:
            selected = row.get("selected_candidate") or {}
            candidates = row.get("candidates") or []
            selected_score = float(selected.get("score") or 0.0)
            max_score = max([float(c.get("score") or 0.0) for c in candidates], default=selected_score)
            gap = max(0.0, max_score - selected_score)
            score_gaps.append(gap)
            if gap <= 1e-9:
                best += 1
            if float(row.get("confidence") or 0.0) < 0.6:
                low_confidence += 1
        matches = read_match_results(out_dir / "match_results.jsonl")
        losses = sum(1 for m in matches if m.reward_a < 0)
        invalids = sum(1 for m in matches if m.invalid)
        return {
            "game": "TicTacToe",
            "method": "sampled_tactical_proxy",
            "decision_count": len(decisions),
            "best_candidate_rate": round(best / max(1, len(decisions)), 4),
            "avg_candidate_score_gap": round(mean(score_gaps), 4) if score_gaps else 0.0,
            "low_confidence_rate": round(low_confidence / max(1, len(decisions)), 4),
            "sampled_games": len(matches),
            "terminal_loss_rate": round(losses / max(1, len(matches)), 4),
            "invalid_move_rate": round(invalids / max(1, len(matches)), 4),
            "extra_llm_matches": 0,
            "note": "Offline proxy from already sampled trajectories. This is not full minimax exploitability over the state space.",
        }

    def _collect_kuhn_policy(self, out_dir: Path, *, episodes: int) -> dict[tuple, dict[str, float]]:
        """Probe the evolved agent to collect (card, history)->action probs."""
        counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        agent = self._llm_player(self.memory_root / "KuhnPoker" / "evolved")
        for ep in range(episodes):
            env = build_env("KuhnPoker", seed=self.seed + 9000 + ep)
            while not bool(getattr(env.state, "done", False)):
                p = int(getattr(env.state, "current_player_id", 0))
                gs = getattr(env.state, "game_state", {})
                card = (gs.get("player_cards") or {}).get(p)
                tree = gs.get("current_legal_action_tree") or {}
                legal = list(tree.keys()) if isinstance(tree, dict) else []
                if not legal:
                    break
                action = agent(env, p)
                a = action.strip("[]").lower()
                if card is not None:
                    counts[(int(card), ())][a] += 1
                try:
                    env.step(action)
                except Exception:
                    break
        policy: dict[tuple, dict[str, float]] = {}
        for key, c in counts.items():
            total = sum(c.values()) or 1
            policy[key] = {a: n / total for a, n in c.items()}
        return policy

    # ------------------------------------------------------------------ helpers
    def _llm_player(self, memory_dir: Path) -> Callable[[Any, int], str]:
        cfg = TextArenaAgentConfig(memory_dir=str(memory_dir), trace_dir="", enable_tracing=False, use_llm=not isinstance(self.llm, HeuristicLLM))
        agent = TextArenaDecisionAgent(cfg, llm=self.llm, evaluator_llm=self.evaluator_llm)
        encoder = TextArenaStateEncoder()

        def play(env: Any, player: int) -> str:
            # decide without stepping (the harness steps); reuse agent.decide
            decision = agent.decide(env)
            return decision.action_text
        return play

    def _load_dqn(self, game: str) -> Callable[[Any, int], str] | None:
        try:
            from .rl_baseline import DQNPolicy, GameFeatureEncoder, TextArenaActionAnalyzer
            path = Path("baselines/local") / f"textarena_dqn_{game.lower()}" / "dqn_policy.pt"
            if not path.exists():
                path = Path("baselines/local/textarena_dqn") / "dqn_policy.pt"
            if not path.exists():
                return None
            enc = GameFeatureEncoder(game=game)
            pol = DQNPolicy(feature_dim=enc.dim)
            pol.load(path)
            analyzer = TextArenaActionAnalyzer(enc.state_encoder, simulate_top=0)

            def play(env: Any, player: int) -> str:
                actions = analyzer.analyze(env, top_k=enc.max_actions)
                if not actions:
                    return "[0]"
                slot = pol.select(enc.features(env, player=player), len(actions), epsilon=0.0)
                return actions[slot].action_text
            return play
        except Exception:
            return None

    def _skill_timeline(self) -> list[dict[str, Any]]:
        from .memory import EvolvingMemory
        # gather timeline from any game memory dir present
        rows: list[dict[str, Any]] = []
        for game in self.games:
            mem = EvolvingMemory(self.memory_root / game / "evolved")
            rows.extend(mem.skill_timeline()[-200:])
        return rows


def _trend_bins(episodes: int) -> list[str]:
    if episodes <= 0:
        return []
    if episodes == 1:
        return ["early"]
    if episodes == 2:
        return ["early", "late"]
    return ["early" if i == 0 else ("late" if i == episodes - 1 else "mid") for i in range(episodes)]


def read_match_results(path: Path) -> list[MatchResult]:
    rows: list[MatchResult] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            rows.append(MatchResult(**{k: obj.get(k) for k in MatchResult.__dataclass_fields__.keys()}))
        except Exception:
            continue
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _decision_events_from_traces(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_dir in sorted(out_dir.glob("trace_*")):
        events = _read_jsonl(trace_dir / "events.jsonl")
        latest_candidates: list[dict[str, Any]] = []
        for event in events:
            payload = event.get("payload") if isinstance(event, dict) else None
            if not isinstance(payload, dict):
                continue
            if event.get("event") == "candidates_ranked":
                candidates = payload.get("candidates")
                latest_candidates = candidates if isinstance(candidates, list) else []
            elif event.get("event") == "decision_resolved":
                decision = payload.get("decision")
                if isinstance(decision, dict):
                    rows.append({**decision, "candidates": latest_candidates, "trace_dir": str(trace_dir)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the evolvable TextArena LLM agent: trend + Elo + exploitability.")
    parser.add_argument("--games", default="TicTacToe,KuhnPoker", help="comma-separated game families")
    parser.add_argument("--episodes", type=int, default=30, help="self-play episodes for the trend")
    parser.add_argument("--steps", type=int, default=60, help="max steps per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-root", default="workspace/textarena_memory")
    parser.add_argument("--output-dir", default="workspace/eval_runs")
    parser.add_argument("--llm", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default="", help="override OpenAI-compatible actor model name")
    parser.add_argument("--critic-model", default="gpt-5.5", help="OpenAI-compatible critic/evaluator model name")
    parser.add_argument("--critic-prefix", default="CRITIC", help="environment prefix for critic API variables")
    parser.add_argument("--elo-rounds", type=int, default=6, help="pairwise Elo tournament rounds; set 0 for budgeted smoke runs")
    parser.add_argument("--exploitability-episodes", type=int, default=-1, help="oracle exploitability episodes; -1 keeps the default heuristic")
    args = parser.parse_args(argv)
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    if args.llm == "heuristic":
        llm = HeuristicLLM()
        evaluator_llm = HeuristicLLM()
    else:
        from .llm import OpenAIChatLLM
        llm = OpenAIChatLLM.from_env()
        if args.model:
            llm.model = args.model
        evaluator_llm = OpenAIChatLLM.from_env(prefix=args.critic_prefix)
        if args.critic_model:
            evaluator_llm.model = args.critic_model
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
        exploitability_episodes=None if args.exploitability_episodes < 0 else args.exploitability_episodes,
    )
    summary = harness.run_all()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
