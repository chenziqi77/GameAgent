"""Hypothesis-driven evaluation per 改进需求.pdf §7.

This module is the Phase-7 evaluation backbone. It provides:

1. ``BaselineSpec`` — declarative description of one of the 9 ablation baselines
   listed in the plan (Random / Heuristic / RawLLM / LLM+RAG /
   LLM+Reflection / LLM+Skill_no_provenance / LLM+Skill_provenance /
   LLM+Skill+Tools / Full). Each spec captures which subsystems are enabled
   and a factory that materialises a callable player ``(env, player) -> str``.

2. ``Hypothesis`` — a hypothesis tying an ID, statement, the baselines that
   form the contrast (``arms``), and the metric pivot that determines pass /
   fail.

3. ``MetricBundle`` — five metric *classes* (棋力 / 学习效率 / 泛化 /
   技能质量 / 系统效率) collapsed into a single typed dict so the report
   writer can render them uniformly without re-knowing each game's vocabulary.

4. ``HypothesisHarness`` — orchestrates: build baselines → run match grid →
   compute metrics → judge each hypothesis. The harness is deliberately
   side-effect-light (writes to ``output_dir`` only) and accepts an injected
   ``match_runner`` so unit tests can deterministically replace the
   environment loop.

5. ``replay_eval(graph, policy_a, policy_b, ...)`` — offline A/B by replaying
   logged ``decision_frame`` rows from the evidence graph instead of running
   live episodes. Useful for cheap "did policy v3 over-fit?" sweeps.

6. ``opponent_pool_snapshot(memory_dir, dest, policy_version)`` — freeze the
   active skill JSONL + a manifest into ``workspace/opponent_pool/<v>/`` so
   later runs can re-load that exact agent as a tournament opponent.

Nothing here calls ``textarena.make`` at import time — that keeps the unit
tests fast and lets the harness be exercised on a mock match runner.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------


BaselineID = str
PlayerFn = Callable[[Any, int], str]
PlayerFactory = Callable[[], PlayerFn]


@dataclass(slots=True, frozen=True)
class BaselineSpec:
    """Static description of one ablation arm.

    ``flags`` mirrors the subsystem toggles in the plan so the report can
    print "✓ memory ✓ reflection ✗ skills" rows. ``factory`` is what the
    harness actually calls to get a player; in tests we hand in a stub
    factory.
    """
    id: BaselineID
    label: str
    description: str
    flags: dict[str, bool]
    factory: PlayerFactory | None = None

    def with_factory(self, factory: PlayerFactory) -> "BaselineSpec":
        return BaselineSpec(
            id=self.id, label=self.label, description=self.description,
            flags=dict(self.flags), factory=factory,
        )


# 9-baseline matrix per plan §7
_BASELINE_TEMPLATES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        id="random", label="Random",
        description="Uniform over legal actions; floor of play strength.",
        flags={"llm": False, "rag": False, "reflection": False, "skill": False, "provenance": False, "tools": False},
    ),
    BaselineSpec(
        id="heuristic", label="Heuristic-v0",
        description="Hand-coded heuristic LLM (HeuristicLLM); no learning.",
        flags={"llm": False, "rag": False, "reflection": False, "skill": False, "provenance": False, "tools": False},
    ),
    BaselineSpec(
        id="raw_llm", label="RawLLM",
        description="LLM with no memory, no reflection, no skills, no tools.",
        flags={"llm": True, "rag": False, "reflection": False, "skill": False, "provenance": False, "tools": False},
    ),
    BaselineSpec(
        id="llm_rag", label="LLM+RAG",
        description="LLM + BM25 episodic retrieval; no reflection, no skills.",
        flags={"llm": True, "rag": True, "reflection": False, "skill": False, "provenance": False, "tools": False},
    ),
    BaselineSpec(
        id="llm_reflection", label="LLM+Reflection",
        description="LLM + Reflexion summary injection; no skill memory.",
        flags={"llm": True, "rag": True, "reflection": True, "skill": False, "provenance": False, "tools": False},
    ),
    BaselineSpec(
        id="llm_skill_no_provenance", label="LLM+Skill_no_provenance",
        description="LLM + skills, but skills can be promoted without evidence-graph SUPPORTS edges.",
        flags={"llm": True, "rag": True, "reflection": True, "skill": True, "provenance": False, "tools": False},
    ),
    BaselineSpec(
        id="llm_skill_provenance", label="LLM+Skill_provenance",
        description="LLM + skills gated by ≥3 SUPPORTS edges in the evidence graph (the upgraded path).",
        flags={"llm": True, "rag": True, "reflection": True, "skill": True, "provenance": True, "tools": False},
    ),
    BaselineSpec(
        id="llm_skill_tools", label="LLM+Skill+Tools",
        description="As above plus the synthesized tool library (Voyager-style five-stage pipeline).",
        flags={"llm": True, "rag": True, "reflection": True, "skill": True, "provenance": True, "tools": True},
    ),
    BaselineSpec(
        id="full", label="Full",
        description="All subsystems enabled — the production agent.",
        flags={"llm": True, "rag": True, "reflection": True, "skill": True, "provenance": True, "tools": True},
    ),
)


def baseline_templates() -> list[BaselineSpec]:
    """Return the 9 baseline templates (no factory wired)."""
    return list(_BASELINE_TEMPLATES)


def baseline_by_id(bid: BaselineID) -> BaselineSpec:
    for spec in _BASELINE_TEMPLATES:
        if spec.id == bid:
            return spec
    raise KeyError(f"Unknown baseline id: {bid}")


# ---------------------------------------------------------------------------
# Hypothesis & metric bundles
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Hypothesis:
    """A testable hypothesis. The harness judges ``arm_a`` against ``arm_b``
    on the named ``metric`` with a configurable margin."""
    id: str
    statement: str
    arm_a: BaselineID
    arm_b: BaselineID
    metric: str
    direction: str = "higher_is_better"  # or "lower_is_better"
    margin: float = 0.05

    def judge(self, value_a: float, value_b: float) -> tuple[bool, str]:
        diff = value_a - value_b
        if self.direction == "higher_is_better":
            passed = diff >= self.margin
        else:
            passed = -diff >= self.margin
        note = (
            f"{self.metric}: arm_a={value_a:.4f} vs arm_b={value_b:.4f}; "
            f"Δ={diff:+.4f} (margin={self.margin}, {self.direction})"
        )
        return passed, note


# Default hypothesis set (H1..H4 in the plan).
DEFAULT_HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(id="H1", statement="Reflection beats raw LLM",
               arm_a="llm_reflection", arm_b="raw_llm", metric="win_rate"),
    Hypothesis(id="H2", statement="Provenance-gated skills beat ungated skills",
               arm_a="llm_skill_provenance", arm_b="llm_skill_no_provenance",
               metric="win_rate"),
    Hypothesis(id="H3", statement="Tool library further lifts win rate",
               arm_a="llm_skill_tools", arm_b="llm_skill_provenance",
               metric="win_rate"),
    Hypothesis(id="H4", statement="Full agent has lower invalid-move rate than RawLLM",
               arm_a="raw_llm", arm_b="full",
               metric="invalid_move_rate", direction="lower_is_better"),
)


@dataclass(slots=True)
class MatchSample:
    """One simulated game result. Stays small so tests can hand-craft rows."""
    arm: BaselineID
    opponent: BaselineID
    game_id: str
    reward: float            # from arm's perspective
    invalid: bool
    turns: int
    seed: int
    # optional system-efficiency fields
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    skill_count: int = 0


@dataclass(slots=True)
class ArmMetrics:
    """All 5 metric classes for a single baseline arm."""
    arm: BaselineID
    # 1. 棋力 (play strength)
    win_rate: float
    draw_rate: float
    loss_rate: float
    avg_reward: float
    elo: float
    invalid_move_rate: float
    # 2. 学习效率 (learning efficiency)
    episodes_to_threshold: int | None
    win_rate_slope: float
    # 3. 泛化 (generalization across games)
    cross_game_win_rate: dict[str, float]
    cross_game_mean: float
    # 4. 技能质量 (skill quality)
    skill_count: int
    skill_provenance_ratio: float
    skill_activation_rate: float
    # 5. 系统效率 (system efficiency)
    avg_latency_ms: float
    avg_prompt_tokens: float
    avg_cached_tokens: float
    cache_hit_ratio: float
    # raw
    n_episodes: int

    def get(self, metric: str) -> float:
        v = getattr(self, metric, None)
        if v is None:
            return 0.0
        if isinstance(v, dict):
            return float(mean(v.values())) if v else 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0


def _aggregate_arm(arm: BaselineID, samples: list[MatchSample],
                   *, win_threshold: float = 0.6, k: float = 16.0) -> ArmMetrics:
    n = len(samples)
    if n == 0:
        return ArmMetrics(
            arm=arm, win_rate=0.0, draw_rate=0.0, loss_rate=0.0, avg_reward=0.0,
            elo=1000.0, invalid_move_rate=0.0,
            episodes_to_threshold=None, win_rate_slope=0.0,
            cross_game_win_rate={}, cross_game_mean=0.0,
            skill_count=0, skill_provenance_ratio=0.0, skill_activation_rate=0.0,
            avg_latency_ms=0.0, avg_prompt_tokens=0.0, avg_cached_tokens=0.0,
            cache_hit_ratio=0.0, n_episodes=0,
        )
    wins = sum(1 for s in samples if s.reward > 0)
    draws = sum(1 for s in samples if s.reward == 0)
    losses = sum(1 for s in samples if s.reward < 0)
    invalids = sum(1 for s in samples if s.invalid)
    win_rate = wins / n
    draw_rate = draws / n
    loss_rate = losses / n

    # cross-game
    by_game: dict[str, list[MatchSample]] = defaultdict(list)
    for s in samples:
        by_game[s.game_id].append(s)
    cross_game_win_rate = {
        g: sum(1 for s in rows if s.reward > 0) / max(1, len(rows))
        for g, rows in by_game.items()
    }
    cross_game_mean = (mean(cross_game_win_rate.values())
                      if cross_game_win_rate else 0.0)

    # learning efficiency — running win-rate over sample order; first index
    # where it crosses ``win_threshold`` is the convergence point.
    running_wins = 0
    crossed: int | None = None
    for i, s in enumerate(samples, start=1):
        if s.reward > 0:
            running_wins += 1
        if crossed is None and running_wins / i >= win_threshold:
            crossed = i
    slope = 0.0
    if n >= 2:
        first_half = samples[: n // 2]
        second_half = samples[n // 2:]
        fh = sum(1 for s in first_half if s.reward > 0) / max(1, len(first_half))
        sh = sum(1 for s in second_half if s.reward > 0) / max(1, len(second_half))
        slope = sh - fh

    # Elo against the implicit population (k-factor pairwise against a 1000-anchor)
    elo = 1000.0
    for s in samples:
        ea = 0.5
        sa = 1.0 if s.reward > 0 else (0.0 if s.reward < 0 else 0.5)
        elo += k * (sa - ea)

    # skill quality + system efficiency — averaged when populated, else 0
    last_skill_count = samples[-1].skill_count
    has_prompt = [s for s in samples if s.prompt_tokens > 0]
    avg_prompt_tokens = mean([s.prompt_tokens for s in has_prompt]) if has_prompt else 0.0
    avg_cached_tokens = mean([s.cached_tokens for s in has_prompt]) if has_prompt else 0.0
    cache_hit_ratio = (avg_cached_tokens / avg_prompt_tokens) if avg_prompt_tokens > 0 else 0.0
    avg_latency = mean([s.latency_ms for s in samples if s.latency_ms > 0] or [0.0])

    return ArmMetrics(
        arm=arm,
        win_rate=win_rate,
        draw_rate=draw_rate,
        loss_rate=loss_rate,
        avg_reward=mean(s.reward for s in samples),
        elo=elo,
        invalid_move_rate=invalids / n,
        episodes_to_threshold=crossed,
        win_rate_slope=slope,
        cross_game_win_rate=cross_game_win_rate,
        cross_game_mean=cross_game_mean,
        skill_count=last_skill_count,
        # provenance & activation ratios are filled later from the graph
        skill_provenance_ratio=0.0,
        skill_activation_rate=0.0,
        avg_latency_ms=avg_latency,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_cached_tokens=avg_cached_tokens,
        cache_hit_ratio=cache_hit_ratio,
        n_episodes=n,
    )


def attach_skill_quality(metrics: ArmMetrics, *, total_skills: int,
                         provenance_supports: int, activated: int) -> ArmMetrics:
    """Mutate-and-return helper so the harness can splice in graph-derived
    skill-quality numbers without having to re-aggregate samples."""
    if total_skills > 0:
        metrics.skill_provenance_ratio = provenance_supports / total_skills
        metrics.skill_activation_rate = activated / total_skills
    return metrics


# ---------------------------------------------------------------------------
# Hypothesis harness
# ---------------------------------------------------------------------------


MatchRunner = Callable[[BaselineSpec, BaselineSpec, str, int], MatchSample]


@dataclass(slots=True)
class HypothesisReport:
    hypothesis_id: str
    statement: str
    arm_a: BaselineID
    arm_b: BaselineID
    metric: str
    direction: str
    margin: float
    value_a: float
    value_b: float
    passed: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HarnessResult:
    arms: list[BaselineID]
    games: list[str]
    n_episodes: int
    arm_metrics: dict[BaselineID, ArmMetrics]
    hypotheses: list[HypothesisReport]
    matches: list[MatchSample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "games": list(self.games),
            "n_episodes": self.n_episodes,
            "arm_metrics": {a: asdict(m) for a, m in self.arm_metrics.items()},
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "matches": [asdict(m) for m in self.matches],
        }


class HypothesisHarness:
    """Run a matrix of (arm × game × seed) games and judge each hypothesis."""

    def __init__(
        self,
        *,
        baselines: Sequence[BaselineSpec],
        hypotheses: Sequence[Hypothesis] = DEFAULT_HYPOTHESES,
        games: Sequence[str] = ("TicTacToe", "KuhnPoker"),
        n_episodes: int = 2,
        seed: int = 0,
        opponent: BaselineID = "random",
        match_runner: MatchRunner | None = None,
        output_dir: str | Path | None = None,
        skill_quality: dict[BaselineID, dict[str, int]] | None = None,
    ) -> None:
        self.baselines = list(baselines)
        self.hypotheses = list(hypotheses)
        self.games = list(games)
        self.n_episodes = int(n_episodes)
        self.seed = int(seed)
        self.opponent = opponent
        self.match_runner = match_runner or _default_synthetic_runner
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.skill_quality = skill_quality or {}

    # -- runtime -----------------------------------------------------------
    def run(self) -> HarnessResult:
        opponent_spec = self._lookup(self.opponent)
        arms = [b.id for b in self.baselines]
        matches: list[MatchSample] = []
        for arm_spec in self.baselines:
            for game in self.games:
                for ep in range(self.n_episodes):
                    seed = self.seed + ep
                    sample = self.match_runner(arm_spec, opponent_spec, game, seed)
                    matches.append(sample)

        arm_metrics: dict[BaselineID, ArmMetrics] = {}
        for arm_spec in self.baselines:
            arm_samples = [m for m in matches if m.arm == arm_spec.id]
            agg = _aggregate_arm(arm_spec.id, arm_samples)
            sq = self.skill_quality.get(arm_spec.id)
            if sq:
                attach_skill_quality(
                    agg,
                    total_skills=int(sq.get("total", 0)),
                    provenance_supports=int(sq.get("with_provenance", 0)),
                    activated=int(sq.get("activated", 0)),
                )
            arm_metrics[arm_spec.id] = agg

        reports = self._judge(arm_metrics)
        result = HarnessResult(
            arms=arms, games=list(self.games), n_episodes=self.n_episodes,
            arm_metrics=arm_metrics, hypotheses=reports, matches=matches,
        )
        if self.output_dir is not None:
            self._persist(result)
        return result

    # -- internal ----------------------------------------------------------
    def _lookup(self, bid: BaselineID) -> BaselineSpec:
        for b in self.baselines:
            if b.id == bid:
                return b
        # opponent might not be one of the arms — fall back to the template
        return baseline_by_id(bid)

    def _judge(self, arm_metrics: dict[BaselineID, ArmMetrics]) -> list[HypothesisReport]:
        out: list[HypothesisReport] = []
        for h in self.hypotheses:
            a = arm_metrics.get(h.arm_a)
            b = arm_metrics.get(h.arm_b)
            if a is None or b is None:
                continue
            va = a.get(h.metric)
            vb = b.get(h.metric)
            passed, note = h.judge(va, vb)
            out.append(HypothesisReport(
                hypothesis_id=h.id, statement=h.statement,
                arm_a=h.arm_a, arm_b=h.arm_b, metric=h.metric,
                direction=h.direction, margin=h.margin,
                value_a=va, value_b=vb, passed=passed, note=note,
            ))
        return out

    def _persist(self, result: HarnessResult) -> None:
        out_dir = Path(self.output_dir)  # type: ignore[arg-type]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "hypothesis_result.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        report_md = render_hypothesis_report(result)
        (out_dir / "hypothesis_report.md").write_text(report_md, encoding="utf-8")
        (out_dir / "matches.jsonl").write_text(
            "".join(json.dumps(asdict(m), ensure_ascii=False, default=str) + "\n"
                    for m in result.matches),
            encoding="utf-8",
        )


def render_hypothesis_report(result: HarnessResult) -> str:
    """Markdown report — used by the hypothesis-report CLI subcommand."""
    lines: list[str] = [
        "# Hypothesis-Driven Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Games: {', '.join(result.games)} | Episodes per arm/game: {result.n_episodes}",
        "",
        "## Hypotheses",
        "",
        "| ID | Statement | Arm A | Arm B | Metric | Δ | Passed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in result.hypotheses:
        diff = h.value_a - h.value_b
        lines.append(
            f"| {h.hypothesis_id} | {h.statement} | {h.arm_a} | {h.arm_b} | "
            f"{h.metric} | {diff:+.4f} | {'✅' if h.passed else '❌'} |"
        )
    lines += [
        "",
        "## Arm Metrics (棋力 / 学习效率 / 泛化 / 技能质量 / 系统效率)",
        "",
        "| Arm | win_rate | invalid_rate | elo | eps_to_thr | slope | "
        "cross_game_mean | skills | provenance | cache_hit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in result.arms:
        m = result.arm_metrics.get(arm)
        if m is None:
            continue
        eps = "-" if m.episodes_to_threshold is None else str(m.episodes_to_threshold)
        lines.append(
            f"| {arm} | {m.win_rate:.3f} | {m.invalid_move_rate:.3f} | "
            f"{m.elo:.1f} | {eps} | {m.win_rate_slope:+.3f} | "
            f"{m.cross_game_mean:.3f} | {m.skill_count} | "
            f"{m.skill_provenance_ratio:.3f} | {m.cache_hit_ratio:.3f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Deterministic synthetic runner — used when no real env factory is supplied
# ---------------------------------------------------------------------------


_STRENGTH_LADDER: dict[BaselineID, float] = {
    "random": 0.10,
    "heuristic": 0.30,
    "raw_llm": 0.40,
    "llm_rag": 0.50,
    "llm_reflection": 0.58,
    "llm_skill_no_provenance": 0.60,
    "llm_skill_provenance": 0.68,
    "llm_skill_tools": 0.74,
    "full": 0.80,
}


def _default_synthetic_runner(arm: BaselineSpec, opponent: BaselineSpec,
                               game: str, seed: int) -> MatchSample:
    """Deterministic synthetic outcome — strength_arm vs strength_opp via a
    fixed hash. This is what the unit tests rely on; production callers should
    pass a real ``match_runner`` that drives ``textarena.make``.

    The function is intentionally side-effect-free and deterministic in
    (arm, opponent, game, seed) so the same run produces identical results.
    """
    s_arm = _STRENGTH_LADDER.get(arm.id, 0.5)
    s_opp = _STRENGTH_LADDER.get(opponent.id, 0.5)
    # deterministic [0,1) draw
    h = abs(hash((arm.id, opponent.id, game, seed))) % 10_000 / 10_000.0
    margin = s_arm - s_opp
    if h < 0.45 + margin:
        reward = 1.0
    elif h < 0.55 + margin:
        reward = 0.0
    else:
        reward = -1.0
    invalid = (h < 0.02) and (not arm.flags.get("llm", False))
    turns = 5 + (seed % 7)
    prompt_tokens = 800 if arm.flags.get("llm", False) else 0
    cached_tokens = int(prompt_tokens * 0.6) if arm.flags.get("skill", False) else 0
    skill_count = 5 if arm.flags.get("skill", False) else 0
    latency_ms = 250.0 if arm.flags.get("llm", False) else 5.0
    return MatchSample(
        arm=arm.id, opponent=opponent.id, game_id=game,
        reward=reward, invalid=invalid, turns=turns, seed=seed,
        latency_ms=latency_ms, prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens, skill_count=skill_count,
    )


# ---------------------------------------------------------------------------
# Offline replay-eval (uses the evidence graph)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReplayEvalResult:
    policy_a: str
    policy_b: str
    episodes_a: int
    episodes_b: int
    win_rate_a: float
    win_rate_b: float
    avg_reward_a: float
    avg_reward_b: float
    diff: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def replay_eval(graph: Any, *, policy_a: str, policy_b: str,
                limit: int = 100) -> ReplayEvalResult:
    """Compute an A/B comparison from already-recorded episodes.

    Reads two cohorts of episodes from the evidence graph (those tagged with
    ``policy_version=policy_a`` and ``policy_version=policy_b``), pulls their
    outcomes from ``attrs_json``, and computes win-rate / avg-reward per
    cohort. This is purely offline — no env is built.
    """
    ep_a = _episodes_for_policy_safe(graph, policy_a, limit=limit)
    ep_b = _episodes_for_policy_safe(graph, policy_b, limit=limit)
    wa, ra = _aggregate_outcomes(ep_a)
    wb, rb = _aggregate_outcomes(ep_b)
    note = ""
    if not ep_a or not ep_b:
        note = (f"Missing cohort: policy_a={len(ep_a)} episodes, "
                f"policy_b={len(ep_b)} episodes; treating empty as 0.")
    return ReplayEvalResult(
        policy_a=policy_a, policy_b=policy_b,
        episodes_a=len(ep_a), episodes_b=len(ep_b),
        win_rate_a=wa, win_rate_b=wb,
        avg_reward_a=ra, avg_reward_b=rb,
        diff=wa - wb,
        note=note,
    )


def _episodes_for_policy_safe(graph: Any, policy_version: str, *,
                              limit: int) -> list[dict[str, Any]]:
    method = getattr(graph, "episodes_for_policy", None)
    if method is None:
        return []
    try:
        return list(method(policy_version, limit=limit))
    except Exception:
        return []


def _aggregate_outcomes(episodes: Iterable[dict[str, Any]]) -> tuple[float, float]:
    """Pull ``outcome`` (win/draw/loss) and ``reward`` from each episode row.

    Tolerant to schema drift: looks at top-level ``outcome``/``reward`` then
    falls back to parsing ``attrs_json`` for the same keys. Returns
    ``(win_rate, avg_reward)``.
    """
    eps = list(episodes)
    if not eps:
        return 0.0, 0.0
    wins = 0
    rewards: list[float] = []
    for row in eps:
        outcome, reward = _outcome_and_reward(row)
        # Accept both "win" and "terminal_win" (engine emits the latter).
        is_win = bool(outcome) and "win" in str(outcome).lower()
        if is_win or (reward is not None and reward > 0):
            wins += 1
        if reward is not None:
            rewards.append(reward)
    win_rate = wins / len(eps)
    avg_reward = mean(rewards) if rewards else 0.0
    return win_rate, avg_reward


def _outcome_and_reward(row: dict[str, Any]) -> tuple[str | None, float | None]:
    outcome = row.get("outcome")
    reward: float | None = None
    raw_reward = row.get("reward")
    if isinstance(raw_reward, (int, float)):
        reward = float(raw_reward)
    if outcome is None or reward is None:
        attrs = row.get("attrs_json") or row.get("attrs")
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except Exception:
                attrs = None
        if isinstance(attrs, dict):
            if outcome is None:
                outcome = attrs.get("outcome")
            if reward is None and isinstance(attrs.get("reward"), (int, float)):
                reward = float(attrs["reward"])
            if reward is None:
                # Engine writes per-player rewards as {"<player_id>": float}
                rewards_dict = attrs.get("rewards")
                player_id = attrs.get("player_id")
                if isinstance(rewards_dict, dict) and rewards_dict:
                    if player_id is not None and str(player_id) in rewards_dict:
                        try:
                            reward = float(rewards_dict[str(player_id)])
                        except Exception:
                            pass
                    if reward is None:
                        try:
                            reward = float(next(iter(rewards_dict.values())))
                        except Exception:
                            pass
    return outcome, reward


# ---------------------------------------------------------------------------
# Opponent pool snapshot
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OpponentPoolSnapshot:
    policy_version: str
    source: str
    dest: str
    files_copied: int
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def opponent_pool_snapshot(memory_dir: str | Path, dest_root: str | Path,
                           *, policy_version: str) -> OpponentPoolSnapshot:
    """Freeze the active skill / patches state for a given policy version.

    Copies the memory directory under ``dest_root/<policy_version>/`` (without
    follow-symlinks magic), writes a manifest with the source path, policy
    version, and a timestamp. Returns the snapshot descriptor.
    """
    src = Path(memory_dir)
    dest_root = Path(dest_root)
    dest = dest_root / policy_version
    dest.mkdir(parents=True, exist_ok=True)
    files_copied = 0
    if src.exists():
        # shallow copy of jsonl + rules + sqlite — refuse to descend forever
        for item in src.iterdir():
            if item.is_dir():
                target = dest / item.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
                files_copied += sum(1 for _ in target.rglob("*") if _.is_file())
            else:
                shutil.copy2(item, dest / item.name)
                files_copied += 1
    manifest = {
        "policy_version": policy_version,
        "source": str(src),
        "dest": str(dest),
        "files_copied": files_copied,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return OpponentPoolSnapshot(
        policy_version=policy_version,
        source=str(src),
        dest=str(dest),
        files_copied=files_copied,
        manifest_path=str(manifest_path),
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "ArmMetrics",
    "BaselineSpec",
    "DEFAULT_HYPOTHESES",
    "HarnessResult",
    "Hypothesis",
    "HypothesisHarness",
    "HypothesisReport",
    "MatchSample",
    "OpponentPoolSnapshot",
    "ReplayEvalResult",
    "attach_skill_quality",
    "baseline_by_id",
    "baseline_templates",
    "opponent_pool_snapshot",
    "render_hypothesis_report",
    "replay_eval",
]
