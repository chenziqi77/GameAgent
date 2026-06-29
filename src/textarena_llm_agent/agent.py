from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from miniagent.tools.base import ToolContext

from .action_analyzer import CandidateAnalysis, TextArenaActionAnalyzer
from .context_packet import ContextBudgeter
from .evaluator import DecisionEvaluator, EvaluationResult, fallback_terminal_lesson
from .llm import DecisionLLM, HeuristicLLM, OpenAIChatLLM
from .memory import EvolvingMemory
from .prompt_builder import GamePromptBuilder, patches_for
from .reflection import Reflector
from .state_encoder import TextArenaStateEncoder
from .tool_library import ToolLibrary
from .tool_loop import ToolLoop
from .tool_synthesis import SafeToolExecutor, ToolNeedDetector, ToolSynthesizer
from .tracing import TextArenaRunTracer, state_snapshot


@dataclass(slots=True)
class TextArenaAgentConfig:
    memory_dir: str = "workspace/textarena_memory"
    top_k_actions: int = 12
    max_valid_actions_in_state: int = 80
    simulate_top: int = 4
    use_llm: bool = False
    allow_evaluator_override: bool = True
    evaluator_min_accept_score: float = 0.45
    evaluator_skip_gap: float = 0.0          # >0 => skip evaluator when confidence high & top-2 gap small
    decision_temperature: float = 0.1
    evaluator_temperature: float = 0.0
    decision_max_tokens: int = 900
    evaluator_max_tokens: int = 800
    prompt_budget_chars: int = 24000
    trace_dir: str = "workspace/textarena_runs/latest"
    enable_tracing: bool = True
    # --- evolution / tooling ---
    enable_tool_synthesis: bool = True
    synthesis_threshold: int = 3
    tool_loop_max_rounds: int = 4
    reflection_enabled: bool = True
    insight_min_episodes: int = 8
    context_budget_chars: int = 24000
    enable_tools_in_loop: bool = True


@dataclass(slots=True)
class Decision:
    action_text: str
    candidate_id: str
    action_index: int
    action_type: str
    confidence: float
    rationale: str
    plan: str
    selected_candidate: dict[str, Any]
    evaluation: dict[str, Any] = field(default_factory=dict)
    raw_llm: dict[str, Any] = field(default_factory=dict)
    memory_excerpt: str = ""
    evaluator_overrode: bool = False
    original_selection: dict[str, Any] = field(default_factory=dict)
    tool_events: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)


class TextArenaDecisionAgent:
    """Evolvable text-first LLM agent for TextArena games.

    Architecture (per decision): encode state -> BM25 memory+reflection recall ->
    (rare) tool synthesis -> context packet -> bounded tool-calling loop ->
    critic/evaluator -> execute step. Per terminal episode: self-reflection,
    ExpeL insight consolidation (batched), and skill promotion/demotion/mutation.
    """

    def __init__(
        self,
        config: TextArenaAgentConfig | None = None,
        *,
        llm: DecisionLLM | None = None,
        evaluator_llm: DecisionLLM | None = None,
        encoder: TextArenaStateEncoder | None = None,
        memory: EvolvingMemory | None = None,
        tracer: TextArenaRunTracer | None = None,
        tool_library: ToolLibrary | None = None,
        executor: SafeToolExecutor | None = None,
    ) -> None:
        self.config = config or TextArenaAgentConfig()
        self.encoder = encoder or TextArenaStateEncoder(max_valid_actions=self.config.max_valid_actions_in_state)
        self.analyzer = TextArenaActionAnalyzer(self.encoder, simulate_top=self.config.simulate_top)
        self.memory = memory or EvolvingMemory(self.config.memory_dir)
        self.tracer = tracer or (TextArenaRunTracer(self.config.trace_dir) if self.config.enable_tracing else None)
        if llm is None:
            llm = OpenAIChatLLM.from_env() if self.config.use_llm else HeuristicLLM()
        if evaluator_llm is None:
            evaluator_llm = llm if self.config.use_llm else HeuristicLLM()
        self.llm = llm
        self.evaluator = DecisionEvaluator(llm=evaluator_llm, memory=self.memory, min_accept_score=self.config.evaluator_min_accept_score)
        self.prompt_builder = GamePromptBuilder()
        self.context_budgeter = ContextBudgeter(total_chars=self.config.context_budget_chars)
        # tooling
        self.executor = executor or (SafeToolExecutor() if self.config.enable_tool_synthesis else None)
        self.tool_library = tool_library or (ToolLibrary(Path(self.config.memory_dir) / "tool_library") if self.config.enable_tool_synthesis else None)
        self.tool_synthesizer = None
        if self.config.enable_tool_synthesis and self.tool_library is not None and self.executor is not None:
            self.tool_synthesizer = ToolSynthesizer(llm=llm, executor=self.executor, library=self.tool_library)
        self.need_detector = ToolNeedDetector(threshold=self.config.synthesis_threshold)
        self.reflector = Reflector(llm=evaluator_llm, memory=self.memory) if self.config.reflection_enabled else None
        # transient episode state
        self._transitions: list[dict[str, Any]] = []
        self._episode_game_id: str = ""
        self._episode_seed: int = 0

    # ------------------------------------------------------------------ main entrypoints
    def decide(self, env: Any) -> Decision:
        self._wait_if_paused()
        state = self.encoder.encode(env, include_actions=True)
        self._snapshot(env, state=state)
        state_text = json.dumps(state, ensure_ascii=False, indent=2, default=str)
        valid_options = self.encoder.valid_actions(env)
        if not valid_options:
            raise RuntimeError("No valid TextArena actions available for current state.")
        game = state.get("game", {})
        game_id = str(game.get("family") or game.get("env_id") or "unknown")
        current_player = int(game.get("current_player", 0))
        turn = game.get("turn")
        phase = _phase_for_turn(turn)
        self._emit("decision_start", {"game_id": game_id, "turn": turn, "current_player": current_player, "valid_actions_count": len(valid_options), "memory_stats": self.memory.memory_stats()})

        candidates = self.analyzer.analyze(env, valid_options, top_k=self.config.top_k_actions)
        self._emit("candidates_ranked", {"candidates": [c.to_prompt_dict() for c in candidates]})

        # game_state snapshot for synthesized tools (frozen view; never the live env)
        game_state_snapshot = copy.deepcopy(getattr(env.state, "game_state", {}) or {})
        visible_text = self.encoder._format_observations(env, current_player)  # noqa: SLF001 — reuse existing encoder

        # BM25 recall + reflections (perspective-correct)
        query = f"game {game_id} player {current_player} turn {turn} " + " ".join(c.action_type for c in candidates[:5])
        memory_excerpt = self.memory.format_for_prompt(query, max_items=8, game_id=game_id, player=current_player, phase=phase)
        self._emit("memory_recalled", {"query": query, "memory_excerpt": memory_excerpt[:6000], "memory_stats": self.memory.memory_stats()})
        reflections = self.memory.retrieve_reflections(query=query, game_id=game_id, top_k=2) if self.config.reflection_enabled else []
        reflection_text = "\n\n".join(f"[{r.outcome}] {r.text}\nLesson: {r.actionable_lesson}" for r in reflections) if reflections else ""
        patches = patches_for(self.memory, game_id)

        # rare tool synthesis (future-turn effect, non-blocking)
        if self.tool_synthesizer is not None and self.config.enable_tool_synthesis:
            self._maybe_synthesize_tool(env, game_id=game_id, phase=phase, candidates=candidates, game_state_snapshot=game_state_snapshot, visible_text=visible_text, state_text=state_text)

        # build context packet + per-game system prompt
        system_prompt = self.prompt_builder.build_system(
            game_id=game_id, phase=phase, reflection=reflection_text, active_patches=patches,
            tool_descriptions=self._tool_descriptions(game_id),
        )
        packet = self.context_budgeter.build(
            spec_system=system_prompt, state_text=state_text,
            candidates=[c.to_prompt_dict() for c in candidates], memory_excerpt=memory_excerpt,
            reflection=reflection_text, patches=patches,
        )
        self._emit("llm_request", {"system_prompt": system_prompt[:4000], "user_prompt_preview": packet.user[:6000], "budget_used": packet.budget_used})

        # tool-calling hybrid loop (falls back to plain JSON for HeuristicLLM / no tools)
        tool_events: list[dict[str, Any]] = []
        raw: dict[str, Any]
        if self.config.enable_tools_in_loop:
            registry = self._build_registry(game_id=game_id)
            ctx = self._build_ctx(env, game_id=game_id, player=current_player, phase=phase, game_state_snapshot=game_state_snapshot, visible_text=visible_text)
            loop = ToolLoop(registry=registry, ctx=ctx, llm=self.llm, max_rounds=self.config.tool_loop_max_rounds, max_tokens=self.config.decision_max_tokens)
            result = loop.run(system=packet.system, user=packet.user, candidate_ids=[c.candidate_id for c in candidates])
            raw = result.decision_json
            tool_events = result.tool_events
            self._emit("tool_loop", {"rounds": result.rounds, "fallback": result.fallback, "events": tool_events})
        else:
            raw = self.llm.complete_json(system=packet.system, user=packet.user, temperature=self.config.decision_temperature, max_tokens=self.config.decision_max_tokens)
        self._emit("llm_response", {"raw": raw})

        selected = self._resolve_decision(raw, candidates)
        original_selected = selected
        decision_obj = self._decision_dict(raw, selected)
        self._emit("decision_selected", {"decision": decision_obj, "selected_candidate": selected.to_prompt_dict()})

        # evaluator (skippable on high-confidence, small-gap decisions for frugality)
        evaluation = self._maybe_evaluate(state_text=state_text, candidates=candidates, decision=decision_obj, selected=selected)
        self._emit("evaluation_complete", {"evaluation": asdict(evaluation)})
        evaluator_overrode = False
        if self.config.allow_evaluator_override and (not evaluation.accept) and evaluation.suggested_candidate_id:
            suggested = next((c for c in candidates if c.candidate_id == evaluation.suggested_candidate_id), None)
            if suggested is not None:
                selected = suggested
                evaluator_overrode = selected.candidate_id != original_selected.candidate_id
                decision_obj = self._decision_dict({**raw, "candidate_id": suggested.candidate_id, "action": suggested.action_text, "rationale": f"Evaluator override: {evaluation.critique}", "confidence": min(0.7, max(0.4, evaluation.score + 0.2))}, selected)
                self._emit("decision_overridden", {"from_candidate_id": original_selected.candidate_id, "to_candidate_id": selected.candidate_id, "evaluation": asdict(evaluation), "decision": decision_obj})
        if evaluation.lesson or evaluation.prompt_patch:
            self.evaluator.learn_from_transition(before_state_text=state_text, game_id=game_id, player=current_player, decision=decision_obj, evaluation=evaluation, outcome="pre_action_evaluation")
            self._emit("memory_written", {"reason": "pre_action_evaluation", "memory_stats": self.memory.memory_stats()})

        result = Decision(
            action_text=str(selected.action_text),
            candidate_id=str(selected.candidate_id),
            action_index=int(selected.action_index),
            action_type=str(selected.action_type),
            confidence=float(decision_obj.get("confidence") or 0.0),
            rationale=str(decision_obj.get("rationale") or ""),
            plan=str(decision_obj.get("plan") or ""),
            selected_candidate=selected.to_prompt_dict(),
            evaluation=asdict(evaluation),
            raw_llm=raw,
            memory_excerpt=memory_excerpt[:4000],
            evaluator_overrode=evaluator_overrode,
            original_selection={} if original_selected.candidate_id == selected.candidate_id else original_selected.to_prompt_dict(),
            tool_events=tool_events,
        )
        self._emit("decision_resolved", {"decision": asdict(result)})
        self._snapshot(env, state=state, decision=result)
        # record transition for terminal reflection
        self._transitions.append({"turn": turn, "action_text": result.action_text, "evaluation": asdict(evaluation), "reward": None, "outcome": None})
        return result

    def act(self, env: Any) -> Decision:
        self._wait_if_paused()
        before_text = self.encoder.encode_text(env, include_actions=True)
        before_state = json.loads(before_text)
        game_id = str(before_state.get("game", {}).get("family") or before_state.get("game", {}).get("env_id") or "unknown")
        player = int(before_state.get("game", {}).get("current_player", 0))
        if self._episode_game_id and self._episode_game_id != game_id:
            self._reset_episode(game_id=game_id)
        if not self._episode_game_id:
            self._episode_game_id = game_id
        decision = self.decide(env)
        self._emit("action_execute_start", {"action_text": decision.action_text, "candidate_id": decision.candidate_id})
        done, info = env.step(decision.action_text)
        reward = _reward_for_player(env, player)
        outcome = _outcome_for_player(env, player, done)
        if self._transitions:
            self._transitions[-1]["reward"] = reward
            self._transitions[-1]["outcome"] = outcome
        exp_id = self.learn_from_outcome(before_state_text=before_text, game_id=game_id, player=player, decision=decision, reward=reward, outcome=outcome)
        self._emit("action_execute_end", {"done": done, "info": info, "reward": reward, "outcome": outcome, "experience_id": exp_id, "memory_stats": self.memory.memory_stats()})
        # terminal evolution: reflection + insight consolidation + skill evolution
        if done and self.config.reflection_enabled and self.reflector is not None:
            self._on_terminal(env, game_id=game_id, player=player, outcome=outcome, reward=reward)
        self._snapshot(env, decision=decision)
        return decision

    def run_episode(self, env: Any, *, max_steps: int = 200, seed: int = 0, verbose: bool = False) -> list[Decision]:
        from .game_specs import canonical_game_id
        game_id = canonical_game_id(str(getattr(env, "env_id", "")))
        self._reset_episode(game_id=game_id)
        self._episode_seed = seed
        decisions: list[Decision] = []
        for step in range(max_steps):
            if bool(getattr(getattr(env, "state", None), "done", False)):
                break
            decision = self.act(env)
            decisions.append(decision)
            if verbose:
                print(f"[{step:03d}] P{getattr(env.state, 'current_player_id', '?')} {decision.candidate_id} -> {decision.action_text}")
        return decisions

    # ------------------------------------------------------------------ learning
    def learn_from_outcome(self, *, before_state_text: str, game_id: str, player: int, decision: Decision, reward: float | None = None, outcome: str | None = None) -> str:
        eval_result = None
        if decision.evaluation:
            eval_result = EvaluationResult(**{k: decision.evaluation.get(k) for k in ["score", "accept", "suggested_candidate_id", "critique", "lesson", "prompt_patch", "raw"]})
        exp_id = self.evaluator.learn_from_transition(
            before_state_text=before_state_text,
            game_id=game_id,
            player=player,
            decision={"action": decision.action_text, "action_text": decision.action_text, "rationale": decision.rationale},
            evaluation=eval_result,
            reward=reward,
            outcome=outcome,
        )
        skill_id = None
        lesson = eval_result.lesson if eval_result is not None else ""
        critique = eval_result.critique if eval_result is not None else ""
        if not lesson and outcome in {"terminal_win", "terminal_loss", "terminal_draw"}:
            lesson = fallback_terminal_lesson(game_id=game_id, outcome=outcome, reward=reward, critique=critique)
        if lesson:
            score = eval_result.score if eval_result is not None else 0.5
            skill_id = self.memory.consolidate_skill_from_lesson(lesson=lesson, game_id=game_id, evidence=f"experience:{exp_id}", reward=reward, evaluator_score=score, outcome=outcome, tags=[game_id, f"player_{player}", "critic_fallback" if not (eval_result and eval_result.lesson) else "critic"])
        # update touched skill usage from the terminal outcome
        if skill_id and outcome in {"terminal_win", "terminal_loss", "terminal_draw"}:
            win = outcome == "terminal_win"
            self.memory.update_skill_usage(skill_id=skill_id, win=win, score=eval_result.score if eval_result else 0.5)
        self._emit("memory_written", {"experience_id": exp_id, "skill_id": skill_id, "game_id": game_id, "player": player, "reward": reward, "outcome": outcome, "memory_stats": self.memory.memory_stats()})
        return exp_id

    # ------------------------------------------------------------------ terminal evolution
    def _on_terminal(self, env: Any, *, game_id: str, player: int, outcome: str, reward: float | None) -> None:
        try:
            reflection = self.reflector.reflect_episode(game_id=game_id, seed=self._episode_seed, outcome=outcome, transitions=self._transitions)
            rid = self.memory.record_reflection(reflection)
            self._emit("reflection_written", {"reflection_id": rid, "outcome": outcome, "lesson": reflection.actionable_lesson[:200]})
        except Exception as exc:
            self._emit("reflection_error", {"error": str(exc)})
        try:
            self.memory.decay_confidence()
        except Exception:
            pass
        # ExpeL batched insight consolidation
        try:
            if self.memory.unconsolidated_count(game_id) >= self.config.insight_min_episodes:
                insights = self.memory.consolidate_insights(game_id=game_id, llm=self.evaluator.llm)
                self._emit("insight_consolidated", {"game_id": game_id, "count": len(insights)})
        except Exception as exc:
            self._emit("insight_error", {"error": str(exc)})
        # skill promotion / demotion / mutation sweep
        try:
            counts = self.memory.evolve_skills(game_id=game_id, llm=self.evaluator.llm)
            if any(counts.values()):
                self._emit("skills_evolved", {"game_id": game_id, **counts})
        except Exception as exc:
            self._emit("evolve_error", {"error": str(exc)})
        self._reset_episode(game_id=game_id)

    # ------------------------------------------------------------------ tool synthesis
    def _maybe_synthesize_tool(self, env: Any, *, game_id: str, phase: str, candidates: list[CandidateAnalysis],
                               game_state_snapshot: dict[str, Any], visible_text: str, state_text: str) -> None:
        intent_tokens = [c.action_type for c in candidates[:3]]
        should, task = self.need_detector.observe(game_id=game_id, phase=phase, intent_tokens=intent_tokens)
        # explicit request from the LLM's last decision (if any)
        last_raw = self._transitions[-1].get("evaluation") if self._transitions else None
        if not should and isinstance(last_raw, dict):
            need = str(last_raw.get("need_tool") or "")
            if need:
                should, task = self.need_detector.from_explicit_request(game_id=game_id, need_desc=need)
        if not should or self.tool_synthesizer is None:
            return
        try:
            name = self.tool_synthesizer.synthesize_and_register(
                task_description=task, game_id=game_id, context_summary=state_text[:3000],
                game_state_snapshot=game_state_snapshot, visible_text=visible_text,
            )
            if name:
                self._emit("tool_synthesized", {"name": name, "task": task, "game_id": game_id})
        except Exception as exc:
            self._emit("tool_synthesis_error", {"error": str(exc), "task": task})

    # ------------------------------------------------------------------ helpers
    def _maybe_evaluate(self, *, state_text: str, candidates: list[CandidateAnalysis], decision: dict[str, Any], selected: CandidateAnalysis) -> EvaluationResult:
        # frugal skip: high confidence + small top-2 gap
        if self.config.evaluator_skip_gap > 0 and len(candidates) >= 2:
            conf = float(decision.get("confidence") or 0.0)
            gap = float(candidates[0].score - candidates[1].score) if len(candidates) > 1 else 0.0
            if conf >= 0.8 and gap < self.config.evaluator_skip_gap:
                return EvaluationResult(score=max(0.6, conf), accept=True, suggested_candidate_id=None, critique="skipped: high confidence, small gap")
        return self.evaluator.evaluate(state_text=state_text, candidates=candidates, decision=decision, temperature=self.config.evaluator_temperature, max_tokens=self.config.evaluator_max_tokens)

    def _build_registry(self, *, game_id: str):
        from .tools import create_textarena_tool_registry
        return create_textarena_tool_registry(encoder=self.encoder, analyzer=self.analyzer, memory=self.memory,
                                              tool_library=self.tool_library, executor=self.executor, game_id=game_id)

    def _build_ctx(self, env: Any, *, game_id: str, player: int, phase: str, game_state_snapshot: dict[str, Any], visible_text: str) -> ToolContext:
        # Build a minimal ToolContext; our tools only use ctx.metadata.
        from miniagent.runtime.workspace import Workspace
        try:
            workspace = Workspace.create(self.config.memory_dir)
        except Exception:
            workspace = None  # tools don't touch the workspace
        return ToolContext(
            workspace=workspace,
            memory=None,
            session_id=uuid4().hex[:12],
            metadata={
                "env": env,
                "game_id": game_id,
                "player": player,
                "phase": phase,
                "game_state_snapshot": game_state_snapshot,
                "visible_text": visible_text,
                "tool_library": self.tool_library,
            },
        )

    def _tool_descriptions(self, game_id: str) -> list[str]:
        if not self.config.enable_tools_in_loop or not self.config.enable_tool_synthesis:
            return []
        descriptions = [
            "textarena_state_summary: compact visible state",
            "textarena_analyze_actions: rank legal actions",
            "textarena_simulate_action: preview a candidate's outcome",
            "textarena_recall_memory: recall lessons/skills",
            "textarena_recall_reflections: recall past-game reflections",
        ]
        if self.tool_library is not None:
            for rec in self.tool_library.active_for(game_id):
                descriptions.append(f"{rec.name}: {rec.description}")
        return descriptions

    def _resolve_decision(self, raw: dict[str, Any], candidates: list[CandidateAnalysis]) -> CandidateAnalysis:
        by_id = {c.candidate_id: c for c in candidates}
        cid = str(raw.get("candidate_id") or "")
        if cid in by_id:
            return by_id[cid]
        action = str(raw.get("action") or "")
        for c in candidates:
            if c.action_text == action:
                return c
        return candidates[0]

    def _decision_dict(self, raw: dict[str, Any], selected: CandidateAnalysis) -> dict[str, Any]:
        return {
            "candidate_id": selected.candidate_id,
            "action": selected.action_text,
            "action_text": selected.action_text,
            "confidence": _float(raw.get("confidence"), 0.0),
            "rationale": str(raw.get("rationale") or f"Selected {selected.candidate_id}: {selected.action_text}"),
            "plan": str(raw.get("plan") or "Use the resulting position or information to improve expected payoff."),
        }

    def _reset_episode(self, *, game_id: str) -> None:
        self._transitions = []
        self._episode_game_id = game_id
        self._episode_seed = 0
        self.need_detector.reset_episode()

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.tracer is not None:
            self.tracer.emit(event, payload)

    def _snapshot(self, env: Any, *, state: dict[str, Any] | None = None, decision: Decision | None = None) -> None:
        if self.tracer is None:
            return
        data = {"snapshot": state_snapshot(env), "memory_stats": self.memory.memory_stats()}
        if state is not None:
            data["state"] = state
        if decision is not None:
            data["decision"] = asdict(decision)
        self.tracer.update_state(data)

    def _wait_if_paused(self) -> None:
        if self.tracer is not None and not self.tracer.wait_for_turn():
            raise RuntimeError("Visualization requested stop.")


def _float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _reward_for_player(env: Any, player: int) -> float | None:
    rewards = getattr(getattr(env, "state", None), "rewards", None)
    if isinstance(rewards, dict) and player in rewards:
        return float(rewards[player])
    return None


def _outcome_for_player(env: Any, player: int, done: bool) -> str:
    if not done:
        return "continued"
    reward = _reward_for_player(env, player)
    if reward is None:
        return "terminal"
    if reward > 0:
        return "terminal_win"
    if reward < 0:
        return "terminal_loss"
    return "terminal_draw"


def _phase_for_turn(turn: Any) -> str:
    try:
        t = int(turn)
    except Exception:
        return "mid"
    return "early" if t < 3 else ("late" if t > 12 else "mid")
