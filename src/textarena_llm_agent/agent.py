from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from miniagent.tools.base import ToolContext

from .action_analyzer import CandidateAnalysis, TextArenaActionAnalyzer
from .context_packet import ContextBudgeter
from .critic_agent import EpisodeCriticAgent
from .evaluator import DecisionEvaluator, EvaluationResult, fallback_terminal_lesson
from .llm import DecisionLLM, HeuristicLLM, OpenAIChatLLM
from .memory import EvolvingMemory
from .prompt_builder import GamePromptBuilder, patches_for
from .prompt_compiler import PromptCompiler
from .reflection import Reflector
from .skill_manager import SkillManager
from .state_encoder import TextArenaStateEncoder
from .tool_library import ToolLibrary
from .tool_loop import ToolLoop
from .tool_synthesis import SafeToolExecutor, ToolNeedDetector, ToolSynthesizer
from .trace_schema import DecisionFrame, EpisodeTrace
from .tracing import TextArenaRunTracer, state_snapshot


@dataclass(slots=True)
class TextArenaAgentConfig:
    memory_dir: str = "workspace/textarena_memory"
    top_k_actions: int = 12
    max_valid_actions_in_state: int = 80
    simulate_top: int = 4
    use_llm: bool = False
    # Phase 4: decision-time evaluator override is permanently disabled.
    # The flag remains for back-compat (tests / external callers may pass it),
    # but the agent.decide() path no longer reads it — see the unconditional
    # ``evaluator_overrode = False`` block. All correction signal is produced
    # off-line by EpisodeCriticAgent in _on_terminal.
    allow_evaluator_override: bool = False
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
    policy_version: str = "v0"
    # Phase 4: Episode Critic Agent owns post-episode evolution.
    enable_critic_agent: bool = True
    critic_max_rounds: int = 8
    critic_max_tokens: int = 1500


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
    # --- Phase 1 logging extensions (defaults preserve backward compatibility) ---
    game_id: str = ""
    episode_id: str = ""
    state_hash: str = ""
    legal_actions: list[str] = field(default_factory=list)
    retrieved_memory_ids: list[str] = field(default_factory=list)
    used_skill_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    policy_version: str = "v0"

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
        # Evidence graph (SQLite index over JSONL truth). Opened once per agent; reused
        # across episodes. JSONL writes remain the source of truth; graph is for queries.
        try:
            from .evidence_graph import EvidenceGraph
            graph_path = Path(self.config.memory_dir) / "evidence_graph.sqlite"
            self.evidence_graph = EvidenceGraph(graph_path)
        except Exception:
            self.evidence_graph = None
        if memory is None:
            self.memory = EvolvingMemory(self.config.memory_dir, graph=self.evidence_graph)
        else:
            self.memory = memory
            if self.evidence_graph is not None and getattr(self.memory, "graph", None) is None:
                self.memory.graph = self.evidence_graph
        # One-shot legacy JSONL import — idempotent (INSERT OR REPLACE keyed on id).
        if self.evidence_graph is not None:
            try:
                self.evidence_graph.bootstrap_from_jsonl(self.config.memory_dir)
            except Exception:
                pass
        # SkillManager owns the proposed→candidate→validated→active lifecycle.
        # Constructed once per agent so Critic Agent (Phase 4) can locate it.
        # When evidence_graph is unavailable (e.g. import failure) the manager
        # is None and skill writes silently fall back to the legacy JSONL path.
        self.skill_manager: SkillManager | None
        if self.evidence_graph is not None:
            self.skill_manager = SkillManager(self.evidence_graph, self.memory, policy_version=self.config.policy_version)
        else:
            self.skill_manager = None
        self.tracer = tracer or (TextArenaRunTracer(self.config.trace_dir) if self.config.enable_tracing else None)
        if llm is None:
            llm = OpenAIChatLLM.from_env() if self.config.use_llm else HeuristicLLM()
        if evaluator_llm is None:
            evaluator_llm = llm if self.config.use_llm else HeuristicLLM()
        self.llm = llm
        self.evaluator = DecisionEvaluator(llm=evaluator_llm, memory=self.memory, min_accept_score=self.config.evaluator_min_accept_score)
        self.prompt_builder = GamePromptBuilder()
        self.context_budgeter = ContextBudgeter(total_chars=self.config.context_budget_chars)
        self.prompt_compiler = PromptCompiler(context_budgeter=self.context_budgeter)
        # tooling
        self.executor = executor or (SafeToolExecutor() if self.config.enable_tool_synthesis else None)
        self.tool_library = tool_library or (ToolLibrary(Path(self.config.memory_dir) / "tool_library") if self.config.enable_tool_synthesis else None)
        self.tool_synthesizer = None
        if self.config.enable_tool_synthesis and self.tool_library is not None and self.executor is not None:
            self.tool_synthesizer = ToolSynthesizer(llm=llm, executor=self.executor, library=self.tool_library)
        self.need_detector = ToolNeedDetector(threshold=self.config.synthesis_threshold)
        self.reflector = Reflector(llm=evaluator_llm, memory=self.memory) if self.config.reflection_enabled else None
        # Phase 4: EpisodeCriticAgent runs post-episode. Game-agent decisions
        # are never overridden mid-episode anymore — all evolution signal flows
        # through the critic's tool calls (propose_skill / propose_tool / etc).
        self.critic_agent: EpisodeCriticAgent | None
        if (
            self.config.enable_critic_agent
            and self.evidence_graph is not None
            and self.skill_manager is not None
        ):
            self.critic_agent = EpisodeCriticAgent(
                llm=evaluator_llm,
                graph=self.evidence_graph,
                skill_manager=self.skill_manager,
                memory=self.memory,
                tool_library=self.tool_library,
                max_rounds=self.config.critic_max_rounds,
                max_tokens=self.config.critic_max_tokens,
                emit=self._emit,
            )
        else:
            self.critic_agent = None
        # transient episode state
        self._transitions: list[dict[str, Any]] = []
        self._episode_game_id: str = ""
        self._episode_seed: int = 0
        self._episode_id: str = uuid4().hex[:12]
        self._episode_step: int = 0
        self._episode_started_at_ms: float = 0.0
        self._episode_decision_frame_ids: list[str] = []
        self._episode_total_prompt_tokens: int = 0
        self._episode_total_completion_tokens: int = 0
        self._episode_total_cached_tokens: int = 0
        self._episode_total_latency_ms: float = 0.0

    # ------------------------------------------------------------------ main entrypoints
    def decide(self, env: Any) -> Decision:
        self._wait_if_paused()
        decide_started = time.perf_counter()
        state = self.encoder.encode(env, include_actions=True)
        state_hash = self.encoder.canonical_state_hash(env)
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
        if not self._episode_game_id:
            self._reset_episode(game_id=game_id)
        legal_actions_preview = [opt.action_text for opt in valid_options[:80]]
        self._emit("decision_start", {"episode_id": self._episode_id, "game_id": game_id, "state_hash": state_hash, "turn": turn, "current_player": current_player, "valid_actions_count": len(valid_options), "memory_stats": self.memory.memory_stats()})

        candidates = self.analyzer.analyze(env, valid_options, top_k=self.config.top_k_actions)
        self._emit("candidates_ranked", {"candidates": [c.to_prompt_dict() for c in candidates]})

        # game_state snapshot for synthesized tools (frozen view; never the live env)
        game_state_snapshot = copy.deepcopy(getattr(env.state, "game_state", {}) or {})
        visible_text = self.encoder._format_observations(env, current_player)  # noqa: SLF001 — reuse existing encoder

        # BM25 recall + reflections (perspective-correct)
        query = f"game {game_id} player {current_player} turn {turn} " + " ".join(c.action_type for c in candidates[:5])
        memory_excerpt = self.memory.format_for_prompt(query, max_items=8, game_id=game_id, player=current_player, phase=phase)
        retrieved_memory_ids = self._recent_retrieved_memory_ids(query=query, game_id=game_id, player=current_player, phase=phase)
        used_skill_ids = self._extract_skill_ids(memory_excerpt)
        self._emit("memory_recalled", {"query": query, "memory_excerpt": memory_excerpt[:6000], "retrieved_memory_ids": retrieved_memory_ids, "used_skill_ids": used_skill_ids, "memory_stats": self.memory.memory_stats()})
        reflections = self.memory.retrieve_reflections(query=query, game_id=game_id, top_k=2) if self.config.reflection_enabled else []
        reflection_text = "\n\n".join(f"[{r.outcome}] {r.text}\nLesson: {r.actionable_lesson}" for r in reflections) if reflections else ""
        patches = patches_for(self.memory, game_id)

        # rare tool synthesis (future-turn effect, non-blocking)
        if self.tool_synthesizer is not None and self.config.enable_tool_synthesis:
            self._maybe_synthesize_tool(env, game_id=game_id, phase=phase, candidates=candidates, game_state_snapshot=game_state_snapshot, visible_text=visible_text, state_text=state_text)

        # build context packet via the four-layer prompt compiler. The compiler
        # emits stable STATIC_PREFIX / GAME_STATIC / POLICY_STATIC layers so the
        # LLM provider's prefix cache hits across decisions within the same
        # (game_id, policy_version) window; USER_DYNAMIC carries state / memory /
        # candidates / reflection and changes every decision.
        compiled = self.prompt_compiler.compile(
            game_id=game_id, phase=phase,
            policy_version=self.config.policy_version,
            state_text=state_text,
            candidates=[c.to_prompt_dict() for c in candidates],
            memory_excerpt=memory_excerpt,
            reflection=reflection_text,
            active_patches=patches,
            tool_descriptions=self._tool_descriptions(game_id),
        )
        packet = compiled.to_packet()
        packet.policy_version = self.config.policy_version
        system_prompt = packet.system
        self._emit("llm_request", {
            "system_prompt": system_prompt[:4000],
            "user_prompt_preview": packet.user[:6000],
            "budget_used": packet.budget_used,
            "layer_hashes": packet.layer_hashes,
            "layer_tokens": packet.layer_tokens,
            "stable_prefix_tokens": packet.stable_prefix_tokens,
        })

        # tool-calling hybrid loop (falls back to plain JSON for HeuristicLLM / no tools)
        tool_events: list[dict[str, Any]] = []
        raw: dict[str, Any]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        if self.config.enable_tools_in_loop:
            registry = self._build_registry(game_id=game_id)
            ctx = self._build_ctx(env, game_id=game_id, player=current_player, phase=phase, game_state_snapshot=game_state_snapshot, visible_text=visible_text)
            loop = ToolLoop(registry=registry, ctx=ctx, llm=self.llm, max_rounds=self.config.tool_loop_max_rounds, max_tokens=self.config.decision_max_tokens)
            result = loop.run(system=packet.system, user=packet.user, candidate_ids=[c.candidate_id for c in candidates])
            raw = result.decision_json
            tool_events = result.tool_events
            usage = dict(result.usage or usage)
            self._emit("tool_loop", {"rounds": result.rounds, "fallback": result.fallback, "events": tool_events, "usage": usage})
        else:
            raw = self.llm.complete_json(system=packet.system, user=packet.user, temperature=self.config.decision_temperature, max_tokens=self.config.decision_max_tokens)
        self._emit("llm_response", {"raw": raw})

        selected = self._resolve_decision(raw, candidates)
        original_selected = selected
        decision_obj = self._decision_dict(raw, selected)
        self._emit("decision_selected", {"decision": decision_obj, "selected_candidate": selected.to_prompt_dict()})

        # evaluator runs as an OPTIONAL self-check; its critique/lesson may
        # still seed memory rows, but it is FORBIDDEN to swap the action.
        # Per Phase 4: all correction signal is off-line via EpisodeCriticAgent.
        evaluation = self._maybe_evaluate(state_text=state_text, candidates=candidates, decision=decision_obj, selected=selected)
        self._emit("evaluation_complete", {"evaluation": asdict(evaluation)})
        evaluator_overrode = False
        if evaluation.lesson or evaluation.prompt_patch:
            self.evaluator.learn_from_transition(before_state_text=state_text, game_id=game_id, player=current_player, decision=decision_obj, evaluation=evaluation, outcome="pre_action_evaluation")
            self._emit("memory_written", {"reason": "pre_action_evaluation", "memory_stats": self.memory.memory_stats()})

        latency_ms = (time.perf_counter() - decide_started) * 1000.0
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        cached_tokens = int(usage.get("cached_tokens", 0) or 0)
        cache_hit_ratio = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0

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
            game_id=game_id,
            episode_id=self._episode_id,
            state_hash=state_hash,
            legal_actions=legal_actions_preview,
            retrieved_memory_ids=retrieved_memory_ids,
            used_skill_ids=used_skill_ids,
            latency_ms=round(latency_ms, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            policy_version=self.config.policy_version,
        )
        self._emit("decision_resolved", {"decision": asdict(result)})

        frame = DecisionFrame(
            game_id=game_id,
            episode_id=self._episode_id,
            turn=int(turn or 0),
            step=self._episode_step,
            state_hash=state_hash,
            current_player=current_player,
            candidate_id=result.candidate_id,
            action_text=result.action_text,
            action_index=result.action_index,
            action_type=result.action_type,
            legal_actions=legal_actions_preview,
            confidence=result.confidence,
            rationale=result.rationale,
            plan=result.plan,
            retrieved_memory_ids=retrieved_memory_ids,
            used_skill_ids=used_skill_ids,
            policy_version=self.config.policy_version,
            latency_ms=result.latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cache_hit_ratio=round(cache_hit_ratio, 4),
            tool_traces=tool_events,
            prompt_trace={
                "system_chars": len(packet.system or ""),
                "user_chars": len(packet.user or ""),
                "prompt_chars": len((packet.system or "") + (packet.user or "")),
                "cached_tokens": cached_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cache_hit_ratio": round(cache_hit_ratio, 4),
                "policy_version": self.config.policy_version,
                "layer_hashes": dict(packet.layer_hashes or {}),
                "layer_tokens": dict(packet.layer_tokens or {}),
                "stable_prefix_tokens": int(packet.stable_prefix_tokens or 0),
            },
            evaluation=asdict(evaluation),
            evaluator_overrode=evaluator_overrode,
        )
        if self.tracer is not None:
            self.tracer.emit_decision_frame(frame.to_dict())
        if self.evidence_graph is not None:
            try:
                self.evidence_graph.ingest_decision_frame(frame.to_dict())
            except Exception:
                pass
        self._episode_decision_frame_ids.append(frame.id)
        self._episode_total_prompt_tokens += prompt_tokens
        self._episode_total_completion_tokens += completion_tokens
        self._episode_total_cached_tokens += cached_tokens
        self._episode_total_latency_ms += result.latency_ms
        self._episode_step += 1

        self._snapshot(env, state=state, decision=result)
        # record transition for terminal reflection
        self._transitions.append({"turn": turn, "action_text": result.action_text, "evaluation": asdict(evaluation), "reward": None, "outcome": None, "decision_frame_id": frame.id, "state_hash": state_hash})
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
        # terminal evolution: EpisodeCriticAgent (Phase 4) replaces the
        # in-line reflector/insight/sweep triple call. The terminal hook fires
        # whenever there is a critic OR a reflector available — the critic
        # itself decides whether to summarize, propose skills, etc.
        if done and (self.critic_agent is not None or (self.config.reflection_enabled and self.reflector is not None)):
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
            tags = [game_id, f"player_{player}", "critic_fallback" if not (eval_result and eval_result.lesson) else "critic"]
            skill_id = self.memory.consolidate_skill_from_lesson(lesson=lesson, game_id=game_id, evidence=f"experience:{exp_id}", reward=reward, evaluator_score=score, outcome=outcome, tags=tags)
            # Mirror the proposal into the SkillManager so the Critic Agent
            # (Phase 4) can promote / validate / activate it. The legacy JSONL
            # write above is the immediate prompt-visible artefact; the graph
            # entry is what gates the lifecycle.
            if self.skill_manager is not None:
                try:
                    self.skill_manager.propose(
                        name=f"{game_id}:{lesson[:60]}", guidance=lesson,
                        trigger=f"game:{game_id} player_{player}",
                        evidence_ids=[exp_id], game_id=game_id,
                        created_by="agent_fallback",
                        skill_id=skill_id,
                    )
                except Exception:
                    pass
        # update touched skill usage from the terminal outcome
        if skill_id and outcome in {"terminal_win", "terminal_loss", "terminal_draw"}:
            win = outcome == "terminal_win"
            self.memory.update_skill_usage(skill_id=skill_id, win=win, score=eval_result.score if eval_result else 0.5)
        self._emit("memory_written", {"experience_id": exp_id, "skill_id": skill_id, "game_id": game_id, "player": player, "reward": reward, "outcome": outcome, "memory_stats": self.memory.memory_stats()})
        return exp_id

    # ------------------------------------------------------------------ terminal evolution
    def _on_terminal(self, env: Any, *, game_id: str, player: int, outcome: str, reward: float | None) -> None:
        episode_id = self._episode_id
        frame_ids = list(self._episode_decision_frame_ids)
        rewards_obj = getattr(getattr(env, "state", None), "rewards", None)
        rewards_dict: dict[str, float] = {}
        if isinstance(rewards_obj, dict):
            for k, v in rewards_obj.items():
                try:
                    rewards_dict[str(k)] = float(v)
                except Exception:
                    continue
        # Memory maintenance (kept — independent of critic): half-life decay
        # of skill confidence so stale advice fades even on critic failure.
        try:
            self.memory.decay_confidence()
        except Exception:
            pass
        # Phase 4: EpisodeCriticAgent owns the evolution loop. The previous
        # reflector / insight / skill_manager.run_evolution_sweep triplet is
        # replaced by a single critic.run() call. The critic itself decides
        # whether to summarize, query the graph, propose skills/tools, mark
        # do-not-learn, or design experiments — via its own bounded tool loop.
        if self.critic_agent is not None:
            try:
                report = self.critic_agent.run(
                    episode_id=episode_id,
                    game_id=game_id,
                    outcome=outcome,
                    transitions=list(self._transitions),
                    policy_version=self.config.policy_version,
                    player=player,
                    rewards=rewards_dict,
                )
                self._emit("critic_report", {
                    "episode_id": episode_id,
                    "report_id": getattr(report, "id", ""),
                    "skill_proposals": len(getattr(report, "skill_proposals", []) or []),
                    "tool_needs": len(getattr(report, "tool_needs", []) or []),
                    "do_not_learn": len(getattr(report, "do_not_learn", []) or []),
                    "fallback": getattr(report, "fallback", False),
                })
            except Exception as exc:
                self._emit("critic_error", {"error": str(exc), "episode_id": episode_id})
        if self.tracer is not None:
            ep_trace = EpisodeTrace(
                episode_id=episode_id,
                game_id=game_id,
                env_id=str(getattr(env, "env_id", "")),
                seed=self._episode_seed,
                player_id=player,
                turns=len(self._transitions),
                decision_frame_ids=frame_ids,
                rewards=rewards_dict,
                outcome=outcome,
                policy_version=self.config.policy_version,
                total_latency_ms=round(self._episode_total_latency_ms, 3),
                total_prompt_tokens=self._episode_total_prompt_tokens,
                total_completion_tokens=self._episode_total_completion_tokens,
                total_cached_tokens=self._episode_total_cached_tokens,
            )
            self.tracer.emit_episode_trace(ep_trace.to_dict())
            if self.evidence_graph is not None:
                try:
                    self.evidence_graph.ingest_episode_trace(ep_trace.to_dict())
                except Exception:
                    pass
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
        self._episode_id = uuid4().hex[:12]
        self._episode_step = 0
        self._episode_started_at_ms = time.perf_counter() * 1000.0
        self._episode_decision_frame_ids = []
        self._episode_total_prompt_tokens = 0
        self._episode_total_completion_tokens = 0
        self._episode_total_cached_tokens = 0
        self._episode_total_latency_ms = 0.0
        self.need_detector.reset_episode()
        if self.tracer is not None:
            self.tracer.emit("episode_start", {"episode_id": self._episode_id, "game_id": game_id, "policy_version": self.config.policy_version})

    def _recent_retrieved_memory_ids(self, *, query: str, game_id: str, player: int, phase: str) -> list[str]:
        try:
            items = self.memory.recall(query, max_items=8, game_id=game_id, player=player, phase=phase)
        except Exception:
            return []
        return [str(it.get("id")) for it in items if it.get("id")]

    def _extract_skill_ids(self, memory_excerpt: str) -> list[str]:
        if not memory_excerpt:
            return []
        import re
        ids: list[str] = []
        for match in re.findall(r"skill:([A-Za-z0-9_-]+)", memory_excerpt):
            if match not in ids:
                ids.append(match)
        return ids[:16]

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
