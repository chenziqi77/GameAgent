"""Evolvable LLM decision agent for TextArena games.

Public surface for the upgraded agent architecture: Voyager-style tool synthesis,
tiered evolving memory (episodic / skill / reflection) with BM25 retrieval, ExpeL
insight consolidation, Reflexion self-reflection, per-game prompt building, bounded
context budgeting, a Double DQN RL baseline, optimal reference agents, and an
evaluation harness (self-play trend, Elo, exploitability).
"""

from .agent import Decision, TextArenaAgentConfig, TextArenaDecisionAgent
from .context_packet import ContextBudgeter, DecisionContextPacket
from .critic_agent import CRITIC_TOOL_SCHEMA, CriticReport, CriticToolCall, EpisodeCriticAgent
from .evaluation import EvaluationHarness, MatchResult
from .hypothesis import (
    ArmMetrics,
    BaselineSpec,
    DEFAULT_HYPOTHESES,
    HarnessResult,
    Hypothesis,
    HypothesisHarness,
    HypothesisReport,
    MatchSample,
    OpponentPoolSnapshot,
    ReplayEvalResult,
    attach_skill_quality,
    baseline_by_id,
    baseline_templates,
    opponent_pool_snapshot,
    render_hypothesis_report,
    replay_eval,
)
from .llm import HeuristicLLM, LLMToolResponse, OpenAIChatLLM
from .memory import EvolvingMemory, Experience, Reflection, SkillMemory
from .optimal_agents import OptimalKuhnBR, OptimalTTT, RandomAgent
from .prompt_builder import GamePromptBuilder
from .prompt_compiler import CompiledPrompt, PromptCompiler
from .reflection import Reflector
from .retrieval import BM25Retriever, ScoredItem
from .insight import InsightExtractor
from .skill_manager import SkillManager, SkillStatus, SkillVersion
from .state_encoder import ActionOption, TextArenaStateEncoder
from .tool_library import SynthesizedToolRecord, ToolLibrary, ToolStatus
from .tool_loop import ToolLoop, ToolLoopResult
from .tool_synthesis import SafeToolExecutor, ToolNeedDetector, ToolProposal, ToolSynthesizer
from .tool_validator import PipelineResult, StageResult, ToolValidator
from .tracing import TextArenaRunTracer

__all__ = [
    "ActionOption",
    "ArmMetrics",
    "BaselineSpec",
    "BM25Retriever",
    "CompiledPrompt",
    "ContextBudgeter",
    "CRITIC_TOOL_SCHEMA",
    "CriticReport",
    "CriticToolCall",
    "Decision",
    "DecisionContextPacket",
    "DEFAULT_HYPOTHESES",
    "EpisodeCriticAgent",
    "EvaluationHarness",
    "EvolvingMemory",
    "Experience",
    "GamePromptBuilder",
    "HarnessResult",
    "HeuristicLLM",
    "Hypothesis",
    "HypothesisHarness",
    "HypothesisReport",
    "InsightExtractor",
    "LLMToolResponse",
    "MatchResult",
    "MatchSample",
    "OpenAIChatLLM",
    "OpponentPoolSnapshot",
    "OptimalKuhnBR",
    "OptimalTTT",
    "PipelineResult",
    "PromptCompiler",
    "RandomAgent",
    "Reflection",
    "Reflector",
    "ReplayEvalResult",
    "ScoredItem",
    "StageResult",
    "SafeToolExecutor",
    "SkillManager",
    "SkillMemory",
    "SkillStatus",
    "SkillVersion",
    "SynthesizedToolRecord",
    "TextArenaAgentConfig",
    "TextArenaDecisionAgent",
    "TextArenaRunTracer",
    "TextArenaStateEncoder",
    "ToolLibrary",
    "ToolLoop",
    "ToolLoopResult",
    "ToolNeedDetector",
    "ToolProposal",
    "ToolStatus",
    "ToolSynthesizer",
    "ToolValidator",
    "attach_skill_quality",
    "baseline_by_id",
    "baseline_templates",
    "opponent_pool_snapshot",
    "render_hypothesis_report",
    "replay_eval",
]
