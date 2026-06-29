"""Evolvable LLM decision agent for TextArena games.

Public surface for the upgraded agent architecture: Voyager-style tool synthesis,
tiered evolving memory (episodic / skill / reflection) with BM25 retrieval, ExpeL
insight consolidation, Reflexion self-reflection, per-game prompt building, bounded
context budgeting, a Double DQN RL baseline, optimal reference agents, and an
evaluation harness (self-play trend, Elo, exploitability).
"""

from .agent import Decision, TextArenaAgentConfig, TextArenaDecisionAgent
from .context_packet import ContextBudgeter, DecisionContextPacket
from .evaluation import EvaluationHarness, MatchResult
from .llm import HeuristicLLM, LLMToolResponse, OpenAIChatLLM
from .memory import EvolvingMemory, Experience, Reflection, SkillMemory
from .optimal_agents import OptimalKuhnBR, OptimalTTT, RandomAgent
from .prompt_builder import GamePromptBuilder
from .reflection import Reflector
from .retrieval import BM25Retriever, ScoredItem
from .insight import InsightExtractor
from .state_encoder import ActionOption, TextArenaStateEncoder
from .tool_library import SynthesizedToolRecord, ToolLibrary
from .tool_loop import ToolLoop, ToolLoopResult
from .tool_synthesis import SafeToolExecutor, ToolNeedDetector, ToolProposal, ToolSynthesizer
from .tracing import TextArenaRunTracer

__all__ = [
    "ActionOption",
    "BM25Retriever",
    "ContextBudgeter",
    "Decision",
    "DecisionContextPacket",
    "EvaluationHarness",
    "EvolvingMemory",
    "Experience",
    "GamePromptBuilder",
    "HeuristicLLM",
    "InsightExtractor",
    "LLMToolResponse",
    "MatchResult",
    "OpenAIChatLLM",
    "OptimalKuhnBR",
    "OptimalTTT",
    "RandomAgent",
    "Reflection",
    "Reflector",
    "ScoredItem",
    "SafeToolExecutor",
    "SkillMemory",
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
    "ToolSynthesizer",
]
