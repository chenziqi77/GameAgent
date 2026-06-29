from __future__ import annotations

import copy
import json
from typing import Any

from miniagent.core.outcome import ToolResult
from miniagent.tools.base import BaseTool, ToolContext
from miniagent.tools.registry import ToolRegistry

from .action_analyzer import TextArenaActionAnalyzer
from .memory import EvolvingMemory
from .state_encoder import TextArenaStateEncoder
from .tool_library import ToolLibrary
from .tool_synthesis import SafeToolExecutor


class TextArenaStateSummaryTool(BaseTool):
    name = "textarena_state_summary"
    description = "Return a compact visible TextArena state summary for LLM decision making."
    parameters = {"type": "object", "properties": {"include_actions": {"type": "boolean", "default": True}}, "required": [], "additionalProperties": False}

    def __init__(self, encoder: TextArenaStateEncoder | None = None) -> None:
        self.encoder = encoder or TextArenaStateEncoder()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        env = _env(ctx)
        return ToolResult(True, self.encoder.encode_text(env, include_actions=bool(args.get("include_actions", True))))


class TextArenaActionAnalysisTool(BaseTool):
    name = "textarena_analyze_actions"
    description = "Rank legal TextArena actions using game-aware heuristics and bounded simulation."
    parameters = {"type": "object", "properties": {"top_k": {"type": "integer", "default": 12, "minimum": 1, "maximum": 80}}, "required": [], "additionalProperties": False}

    def __init__(self, analyzer: TextArenaActionAnalyzer | None = None) -> None:
        self.analyzer = analyzer or TextArenaActionAnalyzer()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        env = _env(ctx)
        top_k = int(args.get("top_k") or 12)
        candidates = self.analyzer.analyze(env, top_k=top_k)
        return ToolResult(True, json.dumps([c.to_prompt_dict() for c in candidates], ensure_ascii=False, indent=2, default=str))


class TextArenaSimulateActionTool(BaseTool):
    name = "textarena_simulate_action"
    description = "Deep-copy a TextArena environment and simulate one legal action text to preview the resulting state."
    parameters = {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False}

    def __init__(self, analyzer: TextArenaActionAnalyzer | None = None) -> None:
        self.analyzer = analyzer or TextArenaActionAnalyzer()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        result = self.analyzer.simulate(_env(ctx), str(args["action"]))
        return ToolResult(True, json.dumps(result, ensure_ascii=False, indent=2, default=str))


class TextArenaMemoryTool(BaseTool):
    name = "textarena_recall_memory"
    description = "Recall durable TextArena lessons, skills, reflections, and prior evaluated experiences for the current situation."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}, "max_items": {"type": "integer", "default": 8}}, "required": ["query"], "additionalProperties": False}

    def __init__(self, memory: EvolvingMemory | None = None) -> None:
        self.memory = memory or EvolvingMemory()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        meta = ctx.metadata
        game_id = str(meta.get("game_id") or "")
        player = meta.get("player")
        phase = meta.get("phase")
        return ToolResult(True, self.memory.format_for_prompt(str(args.get("query") or ""), max_items=int(args.get("max_items") or 8), game_id=game_id, player=player, phase=phase))


class TextArenaReflectionTool(BaseTool):
    name = "textarena_recall_reflections"
    description = "Recall past self-reflections (lessons learned from finished games) relevant to the current situation."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 2}}, "required": ["query"], "additionalProperties": False}

    def __init__(self, memory: EvolvingMemory | None = None) -> None:
        self.memory = memory or EvolvingMemory()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        game_id = str(ctx.metadata.get("game_id") or "")
        items = self.memory.retrieve_reflections(query=str(args.get("query") or ""), game_id=game_id, top_k=int(args.get("top_k") or 2))
        if not items:
            return ToolResult(True, "No relevant reflections yet.")
        text = "\n\n".join(f"[reflection {r.id} outcome={r.outcome}] {r.text}\nLesson: {r.actionable_lesson}" for r in items)
        return ToolResult(True, text)


class SynthesizedToolAdapter(BaseTool):
    """Adapter that runs a library-verified synthesized tool through the safe executor."""

    def __init__(self, record, executor: SafeToolExecutor) -> None:
        self._record = record
        self._executor = executor

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._record.name

    @property
    def description(self) -> str:  # type: ignore[override]
        return self._record.description

    @property
    def parameters(self) -> dict[str, Any]:  # type: ignore[override]
        return self._record.parameters or {"type": "object", "properties": {}, "required": []}

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        meta = ctx.metadata
        game_state = meta.get("game_state_snapshot")
        if game_state is None:
            game_state = {}
        visible_text = str(meta.get("visible_text") or "")
        injected = {"game_state": copy.deepcopy(game_state), "visible_text": visible_text}
        res = self._executor.execute(code=self._record.implementation, args=args or {}, injected=injected)
        library: ToolLibrary | None = meta.get("tool_library")
        if library is not None:
            library.record_use(self._record.name, ok=res.ok)
        if not res.ok:
            return ToolResult(False, "", error=res.error or "tool failed")
        try:
            content = json.dumps(res.value, ensure_ascii=False, default=str)
        except Exception:
            content = str(res.value)
        return ToolResult(True, content)


def create_textarena_tool_registry(*, encoder: TextArenaStateEncoder | None = None,
                                   analyzer: TextArenaActionAnalyzer | None = None,
                                   memory: EvolvingMemory | None = None,
                                   tool_library: ToolLibrary | None = None,
                                   executor: SafeToolExecutor | None = None,
                                   game_id: str = "") -> ToolRegistry:
    encoder = encoder or TextArenaStateEncoder()
    analyzer = analyzer or TextArenaActionAnalyzer(encoder)
    memory = memory or EvolvingMemory()
    tools: list[BaseTool] = [
        TextArenaStateSummaryTool(encoder),
        TextArenaActionAnalysisTool(analyzer),
        TextArenaSimulateActionTool(analyzer),
        TextArenaMemoryTool(memory),
        TextArenaReflectionTool(memory),
    ]
    if tool_library is not None and executor is not None and game_id:
        for rec in tool_library.active_for(game_id):
            tools.append(SynthesizedToolAdapter(rec, executor))
    registry = ToolRegistry()
    for tool in tools:
        if tool.name in registry.tools:
            continue
        registry.register(tool)
    return registry


def _env(ctx: ToolContext) -> Any:
    env = ctx.metadata.get("env")
    if env is None:
        raise RuntimeError("TextArena tools require ctx.metadata['env'] to contain the current environment.")
    return env
