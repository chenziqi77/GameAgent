"""Bounded tool-calling loop whose final emission is a candidate-pick JSON.

Pattern: miniagent AgentLoop — a tool-call -> dispatch -> feed-back loop — but the
final assistant message must be a JSON object selecting a candidate_id. Tools inform
reasoning; they never replace the action contract (TextArena needs an exact bracketed
string, which comes from the selected candidate).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .llm import DecisionLLM, LLMToolResponse, parse_json_object
from miniagent.tools.base import ToolContext
from miniagent.tools.registry import ToolRegistry


@dataclass(slots=True)
class ToolLoopResult:
    decision_json: dict[str, Any]
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    fallback: bool = False
    usage: dict[str, int] = field(default_factory=dict)


class ToolLoop:
    def __init__(self, *, registry: ToolRegistry, ctx: ToolContext,
                 llm: DecisionLLM, max_rounds: int = 4, max_tokens: int = 900) -> None:
        self.registry = registry
        self.ctx = ctx
        self.llm = llm
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens

    def run(self, *, system: str, user: str, candidate_ids: list[str]) -> ToolLoopResult:
        tools = self._openai_tools()
        # If the LLM has no tool-calling support or no tools, use the plain JSON path.
        if not tools or not _llm_supports_tools(self.llm):
            raw = self.llm.complete_json(system=system, user=user, temperature=0.1, max_tokens=self.max_tokens)
            return ToolLoopResult(decision_json=raw, rounds=1)

        messages: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        last_text = ""
        usage_total: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        for rnd in range(1, self.max_rounds + 1):
            try:
                resp = self.llm.complete_with_tools(system=system, user=user if rnd == 1 else "",
                                                    tools=tools, messages=messages or None,
                                                    temperature=0.1, max_tokens=self.max_tokens)
            except Exception:
                # gateway does not support tool-calling: fall back to plain JSON
                raw = self.llm.complete_json(system=system, user=user, temperature=0.1, max_tokens=self.max_tokens)
                return ToolLoopResult(decision_json=raw, tool_events=events, rounds=rnd, fallback=True, usage=usage_total)

            for k, v in (resp.usage or {}).items():
                usage_total[k] = usage_total.get(k, 0) + int(v or 0)

            last_text = resp.text or ""
            if not resp.tool_calls:
                # assistant produced a final answer
                obj = parse_json_object(last_text)
                if _valid_pick(obj, candidate_ids):
                    return ToolLoopResult(decision_json=obj, tool_events=events, rounds=rnd, usage=usage_total)
                # malformed final answer — try once more with a nudge, else fallback
                messages.append({"role": "assistant", "content": last_text or "(no content)"})
                messages.append({"role": "user", "content": "Respond with ONLY a JSON object picking one candidate_id from the provided candidates."})
                continue

            # dispatch tool calls and append results to history
            messages.append(self._assistant_message(resp))
            for tc in resp.tool_calls:
                name = str(tc.get("name") or "")
                args = tc.get("arguments") or {}
                started = time.perf_counter()
                result = self._dispatch(name, args)
                latency_ms = (time.perf_counter() - started) * 1000.0
                events.append({
                    "name": name,
                    "arguments": args,
                    "ok": result.get("ok", False),
                    "content_preview": (result.get("content") or "")[:600],
                    "latency_ms": round(latency_ms, 3),
                    "round_index": rnd,
                    "error": result.get("error"),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(tc.get("id") or name),
                    "name": name,
                    "content": result.get("content") or "",
                })
            # ask for the final decision after tool results
            messages.append({"role": "user", "content": "Using the tool results above, now choose exactly one candidate_id and return JSON only."})

        # exhausted rounds without a valid pick — fall back to top-ranked candidate
        obj = parse_json_object(last_text) if last_text else {}
        if not _valid_pick(obj, candidate_ids):
            obj = {"candidate_id": candidate_ids[0] if candidate_ids else "C0",
                   "rationale": "tool loop exhausted; falling back to top-ranked candidate",
                   "confidence": 0.4, "fallback": True}
            return ToolLoopResult(decision_json=obj, tool_events=events, rounds=self.max_rounds, fallback=True, usage=usage_total)
        return ToolLoopResult(decision_json=obj, tool_events=events, rounds=self.max_rounds, usage=usage_total)

    def _openai_tools(self) -> list[dict[str, Any]]:
        schemas = self.registry.openai_schemas()
        # Chat Completions API wraps each tool as {"type":"function","function":{...}}.
        out: list[dict[str, Any]] = []
        for s in schemas:
            out.append({"type": "function", "function": {"name": s.get("name"), "description": s.get("description"), "parameters": s.get("parameters") or {"type": "object", "properties": {}}}})
        return out

    def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.registry.dispatch(name, args or {}, self.ctx)
            return {"ok": result.ok, "content": result.to_model_output(), "error": None}
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            return {"ok": False, "content": json.dumps({"ok": False, "error": err}), "error": err}

    def _assistant_message(self, resp: LLMToolResponse) -> dict[str, Any]:
        # Re-serialize tool_calls into the Chat Completions assistant-message shape.
        msg: dict[str, Any] = {"role": "assistant"}
        if resp.text:
            msg["content"] = resp.text
        if resp.tool_calls:
            msg["tool_calls"] = [
                {"id": str(tc.get("id") or f"call_{i}"), "type": "function",
                 "function": {"name": str(tc.get("name") or ""), "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False)}}
                for i, tc in enumerate(resp.tool_calls)
            ]
        return msg


def _llm_supports_tools(llm: DecisionLLM) -> bool:
    # HeuristicLLM returns no tool_calls; we still route it through complete_with_tools
    # which short-circuits. Both paths are fine, so we accept anything implementing the method.
    return hasattr(llm, "complete_with_tools")


def _valid_pick(obj: dict[str, Any], candidate_ids: list[str]) -> bool:
    cid = str(obj.get("candidate_id") or "")
    return bool(cid) and (not candidate_ids or cid in candidate_ids)
