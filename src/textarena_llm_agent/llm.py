from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class DecisionLLM(Protocol):
    def complete_json(self, *, system: str, user: str, temperature: float = 0.1, max_tokens: int = 900) -> dict[str, Any]:
        ...

    def complete_with_tools(self, *, system: str, user: str, tools: list[dict[str, Any]],
                            messages: list[dict[str, Any]] | None = None,
                            temperature: float = 0.1, max_tokens: int = 900) -> "LLMToolResponse":
        ...


@dataclass(slots=True)
class LLMToolResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None


@dataclass(slots=True)
class HeuristicLLM:
    """Deterministic offline policy head used for tests and no-key demos."""

    def complete_json(self, *, system: str, user: str, temperature: float = 0.1, max_tokens: int = 900) -> dict[str, Any]:
        candidates = _extract_json_array_after_marker(user, "CANDIDATE_ACTIONS_JSON:")
        if candidates:
            best = candidates[0]
            return {
                "candidate_id": best.get("candidate_id", "C0"),
                "action": best.get("action_text"),
                "confidence": 0.55,
                "rationale": "Offline heuristic selected the highest-ranked legal candidate.",
                "plan": "Execute the legal move and use feedback to update future skills.",
            }
        return {"candidate_id": "C0", "confidence": 0.1, "rationale": "No candidates found in prompt."}

    def complete_with_tools(self, *, system: str, user: str, tools: list[dict[str, Any]],
                            messages: list[dict[str, Any]] | None = None,
                            temperature: float = 0.1, max_tokens: int = 900) -> LLMToolResponse:
        # No-key mode: skip the tool loop entirely — just return the heuristic decision.
        return LLMToolResponse(text=json.dumps(self.complete_json(system=system, user=user, temperature=temperature, max_tokens=max_tokens)), tool_calls=[])


@dataclass(slots=True)
class OpenAIChatLLM:
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls, *, prefix: str = "") -> "OpenAIChatLLM":
        """Build an OpenAI-compatible client from environment variables.

        ``prefix="CRITIC"`` reads CRITIC_MODEL / CRITIC_API_KEY /
        CRITIC_API_BASE first, then falls back to the normal MCP / OpenAI / SCS
        variables. This lets the actor and critic share the same gateway while
        using different, cheaper model names.
        """
        prefix = prefix.strip().upper()
        model_keys: list[str] = []
        api_key_keys: list[str] = []
        base_keys: list[str] = []
        if prefix:
            model_keys.append(f"{prefix}_MODEL")
            api_key_keys.append(f"{prefix}_API_KEY")
            base_keys.append(f"{prefix}_API_BASE")
            base_keys.append(f"{prefix}_BASE_URL")
        model_keys.extend(["MCP_MODEL", "OPENAI_MODEL", "SCS_LLM_MODEL"])
        api_key_keys.extend(["MCP_API_KEY", "OPENAI_API_KEY", "SCS_LLM_API_KEY"])
        base_keys.extend(["MCP_API_BASE", "OPENAI_BASE_URL", "SCS_LLM_BASE_URL"])
        return cls(
            model=_first_env(model_keys) or "gpt-4o-mini",
            api_key=_first_env(api_key_keys),
            base_url=_first_env(base_keys),
        )

    def complete_json(self, *, system: str, user: str, temperature: float = 0.1, max_tokens: int = 900) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing API key. Set MCP_API_KEY, OPENAI_API_KEY, or SCS_LLM_API_KEY.")
        client = self._client()
        response = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return parse_json_object(_response_text(response) or "{}")

    def complete_with_tools(self, *, system: str, user: str, tools: list[dict[str, Any]],
                            messages: list[dict[str, Any]] | None = None,
                            temperature: float = 0.1, max_tokens: int = 900) -> LLMToolResponse:
        if not self.api_key:
            raise RuntimeError("Missing API key. Set MCP_API_KEY, OPENAI_API_KEY, or SCS_LLM_API_KEY.")
        client = self._client()
        # If there are no tools, fall back to the plain JSON path.
        if not tools:
            obj = self.complete_json(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
            return LLMToolResponse(text=json.dumps(obj, ensure_ascii=False), tool_calls=[])
        msg_history: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if messages:
            msg_history.extend(messages)
        else:
            msg_history.append({"role": "user", "content": user})
        # NOTE: do not set response_format here; many gateways reject
        # response_format=json_object together with tools=.
        response = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
            messages=msg_history,
        )
        message = _response_message(response)
        tool_calls = _normalize_tool_calls(_message_tool_calls(message))
        return LLMToolResponse(text=_message_text(message), tool_calls=tool_calls, raw=response)

    def _client(self):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install openai to use OpenAIChatLLM: pip install openai") from exc
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)


@dataclass(slots=True)
class MiniAgentCompatibleLLM:
    client: Any
    model_name: str = "MINIAgent-configured-model"

    def complete_json(self, *, system: str, user: str, temperature: float = 0.1, max_tokens: int = 900) -> dict[str, Any]:
        if hasattr(self.client, "complete_json"):
            return self.client.complete_json(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
        if hasattr(self.client, "create_response"):
            response = self.client.create_response(
                instructions=system + "\nReturn exactly one JSON object.",
                input_items=[{"role": "user", "content": user}],
                tools=[],
                metadata={"agent": "TextArenaDecisionAgent", "model": self.model_name},
            )
            return parse_json_object(getattr(response, "text", "") or "{}")
        raise TypeError("Unsupported MiniAgent client adapter: expected complete_json or create_response.")

    def complete_with_tools(self, *, system: str, user: str, tools: list[dict[str, Any]],
                            messages: list[dict[str, Any]] | None = None,
                            temperature: float = 0.1, max_tokens: int = 900) -> LLMToolResponse:
        obj = self.complete_json(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
        return LLMToolResponse(text=json.dumps(obj, ensure_ascii=False), tool_calls=[])


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else {"value": obj}
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else {"value": obj}
        except Exception:
            pass
    return {"_raw": text, "_parse_error": True}


def _response_text(response: Any) -> str:
    return _message_text(_response_message(response))


def _response_message(response: Any) -> Any:
    if response is None or isinstance(response, str):
        return response or ""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                return choice.get("message") or choice.get("delta") or choice
            return getattr(choice, "message", choice)
        return response.get("message") or response
    choices = getattr(response, "choices", None)
    if choices:
        choice = choices[0]
        return getattr(choice, "message", choice)
    return response


def _message_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if content is None:
            content = message.get("text") or message.get("output_text")
        return _content_to_text(content)
    content = getattr(message, "content", None)
    if content is None:
        content = getattr(message, "text", None) or getattr(message, "output_text", None)
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _message_tool_calls(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("tool_calls")
    return getattr(message, "tool_calls", None)


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for tc in raw:
        try:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                if isinstance(fn, dict):
                    args_raw = fn.get("arguments", "{}")
                    name = fn.get("name", "")
                else:
                    args_raw = getattr(fn, "arguments", "{}")
                    name = getattr(fn, "name", "")
                call_id = tc.get("id", "")
            else:
                fn = tc.function
                args_raw = getattr(fn, "arguments", "{}")
                name = getattr(fn, "name", "")
                call_id = getattr(tc, "id", "")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            except Exception:
                args = {"_raw": args_raw}
            out.append({"id": call_id, "name": name, "arguments": args})
        except Exception:
            continue
    return out


def _first_env(keys: list[str]) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _extract_json_array_after_marker(text: str, marker: str) -> list[dict[str, Any]]:
    idx = text.find(marker)
    if idx < 0:
        return []
    tail = text[idx + len(marker) :]
    start = tail.find("[")
    if start < 0:
        return []
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(tail[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(tail[start : i + 1])
                        return data if isinstance(data, list) else []
                    except Exception:
                        return []
    return []
