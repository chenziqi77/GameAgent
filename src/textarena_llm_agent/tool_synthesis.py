"""Voyager-style tool synthesis with a safe in-process executor.

The agent may ask an LLM to design a reusable Python helper for a recurring
sub-problem (e.g. "enumerate fork-creating TicTacToe moves", "compute Kuhn pot-odds",
"rank negotiation offers by opponent acceptance likelihood"). The proposed code is
AST-validated, executed against a *deep-copied* game_state snapshot, and — if it
passes verification — cached in a tool library and registered for future turns.

Safety: in-process exec is NOT a true sandbox. We mitigate with an AST import
whitelist, a blocked-builtin / dangerous-attribute gate, a restricted ``__builtins__``,
a recursion limit, and a hard timeout (SIGALRM on the main thread, ThreadPoolExecutor
fallback otherwise). Synthesis is rare (triggered by recurrence) and can be disabled.
"""
from __future__ import annotations

import ast
import copy
import signal
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from typing import Any

from .llm import DecisionLLM, parse_json_object
from .prompts import TOOL_SYNTHESIS_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ToolProposal:
    name: str
    description: str
    parameters: dict[str, Any]
    implementation: str
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    game_id: str = ""
    task_description: str = ""


@dataclass(slots=True)
class ToolExecResult:
    ok: bool
    value: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Safe executor
# ---------------------------------------------------------------------------

_ALLOWED_MODULES = {"json", "re", "math", "copy", "collections", "itertools", "functools", "statistics"}
_DANGEROUS_ATTRS = {
    "__globals__", "__builtins__", "__subclasses__", "__class__", "__bases__", "__mro__",
    "__import__", "__code__", "__dict__", "__spec__", "__reduce__", "__getstate__",
}
_BLOCKED_NAMES = {
    "__import__", "open", "eval", "exec", "compile", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir", "breakpoint", "input", "exit", "quit",
    "memoryview", "help",
}

import builtins as _builtins_mod

_SAFE_BUILTIN_NAMES = {
    "len", "range", "int", "float", "str", "list", "dict", "set", "tuple", "frozenset",
    "bool", "isinstance", "issubclass", "enumerate", "zip", "sorted", "reversed",
    "min", "max", "sum", "abs", "round", "any", "all", "map", "filter", "print",
    "type", "format", "repr", "chr", "ord", "hex", "oct", "bin",
    "pow", "divmod", "hasattr", "iter", "next",
}
_SAFE_BUILTINS = {"True": True, "False": False, "None": None}
for _n in _SAFE_BUILTIN_NAMES:
    _v = getattr(_builtins_mod, _n, None)
    if _v is not None:
        _SAFE_BUILTINS[_n] = _v
del _n, _v


class SafeToolExecutor:
    """AST-validated, timeout-bounded in-process executor for synthesized tool code."""

    def __init__(self, *, timeout_s: float = 3.0, recursion_limit: int = 200,
                 max_impl_chars: int = 4000) -> None:
        self.timeout_s = timeout_s
        self.recursion_limit = recursion_limit
        self.max_impl_chars = max_impl_chars

    def validate_ast(self, code: str) -> list[str]:
        """Return a list of policy violations (empty == safe)."""
        violations: list[str] = []
        if len(code) > self.max_impl_chars:
            return [f"implementation exceeds {self.max_impl_chars} chars"]
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"syntax error: {exc.msg}"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = (alias.name or "").split(".")[0]
                    if mod not in _ALLOWED_MODULES:
                        violations.append(f"import of disallowed module: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod not in _ALLOWED_MODULES:
                    violations.append(f"import from disallowed module: {node.module}")
            elif isinstance(node, ast.Attribute):
                if node.attr in _DANGEROUS_ATTRS:
                    violations.append(f"access to dangerous attribute: .{node.attr}")
            elif isinstance(node, ast.Name):
                if node.id in _BLOCKED_NAMES and isinstance(node.ctx, ast.Load):
                    violations.append(f"use of blocked name: {node.id}")
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _BLOCKED_NAMES:
                    violations.append(f"call to blocked function: {func.id}")
                # block open(mode='w'/'a'/'x'/'+')
                if isinstance(func, ast.Name) and func.id == "open":
                    mode = _extract_open_mode(node)
                    if mode and any(c in "wax+" for c in mode):
                        violations.append(f"open() with write/append mode '{mode}'")
        return violations

    def execute(self, *, code: str, fn_name: str = "run",
                args: dict[str, Any] | None = None,
                injected: dict[str, Any] | None = None) -> ToolExecResult:
        violations = self.validate_ast(code)
        if violations:
            return ToolExecResult(ok=False, error="; ".join(violations))
        args = dict(args or {})
        injected = dict(injected or {})
        fn = self._compile_fn(code, fn_name)
        if fn is None:
            return ToolExecResult(ok=False, error="could not extract function 'run'")
        call_args = dict(injected)
        call_args.update(args)
        return self._run_with_timeout(fn, call_args)

    def _compile_fn(self, code: str, fn_name: str):
        # Restricted globals: __builtins__ is a dict of safe names only.
        # (_SAFE_BUILTINS contains True/False/None as values, so place it verbatim
        # rather than dict(_SAFE_BUILTINS), which would try to iterate them as pairs.)
        glb = {"__builtins__": dict(_SAFE_BUILTINS)}
        try:
            exec(compile(code, "<synthesized_tool>", "exec"), glb)  # noqa: S102 - gated by validate_ast above
        except Exception:
            return None
        fn = glb.get(fn_name)
        if not callable(fn):
            return None
        return fn

    def _run_with_timeout(self, fn, call_args: dict[str, Any]) -> ToolExecResult:
        import time
        if threading.current_thread() is threading.main_thread():
            return self._run_signal(fn, call_args)
        return self._run_thread(fn, call_args)

    def _run_signal(self, fn, call_args: dict[str, Any]) -> ToolExecResult:
        import time
        prev_rec = sys.getrecursionlimit()
        sys.setrecursionlimit(self.recursion_limit)
        prev_handler = signal.getsignal(signal.SIGALRM)
        result: dict[str, Any] = {}

        def _handler(signum, frame):
            raise TimeoutError("tool execution timed out")

        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, self.timeout_s)
            start = time.time()
            value = fn(**call_args)
            result["ok"] = True
            result["value"] = value
        except Exception as exc:
            result["ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev_handler or signal.SIG_DFL)
            sys.setrecursionlimit(prev_rec)
            result["elapsed_ms"] = round((time.time() - start) * 1000.0, 2)
        return ToolExecResult(**result)

    def _run_thread(self, fn, call_args: dict[str, Any]) -> ToolExecResult:
        import time
        prev_rec = sys.getrecursionlimit()
        sys.setrecursionlimit(self.recursion_limit)
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn, **call_args)
                start = time.time()
                try:
                    value = fut.result(timeout=self.timeout_s)
                    return ToolExecResult(ok=True, value=value, elapsed_ms=round((time.time() - start) * 1000.0, 2))
                except FutureTimeout:
                    return ToolExecResult(ok=False, error="tool execution timed out")
                except Exception as exc:
                    return ToolExecResult(ok=False, error=f"{type(exc).__name__}: {exc}", elapsed_ms=round((time.time() - start) * 1000.0, 2))
        finally:
            sys.setrecursionlimit(prev_rec)


def _extract_open_mode(call_node: ast.Call) -> str:
    if len(call_node.args) >= 2:
        m = call_node.args[1]
        if isinstance(m, ast.Constant) and isinstance(m.value, str):
            return m.value
    for kw in call_node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return ""


# ---------------------------------------------------------------------------
# Tool synthesizer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ToolSynthesizer:
    """Ask the LLM for a reusable tool, verify it, and register it into the library."""
    llm: DecisionLLM
    executor: SafeToolExecutor
    library: "ToolLibrary"  # type: ignore[name-defined]
    max_impl_chars: int = 4000

    def synthesize(self, *, task_description: str, game_id: str,
                   context_summary: str, game_state_snapshot: dict[str, Any]) -> "ToolProposal | None":
        user = (
            f"Game: {game_id}\nRecurring sub-problem: {task_description}\n\n"
            f"Context summary:\n{context_summary[:3000]}\n\n"
            f"Current game_state snapshot (deep-copied, safe to inspect):\n"
            f"{_short_json(game_state_snapshot)}\n\n"
            "Design a tool whose `run(game_state, visible_text, **args)` returns JSON-serializable insight for this sub-problem."
        )
        try:
            raw = self.llm.complete_json(system=TOOL_SYNTHESIS_SYSTEM_PROMPT, user=user, temperature=0.1, max_tokens=1500)
        except Exception:
            return None
        return _proposal_from_raw(raw, game_id=game_id, task_description=task_description)

    def verify(self, proposal: ToolProposal, *, game_state_snapshot: dict[str, Any],
               visible_text: str = "") -> bool:
        for tc in proposal.test_cases[:3]:
            args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
            res = self.executor.execute(code=proposal.implementation, args=args,
                                        injected={"game_state": copy.deepcopy(game_state_snapshot), "visible_text": visible_text})
            if not res.ok:
                return False
        # smoke run with the real snapshot
        smoke = self.executor.execute(code=proposal.implementation,
                                       injected={"game_state": copy.deepcopy(game_state_snapshot), "visible_text": visible_text})
        return smoke.ok

    def synthesize_and_register(self, *, task_description: str, game_id: str,
                                context_summary: str, game_state_snapshot: dict[str, Any],
                                visible_text: str = "") -> str | None:
        proposal = self.synthesize(task_description=task_description, game_id=game_id,
                                   context_summary=context_summary, game_state_snapshot=game_state_snapshot)
        if proposal is None:
            return None
        if self.library.has(task_description=task_description, game_id=game_id):
            return None
        if not self.verify(proposal, game_state_snapshot=game_state_snapshot, visible_text=visible_text):
            self.library.record_failed_candidate(task_description=task_description, game_id=game_id)
            return None
        rec = self.library.register_verified(proposal)
        return rec.name if rec else None


# ---------------------------------------------------------------------------
# Need detector (throttled trigger)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ToolNeedDetector:
    """Fire tool synthesis only when a sub-problem recurs, capped at 1/episode."""
    threshold: int = 3
    _counts: Counter = field(default_factory=Counter)
    _synthesized_this_episode: bool = False

    def reset_episode(self) -> None:
        self._synthesized_this_episode = False

    def observe(self, *, game_id: str, phase: str, intent_tokens: list[str]) -> tuple[bool, str]:
        """Return (should_synthesize, task_description)."""
        key = (game_id, phase, " ".join(sorted(intent_tokens)))
        self._counts[key] += 1
        if self._synthesized_this_episode:
            return False, ""
        if self._counts[key] >= self.threshold:
            self._synthesized_this_episode = True
            return True, key[2]
        return False, ""

    def from_explicit_request(self, *, game_id: str, need_desc: str) -> tuple[bool, str]:
        if self._synthesized_this_episode or not need_desc:
            return False, ""
        self._synthesized_this_episode = True
        return True, need_desc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_json(value: Any, *, limit: int = 2500) -> str:
    import json
    try:
        s = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        s = str(value)
    return s[:limit] + ("...[truncated]" if len(s) > limit else "")


def _proposal_from_raw(raw: dict[str, Any], *, game_id: str, task_description: str) -> "ToolProposal | None":
    name = str(raw.get("name") or "").strip()
    impl = str(raw.get("implementation") or "")
    if not name or not impl or not _is_snake(name):
        return None
    params = raw.get("parameters")
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}, "required": []}
    tests = raw.get("test_cases") if isinstance(raw.get("test_cases"), list) else []
    return ToolProposal(
        name=name, description=str(raw.get("description") or ""),
        parameters=params, implementation=impl, test_cases=list(tests),
        game_id=game_id, task_description=task_description,
    )


def _is_snake(name: str) -> bool:
    import re
    return bool(re.fullmatch(r"[a-z][a-z0-9_]{2,48}", name))


# Avoid circular import: ToolLibrary imported lazily inside type hints.
from .tool_library import ToolLibrary  # noqa: E402  (placed at end to avoid cycle)
