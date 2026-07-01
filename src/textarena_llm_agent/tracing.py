from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


EventCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class TraceEvent:
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    step: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextArenaRunTracer:
    def __init__(self, root: str | Path = "workspace/textarena_runs/latest") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.state_path = self.root / "latest_state.json"
        self.frames_path = self.root / "state_frames.jsonl"
        self.control_path = self.root / "control.json"
        self.decision_frames_path = self.root / "decision_frames.jsonl"
        self.episode_traces_path = self.root / "episode_traces.jsonl"
        if not self.events_path.exists():
            self.events_path.write_text("", encoding="utf-8")
        if not self.frames_path.exists():
            self.frames_path.write_text("", encoding="utf-8")
        if not self.decision_frames_path.exists():
            self.decision_frames_path.write_text("", encoding="utf-8")
        if not self.episode_traces_path.exists():
            self.episode_traces_path.write_text("", encoding="utf-8")
        if not self.control_path.exists():
            self.write_control({"paused": False, "step_requested": False, "stop_requested": False})

    def emit(self, event: str | dict[str, Any], payload: dict[str, Any] | None = None, *, step: int | None = None) -> dict[str, Any]:
        if isinstance(event, dict):
            obj = TraceEvent(event=str(event.get("event") or event.get("kind") or "event"), payload={k: v for k, v in event.items() if k not in {"event", "kind"}}, step=event.get("step", step)).to_dict()
        else:
            obj = TraceEvent(event=event, payload=payload or {}, step=step).to_dict()
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        return obj

    def emit_decision_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Append a DecisionFrame.to_dict() payload to decision_frames.jsonl.

        Also mirrors a compact summary to events.jsonl so the dashboard
        renders a unified timeline without reading the second file.
        """
        with self.decision_frames_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(frame, ensure_ascii=False, default=str) + "\n")
        summary = {k: frame.get(k) for k in (
            "id", "episode_id", "game_id", "turn", "step", "candidate_id",
            "action_text", "state_hash", "policy_version",
            "latency_ms", "prompt_tokens", "completion_tokens", "cached_tokens",
            "cache_hit_ratio", "evaluator_overrode",
        )}
        return self.emit("decision_frame", summary, step=frame.get("step"))

    def emit_episode_trace(self, episode: dict[str, Any]) -> dict[str, Any]:
        with self.episode_traces_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False, default=str) + "\n")
        return self.emit("episode_trace", episode)

    def update_state(self, state: dict[str, Any]) -> None:
        data = {"updated_at": datetime.now(timezone.utc).isoformat(), **state}
        self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        with self.frames_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

    def read_events(self, *, limit: int = 500) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        rows = []
        for line in self.events_path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, limit) :]:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    def read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}

    def read_control(self) -> dict[str, Any]:
        try:
            data = json.loads(self.control_path.read_text(encoding="utf-8", errors="replace"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def write_control(self, data: dict[str, Any]) -> None:
        current = {"paused": False, "step_requested": False, "stop_requested": False}
        current.update(data)
        self.control_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    def wait_for_turn(self, *, poll_seconds: float = 0.25) -> bool:
        while True:
            control = self.read_control()
            if control.get("stop_requested"):
                return False
            if not control.get("paused"):
                return True
            if control.get("step_requested"):
                control["step_requested"] = False
                self.write_control(control)
                return True
            time.sleep(max(0.05, poll_seconds))


def state_snapshot(env: Any) -> dict[str, Any]:
    state = getattr(env, "state", None)
    gs = getattr(state, "game_state", {}) if state is not None else {}
    board = gs.get("board") if isinstance(gs, dict) else None
    return {
        "env_id": str(getattr(env, "env_id", type(env).__name__)),
        "turn": int(getattr(state, "turn", 0) if state is not None else 0),
        "current_player": int(getattr(state, "current_player_id", 0) if state is not None else 0),
        "done": bool(getattr(state, "done", False) if state is not None else False),
        "rewards": getattr(state, "rewards", None) if state is not None else None,
        "board": _compact_board(board),
        "game_state": _safe_json(gs),
    }


def _compact_board(board: Any) -> Any:
    if board is None:
        return None
    out = []
    try:
        for row in board:
            out.append([_cell_text(cell) for cell in row])
        return out
    except Exception:
        return str(board)


def _cell_text(cell: Any) -> str:
    if cell is None:
        return "."
    if isinstance(cell, dict):
        return f"P{cell.get('player')}:{cell.get('rank', '?')}"
    return str(cell) if cell != "" else "."


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(k): _safe_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_safe_json(v) for v in value]
        return str(value)
