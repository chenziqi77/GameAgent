from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .game_specs import GameSpec, spec_for_env


@dataclass(slots=True)
class ActionOption:
    candidate_id: str
    action_index: int
    action_text: str
    action_type: str
    game_id: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v not in ("", {}, [])}


class TextArenaStateEncoder:
    """Build compact, TextArena-aware state and legal action candidates."""

    def __init__(self, *, max_valid_actions: int = 80, max_observation_chars: int = 8000, max_log_items: int = 30) -> None:
        self.max_valid_actions = max_valid_actions
        self.max_observation_chars = max_observation_chars
        self.max_log_items = max_log_items

    def encode(self, env: Any, *, include_actions: bool = True) -> dict[str, Any]:
        spec = spec_for_env(env)
        state = getattr(env, "state", None)
        current_player = int(getattr(state, "current_player_id", 0) if state is not None else 0)
        data: dict[str, Any] = {
            "game": {
                "env_id": str(getattr(env, "env_id", spec.default_env_id)),
                "family": spec.family,
                "turn": int(getattr(state, "turn", 0) if state is not None else 0),
                "max_turns": getattr(state, "max_turns", None) if state is not None else None,
                "current_player": current_player,
                "num_players": int(getattr(state, "num_players", 2) if state is not None else 2),
                "done": bool(getattr(state, "done", False) if state is not None else False),
                "rewards": getattr(state, "rewards", None) if state is not None else None,
            },
            "rules": spec.rules,
            "strategic_notes": spec.strategic_notes,
            "current_observation": self._format_observations(env, current_player),
            "recent_log": self._recent_log(env),
            "visible_state": self._visible_game_state(env, spec, current_player),
            "game_info": self._safe(getattr(state, "game_info", {}) if state is not None else {}),
        }
        if include_actions:
            actions = self.valid_actions(env)
            data["valid_actions_count"] = len(actions)
            data["valid_actions"] = [a.to_prompt_dict() for a in actions[: self.max_valid_actions]]
            if len(actions) > self.max_valid_actions:
                data["valid_actions_truncated"] = True
                data["valid_actions_note"] = f"Showing first {self.max_valid_actions}; analyzer ranks the full generated set."
        return data

    def encode_text(self, env: Any, *, include_actions: bool = True) -> str:
        return json.dumps(self.encode(env, include_actions=include_actions), ensure_ascii=False, indent=2, default=str)

    def canonical_state(self, env: Any) -> dict[str, Any]:
        """Stable, timestamp-free state dict suitable for hashing and replay.

        Drops volatile fields (rewards, logs, observations) so the same game
        position produces the same hash across runs. Used by Evidence Graph
        and replay-eval to detect identical decision contexts.
        """
        spec = spec_for_env(env)
        state = getattr(env, "state", None)
        current_player = int(getattr(state, "current_player_id", 0) if state is not None else 0)
        canonical = {
            "env_id": str(getattr(env, "env_id", spec.default_env_id)),
            "family": spec.family,
            "turn": int(getattr(state, "turn", 0) if state is not None else 0),
            "current_player": current_player,
            "num_players": int(getattr(state, "num_players", 2) if state is not None else 2),
            "visible_state": self._visible_game_state(env, spec, current_player),
        }
        return _sort_keys_recursive(_safe_json(canonical))

    def canonical_state_hash(self, env: Any) -> str:
        text = json.dumps(self.canonical_state(env), ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def valid_actions(self, env: Any) -> list[ActionOption]:
        spec = spec_for_env(env)
        family = spec.family
        if family == "TicTacToe":
            options = self._tictactoe_actions(env, family)
        elif family == "KuhnPoker":
            options = self._kuhn_actions(env, family)
        elif family == "SimpleNegotiation":
            options = self._negotiation_actions(env, family)
        elif family == "Stratego":
            options = self._stratego_actions(env, family)
        else:
            options = self._actions_from_observation(env, family)
        for i, opt in enumerate(options):
            opt.candidate_id = f"A{i}"
            opt.action_index = i
        return options

    def _tictactoe_actions(self, env: Any, family: str) -> list[ActionOption]:
        board = getattr(env.state, "game_state", {}).get("board", [])
        out: list[ActionOption] = []
        for r in range(3):
            for c in range(3):
                try:
                    if board[r][c] == "":
                        cell = r * 3 + c
                        out.append(ActionOption("", 0, f"[{cell}]", "place_mark", family, f"Place mark at cell {cell}", {"cell": cell, "row": r, "col": c}))
                except Exception:
                    continue
        return out

    def _kuhn_actions(self, env: Any, family: str) -> list[ActionOption]:
        tree = getattr(env.state, "game_state", {}).get("current_legal_action_tree", {})
        keys = list(tree.keys()) if isinstance(tree, dict) else []
        return [ActionOption("", 0, f"[{k}]", "betting_action", family, f"Kuhn legal action {k}", {"move": k}) for k in keys]

    def _negotiation_actions(self, env: Any, family: str) -> list[ActionOption]:
        gs = getattr(env.state, "game_state", {})
        player = int(getattr(env.state, "current_player_id", 0))
        out: list[ActionOption] = []
        current_offer = gs.get("current_offer")
        if current_offer and int(current_offer.get("to_player", 1 - player)) == player:
            net = _trade_net_value(gs, player, current_offer.get("offered_resources", {}), current_offer.get("requested_resources", {}))
            out.append(ActionOption("", 0, "[Accept]", "accept_offer", family, "Accept the pending trade offer.", {"net_value": net, "offer": _safe_json(current_offer)}))
            out.append(ActionOption("", 0, "[Deny]", "deny_offer", family, "Reject the pending trade offer.", {"net_value": 0, "offer": _safe_json(current_offer)}))
        resources = gs.get("player_resources", {}).get(player, {})
        values = gs.get("player_values", {}).get(player, {})
        low_to_high = sorted(resources.keys(), key=lambda r: (float(values.get(r, 0)), -int(resources.get(r, 0))))
        high_to_low = sorted(resources.keys(), key=lambda r: float(values.get(r, 0)), reverse=True)
        seen: set[str] = set()
        for give in low_to_high[:3]:
            for ask in high_to_low[:3]:
                if give == ask or int(resources.get(give, 0) or 0) <= 0:
                    continue
                for give_qty in [1, min(2, int(resources.get(give, 0) or 0))]:
                    if give_qty <= 0:
                        continue
                    ask_qty = 1
                    action = f"[Offer: {give_qty} {give} -> {ask_qty} {ask}]"
                    if action in seen:
                        continue
                    seen.add(action)
                    net = float(values.get(ask, 0)) * ask_qty - float(values.get(give, 0)) * give_qty
                    out.append(ActionOption("", 0, action, "make_offer", family, f"Offer {give} for {ask}.", {"net_value_if_accepted": net, "give": {give: give_qty}, "request": {ask: ask_qty}}))
        if not out:
            out.append(ActionOption("", 0, "[Deny]", "no_offer", family, "No good generated offer; deny or pass."))
        return out[:40]

    def _stratego_actions(self, env: Any, family: str) -> list[ActionOption]:
        player = int(getattr(env.state, "current_player_id", 0))
        board = getattr(env, "board", getattr(env.state, "game_state", {}).get("board", []))
        lakes = set(tuple(x) for x in getattr(env, "lakes", []))
        out: list[ActionOption] = []
        for row in range(10):
            for col in range(10):
                piece = _board_at(board, row, col)
                if not isinstance(piece, dict) or int(piece.get("player", -1)) != player:
                    continue
                rank = str(piece.get("rank", ""))
                if rank.lower() in {"bomb", "flag"}:
                    continue
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                max_dist = 9 if rank.lower() == "scout" else 1
                for dr, dc in dirs:
                    for dist in range(1, max_dist + 1):
                        nr, nc = row + dr * dist, col + dc * dist
                        if not (0 <= nr < 10 and 0 <= nc < 10) or (nr, nc) in lakes:
                            break
                        target = _board_at(board, nr, nc)
                        if target is None:
                            out.append(_stratego_option(family, row, col, nr, nc, rank, "move_empty", {}))
                            continue
                        if isinstance(target, dict) and int(target.get("player", -1)) != player:
                            out.append(_stratego_option(family, row, col, nr, nc, rank, "attack_hidden", {"target": "opponent_hidden"}))
                        break
        return out

    def _actions_from_observation(self, env: Any, family: str) -> list[ActionOption]:
        text = self._format_observations(env, int(getattr(env.state, "current_player_id", 0)))
        actions = []
        for token in re.findall(r"\[[^\[\]\n]{1,80}\]", text):
            if token not in actions:
                actions.append(token)
        return [ActionOption("", 0, a, "observed_legal_action", family, f"Action extracted from observation: {a}") for a in actions[:80]]

    def _format_observations(self, env: Any, player: int) -> str:
        state = getattr(env, "state", None)
        observations = []
        if state is not None:
            observations = list(getattr(state, "observations", {}).get(player, []))
        lines = []
        for from_id, message, obs_type in observations[-25:]:
            name = getattr(obs_type, "name", str(obs_type))
            lines.append(f"[from={from_id} type={name}]\n{message}")
        text = "\n\n".join(lines)
        return text[-self.max_observation_chars :]

    def _recent_log(self, env: Any) -> list[dict[str, Any]]:
        state = getattr(env, "state", None)
        logs = list(getattr(state, "logs", []) if state is not None else [])[-self.max_log_items :]
        return [{"from": x[0], "message": str(x[1])[:1000]} if isinstance(x, (list, tuple)) and len(x) >= 2 else {"message": str(x)[:1000]} for x in logs]

    def _visible_game_state(self, env: Any, spec: GameSpec, player: int) -> dict[str, Any]:
        gs = getattr(getattr(env, "state", None), "game_state", {}) or {}
        if spec.family == "Stratego":
            return {"board": _visible_stratego_board(env, player), "piece_counts": _visible_stratego_counts(env, player)}
        if spec.family == "KuhnPoker":
            safe = dict(gs)
            cards = safe.get("player_cards")
            if isinstance(cards, dict):
                safe["player_cards"] = {player: cards.get(player)}
            return _safe_json(safe)
        return _safe_json(gs)

    def _safe(self, value: Any) -> Any:
        return _safe_json(value)


def _trade_net_value(gs: dict[str, Any], player: int, offered: dict[str, int], requested: dict[str, int]) -> float:
    values = gs.get("player_values", {}).get(player, {})
    gain = sum(float(values.get(res, 0)) * int(qty) for res, qty in offered.items())
    cost = sum(float(values.get(res, 0)) * int(qty) for res, qty in requested.items())
    return gain - cost


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


def _sort_keys_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sort_keys_recursive(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_sort_keys_recursive(v) for v in value]
    return value


def _board_at(board: Any, row: int, col: int) -> Any:
    try:
        return board[row][col]
    except Exception:
        return None


def _stratego_option(family: str, row: int, col: int, nr: int, nc: int, rank: str, action_type: str, extra: dict[str, Any]) -> ActionOption:
    src = f"{chr(row + 65)}{col}"
    dst = f"{chr(nr + 65)}{nc}"
    meta = {"source": src, "destination": dst, "own_piece": rank, "source_row": row, "source_col": col, "dest_row": nr, "dest_col": nc}
    meta.update(extra)
    return ActionOption("", 0, f"[{src} {dst}]", action_type, family, f"Move {rank} from {src} to {dst}.", meta)


def _visible_stratego_board(env: Any, player: int) -> list[list[str]]:
    board = getattr(env, "board", getattr(env.state, "game_state", {}).get("board", []))
    visible: list[list[str]] = []
    for row in range(10):
        out_row = []
        for col in range(10):
            piece = _board_at(board, row, col)
            if piece == "~":
                out_row.append("~")
            elif piece is None:
                out_row.append(".")
            elif isinstance(piece, dict) and int(piece.get("player", -1)) == player:
                out_row.append(str(piece.get("rank", "?")))
            elif isinstance(piece, dict):
                out_row.append("?")
            else:
                out_row.append(str(piece))
        visible.append(out_row)
    return visible


def _visible_stratego_counts(env: Any, player: int) -> dict[str, Any]:
    board = getattr(env, "board", getattr(env.state, "game_state", {}).get("board", []))
    own: dict[str, int] = {}
    hidden = 0
    for row in range(10):
        for col in range(10):
            piece = _board_at(board, row, col)
            if isinstance(piece, dict) and int(piece.get("player", -1)) == player:
                rank = str(piece.get("rank", "?"))
                own[rank] = own.get(rank, 0) + 1
            elif isinstance(piece, dict):
                hidden += 1
    return {"own": own, "opponent_hidden_pieces": hidden}
