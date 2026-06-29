from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from .state_encoder import ActionOption, TextArenaStateEncoder


@dataclass(slots=True)
class CandidateAnalysis:
    candidate_id: str
    action_index: int
    action_text: str
    action_type: str
    score: float
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    simulation: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextArenaActionAnalyzer:
    """Game-specific legal-action ranking with bounded optional simulation."""

    def __init__(self, encoder: TextArenaStateEncoder | None = None, *, simulate_top: int = 4) -> None:
        self.encoder = encoder or TextArenaStateEncoder()
        self.simulate_top = simulate_top

    def analyze(self, env: Any, options: list[ActionOption] | None = None, *, top_k: int = 12) -> list[CandidateAnalysis]:
        options = options if options is not None else self.encoder.valid_actions(env)
        raw = [self.score_action(env, opt) for opt in options]
        raw.sort(key=lambda x: x.score, reverse=True)
        for cand in raw[: max(0, self.simulate_top)]:
            if cand.action_type == "attack_hidden":
                cand.simulation = {"skipped": True, "reason": "hidden-information Stratego attack is not simulated to avoid leaking ranks"}
                continue
            cand.simulation = self.simulate(env, cand.action_text)
            if cand.simulation.get("invalid"):
                cand.score -= 50
                cand.risks.append("simulation reported invalid action")
            else:
                cand.score += float(cand.simulation.get("delta_score", 0.0))
        raw.sort(key=lambda x: x.score, reverse=True)
        for i, cand in enumerate(raw[:top_k]):
            original = cand.candidate_id
            cand.candidate_id = f"C{i}"
            cand.action = dict(cand.action)
            cand.action["candidate_id"] = cand.candidate_id
            cand.action["original_candidate_id"] = original
        return raw[:top_k]

    def score_action(self, env: Any, opt: ActionOption) -> CandidateAnalysis:
        family = opt.game_id
        if family == "TicTacToe":
            score, reasons, risks = self._score_tictactoe(env, opt)
        elif family == "KuhnPoker":
            score, reasons, risks = self._score_kuhn(env, opt)
        elif family == "SimpleNegotiation":
            score, reasons, risks = self._score_negotiation(env, opt)
        elif family == "Stratego":
            score, reasons, risks = self._score_stratego(env, opt)
        else:
            score, reasons, risks = 0.0, ["legal action extracted from TextArena observation"], []
        return CandidateAnalysis(
            candidate_id=opt.candidate_id,
            action_index=opt.action_index,
            action_text=opt.action_text,
            action_type=opt.action_type,
            score=round(float(score), 3),
            reasons=reasons[:6] or ["legal action"],
            risks=risks[:5],
            action=opt.to_prompt_dict(),
        )

    def simulate(self, env: Any, action_text: str) -> dict[str, Any]:
        before = _summary(env)
        try:
            sim = copy.deepcopy(env)
            done, info = sim.step(action_text)
            after = _summary(sim)
            player = int(before.get("current_player", 0))
            before_reward = before.get("rewards") or {}
            after_reward = after.get("rewards") or {}
            delta = 0.0
            if isinstance(after_reward, dict) and player in after_reward:
                delta += float(after_reward[player])
            if isinstance(before_reward, dict) and player in before_reward:
                delta -= float(before_reward[player])
            if after.get("done") and isinstance(after_reward, dict):
                delta += float(after_reward.get(player, 0)) * 5
            return {"before": before, "after": after, "done": done, "info": info, "delta_score": round(delta, 3)}
        except Exception as exc:
            return {"invalid": True, "error": f"{type(exc).__name__}: {exc}"}

    def _score_tictactoe(self, env: Any, opt: ActionOption) -> tuple[float, list[str], list[str]]:
        board = getattr(env.state, "game_state", {}).get("board", [])
        player = int(getattr(env.state, "current_player_id", 0))
        mark = "X" if player == 1 else "O"
        opp = "O" if mark == "X" else "X"
        cell = int(opt.metadata.get("cell", -1))
        r, c = divmod(cell, 3)
        score = 1.0
        reasons: list[str] = []
        risks: list[str] = []
        if cell == 4:
            score += 2.0
            reasons.append("center controls the most lines")
        if cell in {0, 2, 6, 8}:
            score += 1.0
            reasons.append("corner supports fork patterns")
        if _ttt_wins_after(board, r, c, mark):
            score += 100.0
            reasons.append("immediate winning move")
        if _ttt_wins_after(board, r, c, opp):
            score += 50.0
            reasons.append("blocks opponent immediate win")
        if _ttt_allows_opp_win(board, r, c, mark, opp):
            score -= 25.0
            risks.append("may allow opponent an immediate win")
        return score, reasons, risks

    def _score_kuhn(self, env: Any, opt: ActionOption) -> tuple[float, list[str], list[str]]:
        gs = getattr(env.state, "game_state", {})
        player = int(getattr(env.state, "current_player_id", 0))
        card = (gs.get("player_cards") or {}).get(player)
        move = str(opt.metadata.get("move") or "").lower()
        score = 0.0
        reasons: list[str] = []
        risks: list[str] = []
        if card == 2:
            if move in {"bet", "call"}:
                score += 8
                reasons.append("K is the strongest card; value aggression is favored")
            elif move == "fold":
                score -= 10
                risks.append("folding K is usually too conservative")
        elif card == 1:
            if move in {"check", "call"}:
                score += 3
                reasons.append("Q has medium showdown value")
            if move == "bet":
                score += 1
                reasons.append("Q can mix bets to avoid predictability")
        elif card == 0:
            if move in {"check", "fold"}:
                score += 3
                reasons.append("J is weak; avoid investing unless bluffing")
            if move == "call":
                score -= 5
                risks.append("calling with J is usually poor")
            if move == "bet":
                score += 1
                risks.append("J bet is a bluff and should be mixed sparingly")
        return score, reasons, risks

    def _score_negotiation(self, env: Any, opt: ActionOption) -> tuple[float, list[str], list[str]]:
        net = float(opt.metadata.get("net_value", opt.metadata.get("net_value_if_accepted", 0)) or 0)
        score = net
        reasons: list[str] = []
        risks: list[str] = []
        if opt.action_type == "accept_offer":
            if net > 0:
                score += 8
                reasons.append(f"accepting improves own inventory value by {net:.1f}")
            else:
                score -= 8
                risks.append(f"accepting has non-positive own value change {net:.1f}")
        elif opt.action_type == "deny_offer":
            current_offer = opt.metadata.get("offer") or {}
            offered = current_offer.get("offered_resources", {}) if isinstance(current_offer, dict) else {}
            requested = current_offer.get("requested_resources", {}) if isinstance(current_offer, dict) else {}
            deny_net = -_trade_net_from_payload(env, offered, requested)
            if deny_net >= 0:
                score += 6
                reasons.append("denies an unfavorable or unclear offer")
        elif opt.action_type == "make_offer":
            if net > 0:
                score += min(8.0, net / 5.0)
                reasons.append(f"proposal is positive for own values if accepted ({net:.1f})")
            if net > 25:
                risks.append("very one-sided offer may be rejected")
                score -= 2
        return score, reasons, risks

    def _score_stratego(self, env: Any, opt: ActionOption) -> tuple[float, list[str], list[str]]:
        rank = str(opt.metadata.get("own_piece") or "")
        src_row = int(opt.metadata.get("source_row", 0))
        dst_row = int(opt.metadata.get("dest_row", 0))
        player = int(getattr(env.state, "current_player_id", 0))
        score = 1.0
        reasons: list[str] = []
        risks: list[str] = []
        if opt.action_type == "attack_hidden":
            score += 2.0
            reasons.append("attacks a hidden opponent piece to gain material or information")
            if rank in {"Miner", "Marshal", "General", "Colonel"}:
                score += 1.0
                reasons.append("attacking piece has useful tactical role")
            if rank == "Spy":
                risks.append("Spy is fragile except against Marshal")
        else:
            advance = (dst_row - src_row) if player == 0 else (src_row - dst_row)
            if advance > 0:
                score += 1.5
                reasons.append("advances toward opponent side")
            elif advance < 0:
                risks.append("retreats; only useful for repositioning")
                score -= 0.5
        if rank == "Miner":
            score += 0.5
            reasons.append("Miner mobility is valuable for eventual Bomb threats")
        return score, reasons, risks


def _summary(env: Any) -> dict[str, Any]:
    state = getattr(env, "state", None)
    return {
        "turn": int(getattr(state, "turn", 0) if state is not None else 0),
        "current_player": int(getattr(state, "current_player_id", 0) if state is not None else 0),
        "done": bool(getattr(state, "done", False) if state is not None else False),
        "rewards": getattr(state, "rewards", None) if state is not None else None,
    }


def _ttt_wins_after(board: list[list[str]], r: int, c: int, mark: str) -> bool:
    try:
        if board[r][c] != "":
            return False
        b = [row[:] for row in board]
        b[r][c] = mark
        lines = b + [[b[0][i], b[1][i], b[2][i]] for i in range(3)] + [[b[0][0], b[1][1], b[2][2]], [b[0][2], b[1][1], b[2][0]]]
        return any(line == [mark, mark, mark] for line in lines)
    except Exception:
        return False


def _ttt_allows_opp_win(board: list[list[str]], r: int, c: int, mark: str, opp: str) -> bool:
    try:
        b = [row[:] for row in board]
        b[r][c] = mark
        for rr in range(3):
            for cc in range(3):
                if b[rr][cc] == "" and _ttt_wins_after(b, rr, cc, opp):
                    return True
    except Exception:
        return False
    return False


def _trade_net_from_payload(env: Any, offered: dict[str, int], requested: dict[str, int]) -> float:
    gs = getattr(env.state, "game_state", {})
    player = int(getattr(env.state, "current_player_id", 0))
    values = gs.get("player_values", {}).get(player, {})
    return sum(float(values.get(r, 0)) * int(q) for r, q in offered.items()) - sum(float(values.get(r, 0)) * int(q) for r, q in requested.items())
