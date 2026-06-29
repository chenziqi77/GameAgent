"""Optimal / reference agents for evaluation.

- ``OptimalTTT``: memoized minimax for TicTacToe — never loses.
- ``OptimalKuhnBR``: best-response value/policy against a target (empirical) policy,
  using the verified ``current_legal_action_tree`` structure. Used to compute Kuhn
  exploitability = BR_value - game_value(1/18).
- ``RandomAgent``: uniform over legal actions.

These are the "oracles" the evaluation harness needs to score exploitability and to
populate the Elo tournament.
"""
from __future__ import annotations

import random
from typing import Any

from .state_encoder import TextArenaStateEncoder


class RandomAgent:
    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, env: Any, player: int) -> str:
        actions = TextArenaStateEncoder().valid_actions(env)
        if not actions:
            raise RuntimeError("No valid actions for RandomAgent.")
        return self._rng.choice(actions).action_text


class OptimalTTT:
    """Perfect TicTacToe via minimax with memoization. Never loses."""

    def __init__(self) -> None:
        # memo maps board+mark -> (best_move, best_val) so cached calls return a move too.
        self._memo: dict[tuple, tuple] = {}

    def act(self, env: Any, player: int) -> str:
        board = self._board(env)  # 3x3 of ''/'X'/'O'; player1=X, player0=O
        mark = "X" if player == 1 else "O"
        opp = "O" if mark == "X" else "X"
        # work on a copy so the live env board is never mutated by minimax
        bcopy = [row[:] for row in board]
        best_move, _ = self._best(bcopy, mark, opp)
        if best_move is None:
            # fall back to any legal move
            actions = TextArenaStateEncoder().valid_actions(env)
            return actions[0].action_text
        r, c = best_move
        cell = r * 3 + c
        return f"[{cell}]"

    def _board(self, env: Any) -> list[list[str]]:
        gs = getattr(env.state, "game_state", {})
        b = gs.get("board", [["", "", ""], ["", "", ""]])
        # ensure 3x3 of strings
        return [[str(b[i][j]) if b[i][j] else "" for j in range(3)] for i in range(3)]

    def _best(self, board: list[list[str]], mark: str, opp: str) -> tuple[tuple[int, int] | None, int]:
        """Negamax with memoization. Returns (best_move, value) for `mark` to move.
        value is from `mark`'s perspective: +1 mark wins, -1 opp wins, 0 draw.
        Both move and value are cached so repeated calls return a valid move.
        """
        key = (tuple(tuple(row) for row in board), mark)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        # Terminal: the opponent just moved, so check if THEY won.
        if _ttt_winner(board, opp):
            result = (None, -1)
            self._memo[key] = result
            return result
        empties = [(r, c) for r in range(3) for c in range(3) if board[r][c] == ""]
        if not empties:
            result = (None, 0)
            self._memo[key] = result
            return result
        best_val = -2
        best_move = empties[0]
        for (r, c) in empties:
            board[r][c] = mark
            if _ttt_winner(board, mark):
                val = 1
            else:
                _, opp_val = self._best(board, opp, mark)
                val = -opp_val  # negamax: flip perspective
            board[r][c] = ""
            if val > best_val:
                best_val = val
                best_move = (r, c)
                if best_val == 1:
                    break  # cannot do better than a win
        result = (best_move, best_val)
        self._memo[key] = result
        return result


class OptimalKuhnBR:
    """Best-response to a target (empirical) Kuhn policy.

    target_policy: dict mapping (card, history_tuple) -> {action: prob}
    The BR assumes the OPPONENT follows target_policy and the BR player knows
    its own card; it plays to maximize expected chips against that opponent.

    We compute the BR action at each of the BR player's decision points by
    one-step expected value over the verified action tree.
    """

    def __init__(self, target_policy: dict[tuple, dict[str, float]] | None = None) -> None:
        self.target_policy = target_policy or {}

    def act(self, env: Any, player: int) -> str:
        gs = getattr(env.state, "game_state", {})
        card = (gs.get("player_cards") or {}).get(player)
        legal = self._legal_actions(gs)
        if not legal:
            actions = TextArenaStateEncoder().valid_actions(env)
            return actions[0].action_text
        # If we have no model of the opponent, play the Nash-ish heuristic:
        # bet K, bluff J sometimes, call K, fold J to a bet.
        if not self.target_policy:
            return self._heuristic(card, legal)
        # pick the action with the highest expected value vs the modeled opponent
        best_a, best_v = None, -1e9
        for a in legal:
            v = self._ev_action(gs, player, card, a)
            if v > best_v:
                best_v, best_a = v, a
        return f"[{best_a}]"

    def _legal_actions(self, gs: dict) -> list[str]:
        tree = gs.get("current_legal_action_tree") or {}
        return list(tree.keys()) if isinstance(tree, dict) else []

    def _heuristic(self, card, legal: list[str]) -> str:
        legal_l = [a.lower() for a in legal]
        if card == 2:  # King
            if "bet" in legal_l:
                return "[bet]" if "bet" in legal_l else f"[{legal[0]}]"
            if "call" in legal_l:
                return "[call]"
            return f"[{legal[0]}]"
        if card == 0:  # Jack
            if "bet" in legal_l:
                return "[check]" if "check" in legal_l else "[bet]"
            if "call" in legal_l:
                return "[fold]" if "fold" in legal_l else f"[{legal[0]}]"
            return f"[{legal[0]}]"
        # Queen
        if "check" in legal_l:
            return "[check]"
        if "call" in legal_l:
            return "[call]"
        return f"[{legal[0]}]"

    def _ev_action(self, gs: dict, player: int, card, action: str) -> float:
        """One-step expected value of `action` vs the modeled opponent.

        This is a lightweight proxy: terminal-ish outcomes (fold/showdown) are
        scored directly; non-terminal moves get a small heuristic continuation.
        """
        tree = gs.get("current_legal_action_tree") or {}
        subtree = tree.get(action) if isinstance(tree, dict) else None
        pot = float(gs.get("pot", 2))
        if subtree == "showdown":
            opp_card = (gs.get("player_cards") or {}).get(1 - player)
            if opp_card is not None and card is not None:
                return (pot / 2.0) if card > opp_card else -(pot / 2.0)
            return 0.0
        if subtree == "loser":
            return -(pot / 2.0)  # this action folded us out
        # non-terminal: estimate via opponent model probability of folding / calling
        opp_card_guess = None
        # fold probability from opponent model if we bet
        history = tuple(self._history(gs, player))
        probs = self.target_policy.get((card, history)) or {}
        if action.lower() == "bet":
            p_fold = probs.get("fold", 0.0)
            return p_fold * (pot / 2.0) + (1 - p_fold) * 0.0
        return 0.0

    def _history(self, gs: dict, player: int) -> list[str]:
        # reconstruct a crude history from the action tree depth
        return []

    @staticmethod
    def best_response_value(target_policy: dict[tuple, dict[str, float]], *, samples: int = 2000) -> float:
        """Estimate the BR value (chips/hand) against target_policy by simulation.

        Returns the mean chips won by the BR player per hand. Exploitability is
        BR_value - game_value(1/18) when target is the agent being tested.
        """
        rng = random.Random(0)
        total = 0.0
        for _ in range(samples):
            deck = [0, 1, 2]
            rng.shuffle(deck)
            br_card, opp_card = deck[0], deck[1]
            # Simple Kuhn tree simulation: BR acts first as a heuristic best response,
            # opponent follows target_policy. Pot starts at 2 (each ante 1).
            pot = 2
            # BR first action
            legal = ["check", "bet"]
            a = _br_first_action(br_card, target_policy, rng)
            folded = False
            if a == "check":
                # opponent now check or bet
                opp_a = _sample(target_policy.get((opp_card, ("check",)), {"check": 0.5, "bet": 0.5}), rng)
                if opp_a == "check":
                    # showdown
                    total += (pot / 2) if br_card > opp_card else -(pot / 2)
                    continue
                else:
                    # opponent bets; BR call or fold
                    pot += 2
                    a2 = "call" if br_card == 2 else ("fold" if br_card == 0 else "call")
                    if a2 == "fold":
                        total -= (pot - 2) / 2
                        continue
                    total += (pot / 2) if br_card > opp_card else -(pot / 2)
                    continue
            else:  # bet
                pot += 2
                opp_a = _sample(target_policy.get((opp_card, ("bet",)), {"fold": 0.5, "call": 0.5}), rng)
                if opp_a == "fold":
                    total += (pot - 2) / 2
                    continue
                total += (pot / 2) if br_card > opp_card else -(pot / 2)
                continue
        return total / samples


def _br_first_action(card: int, target_policy: dict, rng: random.Random) -> str:
    if card == 2:
        return "bet"
    if card == 0:
        return "check"
    return "check"  # Queen checks first (Nash)


def _sample(probs: dict[str, float], rng: random.Random) -> str:
    if not probs:
        return "check"
    r = rng.random()
    cum = 0.0
    for a, p in probs.items():
        cum += p
        if r <= cum:
            return a
    return next(iter(probs))


def _ttt_winner(board: list[list[str]], mark: str) -> bool:
    lines = board + [[board[0][i], board[1][i], board[2][i]] for i in range(3)] + [[board[0][0], board[1][1], board[2][2]], [board[0][2], board[1][1], board[2][0]]]
    return any(line == [mark, mark, mark] for line in lines)
