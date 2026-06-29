from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GameSpec:
    family: str
    default_env_id: str
    rules: str
    strategic_notes: str
    num_players: int = 2
    action_format: str = ""
    game_theoretic_principles: str = ""


# Stratego rank maps — single source of truth.
# Board cells carry string rank names; player_pieces uses int tuples. Both DQN encoder
# and state-key normalizer must agree, so the canonical mapping lives here.
RANK_NAME_TO_INT: dict[str, int] = {
    "Flag": 0, "Spy": 1, "Scout": 2, "Miner": 3, "Sergeant": 4,
    "Lieutenant": 5, "Captain": 6, "Major": 7, "Colonel": 8, "General": 9, "Marshal": 10, "Bomb": 11,
}
RANK_INT_TO_NAME: dict[int, str] = {v: k for k, v in RANK_NAME_TO_INT.items()}

# Negotiation economy.
RESOURCE_NAMES: list[str] = ["Wheat", "Wood", "Sheep", "Brick", "Ore"]
BASE_VALUES: dict[str, int] = {"Wheat": 5, "Wood": 10, "Sheep": 15, "Brick": 25, "Ore": 40}

# Kuhn cards: 0=J, 1=Q, 2=K. Game value for first player = 1/18.
KUHN_GAME_VALUE: float = 1.0 / 18.0


GAME_SPECS: dict[str, GameSpec] = {
    "TicTacToe": GameSpec(
        family="TicTacToe",
        default_env_id="TicTacToe-v0-raw",
        rules=(
            "TicTacToe is a perfect-information 3x3 game. On your turn output exactly one bracketed cell, "
            "for example [4] (cells are 0..8, row-major). Player 1 plays X, Player 0 plays O. "
            "You win by making three in a row (horizontally, vertically, or diagonally)."
        ),
        strategic_notes=(
            "Prefer immediate wins, then forced blocks, then center, then corners, then edges. "
            "Create forks when possible and never make a move that allows an immediate opponent win."
        ),
        action_format="Exactly one bracketed cell index: [0] through [8]. No other text.",
        game_theoretic_principles=(
            "TicTacToe is solved: perfect play by both sides is a draw. The optimal priority is "
            "win-in-one > block-opponent-win > create-a-fork > center > corner > edge. "
            "Never play a move after which the opponent has an immediate winning reply (the analyzer "
            "flags such self-traps). When defending a fork threat, choose the move that removes the "
            "opponent's double-attack. First mover should open center or corner; second mover should "
            "play center if open, else a corner, and always block immediately."
        ),
    ),
    "KuhnPoker": GameSpec(
        family="KuhnPoker",
        default_env_id="KuhnPoker-v0-raw",
        rules=(
            "Kuhn Poker uses a 3-card deck J < Q < K (internally 0=J, 1=Q, 2=K). Each player antes one chip "
            "and receives one private card. Legal actions depend on the betting tree and are exactly "
            "[check], [bet], [call], or [fold]. The environment reports the current legal action tree."
        ),
        strategic_notes=(
            "With K, value bet and call often. With J, avoid paying off bets but bluff sometimes when "
            "first to act. With Q, mix cautious calls/checks against expected bluffs and stay unpredictable."
        ),
        action_format="Exactly one bracketed action: [check], [bet], [call], or [fold]. No other text.",
        game_theoretic_principles=(
            "Kuhn Poker has a known Nash equilibrium. First player's value is +1/18 chip per hand. "
            "Equilibrium mixing: with King always bet (first to act) and always call a bet; with Queen "
            "always check first and call a bet with probability 1/3; with Jack check first and bet as a "
            "bluff with probability 1/3, and always fold to a bet. If you deviate, a best-response "
            "opponent exploits you. Because the deck is tiny, randomize your bluffs and thin value bets "
            "so your action does not reveal your card. Do not call bets with Jack except as an explicit "
            "mixed bluff-catch."
        ),
    ),
    "SimpleNegotiation": GameSpec(
        family="SimpleNegotiation",
        default_env_id="SimpleNegotiation-v0-raw",
        rules=(
            "Negotiate resource trades. Actions are [Offer: <qty> <Resource>, ... -> <qty> <Resource>, ...], "
            "[Accept], or [Deny]. Resources are Wheat, Wood, Sheep, Brick, Ore. You win if your private "
            "inventory value increased more than your opponent's by the turn limit."
        ),
        strategic_notes=(
            "Accept only positive-value offers for your own valuation unless denial is strategically better. "
            "Offer resources that are low-value to you in exchange for resources that are high-value to you, "
            "while keeping the offer plausible enough to be accepted."
        ),
        action_format=(
            "One bracketed action: [Offer: 3 Sheep -> 5 Brick], [Accept], or [Deny]. "
            "Use exact resource names (Wheat/Wood/Sheep/Brick/Ore) and the ' -> ' separator."
        ),
        game_theoretic_principles=(
            "Each player has private per-resource valuations that perturb the base values "
            "(Wheat 5, Wood 10, Sheep 15, Brick 25, Ore 40). The winner is decided by total "
            "inventory_value change at the turn limit, so trade surplus in your own valuation is what "
            "matters, not raw counts. Rational opponents accept only offers positive in their valuation, "
            "so a profitable trade must be mutually positive: give goods you undervalue for goods you "
            "overvalue relative to base, while leaving the counterparty a small positive margin. Reject "
            "offers whose net value to you is <= 0. On your last turns, prefer locking in accepted surplus "
            "over speculative offers."
        ),
    ),
    "Stratego": GameSpec(
        family="Stratego",
        default_env_id="Stratego-v0-raw",
        rules=(
            "Stratego is a hidden-information 10x10 board game. Move one movable piece using [A0 B0] "
            "coordinate format (source space destination). Flag and Bomb cannot move; lakes are blocked. "
            "Capture the opponent Flag or remove all opponent movable pieces to win. Higher rank captures "
            "lower rank; the special rules are Spy beats Marshal, Miner defuses Bomb."
        ),
        strategic_notes=(
            "Do not assume hidden opponent ranks (they appear as '?'). Develop front-line pieces, preserve "
            "Miners for Bomb threats, use Scouts for cheap information, protect your Flag, and attack when "
            "risk/reward is favorable."
        ),
        action_format="Exactly one bracketed move: [<src> <dst>] e.g. [A0 B0] where rows are A-J and cols 0-9.",
        game_theoretic_principles=(
            "Stratego is a game of information asymmetry. Opponent piece ranks are hidden ('?') until "
            "revealed in combat; deduce ranks from outcomes and from which pieces refuse to move. Piece "
            "value ordering (high to low): Marshal > General > Colonel > Major > Captain > Lieutenant > "
            "Sergeant > Miner > Scout > Spy; Flag and Bomb are immobile. Key tactical rules: Spy defeats "
            "only Marshal (and loses otherwise); Miner defeats Bomb; Scout moves any clear straight line. "
            "Strategic principles: preserve Miners until Bombs are located; do not overextend high-value "
            "pieces into unknown territory; use Scouts to probe cheaply and force reveals; keep Bombs and "
            "the Flag protected in a corner/back row; trade favorably by attacking only when your piece's "
            "expected rank advantage justifies revealing it. Lakes at (4,2),(4,3),(5,2),(5,3),(4,6),(4,7),"
            "(5,6),(5,7) are impassable and channel attacks."
        ),
    ),
}


ALIASES = {
    "tictactoe": "TicTacToe",
    "tic-tac-toe": "TicTacToe",
    "kuhn": "KuhnPoker",
    "kuhnpoker": "KuhnPoker",
    "simple-negotiation": "SimpleNegotiation",
    "simplenegotiation": "SimpleNegotiation",
    "negotiation": "SimpleNegotiation",
    "stratego": "Stratego",
}


def canonical_game_id(game: str) -> str:
    key = (game or "").strip()
    if not key:
        return "TicTacToe"
    if key in GAME_SPECS:
        return key
    base = key.replace("-v0", "").replace("-raw", "").replace("-train", "").replace("-long", "").replace("-short", "").replace("-medium", "").replace("-extreme", "")
    return ALIASES.get(base.lower().replace("_", "-"), ALIASES.get(base.lower().replace("-", ""), base))


def spec_for_env(env: Any) -> GameSpec:
    env_id = str(getattr(env, "env_id", "") or type(env).__name__)
    family = canonical_game_id(env_id)
    return GAME_SPECS.get(family, GameSpec(family=family, default_env_id=env_id, rules="Follow the TextArena game rules in the observation.", strategic_notes="Select a legal action that maximizes expected reward."))


def default_env_id(game: str) -> str:
    family = canonical_game_id(game)
    return GAME_SPECS.get(family, GAME_SPECS["TicTacToe"]).default_env_id
