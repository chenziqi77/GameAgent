"""Traditional deep-RL baseline: Double DQN with per-game structured features.

Algorithm: Double DQN (van Hasselt et al. 2016) —
    target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
The previous baseline selected AND evaluated with the target net (vanilla DQN);
this fixes the overestimation by selecting the next action with the ONLINE net
and evaluating it with the TARGET net.

A single policy net acts for whichever player's turn it is (self-play). Features
are always encoded from the acting player's perspective, and perspective-relative
rewards (+1 own win, -1 own loss, 0 ongoing/draw) make the net learn a single
"value of being the player to move" function. This keeps the RL interface small
and efficient while acknowledging TextArena and LLM agents consume different signals.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .action_analyzer import TextArenaActionAnalyzer
from .cli import build_env
from .game_specs import RESOURCE_NAMES, RANK_NAME_TO_INT, canonical_game_id
from .state_encoder import TextArenaStateEncoder


@dataclass(slots=True)
class RLBaselineResult:
    game: str
    episodes: int
    eval_episodes: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    invalid_move_rate: float
    avg_reward: float
    avg_turns: float
    output_dir: str


# ---------------------------------------------------------------------------
# Per-game structured feature encoder (always acting-player perspective)
# ---------------------------------------------------------------------------

class GameFeatureEncoder:
    def __init__(self, *, game: str, max_actions: int = 80) -> None:
        self.game = canonical_game_id(game)
        self.max_actions = max_actions
        self.state_encoder = TextArenaStateEncoder(max_valid_actions=max_actions)

    @property
    def dim(self) -> int:
        return _FEATURE_DIMS[self.game]

    def encode(self, env: Any, *, player: int) -> np.ndarray:
        fn = _ENCODERS.get(self.game, _encode_generic)
        return fn(env, player)

    def features(self, env: Any, *, player: int) -> np.ndarray:
        """Back-compat alias returning a normalized feature vector."""
        v = self.encode(env, player=player)
        norm = np.linalg.norm(v)
        return (v / norm).astype(np.float32) if norm > 0 else v.astype(np.float32)


_FEATURE_DIMS: dict[str, int] = {
    "TicTacToe": 28,
    "KuhnPoker": 24,
    "SimpleNegotiation": 12,
    "Stratego": 1400,
}


def _encode_tictactoe(env: Any, player: int) -> np.ndarray:
    board = getattr(env.state, "game_state", {}).get("board", [["", "", ""], ["", "", ""], ["", "", ""]])
    out = np.zeros(28, dtype=np.float32)
    # 27 = 9 cells x {empty, own, opponent}; player1=X, player0=O
    for i in range(3):
        for j in range(3):
            try:
                cell = board[i][j]
            except Exception:
                cell = ""
            idx = (i * 3 + j) * 3
            if cell == "":
                out[idx] = 1.0
            elif (cell == "X" and player == 1) or (cell == "O" and player == 0):
                out[idx + 1] = 1.0
            else:
                out[idx + 2] = 1.0
    out[27] = 1.0  # side-to-move bias
    return out


def _encode_kuhn(env: Any, player: int) -> np.ndarray:
    gs = getattr(env.state, "game_state", {})
    out = np.zeros(24, dtype=np.float32)
    card = (gs.get("player_cards") or {}).get(player)
    if card is not None:
        out[int(card)] = 1.0  # 0=J, 1=Q, 2=K one-hot
    # betting-history proxy: depth into the action tree + starting player
    tree = gs.get("current_legal_action_tree")
    out[3] = _tree_depth(tree) / 4.0
    sp = gs.get("starting_player")
    out[4] = 1.0 if sp == player else 0.0
    out[5] = 1.0 if sp != player else 0.0
    # pot / chips (normalized)
    try:
        out[6] = min(1.0, float(gs.get("pot", 0)) / 8.0)
    except Exception:
        pass
    chips = gs.get("player_chips") or {}
    try:
        out[7] = min(1.0, float(chips.get(player, 0)) / 6.0)
        out[8] = min(1.0, float(chips.get(1 - player, 0)) / 6.0)
    except Exception:
        pass
    # whose turn (am I the player to move?)
    out[9] = 1.0 if int(getattr(env.state, "current_player_id", 0)) == player else 0.0
    return out


def _encode_simple_negotiation(env: Any, player: int) -> np.ndarray:
    gs = getattr(env.state, "game_state", {})
    out = np.zeros(12, dtype=np.float32)
    resources = (gs.get("player_resources") or {}).get(player, {})
    values = (gs.get("player_values") or {}).get(player, {})
    for i, res in enumerate(RESOURCE_NAMES):
        out[i] = min(1.0, float(resources.get(res, 0)) / 30.0)
        out[5 + i] = min(1.0, float(values.get(res, 0)) / 50.0)
    # current offer net value to us if we are the responder
    offer = gs.get("current_offer")
    if isinstance(offer, dict) and int(offer.get("to_player", -1)) == player:
        offered = offer.get("offered_resources", {}) or {}
        requested = offer.get("requested_resources", {}) or {}
        gain = sum(float(values.get(r, 0)) * int(q) for r, q in offered.items())
        cost = sum(float(values.get(r, 0)) * int(q) for r, q in requested.items())
        out[10] = min(1.0, max(-1.0, (gain - cost) / 50.0))
    try:
        out[11] = min(1.0, float(getattr(env.state, "turn", 0)) / float(getattr(env.state, "max_turns", 10) or 10))
    except Exception:
        pass
    return out


def _encode_stratego(env: Any, player: int) -> np.ndarray:
    out = np.zeros(1400, dtype=np.float32)
    board = getattr(env, "board", getattr(env.state, "game_state", {}).get("board", []))
    lakes = set(tuple(x) for x in getattr(env, "lakes", []))
    # 12 ranks x 100 cells for OUR pieces; plus 100 opponent-presence; plus 100 lakes
    for r in range(10):
        for c in range(10):
            cell = _board_at(board, r, c)
            base = (r * 10 + c)
            if cell == "~" or (r, c) in lakes:
                out[1200 + base] = 1.0
                continue
            if isinstance(cell, dict):
                if int(cell.get("player", -1)) == player:
                    rank_name = str(cell.get("rank", ""))
                    ri = RANK_NAME_TO_INT.get(rank_name, 11)
                    out[ri * 100 + base] = 1.0
                else:
                    out[1300 + base] = 1.0
    return out


def _encode_generic(env: Any, player: int) -> np.ndarray:
    text = TextArenaStateEncoder().encode_text(env, include_actions=False)
    vec = np.zeros(256, dtype=np.float32)
    for token in text.split():
        vec[hash(token) % 256] += 1.0
    return vec


_ENCODERS = {
    "TicTacToe": _encode_tictactoe,
    "KuhnPoker": _encode_kuhn,
    "SimpleNegotiation": _encode_simple_negotiation,
    "Stratego": _encode_stratego,
}


def _tree_depth(tree: Any) -> int:
    depth = 0
    cur = tree
    while isinstance(cur, dict) and cur:
        depth += 1
        cur = next(iter(cur.values()))
    return depth


def _board_at(board: Any, r: int, c: int) -> Any:
    try:
        return board[r][c]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Double DQN policy
# ---------------------------------------------------------------------------

class DQNPolicy:
    def __init__(self, *, feature_dim: int, max_actions: int = 80, lr: float = 1e-3, gamma: float = 0.95, hidden: int = 128) -> None:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Torch is required for the DQN baseline. Install with: pip install -e .[rl]") from exc
        self.torch = torch
        self.gamma = gamma
        self.max_actions = max_actions
        self.hidden = hidden
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, max_actions))
        self.target = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, max_actions))
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def select(self, feat: np.ndarray, legal_count: int, epsilon: float) -> int:
        if legal_count <= 0:
            return 0
        if random.random() < epsilon:
            return random.randrange(legal_count)
        with self.torch.no_grad():
            q = self.net(self.torch.tensor(feat, dtype=self.torch.float32))[0:legal_count]
            return int(self.torch.argmax(q).item())

    def update(self, batch: list[tuple[np.ndarray, int, float, np.ndarray, int, bool]]) -> None:
        """Double DQN update: select next action with ONLINE net, evaluate with TARGET net."""
        torch = self.torch
        states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        next_counts = [b[4] for b in batch]
        dones = torch.tensor([b[5] for b in batch], dtype=torch.float32)
        q = self.net(states).gather(1, actions[:, None]).squeeze(1)
        with self.torch.no_grad():
            next_online = self.net(next_states)
            # select argmax_a Q_online(s', a) over the legal next slots
            next_actions = torch.stack([next_online[i, : max(1, c)].argmax() for i, c in enumerate(next_counts)])
            next_q_target = self.target(next_states)
            next_q = torch.stack([next_q_target[i, int(next_actions[i])] for i in range(len(batch))])
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = torch.nn.functional.mse_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    def save(self, path: str | Path) -> None:
        self.torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.net.load_state_dict(self.torch.load(path, map_location="cpu"))
        self.target.load_state_dict(self.net.state_dict())


# ---------------------------------------------------------------------------
# Self-play training
# ---------------------------------------------------------------------------

def train_dqn_baseline(*, game: str, episodes: int, eval_episodes: int, max_steps: int, output_dir: str | Path, seed: int = 0, target_sync_every: int = 10) -> RLBaselineResult:
    random.seed(seed)
    np.random.seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = GameFeatureEncoder(game=game, max_actions=80)
    analyzer = TextArenaActionAnalyzer(encoder.state_encoder, simulate_top=0)
    policy = DQNPolicy(feature_dim=encoder.dim, max_actions=encoder.max_actions)
    replay: deque[tuple[np.ndarray, int, float, np.ndarray, int, bool]] = deque(maxlen=8000)
    completed = 0
    for ep in range(episodes):
        env = build_env(game, seed=seed + ep)
        epsilon = max(0.05, 0.5 * (1 - ep / max(1, episodes)))
        first_player = int(getattr(env.state, "current_player_id", 0))
        ep_transitions = []
        for _ in range(max_steps):
            if bool(getattr(env.state, "done", False)):
                break
            actions = analyzer.analyze(env, top_k=encoder.max_actions)
            if not actions:
                break
            player = int(getattr(env.state, "current_player_id", 0))
            feat = encoder.features(env, player=player)
            slot = policy.select(feat, len(actions), epsilon)
            # terminal-aware reward: +1 if this move wins for the mover, -1 if it loses (invalid), 0 otherwise
            pre_reward = _reward(env, player)
            env.step(actions[slot].action_text)
            done = bool(getattr(env.state, "done", False))
            post_reward = _reward(env, player)
            r = _shaped_reward(pre_reward, post_reward, done)
            ep_transitions.append((feat, slot, r, player, done))
            if done:
                # Store the terminal transition with the mover's final perspective reward.
                _assign_terminal_rewards(ep_transitions, env, first_player)
                terminal_r = float(ep_transitions[-1][2]) if ep_transitions else r
                replay.append((feat, slot, terminal_r, feat, 0, True))
                if len(replay) >= 32:
                    policy.update(random.sample(list(replay), 32))
                completed += 1
                break
            # store transition with the NEXT mover's perspective features
            next_player = int(getattr(env.state, "current_player_id", 0))
            next_feat = encoder.features(env, player=next_player)
            next_actions = analyzer.analyze(env, top_k=encoder.max_actions)
            replay.append((feat, slot, r, next_feat, len(next_actions), done))
            if len(replay) >= 32:
                policy.update(random.sample(list(replay), 32))
        if ep % target_sync_every == 0:
            policy.sync_target()
    policy.save(out_dir / "dqn_policy.pt")
    result = evaluate_policy(game=game, policy=policy, encoder=encoder, analyzer=analyzer, episodes=eval_episodes, max_steps=max_steps, seed=seed + 10000, output_dir=out_dir)
    result.episodes = episodes
    (out_dir / "metric_contract.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _assign_terminal_rewards(transitions: list, env: Any, first_player: int) -> None:
    """Stamp the terminal transition with the mover's perspective-relative +1/-1/0."""
    if not transitions:
        return
    rewards = getattr(env.state, "rewards", None) or {}
    last_feat, last_slot, _, last_player, _ = transitions[-1]
    if isinstance(rewards, dict) and last_player in rewards:
        terminal_r = float(rewards[last_player])
    else:
        terminal_r = 0.0
    transitions[-1] = (last_feat, last_slot, terminal_r, last_player, True)


def _shaped_reward(pre: float | None, post: float | None, done: bool) -> float:
    if not done:
        return 0.0
    if post is None:
        return 0.0
    return float(post)


def evaluate_policy(*, game: str, policy: DQNPolicy, encoder: GameFeatureEncoder, analyzer: TextArenaActionAnalyzer, episodes: int, max_steps: int, seed: int, output_dir: Path, opponent: str = "random") -> RLBaselineResult:
    rewards = []
    outcomes = []
    turns_list = []
    invalid = 0
    for ep in range(episodes):
        env = build_env(game, seed=seed + ep)
        first_player = int(getattr(env.state, "current_player_id", 0))
        turns = 0
        # alternate which side the DQN plays against a random opponent
        dqn_side = first_player if opponent == "self" else ep % 2
        for _ in range(max_steps):
            if bool(getattr(env.state, "done", False)):
                break
            player = int(getattr(env.state, "current_player_id", 0))
            actions = analyzer.analyze(env, top_k=encoder.max_actions)
            if not actions:
                break
            if opponent == "random" and player != dqn_side:
                slot = random.randrange(len(actions))
            else:
                slot = policy.select(encoder.features(env, player=player), len(actions), epsilon=0.0)
            env.step(actions[slot].action_text)
            turns += 1
        r = _reward(env, dqn_side if opponent != "self" else first_player)
        if r is None:
            r = 0.0
        rewards.append(r)
        outcomes.append("win" if r > 0 else "loss" if r < 0 else "draw")
        turns_list.append(turns)
        if bool(getattr(env.state, "game_info", {}).get(dqn_side, {}).get("invalid_move", False)) if opponent != "self" else False:
            invalid += 1
    n = max(1, len(outcomes))
    return RLBaselineResult(
        game=game, episodes=0, eval_episodes=episodes,
        win_rate=sum(1 for x in outcomes if x == "win") / n,
        draw_rate=sum(1 for x in outcomes if x == "draw") / n,
        loss_rate=sum(1 for x in outcomes if x == "loss") / n,
        invalid_move_rate=invalid / n,
        avg_reward=float(np.mean(rewards)) if rewards else 0.0,
        avg_turns=float(np.mean(turns_list)) if turns_list else 0.0,
        output_dir=str(output_dir),
    )


def _reward(env: Any, player: int) -> float | None:
    rewards = getattr(env.state, "rewards", None)
    if isinstance(rewards, dict) and player in rewards:
        return float(rewards[player])
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a compact Double DQN baseline on TextArena legal-action slots.")
    parser.add_argument("--game", default="TicTacToe")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="baselines/local/textarena_dqn")
    args = parser.parse_args(argv)
    result = train_dqn_baseline(game=args.game, episodes=args.episodes, eval_episodes=args.eval_episodes, max_steps=args.steps, output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
