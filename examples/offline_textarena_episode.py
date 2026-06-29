from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textarena_llm_agent import TextArenaAgentConfig, TextArenaDecisionAgent  # noqa: E402
from textarena_llm_agent.cli import build_env  # noqa: E402


if __name__ == "__main__":
    env = build_env("TicTacToe", seed=7)
    agent = TextArenaDecisionAgent(TextArenaAgentConfig(memory_dir=str(ROOT / "workspace" / "textarena_mock_memory")))
    for _ in range(5):
        if env.state.done:
            break
        decision = agent.act(env)
        print(decision.to_json())
