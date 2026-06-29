from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from textarena_llm_agent import TextArenaAgentConfig, TextArenaDecisionAgent
from textarena_llm_agent.cli import build_env
from textarena_llm_agent.visualization import TextArenaVisualizationServer


def test_visualization_server_serves_trace_state(tmp_path: Path):
    env = build_env("TicTacToe", seed=9)
    agent = TextArenaDecisionAgent(TextArenaAgentConfig(memory_dir=str(tmp_path / "mem"), trace_dir=str(tmp_path / "trace")))
    agent.act(env)
    server = TextArenaVisualizationServer(agent.tracer, port=0)
    url = server.start(open_browser=False)
    try:
        state = json.loads(urllib.request.urlopen(url + "api/state").read().decode("utf-8"))
        events = json.loads(urllib.request.urlopen(url + "api/events?limit=20").read().decode("utf-8"))
        assert state["snapshot"]["env_id"].startswith("TicTacToe")
        assert any(event["event"] == "decision_resolved" for event in events["events"])
    finally:
        server.stop()
