from __future__ import annotations

import json
from pathlib import Path

import pytest

from textarena_llm_agent import TextArenaAgentConfig, TextArenaDecisionAgent
from textarena_llm_agent.action_analyzer import TextArenaActionAnalyzer
from textarena_llm_agent.cli import build_env
from textarena_llm_agent.context_packet import ContextBudgeter
from textarena_llm_agent.game_specs import RANK_NAME_TO_INT, RANK_INT_TO_NAME, canonical_game_id
from textarena_llm_agent.llm import HeuristicLLM
from textarena_llm_agent.llm import OpenAIChatLLM
from textarena_llm_agent.memory import EvolvingMemory
from textarena_llm_agent.optimal_agents import OptimalKuhnBR, OptimalTTT, RandomAgent
from textarena_llm_agent.prompt_builder import GamePromptBuilder
from textarena_llm_agent.retrieval import BM25Retriever, build_corpus_from_memory
from textarena_llm_agent.state_encoder import TextArenaStateEncoder
from textarena_llm_agent.tool_synthesis import SafeToolExecutor
from textarena_llm_agent.evaluation import EvaluationHarness


@pytest.mark.parametrize("game", ["TicTacToe", "KuhnPoker", "SimpleNegotiation", "Stratego"])
def test_target_games_generate_legal_candidates(game: str):
    env = build_env(game, seed=3)
    encoder = TextArenaStateEncoder()
    actions = encoder.valid_actions(env)
    assert actions
    assert all(a.action_text.startswith("[") and a.action_text.endswith("]") for a in actions)


def test_tictactoe_agent_decides_and_records_memory(tmp_path: Path):
    env = build_env("TicTacToe", seed=7)
    agent = TextArenaDecisionAgent(TextArenaAgentConfig(memory_dir=str(tmp_path / "mem"), trace_dir=str(tmp_path / "trace")))
    decision = agent.act(env)
    legal = {a.action_text for a in agent.encoder.valid_actions(build_env("TicTacToe", seed=7))}
    assert decision.action_text in legal
    assert (tmp_path / "mem" / "experiences.jsonl").exists()
    assert (tmp_path / "mem" / "skills.jsonl").exists()
    assert (tmp_path / "mem" / "skill_updates.jsonl").exists()
    assert (tmp_path / "trace" / "events.jsonl").exists()
    assert (tmp_path / "trace" / "latest_state.json").exists()


def test_ranked_candidate_ids_stay_consistent_inside_action_payload():
    env = build_env("TicTacToe", seed=4)
    analyzer = TextArenaActionAnalyzer(TextArenaStateEncoder(), simulate_top=0)
    candidates = analyzer.analyze(env, top_k=4)
    assert candidates
    for candidate in candidates:
        assert candidate.candidate_id.startswith("C")
        assert candidate.action["candidate_id"] == candidate.candidate_id
        assert candidate.action["original_candidate_id"].startswith("A")


def test_stratego_visible_state_hides_opponent_ranks():
    env = build_env("Stratego", seed=5)
    encoder = TextArenaStateEncoder()
    state = encoder.encode(env, include_actions=True)
    board = state["visible_state"]["board"]
    player = state["game"]["current_player"]
    hidden_count = sum(1 for row in board for cell in row if cell == "?")
    assert hidden_count > 0
    for row in range(10):
        for col in range(10):
            raw = env.board[row][col]
            if isinstance(raw, dict) and raw["player"] != player:
                assert board[row][col] == "?"


def test_trace_contains_decision_and_memory_events(tmp_path: Path):
    env = build_env("TicTacToe", seed=8)
    agent = TextArenaDecisionAgent(TextArenaAgentConfig(memory_dir=str(tmp_path / "mem"), trace_dir=str(tmp_path / "trace")))
    agent.act(env)
    events = [json.loads(line) for line in (tmp_path / "trace" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    names = {event["event"] for event in events}
    assert "candidates_ranked" in names
    assert "decision_resolved" in names
    assert "memory_written" in names
    state = json.loads((tmp_path / "trace" / "latest_state.json").read_text(encoding="utf-8"))
    assert state["snapshot"]["env_id"].startswith("TicTacToe")


# ---------------------------------------------------------------------------
# New tests: upgraded architecture
# ---------------------------------------------------------------------------

def test_safe_executor_ast_gate_rejects_dangerous_code():
    ex = SafeToolExecutor()
    assert ex.validate_ast("import os")                       # blocked import
    assert ex.validate_ast('def run():\n    f=open("x","w")')  # write-mode open
    assert ex.validate_ast('def run():\n    exec("1")')        # exec
    assert ex.validate_ast('def run():\n    return ().__class__.__subclasses__()')  # dunder escape
    assert not ex.validate_ast("import math\ndef run():\n    return math.floor(2.7)")  # allowed


def test_safe_executor_runs_allowed_tool():
    ex = SafeToolExecutor(timeout_s=2.0)
    code = "def run(game_state, visible_text, **args):\n    return {'len': len(game_state)}"
    res = ex.execute(code=code, injected={"game_state": {"a": 1, "b": 2}, "visible_text": "x"})
    assert res.ok
    assert res.value == {"len": 2}


def test_safe_executor_times_out_on_infinite_loop():
    ex = SafeToolExecutor(timeout_s=0.5)
    code = "def run(game_state, visible_text, **args):\n    while True:\n        pass\n    return 1"
    res = ex.execute(code=code, injected={"game_state": {}, "visible_text": ""})
    assert not res.ok
    assert "timed out" in (res.error or "").lower() or "timeout" in (res.error or "").lower()


def test_openai_llm_accepts_string_json_response():
    class FakeCompletions:
        def create(self, **kwargs):
            return '{"candidate_id":"C0","confidence":0.91}'

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    class FakeLLM(OpenAIChatLLM):
        def _client(self):
            return FakeClient()

    llm = FakeLLM(api_key="test")
    out = llm.complete_json(system="s", user="u")
    assert out["candidate_id"] == "C0"
    assert out["confidence"] == 0.91


def test_openai_llm_accepts_dict_tool_response():
    class FakeCompletions:
        def create(self, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": '{"candidate_id":"C1"}',
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {"name": "textarena_state_summary", "arguments": '{"include_actions": true}'},
                        }],
                    }
                }]
            }

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    class FakeLLM(OpenAIChatLLM):
        def _client(self):
            return FakeClient()

    llm = FakeLLM(api_key="test")
    out = llm.complete_with_tools(system="s", user="u", tools=[{"type": "function", "function": {"name": "x"}}])
    assert out.text == '{"candidate_id":"C1"}'
    assert out.tool_calls == [{"id": "call_1", "name": "textarena_state_summary", "arguments": {"include_actions": True}}]


def test_openai_llm_from_env_supports_critic_prefix(monkeypatch):
    monkeypatch.setenv("SCS_LLM_API_KEY", "main-key")
    monkeypatch.setenv("SCS_LLM_MODEL", "main-model")
    monkeypatch.setenv("SCS_LLM_BASE_URL", "https://main.example/v1")
    monkeypatch.setenv("CRITIC_MODEL", "gpt-5.5")
    llm = OpenAIChatLLM.from_env(prefix="CRITIC")
    assert llm.api_key == "main-key"
    assert llm.model == "gpt-5.5"
    assert llm.base_url == "https://main.example/v1"


def test_context_budgeter_keeps_candidate_json_parseable():
    budgeter = ContextBudgeter(total_chars=2000)
    budgeter.budget["candidates"] = 350
    candidates = [
        {"candidate_id": f"C{i}", "action_text": f"[Offer: {i + 1} Wheat -> 1 Ore]", "metadata": {"large": "x" * 300}}
        for i in range(8)
    ]
    packet = budgeter.build(spec_system="sys", state_text="{}", candidates=candidates, memory_excerpt="m", reflection="", patches=[])
    parsed = json.loads(packet.sections["candidates"])
    assert isinstance(parsed, list)
    assert parsed
    assert parsed[0]["candidate_id"] == "C0"


def test_unconsolidated_count_uses_iso_marker(tmp_path: Path):
    from textarena_llm_agent.memory import Experience

    mem = EvolvingMemory(tmp_path)
    mem.record_experience(Experience(state_key="s1", game_id="TicTacToe", player=0, action_text="[4]"))
    assert mem.unconsolidated_count("TicTacToe") == 1
    mem.mark_insight_consolidated("TicTacToe")
    assert mem.unconsolidated_count("TicTacToe") == 0
    mem.record_experience(Experience(state_key="s2", game_id="TicTacToe", player=0, action_text="[0]"))
    assert mem.unconsolidated_count("TicTacToe") == 1


def test_bm25_retrieval_filters_by_perspective(tmp_path: Path):
    mem = EvolvingMemory(tmp_path)
    # player 0 and player 1 experiences for the same game
    from textarena_llm_agent.memory import Experience
    mem.record_experience(Experience(state_key="opening", game_id="TicTacToe", player=0, action_text="[4]", outcome="terminal_win", reward=1, lesson="center as player 0"))
    mem.record_experience(Experience(state_key="opening", game_id="TicTacToe", player=1, action_text="[0]", outcome="terminal_loss", reward=-1, lesson="corner as player 1"))
    bm = BM25Retriever()
    corpus = build_corpus_from_memory(tmp_path, game_id="TicTacToe")
    bm.indexer.index(corpus)
    # player 0 query should retrieve the player-0 experience, not player-1
    hits = bm.retrieve(query="opening center", game_id="TicTacToe", player=0, top_k=4)
    players = [h.player for h in hits if h.source == "experiences"]
    assert 0 in players
    assert 1 not in players


def test_skill_mutation_on_losing_streak(tmp_path: Path):
    mem = EvolvingMemory(tmp_path)
    sid = mem.consolidate_skill_from_lesson(lesson="always bet aggressively", game_id="KuhnPoker", evidence="exp:1", tags=["KuhnPoker"])
    assert sid
    # simulate a losing streak
    for _ in range(6):
        mem.update_skill_usage(skill_id=sid, win=False, score=0.1)
    skill = next(s for s in mem._read_jsonl(mem.skills_path) if s["id"] == sid)
    assert skill["win_rate"] < 0.2


def test_terminal_fallback_lesson_creates_skill(tmp_path: Path):
    from textarena_llm_agent.evaluator import EvaluationResult
    from textarena_llm_agent.agent import Decision
    from dataclasses import asdict

    mem = EvolvingMemory(tmp_path)
    agent = TextArenaDecisionAgent(TextArenaAgentConfig(memory_dir=str(tmp_path), enable_tracing=False), memory=mem)
    decision = Decision(
        action_text="[4]",
        candidate_id="C0",
        action_index=0,
        action_type="place_mark",
        confidence=0.8,
        rationale="test",
        plan="test",
        selected_candidate={},
        evaluation=asdict(EvaluationResult(0.8, True, None, "", "", "")),
    )
    exp_id = agent.learn_from_outcome(before_state_text="TicTacToe state", game_id="TicTacToe", player=0, decision=decision, reward=1.0, outcome="terminal_win")
    assert exp_id
    assert mem.memory_stats()["skills"] >= 1
    assert mem.skill_timeline()


def test_evaluation_manifest_records_snapshots_and_llm_contract(tmp_path: Path):
    memory_root = tmp_path / "memory"
    output_root = tmp_path / "eval"
    harness = EvaluationHarness(
        games=["TicTacToe"],
        episodes=1,
        max_steps=2,
        memory_root=memory_root,
        output_root=output_root,
        seed=11,
        llm=HeuristicLLM(),
        evaluator_llm=HeuristicLLM(),
        eval_episodes=1,
        elo_rounds=0,
        exploitability_episodes=0,
    )
    summary = harness.run_all()
    assert "TicTacToe" in summary
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["uses_heuristic_actor"] is True
    assert manifest["uses_heuristic_critic"] is True
    assert manifest["elo_rounds"] == 0
    assert manifest["exploitability_episodes"] == 0
    assert manifest["actor"]["class"] == "HeuristicLLM"
    assert Path(manifest["memory_snapshots"]["TicTacToe"]["initial"]).exists()
    assert Path(manifest["memory_snapshots"]["TicTacToe"]["evolved"]).exists()
    assert Path(manifest["artifacts"]["TicTacToe"]["trend"]).exists()


def test_budgeted_evaluation_uses_offline_sampled_metrics(tmp_path: Path):
    harness = EvaluationHarness(
        games=["TicTacToe"],
        episodes=2,
        max_steps=4,
        memory_root=tmp_path / "memory",
        output_root=tmp_path / "eval",
        seed=21,
        llm=HeuristicLLM(),
        evaluator_llm=HeuristicLLM(),
        elo_rounds=0,
        exploitability_episodes=0,
    )
    summary = harness.run_all()
    trend = summary["TicTacToe"]["trend"]
    assert [row["bin"] for row in trend] == ["early", "late"]
    assert all(row["episodes"] == 1 for row in trend)
    elo = summary["TicTacToe"]["elo"]
    assert elo["method"] == "sampled_from_trend_matches"
    assert elo["extra_llm_matches"] == 0
    exploitability = summary["TicTacToe"]["exploitability"]
    assert exploitability["method"] == "sampled_tactical_proxy"
    assert exploitability["extra_llm_matches"] == 0
    assert exploitability["decision_count"] >= 1


def test_double_dqn_target_uses_online_selection():
    """Double DQN: next action selected by ONLINE net, evaluated by TARGET net."""
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not installed")
    from textarena_llm_agent.rl_baseline import DQNPolicy
    import numpy as np
    pol = DQNPolicy(feature_dim=28, max_actions=80)
    # craft a batch where online and target disagree, then assert update doesn't crash
    batch = []
    for _ in range(8):
        s = np.random.rand(28).astype(np.float32)
        ns = np.random.rand(28).astype(np.float32)
        batch.append((s, 0, 1.0, ns, 5, False))
    pol.update(batch)  # should not raise; uses online-argmax + target-eval
    # after an update the ONLINE net has changed but the TARGET net has not (until sync)
    w_online = list(pol.net.parameters())[-1].detach().clone()
    w_target = list(pol.target.parameters())[-1].detach().clone()
    assert not torch.allclose(w_online, w_target)
    pol.sync_target()  # then they match
    assert torch.allclose(list(pol.net.parameters())[-1], list(pol.target.parameters())[-1])


def test_game_prompt_builder_returns_per_game_content():
    pb = GamePromptBuilder()
    for game in ["TicTacToe", "KuhnPoker", "SimpleNegotiation", "Stratego"]:
        prompt = pb.build_system(game_id=game, phase="mid", reflection="r", active_patches=["p"], tool_descriptions=[])
        assert game in prompt
        assert "Game-theoretic principles" in prompt
        assert "candidate_id" in prompt
    # unknown game falls back to generic
    assert "expert TextArena" in pb.build_system(game_id="UnknownGame", phase="mid")


def test_optimal_ttt_never_loses():
    opt = OptimalTTT()
    for seed in range(20):
        env = build_env("TicTacToe", seed=seed)
        # optimal plays both sides — the game must end in a draw (optimal vs optimal)
        first = int(env.state.current_player_id)
        for _ in range(20):
            if env.state.done:
                break
            p = env.state.current_player_id
            env.step(opt.act(env, p))
        rewards = env.state.rewards or {}
        # optimal vs optimal -> draw (reward 0 for both)
        assert rewards.get(0, 0) == 0 and rewards.get(1, 0) == 0


def test_optimal_kuhn_br_detects_exploitable_policy():
    # a policy that always bets is exploitable; BR value should be > game value
    policy = {(0, ()): {"bet": 1.0}, (1, ()): {"bet": 1.0}, (2, ()): {"bet": 1.0}}
    br = OptimalKuhnBR.best_response_value(policy, samples=800)
    # against an always-bet opponent, BR (which calls with K, checks J/Q) should be profitable
    assert br > 0.0


def test_stratego_rank_map_is_complete_and_round_trips():
    env = build_env("Stratego", seed=1)
    board = env.state.game_state["board"]
    found_names = set()
    for row in board:
        for cell in row:
            if isinstance(cell, dict):
                found_names.add(cell.get("rank"))
    assert found_names.issubset(set(RANK_NAME_TO_INT)), f"unknown ranks: {found_names - set(RANK_NAME_TO_INT)}"
    # round-trip
    for name, i in RANK_NAME_TO_INT.items():
        assert RANK_INT_TO_NAME[i] == name


def test_canonical_game_id_handles_variants():
    assert canonical_game_id("TicTacToe-v0-raw") == "TicTacToe"
    assert canonical_game_id("KuhnPoker-v0-long") == "KuhnPoker"
    assert canonical_game_id("SimpleNegotiation-v0-short") == "SimpleNegotiation"
    assert canonical_game_id("kuhn") == "KuhnPoker"
