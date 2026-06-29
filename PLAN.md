# TextArena Evolvable Agent — Plan

## Goal

Build a professional, evolvable LLM agent whose central purpose is to
**progressively improve its play over continuous TextArena games** by evolving its
memory and skills, with a traditional deep-RL baseline as a comparison bar, and a
credible evaluation proving the improvement.

## Games (raw, 2-player)

TicTacToe, KuhnPoker, SimpleNegotiation, Stratego.

## Verified TextArena 0.7.4 API invariants (load-bearing)

- `ta.make("<Game>-v0-raw")`, `env.reset(num_players=2, seed=...)`.
- `done, info = env.step(action_str)` — action is an exact bracketed string (`-raw` does NOT auto-wrap).
- `env.get_observation()` clears after one read → the loop reads `env.state.observations` / `env.state.logs` directly.
- `env.state.rewards` is `None` until terminal → `{pid:int}`. `env.close() → (rewards, game_info)`.
- No `get_valid_actions()` → derive from `env.state.game_state`.
- Stratego board cells are dicts `{'rank': <str name>, 'player': <int>}`; `player_pieces[p]` uses int rank tuples; `env.lakes` exists; `env.board is gs["board"]` → a single `RANK_NAME_TO_INT` map is the source of truth.

## Architecture modules

- `game_specs.py` — `GameSpec{rules, action_format, game_theoretic_principles}` + `RANK_NAME_TO_INT`/`RANK_INT_TO_NAME`.
- `prompt_builder.py` — `GamePromptBuilder.build_system(...)` (per-game + Reflexion injection).
- `retrieval.py` — BM25 with perspective filtering (Generative-Agents score blend).
- `memory.py` — tiered memory, skill evolution (version/status/promote/demote/mutate), reflections, versioned prompt patches.
- `reflection.py`, `insight.py` — Reflexion reflection + ExpeL insight consolidation.
- `tool_synthesis.py`, `tool_library.py` — Voyager-style synthesis + safe in-process exec (AST gate + timeout).
- `context_packet.py` — budgeted context packet.
- `tool_loop.py`, `llm.py` — bounded tool-calling loop + `complete_with_tools`.
- `agent.py` — reworked loop integrating all of the above + terminal evolution.
- `tools.py` — `SynthesizedToolAdapter` + extended registry.
- `rl_baseline.py` — Double DQN + per-game structured features + self-play.
- `optimal_agents.py` — OptimalTTT (minimax), OptimalKuhnBR (best response), RandomAgent.
- `evaluation.py` — self-play trend, Elo, exploitability, skill timeline.
- `visualization.py` — dashboard + eval endpoints + 3 charts.

## Evaluation (credible improvement evidence)

- Trend bins (early/mid/late): win-rate, avg-reward, invalid-rate, avg-turns, skill-count.
- Elo tournament across `{llm_evolved, llm_no_memory, dqn, random, optimal}`.
- Exploitability: Kuhn best-response value − game value (1/18); TicTacToe loss-rate vs optimal.
- Persisted: `eval_runs/<game>/{trend,elo,exploitability}.json`, `match_results.jsonl`, `skill_timeline.jsonl`.

## Risks & mitigations

- Safe exec isn't a true sandbox → AST gate + restricted builtins + recursion limit + SIGALRM timeout; `enable_tool_synthesis` toggle.
- Tool-calling gateway inconsistency → `ToolLoop` falls back to `complete_json`; `HeuristicLLM` skips the loop.
- Self-play perspective contamination → every record carries `player`; BM25 filters on `(game_id, player, phase)`.
- Stratego feature/state-key mismatch → single `RANK_NAME_TO_INT` map shared by encoder + normalizer.
- LLM cost blow-up → reflection once/terminal-episode; insight batched; evaluator skippable; synthesis capped 1/episode + recurrence-only.
