# TextArena Evolvable Agent — Upgrade Checklist

- [x] game_specs: action_format + game_theoretic_principles + RANK maps.
- [x] prompt_builder: per-game switchable prompts with game-theoretic content.
- [x] retrieval: BM25 with perspective (game_id, player, phase) filtering.
- [x] memory: tiered memory + skill versioning/evolution + reflections + versioned prompt patches.
- [x] reflection + insight: Reflexion episode reflection + ExpeL batched insight consolidation.
- [x] tool_synthesis + tool_library: Voyager-style synthesis with safe in-process exec (AST gate + timeout).
- [x] llm: complete_with_tools extension (hybrid tool-calling).
- [x] context_packet + tool_loop: budgeted context + bounded tool-calling loop.
- [x] agent: loop rework integrating retrieval/reflection/tool-loop + terminal evolution.
- [x] tools: synthesized tool adapter into the registry.
- [x] rl_baseline: Double DQN (online-select/target-eval) + per-game structured features + self-play.
- [x] optimal_agents + evaluation: OptimalTTT/OptimalKuhnBR/Random + self-play trend + Elo + exploitability.
- [x] visualization: eval endpoints + Evolution & Evaluation panel (win-rate-by-phase, Elo, skill timeline).
- [x] docs + __init__ + tests + verification.

## Verification

- `PYTHONPATH=src pytest -q`: **24 passed** (includes candidate-context compaction and insight-marker regressions).
- Heuristic CLI smoke passes for TicTacToe, KuhnPoker, SimpleNegotiation, Stratego; SimpleNegotiation candidate parsing now reports the normal ranked-candidate rationale.
- `run_textarena_rl_baseline.py --game TicTacToe --episodes 3 --eval-episodes 2 --steps 20` produces `metric_contract.json` + `dqn_policy.pt` in smoke; DQN now stores terminal transitions in replay.
- `run_textarena_evaluate.py --games TicTacToe --episodes 3 --steps 20` produces structured `summary.json`, trend/elo/exploitability/skill_timeline artifacts.

## Notes

- Heuristic (no-key) mode intentionally produces no skills/timeline (it generates no lessons); skill evolution materializes in LLM mode. Trend win-rate and Elo still demonstrate the evaluation plumbing end-to-end.
- Env vars the code reads: `MCP_API_KEY` / `MCP_MODEL` / `MCP_API_BASE` (README corrected).
