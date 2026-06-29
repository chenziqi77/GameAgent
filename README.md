# TextArena LLM Agent — Evolvable Game-Playing Agent Research Library

This library studies whether an LLM agent **progressively improves its play over
continuous games** by evolving its memory and skills. It targets TextArena
two-player games and supplies a traditional deep-RL baseline to compare against.

Supported first-class games (raw variant, 2-player):

- `TicTacToe-v0-raw`
- `KuhnPoker-v0-raw`
- `SimpleNegotiation-v0-raw`
- `Stratego-v0-raw`

## Architecture (upgraded)

The agent implements patterns from professional agent research:

- **Voyager-style tool synthesis** (`tool_synthesis.py`, `tool_library.py`): the
  agent detects recurring sub-problems and asks the LLM to design a reusable
  Python helper. The proposed code is AST-validated, executed against a
  **deep-copied** game-state snapshot, and — if it passes verification — cached in
  a tool library and registered for future turns. Execution is in-process with an
  AST import whitelist, a blocked-builtin/dunder gate, a restricted `__builtins__`,
  a recursion limit, and a hard timeout (SIGALRM on the main thread).
- **Per-game prompts** (`prompt_builder.py`, `game_specs.py`): each game gets a
  switchable system prompt encoding rules, the exact `-raw` action format, and
  game-theoretic principles (Kuhn Nash mixing, TicTacToe minimax priority,
  Stratego information asymmetry, Negotiation valuation). Reflexion-style
  reflections and versioned prompt patches are injected dynamically.
- **Tiered evolving memory** (`memory.py`, `retrieval.py`): episodic experiences,
  evolving skills (versioned, with promotion/demotion/mutation), and
  reflections. Retrieval is BM25 (numpy-only) with **perspective filtering** for
  self-play correctness; final score blends relevance + recency + importance +
  phase match (Generative-Agents synthesis).
- **Reflection & insight** (`reflection.py`, `insight.py`): per-terminal-episode
  self-reflection (Reflexion); batched ExpeL insight consolidation across a
  batch of experiences. Skills mutate when their win-rate drops below threshold
  (verbal reinforcement); `skill_updates.jsonl` is the full audit trail.
- **Context management** (`context_packet.py`): a single budgeted context packet
  per decision (rules/state/candidates/memory/reflection/patches under a char
  budget with principled compaction).
- **Bounded tool-calling loop** (`tool_loop.py`, `llm.py`): tools *inform*
  reasoning; the final emission is a candidate-pick JSON. Falls back to plain
  JSON for `HeuristicLLM` (no-key mode) or gateways without tool-calling — the
  action contract (an exact bracketed string for `-raw`) is always honored.
- **Double DQN baseline** (`rl_baseline.py`): Double DQN (online-select /
  target-evaluate) with per-game structured features and self-play training.
- **Evaluation harness** (`evaluation.py`, `optimal_agents.py`): self-play win-rate
  trend (early/mid/late bins), Elo tournament, Kuhn/TicTacToe exploitability
  (best-response value; loss-rate vs optimal), and a dashboard.

## Run directly

No editable install is required. The root scripts add `src/` to `PYTHONPATH`
automatically. Runtime deps: `textarena`, `numpy`, `openai` (+ `torch` for the RL
baseline).

## Run an agent

Offline heuristic mode (no API key):

```bash
python run_textarena_agent.py --game TicTacToe --llm heuristic --steps 20
python run_textarena_agent.py --game KuhnPoker --llm heuristic --steps 20
python run_textarena_agent.py --game SimpleNegotiation --llm heuristic --steps 20
python run_textarena_agent.py --game Stratego --llm heuristic --steps 20
```

LLM mode (OpenAI-compatible; the code reads `MCP_*`, `OPENAI_*`, or `SCS_LLM_*`):

```bash
export MCP_API_KEY="..."        # or SCS_LLM_API_KEY
export MCP_MODEL="gpt-4o-mini"      # optional; also OPENAI_MODEL or SCS_LLM_MODEL
export MCP_API_BASE="..."          # optional; also OPENAI_BASE_URL or SCS_LLM_BASE_URL
python run_textarena_agent.py --game TicTacToe --llm openai --steps 20

# Optional: use a separate OpenAI-compatible critic/evaluator model.
# Defaults to gpt-5.5 when --llm openai is used; override if your gateway uses a different name.
python run_textarena_agent.py --game TicTacToe --llm openai --model "$SCS_LLM_MODEL" --critic-model gpt-5.5 --steps 20
```

The LLM receives the per-game system prompt (rules + game-theoretic principles),
visible state, ranked legal action candidates, recalled memory + reflections, and
active prompt patches. It may call tools to inform its decision, then returns JSON
selecting one `candidate_id`; the agent executes the candidate's exact bracketed
action string on the `-raw` environment.

## Visualization

For Docker/headless runs, prefer static local artifacts that can be copied to any
machine and opened without a live server:

```bash
python run_textarena_agent.py --game TicTacToe --llm heuristic --steps 20 \
  --trace-dir workspace/textarena_runs/latest \
  --jsonl workspace/textarena_runs/latest/decisions.jsonl

python render_textarena_trace.py \
  --trace-dir workspace/textarena_runs/latest \
  --eval-root workspace/eval_runs \
  --output-dir workspace/textarena_reports/latest
```

This writes one timestamped HTML report, one timestamped Markdown trajectory,
and per-step SVG board frames under `workspace/textarena_reports/latest/`, for
example `report_YYYYMMDDTHHMMSSZ.html` and
`trajectory_YYYYMMDDTHHMMSSZ.md`. The report includes a browser replay control
that animates the SVG frames. Use `--no-timestamp` only when you explicitly want
legacy filenames such as `report.html`.

A live dashboard still exists for environments where localhost is reachable:

```bash
python run_textarena_agent.py \
  --game Stratego --llm heuristic --steps 50 \
  --visualize --trace-dir workspace/textarena_runs/latest
```

## Memory & skill evolution

Default memory root:

```
workspace/textarena_memory/
  rules.md
  prompt_overrides.md          # legacy mirror
  prompt_patches.jsonl         # versioned per-game patches (active/reverted)
  experiences.jsonl            # episodic
  skills.jsonl                 # evolving skills (versioned, status, win_rate)
  skill_updates.jsonl          # full audit trail of every skill event
  reflections.jsonl            # per-episode self-reflections
  retrieval_hits.jsonl
  tool_library/                # synthesized, verified tools
    tools.jsonl
    tool_src/<name>_v<n>.py
```

Per-decision loop: encode state → BM25 memory + reflection recall → (rare, throttled)
tool synthesis → budgeted context packet → bounded tool-calling loop → evaluator/critic
→ execute step. Per terminal episode: self-reflection → record → confidence decay →
batched insight consolidation → skill promote/demote/mutate sweep.

## Experiments (the central improvement evidence)

```bash
python run_textarena_evaluate.py --games TicTacToe,KuhnPoker --episodes 30 --steps 60 \
  --output-dir workspace/eval_runs --llm heuristic

# OpenAI-compatible actor with a separate critic model:
python run_textarena_evaluate.py --games TicTacToe,KuhnPoker --episodes 30 --steps 60 \
  --output-dir workspace/eval_runs --llm openai --model "$SCS_LLM_MODEL" --critic-model gpt-5.5
```

Outputs under `workspace/eval_runs/<game>/`:

- `trend.json` — early/mid/late bin win-rate, avg-reward, invalid-rate, avg-turns, skill-count
- `elo.json` — Elo ratings + history across `{llm_evolved, llm_no_memory, dqn, random, optimal}`. With `--elo-rounds 0`, this is computed offline from already sampled trend matches, so it does not trigger extra LLM games.
- `exploitability.json` — Kuhn best-response value vs game value (1/18); TicTacToe loss-rate vs optimal. With `--exploitability-episodes 0`, this becomes an offline sampled trajectory proxy instead of launching extra LLM-vs-oracle games.
- `match_results.jsonl` — per-match outcomes
- `skill_timeline.jsonl` — derived from `skill_updates.jsonl`
- `../run_manifest_YYYYMMDDTHHMMSSZ.json` — actor/critic model, API base, seed,
  output paths, and initial/evolved memory snapshot paths
- `../memory_snapshots/<game>/{initial,evolved}_YYYYMMDDTHHMMSSZ/` — restorable
  memory/skill state for comparing fresh, evolved, and retrained variants

For aggregate analysis and timestamped HTML/JSON/Markdown reports:

```bash
python analyze_textarena_metrics.py \
  --eval-root workspace/eval_runs \
  --memory-root workspace/textarena_memory \
  --output-dir workspace/analysis_reports/textarena_metrics \
  --critic-llm --critic-model gpt-5.5
```

The analysis report includes Elo plus non-Elo diagnostics: win/draw/loss rates,
average reward and standard error, Wilson win-rate intervals, first/second player
side bias, invalid action rate, turn efficiency, exploitability where available,
and skill/reflection/retrieval growth.

An LLM-mode run shows the improvement curve: win-rate rising across early→mid→late
bins, Elo trending up, and Kuhn exploitability decreasing as skills consolidate.

## RL baseline (Double DQN)

```bash
python run_textarena_rl_baseline.py --game KuhnPoker --episodes 50 --eval-episodes 10 \
  --steps 60 --output-dir baselines/local/textarena_dqn_kuhnpoker
```

Standard Double DQN (online-select / target-evaluate) with per-game structured
features and self-play training. The TextArena adapter still owns legality and
candidate generation; the neural policy chooses among current candidates. TextArena
and the LLM agent consume different signals (raw state vs text + memory), but both
are kept efficient.

## Python API

```python
from textarena_llm_agent import TextArenaAgentConfig, TextArenaDecisionAgent
from textarena_llm_agent.cli import build_env

env = build_env("TicTacToe", seed=7)
agent = TextArenaDecisionAgent(TextArenaAgentConfig(use_llm=False))
decision = agent.act(env)
print(decision.to_json())
```

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Efficiency

Per turn ≈ 1 decision call (+ 0–2 cheap local-tool rounds; synthesized tools run
locally, not via LLM) + 1 evaluator call (skippable on high-confidence decisions).
Per terminal episode: +1 reflection call. Every N episodes: +1 batched
insight-consolidation call. Tool synthesis: ≤1/episode, only on recurrence.
