# References — PDF 7-Chapter Mapping

This index ties each upstream citation in `references/{reflexion,expel,voyager}/` to the
corresponding chapter of `改进需求.pdf` and the specific upgraded module in
`src/textarena_llm_agent/`.

| PDF Chapter | Topic | Reference(s) | Module(s) in this repo | Note |
|---|---|---|---|---|
| Ch. 1 | Decision frame logging | — (own design) | `trace_schema.py`, `tracing.py` | 30+ field DecisionFrame + EpisodeTrace; no external citation needed. |
| Ch. 2 | Evidence graph | — (own design, SQL-only) | `evidence_graph.py` | 14 node tables + 8 edge predicates. Reflexion/ExpeL keep traces as flat JSON; the graph is the differentiator. |
| Ch. 3 | Self-improving loop | **Reflexion** | `agent._on_terminal` → `critic_agent.EpisodeCriticAgent.run` → `skill_manager.run_evolution_sweep` | The Reflexion "trajectory → reflection → next-trial" loop is implemented as decide → episode trace → critic report → skill proposal. |
| Ch. 4 | Skill lifecycle | **ExpeL** | `skill_manager.SkillManager` + `skill_manager.SkillStatus` | Six-state lifecycle (`proposed → candidate → validated → active → deprecated / rejected`) is a strict superset of ExpeL's flat insight library. |
| Ch. 5 | Tool synthesis | **Voyager** | `tool_synthesis.ToolSynthesizer` + `tool_validator.ToolValidator` + `tool_library.ToolLibrary` | Five-stage pipeline (`synthesize_spec → compile_candidate → replay_eval → ab_test → activate`) directly mirrors Voyager's verify-before-publish discipline. |
| Ch. 6 | Critic / evaluator | **Reflexion + ExpeL contrast prompt** | `critic_agent.EpisodeCriticAgent` | Replaces the prior in-loop `evaluator.DecisionEvaluator` override (which was the user's #1 pain point). Critic is the *only* path to `propose → candidate → validated → active`. |
| Ch. 7 | Hypothesis-driven eval | — (own design, hypothesis judging + 9 baselines) | `hypothesis.py` (9 baselines, 4 hypotheses, 5 metric classes, `replay_eval`, `opponent_pool_snapshot`) + `cli.py` (`replay-eval / tournament / hypothesis-report`) + `visualization.py` (`/api/policy_versions`, `/api/cache`, `/api/tool_lifecycle`, `/api/skill_timeline`, `/api/nashconv`) | Goes beyond Reflexion/ExpeL/Voyager which report only aggregate win-rate. |

## Per-reference summary (≤500 words each)

### Reflexion → Ch. 3 (Self-improving loop)
**Citation used.** The "actor → evaluator → self-reflection" three-component design is the
backbone of our `agent._on_terminal` → `critic_agent` → `skill_manager` chain. We replace
Reflexion's binary success label with a structured `CriticReport` and store reflections as
graph nodes (instead of flat text) so the critic can cite the exact `decision_frame` ids
that produced each lesson.

**Not copied:** the HotPotQA / AlfWorld task harness, and the flat reflection buffer (we use
typed graph nodes instead).

### ExpeL → Ch. 4 (Skill lifecycle)
**Citation used.** The success/failure contrast prompt — the critic must enumerate both
`successful_patterns[]` AND `root_causes[]` before emitting a `propose_skill` tool call,
with the hard rule that the proposal carries ≥2 `evidence_ids`. The "insight library"
concept is promoted to first-class `Skill` nodes with a six-state lifecycle.

**Not copied:** the flat insight buffer and the actor's direct write access to insights.
In our design, the actor reads only `active` skills; only the critic can write.

### Voyager → Ch. 5 (Tool synthesis)
**Citation used.** The verify-before-publish discipline and the `tool_id`-stable
versioning (auto-demote old `active_tool` when a new version activates). AST-level safety
check (`tool_synthesis.SafeToolExecutor.validate`) directly mirrors Voyager's forbidden-
import rejection.

**Not copied:** Minecraft's automatic curriculum (no analog in fixed-rule TextArena);
JavaScript code generation (Python here so tools can call our state encoder); embedding
retrieval (BM25 is sufficient at our skill scale).

## How to clone upstream sources locally (optional)

```bash
cd C:/Users/I590008/Desktop/agent-learning/GameAgent
git clone --depth=1 https://github.com/noahshinn/reflexion        references/reflexion/_upstream
git clone --depth=1 https://github.com/LeapLabTHU/ExpeL           references/expel/_upstream
git clone --depth=1 https://github.com/MineDojo/Voyager           references/voyager/_upstream
```

These are NOT checked into the repo — the `_upstream/` subdirectories are gitignored.
The conceptual mapping in each `references/<name>/README.md` is the canonical reference.
