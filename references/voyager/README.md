# Voyager (citation reference)

**Paper:** Wang et al., 2023, "Voyager: An Open-Ended Embodied Agent with Large Language Models"
**Upstream:** https://github.com/MineDojo/Voyager
**arXiv:** https://arxiv.org/abs/2305.16291

## Core idea

Voyager builds a *skill library of executable code* (JavaScript functions that drive a
Minecraft bot). New skills are proposed by the LLM, validated by execution against the
environment, and added to the library only if they pass a self-verification check. The LLM
later retrieves and *composes* learned skills to solve harder tasks.

Three Voyager components:
1. **Automatic curriculum** — LLM proposes the next task given current inventory/state
2. **Iterative prompting with environmental feedback + self-verification** — synthesize code,
   run, observe, repair, validate
3. **Skill library** (the executable kind) — verified skills get persisted by name and
   embedding-retrieved later

## Mapping into this codebase

| Voyager concept | GameAgent module |
|---|---|
| Skill code library | `tool_library.ToolLibrary` (synthesized Python tools) — distinct from
                       the textual *Skill* library described in ExpeL section |
| Iterative synthesize/run/repair | `tool_synthesis.ToolSynthesizer` five-stage pipeline:
                                    `synthesize_spec → compile_candidate → replay_eval →
                                    ab_test → activate` |
| Self-verification | `tool_validator.ToolValidator` (orchestrates the pipeline) + the
                      `unit_tests_passed / replay_score / ab_score` fields on
                      `SynthesizedToolRecord` |
| Skill composition | The actor's `tool_loop.ToolLoop` can chain multiple active tools per
                      decision; each `active_tool` is exposed as an OpenAI tool schema in
                      the L3 (policy_static) prompt layer |
| Curriculum | (deliberately not copied — see below) |

## What we **don't** copy from Voyager

- **Automatic curriculum.** TextArena environments are fixed-rule board/poker games; there is
  no open-ended next-task selection problem. Curriculum is replaced by *cross-game transfer*:
  the same `Skill`/`Tool` is evaluated against all games in `EvaluationHarness.games`.
- **JavaScript code generation.** Tools are Python (so they can call into our state encoder).
- **Embedding retrieval over skills.** We use BM25 (`retrieval.BM25Retriever`) — cheaper, no
  embedding API dependency, fine for the < 1k skill scale of this project.

## What we **do** copy

- The strict gate that "a tool is only `active_tool` after passing replay + A/B" — see
  `tool_library.ToolStatus._FORWARD_TRANSITIONS`. This is the single most important Voyager
  contribution.
- The `tool_id` stability across versions — old `active_tool` is auto-`demoted` when a new
  version activates, avoiding the "two competing implementations of the same primitive"
  failure mode Voyager called out.
- AST-level safety check — Voyager rejects code that imports forbidden modules; we do the
  same in `tool_synthesis.SafeToolExecutor.validate`.

## File-level pointers (PDF 7-chapter map)

- **PDF Chapter 5 (Tool synthesis):** Voyager's pipeline is the direct ancestor of the
  five-stage flow in `tool_synthesis.ToolSynthesizer` + `tool_validator.ToolValidator`.
- **PDF Chapter 7 (Evaluation):** Voyager's "verify before publish" gate is the reason every
  tool carries `replay_score` and `ab_score` columns even before A/B is wired to the live
  tournament.

## Local extract

```bash
git clone --depth=1 https://github.com/MineDojo/Voyager references/voyager/_upstream
```

Not checked in. The Voyager repo's MineDojo dependency is heavy and not needed for citation.
