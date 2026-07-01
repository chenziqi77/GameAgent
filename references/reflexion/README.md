# Reflexion (citation reference)

**Paper:** Shinn et al., 2023, "Reflexion: Language Agents with Verbal Reinforcement Learning"
**Upstream:** https://github.com/noahshinn/reflexion
**arXiv:** https://arxiv.org/abs/2303.11366

## Core idea

After each trajectory the agent writes a *self-reflection* in natural language conditioned on
(trajectory, scalar/binary reward, prior reflections). The reflection is appended to a
"reflection buffer" that is replayed into the next attempt's context. No gradient updates —
all learning is in the language buffer.

Three components:
1. **Actor** — policy LLM that emits trajectories
2. **Evaluator** — scores the trajectory (binary success / heuristic / LM)
3. **Self-Reflection** — distils trajectory + score into a verbal lesson stored in episodic memory

## Mapping into this codebase

| Reflexion concept | GameAgent module |
|---|---|
| Actor LLM | `agent.TextArenaDecisionAgent.decide()` (policy LLM via `OpenAIChatLLM`) |
| Evaluator | `critic_agent.EpisodeCriticAgent` (replaces the prior `evaluator.DecisionEvaluator` override path) |
| Self-Reflection | `reflection.Reflector.reflect_episode` invoked from inside the critic's tool-use loop |
| Reflection buffer | `memory.EvolvingMemory` (`reflections.jsonl` + Phase 2 `EvidenceGraph.reflection` table) |
| Replay into next context | `prompt_compiler.PromptCompiler.user_dynamic(..., reflection=...)` (L4 layer) |

## What we **don't** copy from Reflexion

- Reflexion's binary success label is too coarse for self-play board games. We replace it with
  the critic's structured `CriticReport.outcome ∈ {win, loss, draw, invalid}` plus per-frame
  `evaluation_score` and the SkillManager's `replay_score` / `ab_score`.
- Reflexion stores reflections as flat text. We store them as graph nodes with `SUPPORTS` edges
  to specific `decision_frame` ids → enables the critic to cite *which* turn produced the lesson.

## File-level pointers (PDF 7-chapter map)

- **PDF Chapter 3 (Self-improving loop):** Reflexion's loop is the spine of our four-stage closure
  (decide → episode trace → critic report → skill proposal). See `agent._on_terminal` +
  `critic_agent.EpisodeCriticAgent.run`.
- **PDF Chapter 5 (Memory):** Verbal reflections live in the same tiered store as skills and
  experiences (`memory.py:EvolvingMemory`).

## Local extract

If a working clone is needed for deeper reading, run from the project root:

```bash
git clone --depth=1 https://github.com/noahshinn/reflexion references/reflexion/_upstream
```

We deliberately do **not** check the upstream tree into `references/` — only this README and
the conceptual mapping. The Reflexion repo's HotPotQA / AlfWorld task harnesses do not transfer
to a TextArena setting.
