# ExpeL (citation reference)

**Paper:** Zhao et al., 2024, "ExpeL: LLM Agents Are Experiential Learners"
**Upstream:** https://github.com/LeapLabTHU/ExpeL
**arXiv:** https://arxiv.org/abs/2308.10144

## Core idea

ExpeL extends Reflexion in two directions:
1. **Cross-task insight extraction** — after many task trajectories, an *insight extractor*
   distils trajectory pairs (success vs. failure) into a small set of compact rules
   ("insights"). The insights become a long-lived prompt component, separate from per-task
   reflections.
2. **Pair-wise contrastive learning** — the insight prompt is built by comparing a
   success/failure pair and asking the LLM *what differed*. This is more sample-efficient
   than per-task self-reflection alone.

Three ExpeL components:
- **Trajectory pool** (analog of replay buffer)
- **Insight extractor** (LLM-as-judge with contrast prompt)
- **Insight library** (small, evolves over time, prepended at inference)

## Mapping into this codebase

| ExpeL concept | GameAgent module |
|---|---|
| Trajectory pool | `EvidenceGraph` `episode` + `decision_frame` tables + `transitions` field on each |
| Insight extractor | `insight.InsightExtractor.consolidate` (called from critic loop) |
| Insight library | `memory.SkillMemory` (active skills) + `skill_manager.SkillManager` (lifecycle) |
| Contrast win/loss | `critic_agent.EpisodeCriticAgent` system prompt instructs critic to compare
                       successful_patterns[] vs root_causes[] before emitting `propose_skill` |

## What we **don't** copy from ExpeL

- ExpeL's insight library is flat text. We promote insights to first-class `Skill` nodes with
  a six-state lifecycle (`proposed → candidate → validated → active → deprecated/rejected`) so
  every skill can be replayed against history (`SkillManager.run_evolution_sweep`).
- ExpeL gives the actor read-only access to insights. We give the *critic* (not the actor)
  write access to skills, and the actor reads only `active` skills — preventing the actor
  from drowning in unvalidated insights.

## What we **do** copy

- The success/failure contrast pattern: in `critic_agent.CRITIC_SYSTEM_PROMPT`, the critic is
  explicitly instructed to enumerate `successful_patterns` AND `root_causes`, then call
  `propose_skill` only when both lists are non-empty *and* the skill is supported by ≥2
  evidence ids.
- The "small library" hygiene: `SkillManager` actively `deprecate`s overlapping skills, so the
  actor's prompt budget for L3 stays bounded.

## File-level pointers (PDF 7-chapter map)

- **PDF Chapter 4 (Skill learning):** ExpeL's insight library is the direct ancestor of our
  Skill subsystem. See `skill_manager.SkillManager` + `skill_manager.SkillStatus`.
- **PDF Chapter 6 (Critic):** ExpeL's contrast-prompt design informed
  `critic_agent.CRITIC_SYSTEM_PROMPT`.

## Local extract

```bash
git clone --depth=1 https://github.com/LeapLabTHU/ExpeL references/expel/_upstream
```

Not checked in. The ExpeL repo's HotpotQA / FEVER / WebShop harnesses do not apply.
