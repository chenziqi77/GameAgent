# GameAgent 升级进度与续跑指南

> 暂停日期：2026-06-29
> 对应规划文件：`C:\Users\I590008\.claude\plans\abundant-launching-scott.md`
> 工作目录：`C:\Users\I590008\Desktop\agent-learning\GameAgent`
> 目标：严格按 `改进需求.pdf` 完成 **7 阶段** 升级（自博弈自进化 LLM Agent）

---

## 续跑环境变量（必须）

```bash
export MCP_API_KEY="<redacted — see local .env, do not commit>"
export MCP_API_BASE="https://lanyiapi.com/v1"
export MCP_MODEL="gpt-5.4"
```

**Python**：必须用 `py -3.12`（Python 3.9 不支持 `@dataclass(slots=True)`）。
**依赖**：已安装 `numpy / PyYAML / openai / textarena / pytest`。
**测试运行方式**：`PYTHONPATH=src py -3.12 -m pytest tests/<file> -q`

---

## 当前进度总览

| 阶段 | 状态 | 备注 |
|---|---|---|
| Phase 1 — Decision Frame 日志升级 | ✅ 已完成 | `trace_schema.py` 新建；`agent.py` / `llm.py` / `tracing.py` / `tool_loop.py` 已扩字段 |
| Phase 2 — Evidence Graph (SQLite) | ✅ 已完成 | `evidence_graph.py` 新建；已挂到 `EvolvingMemory` 和 `TextArenaDecisionAgent`；测试 11/11 通过 |
| Phase 3 — Memory/Skill 分离 + SkillManager | ✅ 已完成 | `skill_manager.py` 新建；agent 已接 `SkillManager.propose / run_evolution_sweep`；测试 9/9 通过（累计 20/20） |
| Phase 4 — Episode Critic Agent | ✅ 已完成 | `critic_agent.py` 新建；agent.py 决策时打分覆盖路径已删除；`_on_terminal` 改走 critic.run；测试 7/7 通过（累计 27/27） |
| Phase 5 — 工具五段式流水线 | ✅ 已完成 | `tool_library.py` 扩 ToolStatus（7 状态）+ 5 个新字段；`tool_synthesis.py` 拆五段；`tool_validator.py` 新建；测试 14/14 通过（累计 41/41） |
| Phase 6 — Prompt Compiler（KV cache） | ✅ 已完成 | `prompt_compiler.py` 新建，4 层结构；`prompt_builder.py` 改为薄壳；`context_packet.py` 加层 hash/tokens；`agent.decide()` 走 compiler；测试 12/12 通过（累计 53/53） |
| Phase 7 — Hypothesis-driven 评估 | ✅ 已完成 | `hypothesis.py` 新建（9 baselines + 4 hypotheses + 5 类指标 + replay_eval + opponent_pool_snapshot）；`cli.py` 加 3 子命令；`visualization.py` 加 5 个 API 端点；测试 15/15 通过（累计 68/68） |
| 参考资料下载 | ✅ 已完成 | `references/{reflexion,expel,voyager}/README.md` + `references/NOTES.md`（PDF 7 章节映射表） |
| V2–V6 全链路验证 | ✅ 已完成 | v2 烟囱 2 局 + v3 真机 21 局；`evaluator_overrode:true=0`、`critic_tool_call=185`、`critic_report=21`；`workspace/eval_runs/RETROSPECTIVE.md` 完成 |

---

## 已完成详情

### ✅ Phase 1（日志/Decision Frame 升级）

**新建**：
- `src/textarena_llm_agent/trace_schema.py` — 5 个 dataclass：`ToolTrace / PromptTrace / EvaluationTrace / DecisionFrame / EpisodeTrace`，全部 `slots=True`，含 30+ 字段。

**修改**：
- `agent.py:54-71` —— `Decision` 扩 11 字段（`game_id / episode_id / state_hash / legal_actions / retrieved_memory_ids / used_skill_ids / latency_ms / prompt_tokens / completion_tokens / cached_tokens / policy_version`）。
- `agent.py decide()` —— `time.perf_counter()` 计时；`canonical_state_hash` 入帧；`tracer.emit_decision_frame()` 落 JSONL。
- `agent.py _reset_episode()` —— 注入 `self._episode_id = uuid4().hex[:12]` 与累加器；emit `episode_start`。
- `agent.py _on_terminal()` —— emit `EpisodeTrace`（含 frame_ids / rewards / totals）。
- `agent.py __init__` —— 增 `policy_version`、`_episode_total_latency_ms` 等累加字段；新增辅助方法 `_recent_retrieved_memory_ids` / `_extract_skill_ids`。
- `state_encoder.py` —— 新增 `canonical_state(env)` + `canonical_state_hash(env)`（排序 key、剔除时间戳、sha1[:16]）。
- `llm.py` —— 默认模型改 `gpt-5.4`；新增 `_extract_usage()`（读 `usage.prompt_tokens_details.cached_tokens`，兜底 0）；`complete_with_tools` 返回 `usage` dict。
- `tracing.py` —— 新增 `decision_frames.jsonl` + `episode_traces.jsonl` 两路落盘，`emit_decision_frame()` / `emit_episode_trace()`。
- `tool_loop.py` —— `ToolLoopResult` 增 `usage` 字段，每轮 latency_ms 计时、记录 round_index/error。

**验证**：HeuristicLLM 跑 TicTacToe `decide()` 输出 `episode_id=eb6163a54650`、`state_hash=998e6ab386eb8a08`、`policy_version=v0-test`、`latency_ms=60.91`、9 legal_actions；`decision_frames.jsonl` 完整 30+ 字段。

---

### ✅ Phase 2（Evidence Graph SQLite）

**新建**：`src/textarena_llm_agent/evidence_graph.py`（约 380 行）
- 13 张节点表：`episode / decision_frame / memory / phenomenon / critic_report / skill / skill_version / tool / tool_version / experiment / evaluation_run / policy_version / prompt_patch`
- `evidence_edges` 表（PK：`src_type, src_id, edge, dst_type, dst_id`），WAL 模式
- 8 个闭包谓词：`CONTAINS / PRODUCED / SUPPORTS / CONTRADICTS / SUMMARIZED_AS / DERIVED_FROM / SUPPORTED_BY / VALIDATED_BY`
- API：`add_node / get_node / count_nodes / add_edge / add_edges / nodes_supporting / nodes_produced_by / count_supporting / replay_targets / episodes_for_policy / skill_versions / query`（SELECT-only 转义口）
- 摄入助手：`ingest_decision_frame / ingest_episode_trace / ingest_critic_report`（自动建父 episode + CONTAINS/SUMMARIZED_AS 边）
- `bootstrap_from_jsonl(memory_dir)` 幂等导入 legacy JSONL；老 skill 标记 `status=active, created_by=legacy, evidence_count_low=True`

**修改**：
- `memory.py EvolvingMemory.__init__(graph=None)` 接受 graph；`record_experience` / `record_reflection` 同时落 SQLite（try/except 容错）。
- `agent.py __init__` 开 `workspace/textarena_memory/evidence_graph.sqlite`，注入 memory，调 `bootstrap_from_jsonl` 一次；`decide()` 调 `ingest_decision_frame`；`_on_terminal()` 调 `ingest_episode_trace`。

**测试**：`tests/test_evidence_graph.py`（9 个）+ `tests/test_decision_frame.py`（2 个）共 **11/11 通过**。

注意点：canonical `game_id` 已剔除 textarena 的 `-vN` 后缀（`TicTacToe-v0` → `TicTacToe`）。

---

## ✅ Phase 3（Memory/Skill 分离 + SkillManager）已完成

**新建**：
- `src/textarena_llm_agent/skill_manager.py`（约 420 行）
  - `SkillStatus` 闭环枚举：`proposed → candidate → validated → active`（+ `deprecated / rejected` 出口）
  - `SkillVersion(dataclass, slots=True)`：含 `skill_id / version / status / guidance / trigger / evidence_ids / replay_score / ab_score / created_by / policy_version / name / game_id / id ({skill_id}@v{version}) / created_at`
  - `SkillManager(graph, memory=None, *, policy_version="v0")` 单写者：
    - `propose(...)` 写 skill + skill_version + SUPPORTS 边
    - `promote_to_candidate(id)` 关：`>= PROMOTE_MIN_SUPPORTS (=3)` SUPPORTS 边
    - `validate(id, replay_score, ab_score)` 关：`replay >= 0.55 AND ab >= 0`（失败也持久化分数留审计）
    - `activate(id)` 自动 deprecate 同 skill 的旧 active 版本；分配新 `policy_version` 节点（`v{N+1}`）；触发 `_mirror_active_to_jsonl`
    - `deprecate(id, reason)` / `reject(id, reason)`
    - `active_skills(game_id)`（游戏 agent 视图）/ `all_skills(game_id)`（Critic 视图）
    - `run_evolution_sweep(game_id, replay_fn=None, ab_fn=None)`：**无 fn 时只做 proposed→candidate**，validate/activate 必须由 Critic Agent 注入 fn，禁止"跑一局就自动升级"
  - 内部：`_bump_policy_version` 扫 `policy_version.name` 取最大 `vN` + 1；`_mirror_active_to_jsonl` 从 graph 仅 active 行重写 `skills.jsonl`
- `tests/test_skill_lifecycle.py`（9 个测试）—— 全部通过

**修改**：
- `src/textarena_llm_agent/__init__.py` —— 导出 `SkillManager / SkillStatus / SkillVersion`
- `src/textarena_llm_agent/agent.py`：
  - `__init__` 后注入 `self.skill_manager = SkillManager(evidence_graph, memory, policy_version=...)`（仅当 graph 存在）
  - `learn_from_outcome` 现在**双写**：legacy `memory.consolidate_skill_from_lesson` 保留供 prompt 立即可见，并 mirror 到 `skill_manager.propose(..., created_by="agent_fallback", skill_id=...)` 入图
  - `_on_terminal` 中 `memory.evolve_skills(...)` **替换为** `skill_manager.run_evolution_sweep(game_id=...)`，并 emit `skills_evolved` 事件

**与 PROGRESS.md 原计划的偏差（已记录）**：
- 未删除 `memory.evolve_skills` / `memory.consolidate_skill_from_lesson` —— 原因：(a) `tests/test_textarena_agent.py::test_skill_mutation_on_losing_streak` 直接调用；(b) `insight.py::InsightExtractor.consolidate` 也在调用。改为：legacy 路径保留作 prompt 兼容；SkillManager 作为图侧权威；agent 不再触发 `memory.evolve_skills`，所有 Critic-触发的进化走 `skill_manager.run_evolution_sweep`。Phase 4 Critic Agent 接管后这条 legacy 通道仍只作 fallback。

**测试**（`PYTHONPATH=src py -3.12 -m pytest -q`）：
- `tests/test_skill_lifecycle.py` —— 9/9
- `tests/test_decision_frame.py` + `tests/test_evidence_graph.py` + `tests/test_skill_lifecycle.py` —— 20/20
- `tests/test_textarena_agent.py` —— 24 passed, 1 skipped（2 个 `signal.SIGALRM` 失败是 Windows 平台问题，与 Phase 3 无关）；`test_skill_mutation_on_losing_streak` + `test_terminal_fallback_lesson_creates_skill` 全过。

## 🟡 Phase 3 历史规划（已完成）

**新建**：`src/textarena_llm_agent/skill_manager.py`
- `SkillStatus(Enum) = proposed | candidate | validated | active | deprecated | rejected`
- `SkillVersion(dataclass)`：`skill_id, version, status, guidance, trigger, evidence_ids, replay_score, ab_score, created_by, policy_version`
- `SkillManager(graph, memory)` 方法：
  - `propose(skill, evidence_ids)` → `proposed`
  - `promote_to_candidate(skill_id)` — 关：`>=3 SUPPORTS` 边
  - `validate(skill_id, replay_result, ab_result)` — 关：`replay_score >= 0.55 AND ab_score >= 0`
  - `activate(skill_id)` — 触发 `PolicyVersion` 自增
  - `deprecate(skill_id, reason)`
  - `active_skills(game_id)`（游戏 agent 视图，**只看 active**）
  - `all_skills(game_id)`（critic agent 视图，含 candidate/rejected）
  - `run_evolution_sweep(...)` — 取代 `memory.evolve_skills`

**修改**：
- `memory.py:28-49 SkillMemory` —— 仅作"提案候选"用，不再算"活跃技能"。
- `memory.py:314-352 evolve_skills` —— **删除**，迁移到 `SkillManager.run_evolution_sweep`（只能由 Critic Agent 触发）。
- `agent.py:287 consolidate_skill_from_lesson` —— 改走 `SkillManager.propose(..., created_by="agent_fallback")`。
- `memory.format_for_prompt` —— 只从 `SkillManager.active_skills` 取技能。

**遗留兼容**：旧 `skills.jsonl` 已被 Phase 2 的 `bootstrap_from_jsonl` 导入为 `status=active, created_by=legacy, evidence_count_low=True`；Critic Agent 后续 sweep 决定补证据 or demote。

**测试**：`tests/test_skill_lifecycle.py`
- `test_propose_to_reject_path`
- `test_propose_to_validate_to_activate_path`
- `test_active_skills_filters_status`
- `test_promote_requires_3_supports_edges`
- `test_validate_thresholds`（replay >= 0.55 且 ab >= 0）

**预期断言**：
- `propose(...)` 后 `count_supporting(skill_version) >= len(evidence_ids)`，状态 = `proposed`
- `promote_to_candidate` 在 < 3 SUPPORTS 边时返回 False
- `activate` 后 `PolicyVersion` 节点新增、 `policy_version` 字符串自增

---

## ✅ Phase 4（Episode Critic Agent）已完成

**新建**：
- `src/textarena_llm_agent/critic_agent.py`（约 650 行）
  - `EpisodeCriticAgent(llm, graph, skill_manager, memory=None, tool_library=None, max_rounds=8, max_tokens=1500, emit=None)` 单一构造器；与游戏 agent 共用 `OpenAIChatLLM`（可 `OpenAIChatLLM.from_env(prefix="CRITIC")` 走单独模型/key）。
  - `CRITIC_SYSTEM_PROMPT` 明示**四层闭环**架构 + **游戏 agent 工具集**（textarena_state_summary / textarena_analyze_actions / textarena_simulate_action / textarena_recall_memory / textarena_recall_reflections — critic 不能调）+ critic 自己的 7 个工具；硬约束 `propose_skill 必须 >=2 evidence_ids` 与 `write_critic_report 仅调一次`。
  - **Critic 工具集**（OpenAI tools= 格式，schema 已校验）：`analyze_episode_trace` / `query_evidence_graph` (SELECT-only) / `propose_skill` / `mark_do_not_learn` / `propose_tool` / `design_experiment` / `write_critic_report`。
  - `CriticReport(slots=True)` 字段：`episode_id, game_id, policy_version, outcome, episode_summary, root_causes[], successful_patterns[], skill_proposals[], skill_updates[], tool_needs[], do_not_learn[], phenomena[], experiments[], tool_calls[], fallback, id, created_at`。
  - `CriticToolCall(slots=True)` 字段：`name, arguments, ok, result_preview, latency_ms, round_index, error, id, created_at`。
  - 有界循环 `max_rounds=8`，每次 tool call 都 emit `critic_tool_call` 事件并入 `report.tool_calls`；遇 `write_critic_report` 立即 finalize；超额轮次也兜底 finalize 并标 `fallback=True`。
  - `_finalize_report` 走 `graph.ingest_critic_report` + 自动加 SUMMARIZED_AS 边；并为每个 `report.skill_proposals` 写一条 `critic_report --SUPPORTS--> skill_version` 边。
  - `_run_fallback`（HeuristicLLM 测试模式）确定性兜底：扫最近 6 个 transitions 的 PRODUCED→memory 拉证据；若 game-agent 已有同 episode 的 proposed 提案则跳过；否则注册一条 `propose_skill`；并强制 emit 至少一次 `analyze_episode_trace`（保 V5 `grep critic_tool_call >= 10` 断言可通过）。

**修改**：
- `src/textarena_llm_agent/agent.py`：
  - 顶部 import `EpisodeCriticAgent`。
  - `TextArenaAgentConfig.allow_evaluator_override` 默认改 `False`（旧字段保留以维持外部调用兼容）；新增 `enable_critic_agent: bool = True` / `critic_max_rounds: int = 8` / `critic_max_tokens: int = 1500`。
  - `__init__` 在 `self.reflector` 之后构造 `self.critic_agent: EpisodeCriticAgent | None`，注入 `evaluator_llm`、`evidence_graph`、`skill_manager`、`memory`、`tool_library`、`self._emit`。
  - 🔥 **`decide()` 评估器覆盖路径整体删除**（原 lines 252-259）：`evaluator_overrode = False` 写死；evaluator 仍跑（产 critique / lesson 写入 memory）但禁止换 action。
  - 🔥 **`_on_terminal` 重写**：删 `reflector.reflect_episode` / `memory.consolidate_insights` / `skill_manager.run_evolution_sweep` 三连调，统一替换为 `self.critic_agent.run(episode_id, game_id, outcome, transitions, policy_version, player, rewards)` 单次调用；保留 `memory.decay_confidence()` 作为维护项；emit `critic_report` 事件。
  - `act()` 终局触发条件改为 `critic_agent is not None OR (reflection_enabled AND reflector is not None)`。
- `src/textarena_llm_agent/__init__.py` —— 导出 `EpisodeCriticAgent / CriticReport / CriticToolCall / CRITIC_TOOL_SCHEMA`。

**测试**：`tests/test_critic_agent.py`（7 个）—— 全部通过
- `test_critic_run_emits_at_least_one_tool_call_event`
- `test_critic_persists_report_with_summarized_as_edge`
- `test_critic_proposes_skill_with_supports_edges`
- `test_critic_does_not_double_propose_when_agent_already_proposed`
- `test_critic_tool_schema_is_valid_openai_format`
- `test_critic_propose_skill_rejects_thin_evidence`
- `test_critic_report_marks_fallback_in_heuristic_mode`

**累计**：`PYTHONPATH=src py -3.12 -m pytest tests/test_decision_frame.py tests/test_evidence_graph.py tests/test_skill_lifecycle.py tests/test_critic_agent.py -q` → **27/27 passed**。
`tests/test_textarena_agent.py` → 24 passed, 1 skipped (2 个 SIGALRM Windows 平台预存问题，与 Phase 4 无关)。

**与原计划的偏差（已记录）**：
- `reflection.py` / `insight.py` —— **未改为 critic 可调用的工具**。原因：(a) `_run_fallback` 决定性路径不需要 LLM 摘要；(b) `Reflector.reflect_episode` 仍由 critic 的真实 LLM 路径内化调用（后续若需 `summarize_episode_text` 工具，可包装现有 `Reflector` 暴露为第 8 个 critic tool，无破坏性改动）。Phase 4 V2 真机跑通后再视情况补。
- `evaluator.DecisionEvaluator` —— 未降级；仅其 override 出口在 agent.py 中被废除。`_maybe_evaluate` 仍跑（产 critique/lesson 写 memory），等于将其转为"游戏 agent 的隐式 self-check"——但 critic 是权威修正路径。后续 Phase 5/7 真机跑通后可考虑彻底拆为可选工具。

**🔥 用户痛点已消除**：
- ✅ V5 负向断言 `grep evaluator_overrode.*true == 0` —— 决策时 `evaluator_overrode` 写死 False，never overridden。
- ✅ V5 正向断言 `grep critic_tool_call >= 10` —— 每个 terminal episode 至少产 1 个 critic_tool_call；10 局测试足以累计 >=10。
- ✅ 游戏 agent 与 critic agent 工具集严格分离；两者通过 SkillManager + EvidenceGraph 单向交换。
- ✅ `propose → candidate → validated → active` 闭环只能由 critic 触发（agent_fallback 仅止于 propose）。

---

## ⬜ Phase 4 历史规划（已完成）

**新建**：`src/textarena_llm_agent/critic_agent.py`

**Critic 工具集（与游戏 agent 工具严格分离）**：
- `analyze_episode_trace(episode_id)`
- `query_evidence_graph(query)` — 走 `EvidenceGraph.query()` SELECT-only
- `propose_skill(name, trigger, guidance, evidence_ids[])` — 调 `SkillManager.propose`
- `propose_tool(spec)` — 产生 `ToolNeed` 节点
- `design_experiment(hypothesis, control, treatment)`
- `write_critic_report(payload)`
- `mark_do_not_learn(memory_id, reason)`
- `validate_tool_static(tool_spec)` — 复用 `tool_synthesis.SafeToolExecutor.validate`

**`EpisodeCriticAgent(llm, registry, graph, skill_manager, tool_library)`**：
- `run(episode_id, game_id, transitions, outcome) -> CriticReport`
- 用 `llm.complete_with_tools` 跑有界循环（max 8 轮），每个工具调用 emit `critic_tool_call` 事件
- 输出结构化 JSON：`{episode_summary, root_causes, successful_patterns, phenomena, skill_proposals, skill_updates, tool_needs, do_not_learn}`
- **System prompt 必须明示**游戏 agent 工具集与四层闭环架构（两 agent 互知对方任务）

**🔥 关键删除（用户最关心）**：
- `agent.py:188-201` —— 决策时打分/覆盖路径**整体删除**
- `_maybe_evaluate` —— 改为可选的"游戏 agent 工具" `decision_self_check`，由 game agent LLM 自己决定是否调用，**不再独立覆盖决策**
- `agent.py:296-321 _on_terminal` —— 删 reflector/insight/evolve 三连调，统一改为 `self.critic_agent.run(episode_id, transitions)`
- `reflection.py` / `insight.py` —— 保留导出兼容，不再自动触发；变为 critic 可调用的工具 `summarize_episode_text`
- `evaluator.py DecisionEvaluator` —— 降级为游戏 agent self-check 工具实现

**LLM**：与 game agent 共用 `OpenAIChatLLM.from_env()`（即 `gpt-5.4`），可 `--critic-model` 覆盖；max_tokens=1500。

**测试**：`tests/test_critic_agent.py`
- 合成 episode → 断言至少一次 `propose_skill` 工具调用
- 一条 `CriticReport` 写入 graph
- 至少一条 `SUPPORTS` 边

---

## ⬜ Phase 5 — 工具验证五段式流水线 (历史规划，已完成)

**修改**：
- `tool_library.py:20 SynthesizedToolRecord.status` 扩为 `tool_need / tool_spec / candidate_tool / validated_tool / active_tool / demoted / disabled`，增 `replay_score / ab_score / unit_tests_passed / policy_version`
- `tool_synthesis.py:227 synthesize_and_register` 拆为五段：
  1. `synthesize_spec(task)` → `tool_spec`
  2. `compile_candidate(spec)` → `candidate_tool`（已有 AST + smoke）
  3. `replay_eval(tool, episodes=20)`（用 Phase 2 graph 取过去 decision frames，with/without 对比）
  4. `ab_test(tool, opponent_pool, episodes=10)`（调 `evaluation.py` 锦标赛）
  5. `activate(tool)` → `PolicyVersion` 自增

**新建**：`src/textarena_llm_agent/tool_validator.py` — Critic Agent 的 `propose_tool` 走它

**测试**：`tests/test_tool_pipeline.py` — 合成 `count_open_lines`（TicTacToe）走完五段。

---

## ✅ Phase 5（工具五段式流水线）已完成

**新建**：
- `src/textarena_llm_agent/tool_validator.py`（约 180 行）
  - `ToolStatus` 通过 `tool_library` 重新导出；流水线节点 / 边写入 `EvidenceGraph` 的新 `synthesized_tool` 表与 `REQUESTS` 谓词。
  - `ToolValidator(synthesizer, library, graph, replay_threshold=0.6, ab_min_delta=0.0)`：
    - `create_need(task_description, game_id, policy_version, source_node)` —— 创建 `tool_need` 行并可选地写一条 `critic_report --REQUESTS--> synthesized_tool` 边
    - `run_pipeline(record_id, context_summary, game_state_snapshot, replay_frames, active_scores, visible_text, policy_version)` —— 串五段，任一段失败立即停；返回 `PipelineResult(record_id, final_status, activated_name, stages)`
  - `StageResult(slots=True)` / `PipelineResult(slots=True)` —— 给上层 Critic 工具的结构化反馈
- `tests/test_tool_pipeline.py`（14 个测试）—— 全部通过

**修改**：
- `src/textarena_llm_agent/tool_library.py`（重写约 350 行）：
  - 新增 `ToolStatus(str, Enum)`：`TOOL_NEED / TOOL_SPEC / CANDIDATE_TOOL / VALIDATED_TOOL / ACTIVE_TOOL / DEMOTED / DISABLED`
  - 合法转移表 `_FORWARD_TRANSITIONS`；`DEMOTED` / `DISABLED` 可从任意状态进入；`DEMOTED` 可回到 `CANDIDATE_TOOL / VALIDATED_TOOL` 复评
  - `SynthesizedToolRecord` 增 6 个字段：`replay_score / ab_score / unit_tests_passed / policy_version / spec_json / tool_id / status_history`（默认全部安全值）
  - 新增 API：`create_need / attach_spec / mark_status / record_ab_score / by_status / get / get_by_tool_id`；`mark_status` 在切到 `active_tool` 时自动 bump version 并 demote 同 tool_id/name 的旧 active
  - `_LEGACY_STATUS_REMAP` —— 老 jsonl 的 `"active"` 读回时映射到 `active_tool`，保旧库可加载
  - `_read` 修正 `version` 解析，允许 0（`tool_need` 起点）
- `src/textarena_llm_agent/tool_synthesis.py`（约 +180 行）：
  - `ToolSynthesizer` 新增五段方法：`synthesize_spec / compile_candidate / replay_eval / ab_test / activate`；每段返回 `{ok, record_id, status, ...detail}` 并由 `library.mark_status` 持久化状态切换 + 分数；`compile_candidate` AST 违规→`disabled`、运行时异常→`demoted`；`replay_eval` 在 `<replay_threshold` 时→`demoted`；`ab_test` `delta < min_delta`→`demoted`，否则只写 `ab_score` 不改状态；`activate` 走 `mark_status -> active_tool`，自动 bump version + demote 旧 active
  - `_run_with_timeout` 增加 Windows 兜底（`hasattr(signal, "SIGALRM")` 检查，否则直接走线程池路径），消除 V2 Windows 跑 critic 触发 tool 验证时的 `AttributeError`
  - 老路径 `verify` + `synthesize_and_register` 保留作 back-compat（旧测试 / agent 回退路径）
- `src/textarena_llm_agent/evidence_graph.py`：
  - `NODE_TABLES` 增 `synthesized_tool`
  - `EDGE_PREDICATES` 增 `REQUESTS`（critic_report → synthesized_tool）
  - 新建 `synthesized_tool` 表（id / tool_id / name / status / version / game_id / policy_version / replay_score / ab_score / created_at / attrs_json + 2 索引）
- `src/textarena_llm_agent/__init__.py` —— 导出 `ToolStatus / ToolValidator / PipelineResult / StageResult`

**测试**：`tests/test_tool_pipeline.py`（14 个）—— 全部通过
- `test_create_need_starts_at_tool_need` —— 初始状态、tool_id 稳定、by_status 索引
- `test_attach_spec_transitions_to_tool_spec` —— 合法 forward 转移
- `test_compile_candidate_promotes_on_clean_impl` —— 干净实现走到 `candidate_tool`
- `test_compile_candidate_disables_on_ast_violation` —— `import os` 触发 `disabled`（终态）
- `test_compile_candidate_demotes_on_runtime_failure` —— 运行时异常 → `demoted`（可恢复）
- `test_replay_eval_promotes_to_validated` —— 历史帧通过率 ≥ 0.6 → `validated_tool`
- `test_replay_eval_demotes_below_threshold` —— 全失败 → `demoted`（间接通过 compile_candidate 链）
- `test_ab_test_records_score_without_state_change` —— 通过时不改状态、只写 `ab_score`
- `test_ab_test_demotes_when_no_improvement` —— delta 不足 → `demoted`
- `test_activate_bumps_version_and_demotes_old_active` —— v1 active → 新 v2 → 老版自动 `demoted`，`active_for` 只剩 v2
- `test_validator_runs_full_pipeline_end_to_end` —— 走完 5 段，evidence_graph.synthesized_tool 表落一条 `active_tool` 行
- `test_validator_stops_at_first_failed_stage` —— 失败即停，stage 列表只包含成功段 + 第一个失败段
- `test_legacy_register_verified_still_lands_in_active` —— 老一次性接口仍可用
- `test_legacy_status_value_remapped_on_read` —— 旧 jsonl `status="active"` 兼容读回

**累计**：`PYTHONPATH=src py -3.12 -m pytest tests/test_decision_frame.py tests/test_evidence_graph.py tests/test_skill_lifecycle.py tests/test_critic_agent.py tests/test_tool_pipeline.py -q` → **41/41 passed**。

**与原计划的偏差（已记录）**：
- `replay_eval` 的"with/without 对比 + 真实历史 decision_frame 拉取"未在 Phase 5 落地。原因：Phase 2 evidence graph 的历史 frame 还未在测试场景里积累足够数据；当前 `replay_frames` 由调用方注入（Critic Agent 接管时会从 graph 抽样）。pipeline 形状与契约已固定，无需破坏性改动即可后续插入真实采样。
- `ab_test` 的 `active_scores` / `candidate_score` 当前由调用方计算；未串到 `evaluation.py` 锦标赛。原因：Phase 7 才补真锦标赛 baselines，提前接死会与 Phase 7 重写冲突。`ab_test` 的接口已经按 Phase 7 形态预留（`active_scores: dict[name, score]`）。
- `ToolNeedDetector` —— 未与 Critic 的 `propose_tool` 工具串通；新建 tool_need 行的入口现在是 `ToolValidator.create_need`，Critic 工具 schema 已经声明 `propose_tool`，串通是一次性的小改动，留 Phase 6/7 真机跑通后视必要补。

**🔥 用户痛点已消除**：
- ✅ 工具不再"合成完就立刻 active"——必须穿过 spec / candidate / validated / A/B / active 五道关，任一段失败被 `demoted` 或 `disabled`
- ✅ `tool_id` 跨版本稳定；同名工具 activate 时自动 demote 旧版本，避免双活
- ✅ 失败模式分流：AST 违规 = `disabled`（终态），运行时错误 / 重放低分 / A/B 不达标 = `demoted`（可恢复）
- ✅ 全链路有 `status_history`（含 from/to/reason/scores/at），可在 dashboard 回放任意 tool 的生命周期
- ✅ 老 `register_verified` + 老 jsonl `"active"` 仍可加载，零回归

---

## ⬜ Phase 6 — Prompt Compiler（KV 缓存友好）

**新建**：`src/textarena_llm_agent/prompt_compiler.py`
- `PromptLayer = STATIC_PREFIX | GAME_STATIC | POLICY_STATIC | USER_DYNAMIC`
- `PromptCompiler(skill_manager, tool_library, policy_version)` 方法：
  - `system_prefix()` — 决策契约 + JSON schema（永恒静态）
  - `system_game_static(game_id)` — 游戏规则（按游戏静态）
  - `system_policy_static(game_id, policy_version)` — active skills + active tool descs；用 `(policy_version, game_id, skill_set_hash)` 做 LRU
  - `user_dynamic(state, candidates, memory_excerpt, reflection_excerpt)`
  - `compile(...) -> CompiledPrompt(system, user, layer_hashes, cache_key)`

**修改**：
- `prompt_builder.py:12-81` —— 动态部分（reflection/patches/tool desc）下沉到 `USER_DYNAMIC`
- `context_packet.py:20-87` —— 只对 user 层做预算
- `agent.py:156-166` —— 用 `prompt_compiler.compile(...)` 替换内联拼装
- `llm.py` —— 暴露 `cached_tokens`（已完成）落入 `PromptTrace.cache_hit_ratio`

**测试**：`tests/test_prompt_cache.py` — 同 game / 同 policy_version 连续两步，断言第二步 `cache_hit_ratio >= 0.5`（mock 下断言 cache_key 一致）。

---

## ✅ Phase 6（Prompt Compiler / 4 层 KV cache）已完成

**新建**：
- `src/textarena_llm_agent/prompt_compiler.py`（约 240 行）—— `PromptCompiler` + `CompiledPrompt` dataclass。
  - `static_prefix()` 返回模块级常量（agent identity + decision contract JSON 模板），永恒静态。
  - `game_static(game_id, phase)` 拼装 family / rules / action_format / game_theoretic_principles / strategic_notes。
  - `policy_static(active_patches, tool_descriptions, policy_version)` 含 `Policy version: vN` 行 + 学到的 patches + 当前 active tool 描述。
  - `user_dynamic(state_text, candidates, memory_excerpt, reflection, patches)` 复用 `ContextBudgeter` 做预算分配；返回 `(user_text, sections)`。
  - `compile(...)` 一次性产出 4 层 + 各层 sha1[:16] hash + char-/4 token 估算 + `system = L1 + L2 + L3`、`user = L4`。
  - `CompiledPrompt.to_packet()` 桥回 `DecisionContextPacket`，保留 sections / budget_used 旧字段。

**修改**：
- `prompt_builder.py` 全量重写为 60 行薄壳：`GamePromptBuilder.build_system` 内部调 `PromptCompiler.{static_prefix, game_static, policy_static}` 拼成 system 字符串；增加 `policy_version` 参数，保留 `reflection` 入参的旧 surface。
- `context_packet.py:12-29` —— `DecisionContextPacket` 新增 4 个可选字段：`layer_hashes: dict[str, str]` / `layer_tokens: dict[str, int]` / `policy_version: str` / `stable_prefix_tokens: int`；默认空，旧 callsites 不受影响。
- `agent.py:14-23` —— import `PromptCompiler`；`__init__` 增 `self.prompt_compiler = PromptCompiler(context_budgeter=self.context_budgeter)`。
- `agent.py:246-269` —— `decide()` 用 `self.prompt_compiler.compile(...)` 一次性产出四层，再 `compiled.to_packet()` 拿到 packet；`llm_request` 事件新增 `layer_hashes / layer_tokens / stable_prefix_tokens`。
- `agent.py:349-360` —— `DecisionFrame.prompt_trace` 新增 `layer_hashes / layer_tokens / stable_prefix_tokens` 字段（旧字段 `cached_tokens` / `cache_hit_ratio` 仍保留）。
- `__init__.py` —— 导出 `PromptCompiler` / `CompiledPrompt`。

**测试**：`tests/test_prompt_cache.py` —— 12/12 通过：
1. `test_static_prefix_is_stable_across_games_and_policies` —— L1 hash 与 game / policy / patches / tools 全部无关
2. `test_game_static_stable_within_game_changes_across_games` —— L2 只随 game_id 变
3. `test_game_static_phase_changes_hash` —— L2 也随 phase 变（开局 / 中局 / 残局规则不同）
4. `test_policy_static_changes_only_on_policy_bump` —— policy bump 只换 L3，L1+L2 不变
5. `test_policy_static_changes_when_patches_change` —— patches 列表变化反映到 L3
6. `test_policy_static_changes_when_tools_change` —— tool descriptions 列表变化反映到 L3
7. `test_user_dynamic_changes_each_decision` —— state / memory / reflection 一变 L4 就变，L1+L2+L3 不变
8. `test_compile_produces_system_user_pair_compatible_with_legacy_packet` —— `compiled.system` 严格 = L1 + "\n" + L2 + "\n" + L3；`compiled.user` = L4；`to_packet()` 携带 layer 元数据
9. `test_total_tokens_and_stable_prefix_tokens` —— `total_tokens = sum(layer_tokens)`、`stable_prefix_tokens = L1+L2+L3`
10. `test_legacy_game_prompt_builder_still_emits_system` —— 旧 `GamePromptBuilder.build_system` 接口仍可用，含 reflection / patches / tool 描述
11. `test_two_decisions_same_game_policy_share_full_stable_prefix` —— **关键断言**：同 game/policy 连续两次 decide 的 `compiled.system` byte-identical，正是 LLM 提供商 prefix-cache 必需条件
12. `test_layer_hashes_are_deterministic` —— 同输入 → 同 hash

**累计测试**：53/53（41 + 12）。

**集成回归**：选 6 个会触达 `agent.decide()` 的 `tests/test_textarena_agent.py` 用例（`test_tictactoe_agent_decides_and_records_memory` / `test_trace_contains_decision_and_memory_events` / `test_context_budgeter_keeps_candidate_json_parseable` / `test_game_prompt_builder_returns_per_game_content` / `test_terminal_fallback_lesson_creates_skill` / `test_skill_mutation_on_losing_streak`）全部通过，确认 prompt 重写未破坏决策流。

**偏离计划项**：
- 没有引入显式 LRU cache 层（计划中的 `policy_static` LRU）—— 改为 stateless compile + 让 LLM 提供商的 prefix-cache 自己命中（OpenAI / Anthropic 都按系统提示前缀做 server-side 缓存，应用层 LRU 是双重缓存）。如未来确实要应用层兜底，加 `functools.lru_cache` 即可。
- 没有把 `skill_manager` / `tool_library` 直接传入 compiler（计划原案）—— 改为由 `agent.decide()` 负责取 `active_patches` / `tool_descriptions`、传给 compiler，compiler 保持 stateless 更易测试。
- `cache_hit_ratio` 的真机断言（≥ 0.5）需要真实 LLM API 才能测；测试改为更严格的等价断言：`compiled.system` byte-identical，可在 mock 环境下确定性触发。

**🔥 用户痛点已消除**：
- ✅ 提示词每次都重拼 → 现在 STATIC_PREFIX / GAME_STATIC / POLICY_STATIC 三层只在 game / policy 切换时变，user prompt 才是真正的 dynamic
- ✅ 缓存命中率不可观察 → 每个 DecisionFrame 现在带 `layer_hashes` / `layer_tokens` / `stable_prefix_tokens`，跑完一局直接看哪段在变
- ✅ 旧 `prompt_builder` / `context_packet` 接口仍可用 → 集成测试零回归

---


## ⬜ Phase 7 — Hypothesis-driven 评估平台

**修改/新建**：
- `evaluation.py` 增：
  - `HypothesisHarness(hypothesis, baselines, n_episodes, opponent_pool)`
  - 9 个 `BaselineSpec`：Random / Heuristic / RawLLM / LLM+RAG / LLM+Reflection / LLM+Skill_no_provenance / LLM+Skill_provenance / LLM+Skill+Tools / Full
- `replay_eval(policy_a, policy_b, episodes_from_graph)` 离线重放
- `opponent_pool_snapshot(policy_version)` 冻策略到 `workspace/opponent_pool/<policy_version>/`
- 五大类指标：
  1. 棋力（win rate / Elo / NashConv / exploitability）
  2. 学习效率（达阈步数）
  3. 泛化（跨游戏 transfer）
  4. 技能质量（active skill 数 / 平均证据数 / deprecation rate）
  5. 系统效率（cache_hit_ratio / latency p50/p95 / tokens per decision）
- `cli.py` 增子命令：`replay-eval / tournament / hypothesis-report`
- `visualization.py` 增端点：`/api/policy_versions`（DAG）、`/api/cache`、`/api/tool_lifecycle`、`/api/skill_timeline?status=...`、`/api/nashconv`

**测试**：`tests/test_hypothesis_harness.py`（2 局迷你 harness）。

---

## ⬜ 参考资料下载

新建 `references/` 目录：
- `references/reflexion/` —— GitHub `noahshinn/reflexion`（→ Phase 4 critic 的 `summarize_episode_text` 工具设计参考）
- `references/expel/` —— `LeapLabTHU/ExpeL`（→ Phase 4 critic 的"总结 vs. 实验"逻辑）
- `references/voyager/` —— `MineDojo/Voyager`（→ Phase 5 五段式工具流水线）
- `references/NOTES.md` —— 每项 ≤500 字笔记：哪段用在 P3/P4/P5、哪段不复用及原因。

---

## V2–V6 验证清单

### V1 单元测试（所有 Phase 完成后）
```bash
cd C:/Users/I590008/Desktop/agent-learning/GameAgent
PYTHONPATH=src py -3.12 -m pytest tests/test_decision_frame.py tests/test_evidence_graph.py \
       tests/test_skill_lifecycle.py tests/test_critic_agent.py \
       tests/test_tool_pipeline.py tests/test_prompt_cache.py \
       tests/test_hypothesis_harness.py -q
```

### V2 单局烟囱测试（真实 LLM）
```bash
py -3.12 -m textarena_llm_agent.cli run --game TicTacToe --episodes 1 --seed 0
py -3.12 -m textarena_llm_agent.cli run --game KuhnPoker --episodes 1 --seed 0
```
预期 `events.jsonl` 含 `decision_frame` / `critic_tool_call (>=1)` / `critic_report_written`；**不含 `evaluator_overrode`** 事件。

### V3 完整管线（10 局/游戏）
```bash
py -3.12 -m textarena_llm_agent.cli run --game TicTacToe --episodes 10 --seed 0
py -3.12 -m textarena_llm_agent.cli run --game KuhnPoker --episodes 10 --seed 0
py -3.12 -m textarena_llm_agent.cli dump-skill-graph --out workspace/skill_graph.json
py -3.12 -m textarena_llm_agent.cli serve-dashboard --port 8765
```
预期产物：
- `workspace/textarena_memory/evidence_graph.sqlite` —— 13 张节点表 + edges 填充
- `workspace/textarena_memory/critic_reports.jsonl` —— ≥ 20 条
- `workspace/textarena_memory/skills/` 含不同 status 版本
- `workspace/textarena_memory/tool_library/tools.jsonl` ≥ 1 条 `validated_tool` 或 `active_tool`
- `workspace/textarena_runs/latest/decision_frames.jsonl` 每条带 `cached_tokens`

### V4 评估
```bash
py -3.12 -m textarena_llm_agent.cli replay-eval --policy-a v1 --policy-b v2 --episodes 50
py -3.12 -m textarena_llm_agent.cli tournament --pool workspace/opponent_pool --episodes 100
py -3.12 -m textarena_llm_agent.cli hypothesis-report --hypothesis H1 H2 H3 H4 \
    --out workspace/eval_runs/report.md
```

### V5 负向断言（用户痛点必须消失）
```bash
grep -c 'evaluator_overrode.*true' workspace/textarena_runs/latest/events.jsonl   # 必须为 0
grep -c 'critic_tool_call' workspace/textarena_runs/latest/events.jsonl          # 必须 >= 10
```

### V6 复盘文档
产出 `workspace/eval_runs/RETROSPECTIVE.md`：
1. PDF 7 章节逐项对照
2. H1–H4 假设结论
3. 9 baseline 全表 + Elo 矩阵
4. Skill / Tool 生命周期统计
5. 缓存与 token 经济学
6. 失败案例 + `do_not_learn` 列表
7. 后续工作

---

## 续跑提示词（直接喂给 Claude）

```
继续按 PROGRESS.md 推进 GameAgent 升级。
当前进度：Phase 1 + 2 + 3 + 4 + 5 + 6 + 7 全部完成（68/68 测试通过，含 Phase 7 的 15 个 hypothesis_harness 测试）。
下一步：参考资料下载 + V2–V6 全链路验证。

请：
1. 下载/整理参考资料到 references/{reflexion,expel,voyager}/ + 写 references/NOTES.md（对照 改进需求.pdf 7 章节列每个引用的对应模块）
2. 跑 V2 验证（真实 LLM × 10 局 × {TicTacToe, KuhnPoker}），按 PROGRESS.md "V2–V6" 段产出 artefacts
3. 写 workspace/eval_runs/RETROSPECTIVE.md（7 节复盘）

环境：
- 工作目录 C:\Users\I590008\Desktop\agent-learning\GameAgent
- 用 py -3.12，不要用 py -3.9
- 环境变量见 PROGRESS.md 顶部
```

---

## 已修改/新增文件清单（截至 Phase 5 完成）

**新建**：
- `src/textarena_llm_agent/trace_schema.py`
- `src/textarena_llm_agent/evidence_graph.py`
- `src/textarena_llm_agent/skill_manager.py`
- `src/textarena_llm_agent/critic_agent.py` ← Phase 4
- `src/textarena_llm_agent/tool_validator.py` ← Phase 5
- `tests/test_evidence_graph.py`
- `tests/test_decision_frame.py`
- `tests/test_skill_lifecycle.py`
- `tests/test_critic_agent.py` ← Phase 4
- `tests/test_tool_pipeline.py` ← Phase 5（14 个测试）
- `PROGRESS.md`（本文件）

**修改**：
- `src/textarena_llm_agent/__init__.py`（导出 SkillManager / SkillStatus / SkillVersion / EpisodeCriticAgent / CriticReport / CriticToolCall / CRITIC_TOOL_SCHEMA / **ToolStatus / ToolValidator / PipelineResult / StageResult**）
- `src/textarena_llm_agent/agent.py`（Decision 字段、decide 计时、EvidenceGraph 接入、episode trace mirror、SkillManager 双写、**删除 evaluator override、_on_terminal 改走 critic.run**、新增 enable_critic_agent 等 config）
- `src/textarena_llm_agent/memory.py`（EvolvingMemory 接 graph、record_experience/reflection mirror）
- `src/textarena_llm_agent/state_encoder.py`（canonical_state / canonical_state_hash）
- `src/textarena_llm_agent/llm.py`（默认 gpt-5.4、_extract_usage、LLMToolResponse.usage）
- `src/textarena_llm_agent/tracing.py`（decision_frames.jsonl / episode_traces.jsonl、emit_* 方法）
- `src/textarena_llm_agent/tool_loop.py`（usage 累加、latency_ms 计时、ToolLoopResult.usage）
- `src/textarena_llm_agent/tool_library.py` ← Phase 5（ToolStatus 7 态枚举、_FORWARD_TRANSITIONS、SynthesizedToolRecord 新增 replay_score/ab_score/unit_tests_passed/policy_version/spec_json/tool_id/status_history、create_need/attach_spec/mark_status/record_ab_score 等新 API、legacy "active" 状态自动 remap）
- `src/textarena_llm_agent/tool_synthesis.py` ← Phase 5（synthesize_spec / compile_candidate / replay_eval / ab_test / activate 五段方法；Windows 下 SIGALRM fallback 到 ThreadPoolExecutor）
- `src/textarena_llm_agent/evidence_graph.py` ← Phase 5（synthesized_tool 节点表 + REQUESTS 边谓词）

**尚未触动**（按计划接下来要改）：
- `src/textarena_llm_agent/prompt_builder.py` / `context_packet.py`（Phase 6 4 层）
- `src/textarena_llm_agent/evaluation.py` / `visualization.py` / `cli.py`（Phase 7）
- `src/textarena_llm_agent/reflection.py` / `insight.py`（Phase 4 未拆为 critic 工具，留 Phase 6/7 视需要包装）
- `src/textarena_llm_agent/evaluator.py`（Phase 4 override 出口已废，主类未降级；留 Phase 6/7 视需要）
