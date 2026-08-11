# NEXT_STAGE_RESEARCH_SURVEY — Skill Self-Evolution 调研

**Date:** 2026-08-08
**Branch:** research/next-stage-final
**范围:** 任务1 Skill 自学习 + 任务2/3/4 相关调研

## 1. 核心问题
"固定 Skill 规则能否变成可验证、自我更新的 Skill artifact？"

## 2. 调研结果（联网核实，arXiv/官方仓库）

### 2.1 SkillOpt（arXiv:2605.23904, Microsoft）
- **原文**: skill 是自然语言文档 `best_skill.md`（300-2000 tokens），非权重；目标模型冻结；optimizer 产生 bounded edit（lr=4 cosine）；held-out validation（D_tr/D_sel/D_test）；**严格 improvement gate（平局拒绝）**；rejected edit buffer；hash 版本追踪
- **借鉴**: 完整模板——external artifact + frozen executor + bounded edit + held-out gate + rejected buffer
- **不照搬**: 每候选重跑 held-out（数字环境便宜）；skill 是纯 NL 单文档

### 2.2 EmbodiSkill（arXiv:2605.10332）
- **原文**: training-free；四类 reflection（DISCOVERY/OPTIMIZATION/SKILL DEFECT/EXECUTION LAPSE）；body/appendix 双通道——**EXECUTION LAPSE 只更新 appendix 不破坏 body**
- **借鉴**: defect vs lapse 归因 + body/appendix 防误改
- **不照搬**: 无 held-out gate，信任 LLM 归因

### 2.3 RoboSkill（FlagOpen/RoboSkill GitHub）
- **原文**: MCP 标准化技能商店，Manufacturer→Model 分层注册——**非 self-evolution**
- **借鉴**: skill 注册/分层存储载体（可并入 SkillChannel/USR 作为 Skill Registry）

### 2.4 补充
| 工作 | 机制 | 验证 |
|------|------|------|
| EvoSkill（2603.02766）| 3-agent（Executor/Proposer/Skill-Builder）失败驱动 | Pareto frontier + held-out |
| SkillOS（2605.06614）| trainable curator 管理 SkillRepo（insert/update/delete），GRPO | composite reward |
| ASPIRE（2607.00272）| code-as-policy，细粒度轨迹 failure diagnosis | **执行引擎重跑验证** |
| MUSE-Autoskill（2605.27366）| skill 生命周期（create/memory/manage/eval/refine）| per-skill 经验档案 |

## 3. "如何判断变好"的主流机制
1. **held-out 严格门控**（SkillOpt/EvoSkill）
2. **事后评测**（EmbodiSkill）
3. **执行重跑验证**（ASPIRE）
4. **延迟奖励 RL**（SkillOS）

## 4. 当前项目适用组合（推断）
- 主干: **SkillOpt 模板**（artifact + frozen executor + bounded edit + held-out gate + rejected buffer）
- 修正: **EmbodiSkill defect/lapse 归因**（防物理抖动误改）
- 可选: **ASPIRE 执行重跑**作门控信号

## 5. 结论
**"固定 Skill 规则可以变成可验证自我更新 artifact"——成立**，前提有 held-out 评测套件与可评分执行。RoboSkill 提供注册载体，SkillOpt 提供演化机制，EmbodiSkill 提供防误改归因。

---

# Part B: Task 2 Skill 并行 / 异步执行 调研

## 核心问题
"SkillChannel 上多 skill 能否并行/异步执行，用 AgenticCache/Tool-Aligned 思路提升 throughput + 减少 latency？"

## 调研结果（联网核实）

### B.1 AgenticCache（arXiv:2604.24039, MLSys 2026）
- **原文**: embodied 任务有 plan locality（下一 plan 可从当前预测）；runtime cache of frequent plan transitions；**background Cache Updater 异步调 LLM 验证/refine cached entries**
- **结果**: SR +22%（12 configs），sim latency -65%，tokens -50%
- **借鉴**: 缓存 + 异步后台更新——把"每步都要 LLM"改为"cache hit 直出 + 异步 refine"。RoboAgent 的 OG/SD/EG 高频调用可复用此思路（state→skill 调用的 cache）
- **不照搬**: AgenticCache 是纯 planning 复用，不涉及多 skill 并行执行

### B.2 Tool-Aligned VLA（arXiv:2605.13119）
- **原文**: VLAs-as-Tools——高维 VLM agent 做时序推理 + 家族 specialized VLA tool 做局部操作；**VLA tool-family interface 暴露 explicit tool selection + in-execution progress feedback**，event-triggered agent replanning（不连续轮询）；**TAPT**（invocation-aligned training units + tool-family residual adapters）提高 invocation fidelity
- **结果**: LIBERO-Long +4.8，RoboTwin +23.1，invocation fidelity +15（Non-biased Rate）
- **借鉴**: tool-family residual adapters 与我们的分层 DA adapter 对应；in-execution progress feedback + event-triggered replan 与 RoboAgent 的 skill 反馈→Brain 重规划一致
- **不照搬**: 物理 VLA tool 动作空间，非符号 skill 调用

### B.3 Robotouille（arXiv:2502.05227）
- **原文**: 异步 planning benchmark；同步/异步两套数据集测 LLM 并行/时序规划
- **关键数字**: ReAct(gpt4-o) **sync 47% → async 11%**——异步 planning 是 LLM 明显短板
- **借鉴**: 证明"并行/异步规划"是真实 open problem；RoboAgent 的 SkillChannel 若做并行需显式处理时序依赖
- **教训**: 需要 long-horizon feedback + self-audit reasoning（失败模式分析）

### B.4 Survey: Embodied Agentic AI（arXiv:2508.05294, v4）
- **原文**: LLM/VLM-driven 机器人自治综述；taxonomy 分类 agent 角色：coordinator / planner / perception actor / generalist interface
- **借鉴**: RoboAgent 定位为 coordinator（多 skill 协调）→ 支持"并行 skill 执行 + Brain 协调"叙事；架构定位佐证

## 结论（Task 2 方向）
1. **异步/并行是真实 open problem**（Robotouille async 11%），有论文支撑（AgenticCache 提升 65% latency）
2. RoboAgent 可做**轻量并行**：状态→skill 调用的 cache（AgenticCache 式）+ 多 skill 异步触发 + Brain event-triggered 重规划（Tool-Aligned 式）
3. 风险：ALFRED 单 agent 串行动作（grasp/clean 不能真并行）——**并行收益主要来自 latency 侧**（cache hit 免 LLM 调用）而非任务时长；需诚实标注局限
