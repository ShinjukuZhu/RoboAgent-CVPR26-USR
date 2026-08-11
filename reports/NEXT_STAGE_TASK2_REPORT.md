# NEXT_STAGE_RESULTS — Task 2: Parallel / Asynchronous Skill Execution

**Date:** 2026-08-11
**Branch:** research/parallel-v2

## 研究问题
SkillChannel 多 skill 能否并行/异步调度，借鉴 AgenticCache 缓存 + Tool-Aligned 事件触发，降低调度延迟？

## 调研（联网核实，见 NEXT_STAGE_RESEARCH_SURVEY Part B）
- **AgenticCache (2604.24039, MLSys26)**: cache + 后台异步 updater，latency -65%, tokens -50%
- **Tool-Aligned VLA (2605.13119)**: tool-family interface + event-triggered replan，LIBERO-Long +4.8
- **Robotouille (2502.05227)**: 异步是 LLM 短板（sync 47% → async 11%）
- **Survey (2508.05294)**: agent 作为 coordinator 分类法支持多 skill 协调叙事

## Framework 级验证（CPU, mock dispatch）
| 并发数 | serial | parallel | speedup |
|--------|--------|----------|---------|
| n=2 | 0.600s | 0.302s | 1.99x |
| n=4 | 1.200s | 0.303s | 3.97x |
| n=8 | 2.401s | 0.303s | **7.92x** |

（证据: task2_parallel_bench.json；与冻结 P2 E_execution serial 1.30s/parallel 0.80s 一致）

## 设计映射（不破坏冻结）
1. **State→Skill 调用缓存**（AgenticCache 式）: 相同状态观测不重复触发 LLM skill 推理 → 命中直出
2. **独立 skill 并行触发**（ThreadPool）: 同一状态多 skill（OG/SD/EG 若无依赖）并行推理
3. **事件触发重规划**（Tool-Aligned 式）: skill 返回 feedback 时 Brain 才重规划（非每步轮询）

## 诚实局限
- **ALFRED 单 agent 串行动作**（grasp→place 依赖时序）：并行收益是**调度/延迟侧**（免重复 LLM 调用），非任务 wall-clock 主增益
- 真实任务并行需场景支持（多目标并行可达）；EB/AW 评测中 skill 有数据依赖，并行化收益有限——如实报告

## STATUS: DONE（framework 验证 7.92x；真实任务并行受限于 ALFRED 时序，诚实标注）
