# P1a Report — Parallel Skill Scheduler 框架

**Date:** 2026-08-08
**分支:** research/parallel-skill
**Hypothesis:** 无数据依赖的 Skill 并行执行，在 SR 不下降前提下降低时间。

## 做了什么
实现 ParallelSkillScheduler：
- **dependency graph**：eg/lpe/og/sd/lpm 的依赖（EG→LPE→OG→SD→LPM），EG/OG 独立可 overlap
- **并行执行**：asyncio 并发跑 ready 任务（无依赖冲突）
- **stale discard**：observation snapshot 变化时丢弃过期结果（不喂给 Brain）
- **serial baseline**：严格串行

## 结果（框架测试）
- **parallel 0.80s vs serial 1.30s（38% 时间节省）** ✅
- dependency graph 正确（EG/OG 独立、LPM 依赖 SD、LPE 依赖 EG）✅
- stale discard 正确（observation 更新→丢弃）✅
- serial 正确全串行（1.30s）✅
- debug 确认 6 个 skill 全 done ✅

## 结论
**框架验证成立**：无数据依赖的 Skill（EG/OG 等）可安全 overlap（38% 时间节省），依赖链（LPM 等）保持顺序，stale 结果被丢弃。这证明「计算 overlap」而非「盲目异步」是正确路线。

**限制**：这是框架级验证（模拟延迟）。真实 agent 中的并行需将 Skill 调用改为 worker（独立 GPU 或多进程），且 GPU 共享时需测量 contention。SR 等价性需端到端验证（框架保证 stale discard，但未跑真实 episode）。

## 下一步
- 真实 agent 中接 ParallelSkillScheduler（worker + GPU 分配）
- EB/AW smoke test（10 ep）+ 50 ep 验证 SR 等价 + 时间收益

## 版本
- registry P1A_SCHED_FRAMEWORK done
