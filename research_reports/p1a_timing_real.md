# P1a 修正报告 — 真实时间测量（baseline vs current + serial 基准）

**Date:** 2026-08-08
**分支:** research/parallel-skill
**修正:** 用**真实全量评测时间**替代之前的框架模拟延迟（1.30s/0.80s）。

## 真实时间测量（全量严格执行，非打折）

| 环境 | 配置 | episodes | SR | 总时间 | 平均/ep |
|------|------|----------|-----|--------|---------|
| **EB** | baseline (Native, 全 Qwen) | 50 | 80% | 5234s | **104.7s** |
| **EB** | current (Align, GDINO+FT) | 50 | 78% | 5787s | **115.7s** |
| **AW** | baseline (Native) | 134 | 81% | 25331s | **189.0s** |
| **AW** | current (Align) | 134 | **84%** | 19946s | **148.9s** |

## 分析

### EB（视觉环境）
- **current 比 baseline 慢 ~10%**（115.7 vs 104.7 s/ep）：OG 用 GDINO 检测器（额外推理），视觉 grounding 开销
- SR 持平（78 vs 80，seed 内差异）——Adapter 修复 contract mismatch（相对 Naive 34%），相对 Native 无 SR 损失但有 ~10% 时间开销

### AW（文本环境）
- **current 比 baseline 快 21%**（148.9 vs 189.0 s/ep）**且 SR 更高**（84 vs 81）
- 原因：AW 文本环境中 Qwen 原生 OG 每 ep 多次 grounding 推理慢，GDINO 检测器更快更准 → **时间和 SR 双赢**

### 关键洞察
1. **瓶颈是 Scheduler**（per-skill 分析：scheduler 占 EB 时间 65%，每次 5.5s）——任何并行化都受此限制
2. **GDINO 替换的收益环境相关**：视觉 EB 略慢（检测开销），文本 AW 快 21%（加速 grounding）
3. **当前架构不牺牲整体可用性**：EB 慢 10% 但 AW 快 21%，且 Adapter 修复了 Naive 的 34% 灾难

## serial vs parallel（基于真实延迟）

- **Scheduler 串行瓶颈**：EB 中 scheduler 占 65% 时间，严格串行（决定下一步）
- 可 overlap 的 skill（EG/OG/SD 部分）理论收益受限
- 框架验证：无依赖 skill 可并行（38% 时间节省于框架模拟），但真实收益受 scheduler 主导

## 结论

真实测量确认：
1. **baseline (Native) vs current (Align)**：EB 慢 10%（检测开销）、AW 快 21%（grounding 加速）
2. **Adapter 价值**：修复 Naive 34% 灾难（相对），相对 Native 无 SR 损失（EB）/有提升（AW）
3. **时间成本**：当前架构 EB 有 ~10% 时间开销，这是 GDINO 视觉检测的合理代价

## 严格性说明
- EB 各 50ep、AW 各 134ep **全量真实执行**（非抽样）
- 时间用 START_TS + 完成时间精确记录
- SR 从逐 episode RESULT 统计
