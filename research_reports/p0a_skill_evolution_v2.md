# P0a Report v2 — Skill Evolution with LLM Reflection

**Date:** 2026-08-08
**分支:** research/skill-evolution
**Hypothesis:** LLM reflection（SkillOpt 式 optimizer）能从失败轨迹生成有效候选规则，经 validation gate 稳定提升。

## 做了什么
在 v1（规则启发式）基础上，用 **FT Qwen 作为 reflection LLM**：对缺失失败 family，Qwen 分析「该目标最可能在哪」→ 生成候选规则 → validation gate（严格提升才接受）。

## 结果
| 版本 | seed 比例 | test 分数 | evolution |
|------|----------|-----------|-----------|
| v2 seed 60% | 60% | 0.700 | 全 reject（tie）|
| v2 seed 40% | 40% | 0.632 | 全 reject（tie）|

**LLM reflection 正确**：`somewhere knife` → Qwen 生成 `target DiningTable`（**正确**——刀应在餐桌）。
**gate 行为正确**：validation tie（0.667→0.667）被严格拒绝。

## 为什么无提升（诚实分析）
1. **validation 无区分度**：197 训练样本 split 后，validation（~60）对某些 family 无失败样本 → 加规则不改 validation 分数 → tie → reject
2. **family 提取噪音**：`somewhere knife`（2 词拼接）与真实目标 `somewhere to put knife` 不完全匹配，seed 已覆盖主要 family

## 失败原因
- **数据量限制**：SkillOpt 需要每 family 在 train/validation 都有足够样本；197 样本对规则演化偏小
- **gate 无提升时的正确行为**：拒绝——这证明 gate 机制有效（不盲从候选），但无法展示「提升」

## 结论
- ✅ **Skill Evolution 框架可运行**：rollout→reflection（LLM）→bounded edit→validation gate→accept/reject 全链路工作
- ✅ **LLM reflection 生成正确候选**
- ✅ **gate 严格拒绝 tie**（SkillOpt 式，regression=0）
- ❌ **小数据下无统计提升**——数据量不足以让 evolution 展示收益

## 下一步
- 用更大 EG 数据（全部 episode 提取）+ 每 family 充足 validation 样本
- 或转向 **Adapter 规则演化**（det_query→canonical 映射），该任务有更多失败样本
- P0a 结论：**机制验证成立（框架+gate+reflection），数据量限制记录为负结果**

## 版本
- tag p0a-evolution-v1（规则版）
- commit 待 v2 固化
