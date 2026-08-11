# TOPCONF_OPTIMIZATION_REPORT — 优化实验综合报告

**Date:** 2026-08-08
**研究问题:** 当具身 Agent 的感知 Skill 可独立替换后，如何让 Skill 可组合、可并行、可从执行经验中持续改进，并让 Brain 持续适应？
**冻结基础:** Adapter 解决 contract mismatch (34→80)；USR/SkillChannel 提供统一可审计通信。

---

## P0a — Skill Evolution（EG/Adapter 自学习）

**做了什么:** SkillOpt 式循环（rollout→LLM reflection→bounded edit→validation gate→accept/reject），RuleSkill 作为可版本化可学习对象（不碰 Brain 权重）。
**结果:**
- v1 规则启发式: gate 正确（tie 拒绝），但启发式 guess 弱 → 无提升（0.682）
- v2 LLM reflection: Qwen 正确生成候选（knife→DiningTable），但 197 样本 validation 无区分度 → 全 reject
**结论:** **机制验证成立**（框架+LLM reflection+gate 全链路），**数据量限制**（validation 无 family 区分度）记录为负结果。
**失败原因:** 小数据（197）下 validation split 的 family 覆盖不足；启发式 guess 无法从失败推导正确规则。

## P0b — Brain Post-training

**做了什么:** 统一评估 Brain 模型（原 FT / DA-full / DA-layered）在 175 反事实 + scheduler contract。
**结果:**
| Brain | counterfactual_acc |
|-------|-------------------|
| raw-FT | 40% |
| **DA-full**（全层 LoRA）| **98%** |
| DA-layered（layer 20-35）| 53% |
**结论:** **Decision-aware post-training 显著提升信号消费**（40→98%），分层折衷（53% 保 scheduler）。shuffled/canonical/signals 对照已有（USR≈canon+signals）。

## P1a — Parallel Skill Scheduler

**做了什么:** dependency graph（EG/OG 独立可 overlap）+ asyncio 并行 + stale discard + serial baseline。
**结果:** **parallel 0.80s vs serial 1.30s（38% 时间节省）**，dependency 正确，stale 丢弃正确，serial 全串行正确。
**结论:** 无数据依赖的 Skill 可安全 overlap（计算 overlap 非盲目异步），stale 结果被丢弃。**框架验证成立**（真实 agent 接入留作后续）。

## P1b — 7B Brain Scale

**结果:** **BLOCKED**——所有代理端口（7892/7890/7897）拒绝连接，7B 权重无法下载。
**结论:** 资源受限，记录为负结果。3B 已有完整证据链。

## P2 — 系统性消融

| 维度 | 结果 |
|------|------|
| A. Skill replacement | Native 78 / Naive 34 / **Align 80** |
| B. Interface | raw 35 / canonical 48 / **canon+signals 61** / USR 58 / shuffled 28 |
| C. Adapter | none 34 / **remap-v3 80** |
| D. Brain training | raw-FT 40 / **DA-full 98** / DA-layered 53 |
| E. Execution | serial 1.30s / **parallel 0.80s** |
| F. Evolution | fixed 0.682 / random 0.682 / gated 0.682（数据限制）|
| G. Scale | 3B ✓ / **7B blocked** |
| H. Composition | OG+USR 78 / **OG+EG+SD+USR 80** |

## 关键增量结论

1. **Brain 能通过 post-training 适应 decision signals**（DA-full 98% 反事实）——回答「Brain 持续适应」
2. **无依赖 Skill 可安全并行**（38% 时间节省，stale 丢弃）——回答「可并行」
3. **Skill 自学习机制成立但受数据量限制**（框架+gate+reflection 正确，197 样本无统计提升）——回答「可自我改进」的部分验证
4. **7B 受阻**（网络限制）——资源边界

## 实验事实 vs 机制解释 vs 合理推断

- **实验事实**: 消融矩阵 A-H；DA-full 98%；parallel 38%；evolution 无提升
- **机制解释**: Decision-aware 训练使 Brain 消费信号；dependency graph 使 Skill 安全 overlap
- **合理推断**: 更大数据下 evolution 可能提升（SkillOpt 范式）；真实 agent 并行需 GPU worker

## 产物

- experiments/{skill_evolution,parallel,brain_posttrain,brain_scale,ablation}/v1/
- reports/P2_ABLATION_MATRIX.json + 各阶段 report
- tags: p0a-evolution-v1/v2, research-p0b-p1a-p2
- EXPERIMENT_REGISTRY.csv（完整记录）
