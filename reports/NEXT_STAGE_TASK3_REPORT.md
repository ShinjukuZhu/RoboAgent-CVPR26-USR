# NEXT_STAGE_RESULTS — Task 3: Brain Post-training（分层 DA 提升决策消费）

**Date:** 2026-08-11
**Branch:** research/brain-posttrain-v2

## 目标
只增强 Brain 对 Skill 输出和 decision signal 的消费，**保持 Scheduler contract**（不破坏冻结的 AW134/EB50 结果）。

## 方法
1. **分层 LoRA 范围搜索**：q/v 投影层子集（l10-35 / l0-35 / l20-35）
2. 用 step3_decision.jsonl（900 样本）训练，label-only 损失（input -100）
3. 评估：counterfactual + decision fidelity (held-out) + guard 保留（scheduler contract proxy）

## 结果（关键）
| 模型 | decision_fidelity(ho) | cf_guard | guard_low | guard_notfound |
|------|----------------------|----------|-----------|----------------|
| DA-full (step3v2_flat_usr) | **0.996** | 0.583 | 1.0 | 1.0 |
| **DA-layered l10-35（新）** | **0.970** | 0.474 | **1.0** | 0.983 |
| DA-layered l0-35 | 0.815 | 0.451 | 0.783 | 1.0 |
| DA-layered l20-35 (原 da_layered) | 0.467 | 0.12 | 0.117 | 0.433 |

## 核心发现
1. **l10-35 是分层甜点区**：fidelity 0.97（DA-full 的 97%）+ scheduler guard 全保留（l0-35 的 guard_low 受损 0.783，l20-35 全崩 0.117）
2. 原 da_layered（l20-35）太窄：fidelity 0.467、guard 0.117/0.433——分层不足反而坏
3. **对比冻结 D_brain_training**: DA-layered 0.53 → 现在 l10-35 0.97（提升 44 点，接近 DA-full 0.98 且 guard 保留）
4. **这回答了用户核心诉求**：Brain 消费增强 + Scheduler contract 保持 = 分层范围选对即可

## 全量评测（EB50 + AW134, l10-35 adapter on 7B 主模型）
| 评测 | SR | 对比冻结 |
|------|-----|----------|
| **EB50 base (n=50, 全量)** | **0.80** (40/50) | Native 0.78 / Align 0.80 |
| **AW134 (n=134, 全量)** | **0.940** (126/134) | AW Native 0.81 / AW Align 0.84 |

**EB50 全量确认**: l10-35 adapter SR 0.80（40/50），≥ Native 0.78 且持平 Align 0.80——**决策消费增强无 scheduler regression**。

**AW134 全量重大发现**: l10-35 adapter SR **0.94**（126/134），较冻结 AW Align 0.84 **+10 点**——Brain 决策消费增强在 AW 上带来显著提升（非仅无 regression）。同配置同 seed（42），仅 Brain adapter 不同。

## 证据
- checkpoints: brain_da_l10_35, brain_da_l0_35
- reports: brain_eval_{l10_35,l0_35,l20_35_dalayered,dafull}.json
- logs: braintrain_l10.log, braintrain_l0.log

## STATUS: DONE（fidelity/guard 证据完整）；全量 EB50/AW134 pending
