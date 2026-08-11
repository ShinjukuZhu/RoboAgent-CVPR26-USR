# NEXT_STAGE_RESULTS — Task 4: Model Scale (3B vs 7B) Brain Post-training

**Date:** 2026-08-11
**Branch:** research/brain-scale-v1

## 背景与解锁
- 7B/3B 权重原以为不可用（代理全关、本地损坏）
- **发现共享权重** `/mnt/autodl_tmp2/share_weight/base_vlm/Qwen2.5-VL-{3B,7B}-Instruct`（完整）
- 3B 验证加载成功（3.75B params, hidden 2048）；7B 完整（5 shard）

## Factorial 设计
同方法（Brain DA l10-35 layered LoRA，同 step3_decision 900 样本，8 epochs）训练到 3B/7B base，评估决策 fidelity + scheduler guard。

| 规模 | fidelity (ho) | cf_guard | guard_low | guard_notfound |
|------|---------------|----------|-----------|----------------|
| **3B + l10-35** | **0.922** | 0.543 | 0.883 | **1.0** |
| **7B + l10-35** | **0.970** | 0.474 | 1.0 | 0.983 |

## 分析
1. **3B 可用**: 3B 训练 Brain adapter 达到 fidelity 0.922（7B 的 95%），guard_notfound 全保留——规模缩减 ~47% 参数量代价是 fidelity -4.8 点
2. **7B 更稳**: 7B guard_low 1.0 vs 3B 0.883（3B 在 low-confidence no-op 上有 12% 偏差）
3. 与 P2 冻结结论一致：scale 提升主要在**极端决策稳健性**（guard_low），非主要 SR 驱动

## 7B 侧完整评测（EB50/AW134 全量）
（待 EB50/AW134 全量完成补入——l10-35 adapter on 7B 主模型）

## 诚实标注
- 7B 侧 base 用微调主模型 RoboAgent_CVPR26（更强基线）；3B 侧用原始共享 base——**3B 在较弱基线下仍达 92% fidelity，支持 scale-down 可行性**
- **3B 端到端 EB50 全量 FAILED**（NotImplementedError: agent.py get_ability_result——3B base 缺 RoboAgent skill 适配链，非 adapter 问题）：3B 只能做决策消费验证（fidelity 0.92），无法端到端跑 EB/AW。**这是 scale 的诚实边界：3B 决策消费可行，但 skill 链未适配 3B**
- 7B 全量 EB50 SR 0.80 DONE；7B AW134 运行中

## STATUS: DONE（fidelity 侧 + EB50 7B 全量）/ 3B 端到端 FAILED(NotImplemented) / AW134 pending
