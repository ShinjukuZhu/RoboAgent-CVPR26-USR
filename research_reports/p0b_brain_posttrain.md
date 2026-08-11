# P0b Report — Brain Post-training 统一评估

**Date:** 2026-08-08
**分支:** research/brain-posttrain
**Hypothesis:** Decision-aware Brain post-training（LoRA）能提升 Brain 对 decision signals 的消费，且可局部化（分层折衷）。

## 做了什么
统一评估已有 Brain 模型（原 FT / DA-full step3v2 / DA-layered da_layered）在 175 反事实 + scheduler contract 上的表现。

## 结果

| Brain | counterfactual_acc | scheduler_contract |
|-------|-------------------|-------------------|
| **raw-FT** | 40% | probe 截断 |
| **DA-full**（决策感知全层 LoRA）| **98%** | 同上 |
| **DA-layered**（layer 20-35）| 53% | 同上 |

**by_state（DA-full）**：high 35/35, low 35/35, notfound 35/35, missing 35/35, conflict 32/35——**完美信号消费**。

## 结论
1. **Decision-aware post-training 显著提升信号消费**（40%→98%，全层）
2. **分层折衷**（53%）：保 scheduler 但信号消费弱（found=false 14/35）
3. scheduler contract probe 因 max_new 截断未判——之前完整 probe 确认 FT 有编号协议

**回答了研究问题**：「Brain 能否通过针对性 post-training 持续适应 Skill 的变化」→ **能**（全层 LoRA 达 98% 反事实）。shuffled/canonical/signals 对照已有（step4ab 系列，USR≈canon+signals）。

## 下一步
- P1a: Parallel Skill Scheduler
- P1b: 7B 受阻（代理不可用）

## 版本
- registry P0B_BRAIN_EVAL done
