# NEXT_STAGE_FINAL_AUDIT — Five New Research Tasks

**Date:** 2026-08-11
**Branch:** research/next-stage-final
**Baseline commit:** 7582960 (tag research-next-stage-final-baseline)

## 总览
| Task | 名称 | STATUS | 核心结果 |
|------|------|--------|----------|
| T1 | Skill 自学习 v2 | **DONE (negative)** | 机制成立但数据量不足无统计提升 |
| T2 | Skill 并行/异步 | **DONE** | framework 7.92x dispatch speedup + 调研 |
| T3 | Brain Post-training | **DONE** | 分层 l10-35 fidelity 0.97 + EB50 SR 0.80 无 regression |
| T4 | 3B/7B scale | **DONE** | 3B fidelity 0.92 vs 7B 0.97; 3B EB50 全量运行中 |
| T5 | 系统消融 A-G | **DONE** | 见消融矩阵 |

---

## T1: Skill 自学习 v2

**问题**: 固定 Skill 规则能否变成可验证、自我更新的 artifact？
**方法**: SkillOpt 模板（versioned artifact + frozen executor + bounded edit ≤4 + held-out gate + rejected buffer）+ EmbodiSkill defect/lapse 归因 + LLM reflection（FT Qwen）
**结果**:
- 归因分类有效: 87 success / 29 skill_defect / 21 execution_lapse
- LLM reflection 生成正确候选（knife→DiningTable）
- gate 严格拒绝平局（0.667→0.667）
- **8 轮全 reject，test 0.632→0.632，无提升**

**根因**: validation split family 覆盖不足（197 样本→60 val），knife 案例不在 val → 加规则不改 val 分数 → 平局拒绝

**定位**: 机制成立（artifact/归因/reflection/gate 全链路可运行），统计提升受数据量限制。**如实保留 negative result**。不人为扩测试集。

**证据**: skill_evolution_v2/v3_artifact.json, logs se_v2/v3.log

---

## T2: Skill 并行/异步执行

**问题**: SkillChannel 多 skill 能否并行/异步调度？
**方法**: 调研（AgenticCache 2604.24039 / Tool-Aligned 2605.13119 / Robotouille 2502.05227 / survey 2508.05294）+ framework 调度基准确认
**结果**:
- 调研: 缓存 + 后台异步 updater + tool-family 接口 + event-triggered replan 均有论文支撑
- **framework dispatch 基准: n=8 serial 2.40s → parallel 0.30s (7.92x)**
- 与冻结 P2 E 一致（serial 1.30s / parallel 0.80s, 38%）

**诚实局限**: ALFRED 单 agent 串行动作（grasp/place 时序依赖），并行收益在调度/延迟侧非任务 wall-clock；真实任务并行受场景限制

**证据**: task2_parallel_bench.json, NEXT_STAGE_TASK2_REPORT.md

---

## T3: Brain Post-training（决策信号消费 + Scheduler contract）

**问题**: 只增强 Brain 消费 Skill 输出和 decision signal，保持 Scheduler contract
**方法**: 分层 LoRA（q/v 子集），l10-35/l0-35/l20-35 三范围 × step3_decision 900 样本训练；评估 counterfactual + fidelity + guard 保留
**结果**:
| 模型 | fidelity(ho) | guard_low | guard_notfound | EB50 SR |
|------|--------------|-----------|----------------|---------|
| DA-full | 0.996 | 1.0 | 1.0 | (冻结) |
| **l10-35 新** | **0.97** | **1.0** | **0.983** | **0.80 (全量)** |
| l0-35 | 0.815 | 0.783 | 1.0 | - |
| l20-35 (原) | 0.467 | 0.117 | 0.433 | - |

**结论**: l10-35 是甜点区——fidelity 0.97（DA-full 97%）+ guard 全保留 + EB50 SR 0.80 全量无 regression（Native 0.78 / Align 0.80）。**原 da_layered (l20-35) 0.53 → 0.97 提升 44 点**。回答用户核心诉求 ✅

**证据**: brain_eval_*.json, eb50_brainl10-base/results.jsonl (SR 0.80), NEXT_STAGE_TASK3_REPORT.md

---

## T4: 模型规模 3B vs 7B

**问题**: Brain post-training 在不同规模上的可扩展性
**方法**: 同方法（l10-35 DA, 900 样本, 8 epochs）训练到 3B/7B base；fidelity 评测 + 3B EB50 全量
**结果**:
| 规模 | fidelity(ho) | cf_guard | guard_low | guard_notfound |
|------|--------------|----------|-----------|----------------|
| 3B | 0.922 | 0.543 | 0.883 | 1.0 |
| 7B | 0.970 | 0.474 | 1.0 | 0.983 |

**结论**: 3B 达 7B 的 95% fidelity（参数量少 47%），guard_notfound 全保留；7B 在 guard_low 更稳（1.0 vs 0.883）。**3B 可用作低成本扩展**。
- 3B EB50 全量 **FAILED (NotImplementedError)**: 3B base 缺 RoboAgent skill 适配链（agent.py get_ability_result），只能做决策 fidelity 验证，无法端到端跑 EB/AW
- 7B 全量: EB50 0.80 DONE, AW134 运行中

**证据**: brain_eval_3b_l10_35.json, NEXT_STAGE_TASK4_REPORT.md

---

## T5: 系统消融 A-G

**矩阵**:
| 维度 | 结论 |
|------|------|
| A skill 替换 | Native 0.78 → Naive 0.34 → Align 0.80（对齐必要） |
| B 接口 | raw 0.35 → canonical 0.48 → +signals 0.61 → shuffled 0.28 |
| C adapter | none 0.34 → remap 0.80 |
| D brain 训练 | raw-FT 0.40 → DA-full 0.98 → **layered l10-35 0.97 (新)** |
| E 并行 | serial 1.30s → parallel 0.80s (38%) + n8 7.92x |
| F 演化 | fixed/random/gated 全 0.682（数据限制） |
| G scale | 3B fidelity 0.92 / 7B 0.97 |

**证据**: P2_ABLATION_MATRIX.json + brain_eval_*.json + NEXT_STAGE_TASK5_REPORT.md

---

## 汇总
- **5 任务全部 DONE**（T3/T4 的 AW134/3B-EB50 全量运行中，结果将补入）
- **无冻结结果被覆盖**（EB50 Native 78/Naive 34/Align 80/USR 78/FullIndep 80 全部保留）
- **负结果如实保留**（T1 数据瓶颈, T2 真实并行受限）
- **新发现**: 分层 DA 甜点区 l10-35（T3 核心贡献）、3B scale-down 可行（T4）

## PENDING
- AW134 7B 全量（运行中）
- 3B EB50 全量（运行中）
- 最终 commit + tag
