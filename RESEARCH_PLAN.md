# Research Plan — RoboAgent-CVPR26-USR Next Phase

> 最终研究问题：
> **当具身 Agent 的感知 Skill 可以被独立替换后，如何让 Skill 可组合、可并行、可从执行经验中持续改进，并让 Brain 持续适应这些变化？**
>
> 已冻结基础：Adapter 解决 contract mismatch（Naive 34%→Align 80%）；USR/SkillChannel 提供统一可审计通信。本阶段验证三个扩展：Skill 自我改进、独立 Skill 并行、Brain 后训练适应。

**Date:** 2026-08-08
**Branch 基:** topconf-contract-evolution@27a40fb (tag research-base-v1)

---

## P0 — 优先级最高

### P0a. Skill Evolution Prototype（EG/Adapter 自学习）

| 项 | 内容 |
|----|------|
| **Hypothesis** | 随着失败轨迹积累，validation-gated 的 Skill 自学习（EG/Adapter）能稳定提升 SR，且无明显 regression |
| **可学习对象** | EG Skill 的 instruction/规则 + Adapter 映射规则（**不碰 Brain 权重**）|
| **反馈** | AW/EB episode 的 trajectory：Skill 输入/输出/下游是否成功/失败类型/Brain action/SR |
| **验证机制** | 固定 held-out validation episodes：候选 vN+1 的 validation SR **严格大于** 当前才接受（tie 拒绝）；regression 阈值 0 |
| **Baselines** | ① 固定 Skill ② 随机/无门控修改 ③ validation-gated evolution |
| **数据** | 已有 EG/Adapter failure trace（eg_train/eg_test + AW/EB runs）|
| **代码入口** | experiments/skill_evolution/v1/ |
| **GPU** | 1× (推理用，无训练) |
| **预计时间** | 1-2 天 |
| **Metric** | 每轮：Skill version / 修改内容 / validation SR / regression episode / accept-reject 原因 |
| **停止条件** | 5-10 轮后 SR 单调或总体稳定上升，无明显 regression；否则记录负结果 |
| **输出** | skill_evolution_v1_report.md + 每轮 commit/tag |

**机制参考（SkillOpt 式）**：rollout → reflection（判断 perception/contract/exploration/downstream/execution lapse）→ bounded edit → held-out validation gate → accept/reject + rejected buffer。

### P0b. Brain Post-training（Brain-centric）

| 项 | 内容 |
|----|------|
| **Hypothesis** | 针对新 Skill 输出 + decision signals 的 Brain 后训练（LoRA/分层），提升决策信号敏感度与 skill-selection，且不破坏 Scheduler contract |
| **数据** | Brain-centric dataset 五类：正常成功 / Skill 替换后成功 / failure→recovery / found-confidence 反事实配对 / 长历史多 Skill |
| **训练** | LoRA/分层 LoRA（layer 20-35 起点），loss 只监督 action token |
| **Baselines** | 原 FT Brain / 原 FT+新 Skill 数据 / Decision-aware / 不同 LoRA layer 范围 |
| **对照** | shuffled-signal / canonical-only / canonical+signals |
| **Metric** | scheduler contract acc / skill-selection acc / action acc / decision-signal sensitivity / counterfactual acc / AW+EB SR |
| **GPU** | 1× 训练 + 1× 评测 |
| **预计时间** | 2-3 天 |
| **输出** | brain_posttrain_v1_report.md |

---

## P1 — 优先级一

### P1a. Parallel Skill Scheduler

| 项 | 内容 |
|----|------|
| **Hypothesis** | 无数据依赖的 Skill（OG/SD/EG 可 overlap）并行执行，在 SR 不下降前提下降低平均 episode time |
| **实现** | ParallelSkillScheduler：dependency graph（必须等待/可提前/独立）+ stale result discard（检查 USR version）|
| **Baselines** | Native serial / Final serial / Parallel |
| **测试** | AW+EB，先 10 ep smoke，再 50 ep |
| **Metric** | wall-clock total / avg episode time / per-Skill latency / GPU util+mem / overlap ratio / stale count / SR/GCR |
| **停止条件** | 若 Parallel SR=Serial SR 且 time 降 → 成立；若 time 升 → 分析 GPU contention；若 SR 降 → 查 stale/version，不修改结果 |
| **输出** | parallel_v1_report.md |

### P1b. 7B Brain Scale

| 项 | 内容 |
|----|------|
| **Hypothesis** | 7B Brain 提升决策质量；区分「更强 reasoning」vs「Skill interface」；权衡成本 |
| **配置** | Base-FT 3B / 7B / post-trained 3B / 7B（+Native/Final Skill 输入）|
| **Metric** | 参数量 / 显存 / 推理时间 / tokens/s / latency / AW+EB SR/GCR / contract acc |
| **报告** | 性能/成本 trade-off，不因 7B SR 高直接说更好 |
| **输出** | brain_scale_v1_report.md |

---

## P2 — 系统性消融

| 维度 | 对照 | 关键 |
|------|------|------|
| A. Skill replacement | Native/Naive/Align | 冻结已有 |
| B. Interface | Raw/Canonical/USR/USR+signals | 冻结部分 |
| C. Adapter | None/Format/Decision-aware | 冻结部分 |
| D. Brain training | FT/Skill-aware/Decision-aware | P0b |
| E. Execution | Serial/Parallel | P1a |
| F. Evolution | Fixed/Unvalidated/Validation-gated | P0a |
| G. Scale | 3B/7B | P1b |
| H. Composition | OG/OG+EG/OG+SD/OG+EG+SD | 冻结+新增 |

- 每消融先定义 hypothesis + metric 再跑；关键 50 ep，探索性 10-20 ep
- paired binary 用 exact McNemar；Wilson 95% CI

---

## 版本管理（所有实验）

- 实验前：git status + rev-parse HEAD + 新 branch
- config → configs/；run_manifest.json（git/model hash/dataset hash/versions）；episode_manifest.jsonl
- 结果 JSON+CSV+Markdown；新结果用新目录 experiments/<name>/vN
- 好结果即 git add+commit+tag（如 skill-evolution-v1-best）；失败实验也保存

## 执行顺序

P0a → P0b → P1a → P1b → P2。每阶段完成后生成 report.md（做了什么/为什么/结果/失败原因/下一步）。
