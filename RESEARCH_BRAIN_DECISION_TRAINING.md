# Research — Does the baseline (fine-tuned Qwen) train the "decision" part? What data?

> 调研问题（学长 Task 2）：RoboAgent 的 baseline（微调 Qwen2.5-VL-3B）是否训练了"decision"（决策/调度/动作选择）这部分？它用了什么数据？
>
> 结论一句话：**官方 RoboAgent 的微调 Brain 确实训练了完整的决策链路（调度 scheduler + 动作选择 LPM + 感知 skill 的 canonical 词表），但训练代码与数据均未开源**；本文通过 checkpoint 身份 + 代码逆向 + 运行日志推断其监督来源，并确认"决策语义"并未显式建模（无 reward / 无 signal 监督）。

**Date:** 2026-08-08
**Scope:** baseline = RoboAgent 官方微调 Qwen（`woyut/RoboAgent_CVPR26`，全量微调，非本项目训练产物）

---

## 1. 结论速览

| 子问题 | 结论 |
|--------|------|
| baseline 是否训练了调度/决策？ | ✅ 训练了。微调 Brain 能输出 scheduler 的 `Think→Query: N. skill(args)` 协议 + LPM 的原子动作列表（Base Qwen 做不到）|
| baseline 是否训练了"decision signal"（置信/不确定性）？ | ❌ 没有显式建模。微调 Brain 无 reward / 无 signal 监督，不会根据 found/confidence 改变行为（反事实实验 0/35 signal-sensitive）|
| 用了什么数据？ | 官方：BC ~640k + DAgger ~690k + EIPO ~25k（作者声明，未开源）。数据源自 ALFRED 风格长程轨迹 + 特权场景图(SG)/分割掩码(mask) 监督 |
| 可复现吗？ | ❌ 训练代码/数据均未发布（服务器全盘搜索无 dagger/eipo/train 代码）|
| 对"Skill 替换"意味着什么？ | 微调 Brain 学的是 **RoboAgent 特有输出契约 + canonical 词表**（如 CellPhone/KeyChain），不是通用语义；替换 Skill 时 label 分布不匹配 → 需要 Adapter/USR 对齐 |

---

## 2. Checkpoint 身份（已核实）

| 项 | 证据 |
|----|------|
| Fine-tuned Brain | HF `woyut/RoboAgent_CVPR26`，全量微调权重（7.6G），mtime 2026-07-23（早于本项目）|
| Base Qwen | 官方 `Qwen2.5-VL-3B-Instruct.git`（git clone + LFS，3.98G+3.53G），加载验证 `LOAD_OK qwen2_5_vl 3.75B` |
| 是否含 LoRA | FT 目录无 adapter 残留（官方全量微调，已合并）；本项目 LoRA 独立存放 |

> 关键：官方微调权重是**下载**的，本项目从未重训 Base→FT；本项目所有训练 = 在官方 FT 上叠加 LoRA（r=16）。

---

## 3. "decision" 部分是否被训练？——三层证据

### 3.1 调度（scheduler）：✅ 训练了

- 微调 Brain 能输出：`Think: ... Query: 1. exploration_guidance(ladle)`（编号 skill 调用协议）
- Base Qwen 输出自然语言 `Query: How do I rinse off a ladle?`（无编号、不调用 skill）→ 调度协议是**微调习得**
- 证据：Base/FT 六层 probe（L4 scheduler）：FT 有编号 Query，Base 无

### 3.2 动作选择（LPM/manipulation planner）：✅ 训练了

- FT 输出 `[pick up the Ladle 1]`（精确动作），Base 输出近似文本
- 训练-推理失配来源：FT 学到"OG label → 动作"映射，OG label 变化会让 Brain 困惑

### 3.3 决策信号（found/confidence/uncertainty）：❌ 未显式训练

- **反事实实验**：固定 object.class+target，只变 found/confidence
  - raw-FT：0/35 signal-sensitive（所有信号状态都输出相同动作）
  - DA-FT（本项目额外训练后才）：34/35 signal-sensitive，98% decision accuracy
- 结论：**baseline 微调 Brain 的"决策语义"没有训练**——它把 skill 输出当普通文本消费，不区分置信/不确定性

---

## 4. 用了什么数据？（官方声明 + 逆向推断）

### 4.1 官方三阶段（作者声明，未开源）

| 阶段 | 数据量 | 内容 | 监督来源（推断）|
|------|--------|------|----------------|
| Stage1 BC (Expert-SFT) | ~640k | 行为克隆 | ALFRED 风格 expert 轨迹；scheduler 用 expert 计划 CoT 模板；LPM 用 expert low-level 动作序列 |
| Stage2 DAgger-SFT | ~690k | 自我 rollout + 开放词汇改写 | 在策略分布下聚合专家纠正，缓解分布漂移 |
| Stage3 RFT/EIPO | ~25k | 强化 scheduler | 无公开同名方法（EIPO 需定义）；推测为对 scheduler 的强化/拒绝采样 |

### 4.2 从代码逆向的监督结构

- **Scheduler**：`prompt_ct(task_instruction, core_history)` → 输出 Query 序列；监督 = expert 计划的 CoT（子目标列表 + 已完成 + 下一步）
- **LPM**：`prompt_lpm(holding, at, scene_description, subgoal)` → 输出原子动作；监督 = expert 轨迹 low-level 动作
- **OG**：输出 label/found 进 core_history（`"Grounding feedback: the target object (ladle) is found at Ladle 1"`）；监督 = 特权分割掩码(segmentation mask) 的 bbox
- **SD/EG**：监督来自特权场景图(SG)（物体位置/关系）

### 4.3 数据是否可用？

| 数据 | 状态 |
|------|------|
| RoboAgent 官方训练数据 (640k/690k/25k) | ❌ 未发布 |
| ALFRED 数据集 (json_2.1.1) | ✅ 本地有 6374 训练轨迹（可作为替代训练基础）|
| 特权 SG/mask | ⚠️ 需在模拟器重放 expert 轨迹抓取 |

---

## 5. 关键机制证据

| 证据 | 值 | 说明 |
|------|-----|------|
| det_label ≠ 微调 Qwen label | 50% (93/187) | 微调 Brain 学的是 canonical 词表（非 detector 原始输出）|
| abstain (no_det_false) | 6.4% | 微调 Brain 学会保守 abstain（对"没找到"的默认行为）|
| Base Qwen 直接跑 EB | 崩溃（scheduler 空输出）| 未微调无调度协议 |
| LoRA canonicalization | 61% → 89% | 本项目在 FT 上微调可达 |
| 反事实 signal-sensitive | raw-FT 0/35 → DA-FT 34/35 | baseline 未消费 decision signals，需额外训练 |

---

## 6. 对"模块拆分成 Skill 串联"的含义

1. 微调 Brain 的能力 = **RoboAgent 特有输出契约 + canonical 词表 + 调度协议**（Base 无这些）
2. 替换 Skill（如 OG 用 GDINO）时，**label 词表/格式与 Brain 期望不匹配** → 直接替换崩（EB 78%→34%）
3. 需要 **Decision-Compatible Adapter**（对齐 label）+ **USR**（统一语义接口 + 决策信号）+ 可选 **decision-aware training**（让 Brain 学会消费信号）
4. 官方未开源训练 → 本项目用"冻结官方 FT 作为 Brain + 外部 FM 作为 Skill + Adapter/USR 桥接"的方案，不重训 Brain

---

## 7. 已确认 vs 合理推断

| 项 | 已确认 | 合理推断 |
|----|--------|---------|
| FT 来源 | ✅ HF 官方全量微调 | — |
| 官方训练未开源 | ✅ 服务器无训练代码 | — |
| scheduler/LPM 被训练 | ✅（Base/FT probe 差异）| — |
| decision signal 未训练 | ✅（反事实 0/35）| — |
| BC/DAgger/EIPO 具体数据 | — | 源自 ALFRED + SG/mask 监督 |
| EIPO 定义 | — | 无公开出处 |
| 各阶段学到什么 | scheduler 调度/LPM 动作映射（有证据）| 具体损失函数/超参 |
