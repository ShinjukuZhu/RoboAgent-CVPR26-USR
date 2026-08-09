# 学长调研报告：SOTA 方案、RoboAgent 微调 Qwen 训练调研、模块拆分与 Skill 串联

**Date:** 2026-08-09
**对应任务:** ① SOTA 方案调研（工具+环境）② 调研 baseline 微调 Qwen 的训练部分（是否训练了 decision、用了什么数据）③ 拆成不同模块后如何与大脑通过 Skill 串联

---

## 一、SOTA 方案调研

### 1.1 代表性具身 Agent 系统

| 系统 | 结构 | Brain/Skill 分工 | 来源 |
|------|------|-----------------|------|
| **AutoRT** | VLM 场景理解 + LLM 指令生成 + 已有策略执行 | Brain(LLM 编排) + Skill(策略) | arXiv:2401.12963 |
| **GRID** | scene graph + LLM+GAT 指令分解 | Brain(LLM 规划) + 外部场景图 | arXiv:2309.07726 |
| **SayPlan** | 3D scene graph + LLM 规划器 + 反馈闭环 | Brain(LLM) + 结构化感知 | arXiv:2307.06135 |
| **Voyager** | LLM + 可增长 skill library（代码块）+ 自反思 | Brain(LLM) + Skill(代码库) | arXiv:2305.16291 |
| **RT-2 / OpenVLA / Octo** | 单一 VLA 直接输出动作 | **不分层 skill** | arXiv:2307.15818 / 2406.09246 / 2405.12213 |
| **SayCan** | LLM 候选 × skill value function 加权 | Brain(LLM) + 现成 skill | arXiv:2204.01691 |

**结论**：多数工作要么「单一 VLM 直接出动作」（RT-2/OpenVLA，无 Skill 层），要么「LLM 编排 + 现成 skill」（SayCan/Voyager，skill 不可替换训练）。**RoboAgent 属于罕见的「单一微调 VLM 承担全部 7 个 Skill 角色」**——skill 与 Brain 共享同一权重，这是其可替换性问题的根源。

### 1.2 Foundation Model 作为 Skill（工具与环境）

| 角色 | 候选模型 | 特点 |
|------|---------|------|
| OG（物体接地）| **Grounding DINO** (0.2B), **LLMDet**, OWL-ViT, Florence-2 | 开放词表检测，可插拔 |
| SD（场景描述）| **Florence-2** (0.77B), GPT-4V | prompt 统一多任务 |
| EG（探索方向）| Qwen2.5-VL 系 / 独立 LoRA | 空间-物体关系推理 |

**Benchmark**：ALFRED / EmbodiedBench (EB-ALFRED, EB-Habitat 等) / ALFWorld (文本版)。评测协议：SR / GCR / PLWSR。

---

## 二、调研 baseline（RoboAgent 微调 Qwen）的训练部分 — 重点

### 2.1 它是否训练了 decision 部分？—— 是，但决策被隐式耦合进整个微调

RoboAgent 用**单一微调 Qwen2.5-VL-3B** 同时承担决策（scheduler/LPM）和感知（OG/SD）角色。它的「决策能力」不是独立模块，而是**三阶段训练注入整个模型**：

| 阶段 | 数据量 | 内容 | 对 decision 的作用 |
|------|--------|------|-------------------|
| **Stage1 BC (Expert-SFT)** | ~640k | 特权 SG/mask 监督下行为克隆 | 学「任务→skill 调用序列」(scheduler CoT) + 「skill 输出→原子动作」(LPM) |
| **Stage2 DAgger-SFT** | ~690k | 自 rollout + 开放词汇改写 | 学「从失败反馈改计划」（决策纠错）|
| **Stage3 RFT/EIPO** | ~25k | 主要强化 Scheduler | 强化调度决策 |

**决策相关的监督来源**（audit_training_data.md）：
- **Scheduler 决策** ← expert 计划的 CoT 模板（子目标列表 + 已完成 + 下一步）
- **LPM/动作决策** ← expert 轨迹的 low-level 动作序列
- **OG/EG/SD** ← 模拟器特权信息（mask/场景图/物体关系）

**关键**：决策监督用的是 **expert 轨迹 + 特权模拟器信息**，不是「人类反馈」或「在线 RL 奖赏」。

### 2.2 用了什么数据？

| 数据 | 说明 |
|------|------|
| 官方训练数据 (640k/690k/25k) | ❌ **未开源**（GitHub 仅 inference 代码）|
| ALFRED json_2.1.1 | ✅ 本地有 6374 训练轨迹（可作替代）|
| 模拟器特权信息 (SG/mask) | ⚠️ 需在模拟器中重放获取 |

### 2.3 代码级证据（本次审计）

- 服务器 `find -iname "*dagger*" -o -iname "*eipo*"` → **无结果**：官方训练代码未开源
- `code/RoboAgent_CVPR26/` grep train/dagger/rl → **无训练代码**：只有 inference
- checkpoint `RoboAgent_CVPR26`（mtime 2026-07-23）= 官方下载的全量微调权重，**无 LoRA 残留**（本项目 LoRA 独立存放）

### 2.4 微调 Brain 到底「学到了 decision 的什么」

**已确认机制证据**：

| 证据 | 值 | 含义 |
|------|-----|------|
| det_label ≠ 微调 Qwen label | 50% (93/187) | 微调 Brain 学到了**自己的 canonical label 词表**（非检测器词表）|
| abstain (no_det_false) | 6.4% | 微调 Brain 学会**保守 abstain**（找不到就说不找到）|
| Base Qwen 直接跑 EB | **崩溃**（scheduler 空输出）| 未微调 Qwen **没有决策协议**（Think/Query）|

**→ 结论：baseline 的 decision 部分 = 通过 BC+DAgger+EIPO 全模型微调注入的「调度协议 + 动作映射 + label 词表 + 保守行为」。它不是一个可单独替换的 decision 模块，而是与整个模型耦合的隐式能力。**

### 2.5 这对我们的意义

正因为 baseline 的 decision 是「隐式耦合 + 特定词表 + 无独立接口」，所以：
1. 直接换 OG 检测器 → label 不匹配 Brain 词表 → **决策崩（34%）**
2. 需要 **Decision-Compatible Adapter** 把异构输出对齐回 Brain 词表 → **恢复 80%**
3. 需要 **USR** 把决策信号（found/confidence）显式化，供 Brain 消费（反事实 98%）

---

## 三、拆成不同模块后，如何与大脑通过 Skill 串联

### 3.1 现状：RoboAgent 的 7 个 Skill 都是同一权重里的「prompt 分派」

```
agent.py 主循环:
  get_qwen_action_raw()
    └─ ability_buffer 空? → get_core_result()  # Scheduler (prompt_ct)
         parse "Query:" → [[skill_name, args]]
    └─ pop → get_ability_result(name, args)     # 调对应 prompt
```

| Skill | prompt 契约 | 下游消费 |
|-------|------------|---------|
| Scheduler | prompt_ct(task, core_history) | 产出 skill 调用序列 |
| OG | prompt_og(图像, target) → label/bbox | 写 core_history |
| SD | prompt_sd(图像, label) | scene_description → LPM |
| EG | prompt_eg(objects) → 方向 | exploration_subgoal → LPE |
| LPM | prompt_lpm(scene, subgoal) | 原子动作 |

**Skill 间唯一共享状态 = `core_history` 字符串 + `last_grounding_label` + `scene_description`**——这就是「串联」的现状：**通过自由文本反馈，无结构化接口**。

### 3.2 我们的方案：USR + SkillChannel 作为结构化串联层

```
Foundation Model → Skill Logic + Adapter → USR → SkillChannel → Brain
  (OG/SD/EG)      (remap v3/parse)     (公共语义+信号) (publish/consume门控)
```

- **publish(skill, USR)**：每个 Skill 执行后发布结构化 USR（environment_facts / task_semantics / decision_signals / provenance）
- **consume(skill, field)**：下游只能读 PUBLIC_FIELDS 白名单，raw 模型输出被阻断
- **验证**：OG→USR→SD 链路 100%（SD 输入 == OG class）；OG found=false → 100% 触发重新探索；DA-FT 反事实 98%

### 3.3 三条真实传播路径（已审计）

| 路径 | 证据 |
|------|------|
| A: OG found=false → EG/reobserve | GDINO 10/10 下一个 skill 是 exploration_guidance |
| B: OG class → SD → LPM | episode 0, ladle→SD input=ladle→LPM pick up |
| C: USR signal → DA-Brain → execute/guard/reobserve | 反事实 34/35 signal-sensitive |

---

## 四、核心结论

1. **SOTA 无现成可替换 skill 范式**：主流是「单一 VLA」或「LLM 编排现成 skill」，RoboAgent 是「单一微调 VLM 全 skill 耦合」。
2. **baseline 的 decision = 隐式耦合能力**（BC+DAgger+EIPO 注入调度/动作/词表），**无独立接口、训练数据未开源、决策监督来自 expert 轨迹 + 特权模拟器信息**。
3. **Skill 串联的正确方式 = 结构化 USR + SkillChannel 门控**（取代自由文本 core_history），让异构 FM 输出经 Adapter 对齐后，以 typed/temporal/auditable 的接口被 Brain 消费。

> 一句话：baseline 微调 Qwen 的「决策」是 BC+DAgger+EIPO 全模型微调注入的隐式能力（调度协议+动作映射+label 词表，数据未开源，监督来自 expert 轨迹+特权模拟器信息）；把它拆开后，必须用 Decision-Compatible Adapter 对齐异构 Skill 输出，再用 USR+SkillChannel 作为结构化串联层让 Brain 可靠消费。
