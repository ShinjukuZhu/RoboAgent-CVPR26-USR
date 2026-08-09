# Research — SOTA 方案调研（具身 Agent / Foundation Model / Skill 接口 / 训练范式）

> 调研问题（学长 Task 1）：代表性具身 Agent 的 SOTA 方案有哪些？Foundation Model 如何作为 Skill 接入？训练范式与 Benchmark 如何？
>
> 来源标注：[已核实]=arXiv/官方仓库核实；[作者声明]=RoboAgent 自身信息；[推断]=合理推导。

---

## 1. 代表性具身 Agent 系统

### 多模块 / 多 Skill 架构（对照 RoboAgent 的 scheduler+7 skill）

| 系统 | 结构 | Brain/Skill 分工 | 来源 |
|------|------|-----------------|------|
| AutoRT | VLM 场景理解 + LLM 指令生成 + 策略执行 | Brain(LLM) + Skill(策略) | arXiv:2401.12963 |
| GRID | scene graph + LLM 分解指令 | Brain(LLM) + 外部场景图 | arXiv:2309.07726 |
| Voyager | LLM + 可增长 skill library + 自反思 | Brain(LLM) + Skill(代码库) | arXiv:2305.16291 |
| SayPlan | 3D scene graph + LLM 规划 + 反馈 | Brain(LLM) + 感知 | arXiv:2307.06135 |

### 统一 VLM / VLA（对照 RoboAgent 单一 VLM）

| 系统 | 结构 | 训练 | 来源 |
|------|------|------|------|
| RT-2 | action=文本 token, VLM co-finetune | robot+VQA | arXiv:2307.15818 |
| OpenVLA | VLM 微调输出 tokenized action | 970k OpenX, LoRA 1.4% | arXiv:2406.09246 |
| Octo | diffusion policy | 800k 轨迹 | arXiv:2405.12213 |
| PaLM-E | 状态序列化进 LLM | 多任务 | arXiv:2303.03378 |

### 语言 grounding / 反馈回路

| 系统 | 思路 | 来源 |
|------|------|------|
| SayCan | LLM 候选 × skill affordance 加权 | arXiv:2204.01691 |
| InnerMonologue | LLM 规划 + 闭环语言反馈 | arXiv:2207.05608 |

> **RoboAgent 定位**：单一微调 VLM 承担 scheduler/EG/OG/SD/LPM/ES/QA 七角色——介于"纯动作 VLA"与"LLM 编排+现成 skill"之间。

---

## 2. Foundation Model 作为 Skill

| 角色 | 候选模型 | 能力 | 来源 |
|------|---------|------|------|
| OG 开放检测 | Grounding DINO / LLMDet / OWL-ViT | 语言条件开放集检测 | arXiv:2303.05499 / 2501.18954 |
| SD 统一视觉 | Florence-2 | caption/detect/ground/segment | arXiv:2311.06242 |
| Brain/QA | GPT-4V / Qwen2.5-VL | 多模态推理 | 官方文档 / arXiv:2502.13923 |

---

## 3. Skill/Tool 接口与组合

- **Function calling**：LLM 按 JSON Schema 输出可调用参数（OpenAI）
- **HuggingGPT**：LLM controller 编排 AI 模型库（arXiv:2303.17580）
- **Adapter/对齐层**：LoRA（arXiv:2106.09685）、EmbodiedBench 动作离散化、RT-2 动作=token
- **[推断]** "接口契约/统一 Skill 表示"作为正式方法无公开同名——本工作 Decision-Aware Adapter + USR 为原创

---

## 4. 训练范式

| 范式 | 定义 | 来源 |
|------|------|------|
| BC | 专家演示行为克隆 | 常识 |
| DAgger | 在策略分布下聚合专家数据 | arXiv:1011.0686 |
| PPO | 策略梯度 + surrogate | arXiv:1707.06347 |
| LoRA SFT | 低秩适配微调 | arXiv:2106.09685 |

---

## 5. 常用 Benchmark

| 基准 | arXiv | 要点 |
|------|-------|------|
| ALFRED | 1912.01734 | 25k 指令, AI2-THOR 长程家居 |
| ALFWorld | 2010.03768 | 文本版 ALFRED |
| EmbodiedBench | 2502.09560 | 4 环境 6 能力; GPT-4o 平均 28.9% |
| RoboBench | 2510.17801 | MLLM-as-brain |

评测协议：SR / GCR / PLWSR（ALFRED leaderboard），EB 沿用 SR + Subgoal SR。
