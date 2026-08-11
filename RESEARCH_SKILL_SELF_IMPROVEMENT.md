# Research — Skill Self-Improvement from Execution Trajectories

> 调研问题：Skill 的"可学习对象"是什么、反馈来自哪里、更新后如何判断真的变好了？
> 对标 SkillOpt / EmbodiSkill / RoboSkill + 补充工作。

---

## 1. SkillOpt（微软）— 【已核实 arXiv:2605.23904 / aka.ms/skillopt】

- **可学习对象**：一个自然语言文档（`best_skill.md`，300–2000 tokens），**不是权重**。目标 Agent、执行 harness、评分器全部冻结。
- **反馈来源**：训练集 rollout 的带分数结果 `r∈[0,1]`；optimizer 对失败/成功轨迹做 minibatch reflection → bounded add/delete/replace 编辑 → 按"文本学习率预算"裁剪 top-k。**被拒绝的编辑进入 rejected-edit buffer 作为负反馈**。
- **验证**：候选 Skill 在**不相交 selection split** 上跑同一冻结模型；**只有 selection 分数严格大于当前才接受（tie 拒绝）**；最终分数只在 test split 报告。

> **借鉴**：rollout → reflection → bounded edit → held-out validation gate 循环。这是我们的 Skill Evolution Prototype 的机制原型。

## 2. EmbodiSkill — 【已核实 arXiv:2605.10332】

- **可学习对象**：skill 文档（body 规则 + appendix 强调区）；executor 冻结、训练-free。
- **反馈**：轨迹 0/1 成功信号；reflection 分类为 `DISCOVERY / OPTIMIZATION / SKILL DEFECT / EXECUTION LAPSE`。
- **关键差异**：`EXECUTION LAPSE`（失败但 Skill 有效、是 agent 没遵守）→ **只更新 appendix 不破坏 body**，避免把执行噪声误判成 Skill 缺陷。
- **验证**：无逐编辑 gate；演化 N 轮后一次性 held-out 评测 SR。

> **借鉴**：区分「Skill 缺陷 vs 执行噪声」——失败分析必须分类，不能一律改 Skill。

## 3. RoboSkill — 【已核实 github.com/FlagOpen/RoboSkill】

- **定位**：基于 MCP 的具身机器人 **Skill 标准化注册 + 组合复用**（skill store），**不是自学习算法**。
- 配套 RoboOS（arXiv:2505.03673）的 Cerebellum Skill Library 也是组合复用定位。
- **不误述为自学习**。

> **借鉴**：Skill 标准化注册（skill_id/version/schema/backend/adapter/validator/metrics/hash）可与 USR/SkillChannel 合并为 Skill Registry。

## 4. 补充：从轨迹改进的工作

| 工作 | 可学习对象 | 反馈 | 验证 |
|------|-----------|------|------|
| **EvoSkill**（arXiv:2603.02766）| skill folder（代码+工作流）| 迭代 failure analysis | Pareto frontier + held-out gate |
| **Branch2Skill**（arXiv:2608.08677）| skill 更新 | 失败轨迹密集监督（MCTS 推理树对比）| 任务性能 + token 效率 |
| **SkillForge**（arXiv:2604.08618）| 技能文档 | 部署后批量失败 | 与专家一致性单调提升（离线）|
| **Voyager**（arXiv:2305.16291）| 代码技能库 | 环境反馈+报错 | 自验证 checker 通过才入库 |
| **GEPA**（arXiv:2507.19457）| prompt | 轨迹 NL reflection | Pareto frontier |

## 5. 「如何判断变好」的主流机制

| 机制 | 代表 | 逐编辑 gate |
|------|------|------------|
| **Held-out validation gate（严格大于，tie 拒绝）** | SkillOpt / EvoSkill / GEPA | 是 |
| 事后端到端 held-out 评测（无 gate）| EmbodiSkill | 否 |
| 验证器/单元测试 | Voyager | 是 |

> 结论：主流是 **"held-out 集严格提升即接受（regression=0）"**。我们的 Prototype 采用此机制。

## 6. 对 RoboAgent 的落地映射

- **可学习对象** = EG Skill 的 instruction/规则 + Adapter 映射规则（不碰 Brain 权重）
- **反馈** = AW/EB trajectory（Skill 输入/输出/下游成功/失败类型/Brain action/SR）
- **验证** = 固定 held-out validation episodes：候选 SR 严格大于当前才接受
- **失败分类**（EmbodiSkill 式）：perception / contract mismatch / wrong exploration / downstream misuse / execution lapse——只 execution lapse 不改 Skill body
