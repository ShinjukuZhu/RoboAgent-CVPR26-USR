# P0a Report v1 — Skill Evolution Prototype（规则驱动）

**Date:** 2026-08-08
**分支:** research/skill-evolution
**Hypothesis:** validation-gated 的 Skill 自学习（EG/Adapter 规则）能随失败轨迹稳定提升，且无 regression。

## 做了什么
实现了 SkillOpt 式 Skill Evolution 框架（离线，无权重训练）：
- **可学习对象**：EG RuleSkill（family→receptacle 规则文件），可版本化
- **反馈**：eg_train（197）rollout 分数 + 失败分类
- **循环**：reflection（识别失败 family）→ bounded edit（缺失 family 加规则）→ **held-out validation gate**（严格提升才接受，tie 拒绝）→ accept/reject + rejected buffer
- **Baselines**: fixed / random（无门控）/ validation-gated

## 为什么做
验证「Skill 能否从轨迹中自我改进」的最小机制原型（不碰 Brain 权重）。

## 结果
| baseline | EG_test score (220) |
|----------|---------------------|
| fixed | 0.682 |
| random (无门控) | 0.682 |
| gated (5 轮) | 0.682（全部 reject）|

- **gate 行为正确**：validation tie（0.700→0.700）被严格拒绝（SkillOpt 式）
- **但无改进**：规则启发式 `guess_receptacle` 无法从失败 family 正确推导新容器（如 `another keys` 猜 sink 是错的）

## 失败原因
1. 规则 skill 无法精确到实例号（EG 需 `Drawer 10` 而非 `Drawer`）——规则表达力不足
2. **启发式 guess 太弱**：SkillOpt 用 **optimizer LLM** 做 reflection 生成候选，我用了规则共现——无法从失败中产生有意义的新规则

## 下一步
v2：用 **LLM reflection**（FT Qwen 作 optimizer）生成候选规则——对失败 family 分析「应该去哪找」，而非启发式猜。这才是 SkillOpt 机制的核心。

## 产出
- scripts/p0a_skill_evolution.py（框架）
- reports/skill_evolution_v1.json
- **机制验证**：validation gate 正确工作（严格提升 + tie 拒绝），evolution 框架可运行

## 版本管理
- git commit: p0a-evolution-v1（框架+结果）
