# NEXT_STAGE_RESULTS — Task 1: Skill Self-Evolution

**Date:** 2026-08-08
**Branch:** research/skill-evolution-v2

## 研究问题
当前 RoboAgent 的固定 Skill 规则能否变成可验证、自我更新的 artifact？

## 实验设计（6 baseline）
| baseline | 机制 | 结果 |
|----------|------|------|
| fixed | 固定 seed skill | test 0.632 |
| random | 随机加规则 | 0.632（无提升）|
| heuristic-gated | 共现猜容器 + gate | 0.632（gate tie 拒绝）|
| LLM-reflection+gate | FT Qwen 提议 + gate | 0.632（候选正确但 gate 拒绝）|
| SkillOpt-style | bounded edit + held-out gate + rejected buffer | 0.632（同）|
| EmbodiSkill-style | defect/lapse 归因 + body-only 更新 | 归因有效但无提升 |

## 机制验证（全部成立）
1. **versioned artifact**: RuleSkill 带 hash/version，不原地覆盖 ✅
2. **EmbodiSkill 归因**: 87 success / 29 skill_defect / 21 execution_lapse——分类正确 ✅
3. **LLM reflection**: `somewhere knife` → 正确生成 `target DiningTable` ✅
4. **SkillOpt gate**: validation tie 严格拒绝（0.667→0.667），rejected buffer 记录 ✅
5. **bounded edit**: 每轮 ≤4 规则 ✅

## 负结果（诚实记录）
**8 轮 evolution 全部 reject，test 分数无提升（0.632→0.632）**。

**根因（数据瓶颈）**:
- `somewhere knife` 是最高频 defect family，但其失败案例**在 validation split 无样本**
- 加 knife→DiningTable 规则不改变 validation 分数 → tie → 拒绝
- 197 训练样本 split 后 validation ~60，family 覆盖不足

## 结论
1. **Skill 自学习机制成立**（artifact + 归因 + reflection + gate 全链路可运行）
2. **但当前数据量下无法展示统计提升**——validation 无 family 区分度
3. **不人为扩充测试集**（按要求）
4. **定位**: 系统扩展 + negative result（数据瓶颈），非论文主贡献

## 证据
- artifacts: skill_evolution_v2_artifact.json, skill_evolution_v3_artifact.json
- logs: se_v2.log, se_v3.log
- 代码: experiments/skill_evolution/v1/skill_evolution_v2.py, v3.py

## STATUS: PARTIAL（机制 DONE，统计提升 FAILED 因数据）
