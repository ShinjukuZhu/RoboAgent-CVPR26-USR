# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~42 / 134 | ~0.833 | running |
| EB base | **50 / 50** | **0.74** | sealed; paraphrase reeval promoted 1+7; more on GPU6/7 |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | ~16 / 20 | ~0.688 | baseline skill eval |

Landed: survey + USR Skill + SkillOpt gate + grounding paraphrase hotfix + parallel reeval. Target EB ≥0.78 (need ~2 more promotions). Pending: finish reeval/AW134/SkillOpt history + final md.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
