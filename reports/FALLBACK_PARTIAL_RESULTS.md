# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~43 / 134 | ~0.814 | running (long episode earlier) |
| EB base | **50 / 50** | **0.76** | reeval promoted 1+7+25; need 1 more win for ≥0.78 |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | ~18 / 20 | ~0.722 | nearly done baseline eval |

Landed: survey + USR Skill + SkillOpt gate + paraphrase reeval. Pending: +1 EB win, AW134, SkillOpt history, final md.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
