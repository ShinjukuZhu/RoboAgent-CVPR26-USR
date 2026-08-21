# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~41 / 134 | ~0.829 | approaching Align 0.84 |
| EB base | 47 unique / 50 | 0.723 | miss {44,47,49}; finish-missing order 48✓→49→44→47; then paraphrase reeval on GPU6 |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | ~14 / 20 | ~0.714 | baseline skill eval in progress (start 29) |

Landed: survey + USR Skill + SkillOpt gate + `eb_finish_missing`. Pending: seal AW134/EB50 + history + final md.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
