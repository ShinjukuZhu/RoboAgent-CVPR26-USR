# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~37 / 134 | ~0.81 | approaching Align 0.84 |
| EB base | 46 unique / 50 | 0.717 | miss {44,47–49}; then paraphrase reeval on GPU6 |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | ~13 / 20 | ~0.69 | evolving |

Landed: survey + USR Skill + SkillOpt gate. Pending: seal AW134/EB50 + history + final md.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
