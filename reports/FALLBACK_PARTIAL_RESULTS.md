# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~41 / 134 | ~0.829 | approaching Align 0.84 |
| EB base | 48 unique / 50 | 0.729 | 48/49 done (SR=1); running stuck 44 then 47 (30m timeout → fail stub) |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | ~14 / 20 | ~0.714 | baseline skill eval in progress |

Landed: survey + USR Skill + SkillOpt gate + finish-missing + keep-alive marker. Pending: seal AW134/EB50 + paraphrase reeval + history + final md.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
