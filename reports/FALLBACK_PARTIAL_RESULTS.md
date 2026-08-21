# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~42 / 134 | ~0.83 | running |
| EB base | **50 / 50** | **0.70** | sealed (44 Thor hang + 47 put-loop stubbed); paraphrase reeval on GPU6 in progress |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | ~15 / 20 | ~0.71 | baseline skill eval |

Landed: survey + USR Skill + SkillOpt gate + grounding paraphrase hotfix + early reeval. Pending: reeval splice → EB ≥0.78 target; AW134; SkillOpt history; final md.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
