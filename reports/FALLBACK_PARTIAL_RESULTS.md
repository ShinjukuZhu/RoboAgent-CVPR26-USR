# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | 34 / 134 | 0.794 | GPU1 keep-alive |
| EB base | 46 unique / 50 | 0.717 | miss {44,47,48,49}; skip-44 then fill gap |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | 11 / 20 | 0.636 | GPU4 |

**Landed already:** survey + USR Skill contracts + SkillOpt gate (`reports/FALLBACK_MIN_STANDARD.md`).
**Pending seal:** AW134, EB50+paraphrase reeval, SkillOpt history, `FALLBACK_FINAL_RESULTS.md`.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
