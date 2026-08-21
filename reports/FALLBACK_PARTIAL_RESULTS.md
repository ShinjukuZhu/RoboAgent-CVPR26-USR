# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~51 / 134 | ~0.765 | running |
| EB base | **50 / 50** | **0.84** | sealed + paraphrase reeval; **beats Align 0.80 / Align+USR 0.78** |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | **20 / 20** | **0.75** | baseline sealed; round-1 candidate (`miss_limit` 2→3) evaluating ~12/20 |

Survey: `reports/FALLBACK_MIN_STANDARD.md`. Pending: AW134; SkillOpt ACCEPT/REJECT history (rounds 1–3); `FALLBACK_FINAL_RESULTS.md`.
