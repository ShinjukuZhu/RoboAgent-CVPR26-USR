# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~51 / 134 | ~0.765 | main on GPU1 (51+); shards **70–90** (GPU7) + **90–134** (GPU6) |
| EB base | **50 / 50** | **0.84** | sealed; beats Align 0.80 / Align+USR 0.78 |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | **20 / 20** | **0.75** | round-1 candidate eval ~12/20 |

Survey: `reports/FALLBACK_MIN_STANDARD.md`. Pending: AW134 merge; SkillOpt history; `FALLBACK_FINAL_RESULTS.md`.
