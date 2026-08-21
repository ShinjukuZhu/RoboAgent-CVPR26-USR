# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | ~43 / 134 | ~0.814 | running |
| EB base | **50 / 50** | **0.78** | sealed + reeval; matches Align+USR 0.78 (promoted 1,7,25,28) |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | ~19 / 20 | ~0.737 | nearly done; then ACCEPT/REJECT history |

Pending: finish AW134; SkillOpt evolution history; optional further EB lifts (22/29/37/46 still in reeval); `FALLBACK_FINAL_RESULTS.md`.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
