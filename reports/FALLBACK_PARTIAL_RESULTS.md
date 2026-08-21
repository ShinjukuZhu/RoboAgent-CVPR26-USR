# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | 29 / 134 | 0.793 | GPU1; keep-alive armed |
| EB base | 44 / 50 (+task 45 running) | 0.727 | skipped stuck 44 → finish 45–49 then retry 44 |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | 9 / 20 | 0.667 | GPU4 from 29 |

**Hotfix:** compact `ontable` abstain; Align `kitchenisland↔countertop`, `tvstand↔dresser`.
EB paraphrase reeval queued for `{1,7,19,22,25,28,29,37}` after EB seals.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
