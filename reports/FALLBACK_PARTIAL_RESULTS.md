# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`
(V2 hard-codes of `fallback_usr_skillopt` miss this path.)

| Split | n / target | SR so far | Resume |
|---|---|---|---|
| AW OOD | 27 / 134 | 0.815 | from 27 (GPU1); grounding hotfix applied for remaining |
| EB base | 44 / 50 | ~0.72 | finishing 39–50 (GPU2); paraphrase reeval queued |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | 4 / 20 | 0.500 | from 23 (GPU6) |

**Hotfix (2026-08-21):** compact location tokens (`ontable`), Align-compatible
`kitchenisland↔countertop` and `tvstand↔dresser`. EB reeval tasks expanded to
`1 7 19 22 25 28 29 37`.

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
