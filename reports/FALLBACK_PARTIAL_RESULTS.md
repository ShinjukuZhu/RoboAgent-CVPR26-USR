# Fallback partial results (in-progress)

**Incident:** V2 freeze again killed sealed jobs and moved
`fallback_usr_skillopt` → `…_pre_freeze_contaminated_20260821_0435EDT`.
Data restored under **`/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`**
(path V2 hard-codes do not match). Jobs relaunched.

| Split | n / target | SR so far | Resume |
|---|---|---|---|
| AW OOD | 26 / 134 | 0.846 | from 26 (GPU1) |
| EB base | 39 / 50 | 0.692 | from 39 (GPU2) |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | 3 / 20 | — | from 23 (GPU6) |

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
