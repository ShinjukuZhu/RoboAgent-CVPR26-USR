# Fallback partial results (in-progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

| Split | n / target | SR so far | Notes |
|---|---|---|---|
| AW OOD | 30 / 134 | 0.767 | GPU1; recent fails are take/holding misses (not Skill grounding) |
| EB base | 45 / 50 (gap@44) | 0.733 | task 45 OK; on 46–49 then retry 44; paraphrase reeval queued |
| SkillOpt D_tr | 20 / 20 | 0.650 | done |
| SkillOpt D_sel | 9 / 20 | 0.667 | GPU4 mid task 29 |

**Already landed (code/docs, independent of sealed SR):**
- Survey + reproduced-workload analysis → `reports/FALLBACK_MIN_STANDARD.md`
- USR Skill contracts (stale perception / suffix / grounding / skip confirmed)
- SkillOpt gate (`training/skillopt_evolve.py`, ACCEPT only on strict D_sel SR↑)

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.
