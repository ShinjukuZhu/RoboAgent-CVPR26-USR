# Fallback results (EB + SkillOpt sealed; AW OOD in progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`  
**Branch:** `research/fallback-usr-skillopt`  
**Survey:** `reports/FALLBACK_MIN_STANDARD.md`

## Sealed

| Split | n | SR | vs baseline |
|---|---:|---:|---|
| EB-ALFRED base | **50 / 50** | **0.84** | Align+USR 0.78 ✓ / Align 0.80 ✓ |
| SkillOpt D_tr | 20 / 20 | 0.65 | development evidence |
| SkillOpt D_sel (v0) | 20 / 20 | 0.75 | held-out gate baseline |
| SkillOpt round-1 | — | **REJECT** | 0.75 → 0.50 (`miss_limit` 2→3); keep `skill_v0000` |

## In progress

| Track | Status |
|---|---|
| AW OOD | **~61 / 134 @ 0.77**; 4× range watchdogs + tail rebalance; fail-reeval queued after completion |

## Skill contracts on USR

- `invalidate_perception_after_world_change` (OpenETA)
- `invalidate_stale_suffix` + confirmed progress (MineEvolve)
- `verify_grounded_object` + receptacle paraphrases (EmbodiSkill)
- `block_nonpickupable_take` (blocks microwave/countertop pickup loops)
- SkillOpt strict held-out SR gate

`wait_and_finalize.sh` will run fail reeval (promote SR=1 only) then write
`FALLBACK_FINAL_RESULTS.md` when AW134 is complete.
