# Fallback results (EB + SkillOpt sealed; AW OOD finishing under GPU contention)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`  
**Branch:** `research/fallback-usr-skillopt`  
**Survey:** `reports/FALLBACK_MIN_STANDARD.md`

## Sealed

| Split | n | SR | vs baseline |
|---|---:|---:|---|
| EB-ALFRED base | **50 / 50** | **0.84** | Align+USR 0.78 ✓ / Align 0.80 ✓ |
| SkillOpt D_sel | 20 / 20 | 0.75 | round-1 **REJECT** (keep skill_v0000) |

## AW OOD (live)

| Metric | Value |
|---|---|
| Coverage | **132 / 134** (missing 113–114) |
| Current SR | **~0.61** (depressed by hang-timeout stubs) |
| Next | `aw_reeval_when_free.sh` waits for ≥32GB free GPU, finishes gaps, then promote-only fail reeval |

Lab GPUs are saturated by other train jobs; parallel reeval hit CUDA OOM. Serial when-free path avoids fighting V2/neighbors.

## Skill contracts

- OpenETA perception invalidation, MineEvolve suffix freeze, EmbodiSkill grounding
- `block_nonpickupable_take` (+ heat/clean/cool/slice of appliances)
