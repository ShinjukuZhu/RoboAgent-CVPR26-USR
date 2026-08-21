# Fallback results (EB + SkillOpt sealed; AW blocked on GPU)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`  
**Branch:** `research/fallback-usr-skillopt`  
**Survey:** `reports/FALLBACK_MIN_STANDARD.md`

## Sealed (verified)

| Split | n | SR | vs baseline |
|---|---:|---:|---|
| EB-ALFRED base | **50 / 50** | **0.84** | Align+USR 0.78 ✓ / Align 0.80 ✓ |
| SkillOpt D_tr | 20 / 20 | 0.65 | development |
| SkillOpt D_sel v0 | 20 / 20 | 0.75 | held-out gate |
| SkillOpt round-1 | — | **REJECT** | keep `skill_v0000`; history sealed |

## AW OOD (incomplete — GPU contention)

| Metric | Value |
|---|---|
| Coverage | **132 / 134** (missing **113, 114**) |
| SR so far | **0.6061** (35+ hang stubs depressed the score) |
| Blocker | Lab GPUs saturated; need ≥50 GiB free for RoboAgent load |
| Waiter | `training/aw_reeval_when_free.sh` polling; then finish gaps + promote-only fail reeval |

## Skill landed on USR

- `invalidate_perception_after_world_change`, `invalidate_stale_suffix`, `verify_grounded_object`
- `block_nonpickupable_take` (also blocks heat/clean/cool/slice of appliances)
- SkillOpt strict held-out SR gate (`training/skillopt_evolve.py`)

Do not treat AW as sealed until 134/134 and post-reeval SR are written by
`wait_and_finalize.sh` → `FALLBACK_FINAL_RESULTS.md`.
