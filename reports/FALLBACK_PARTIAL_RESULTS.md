# Fallback results (EB sealed; AW post-reeval)

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

## AW OOD (134/134; stub batch finishing)

| Metric | Value |
|---|---|
| Coverage | **134 / 134** |
| SR (live) | **0.6791** (91/134) — up from 0.606 pre-reeval |
| Promoted so far | **9** hang-stub tasks via `aw_stub_batch_reeval.sh` |
| vs Native 0.81 | still below |
| vs Align 0.84 | still below — need ~22 more successes for parity |
| Ops | last chunk `125,126,127,128,130` running; then re-finalize |

Early `FALLBACK_FINAL_RESULTS.md` (SR 0.664) was written before later chunks promoted; wait for stub batch + `wait_and_finalize.sh` for the sealed final.

## Skill landed on USR

- `invalidate_perception_after_world_change`, `invalidate_stale_suffix`, `verify_grounded_object`
- `block_nonpickupable_take` (also blocks heat/clean/cool/slice of appliances)
- SkillOpt strict held-out SR gate (`training/skillopt_evolve.py`)
