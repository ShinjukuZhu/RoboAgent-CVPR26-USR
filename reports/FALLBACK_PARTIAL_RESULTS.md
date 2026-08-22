# Fallback results (EB sealed; AW reeval in progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`  
**Branch:** `research/fallback-usr-skillopt`  
**Survey:** `reports/FALLBACK_MIN_STANDARD.md`  
**Final (live):** `reports/FALLBACK_FINAL_RESULTS.md`

## Sealed (verified)

| Split | n | SR | vs baseline |
|---|---:|---:|---|
| EB-ALFRED base | **50 / 50** | **0.84** | Align+USR 0.78 ✓ / Align 0.80 ✓ |
| SkillOpt D_tr | 20 / 20 | 0.65 | development |
| SkillOpt D_sel v0 | 20 / 20 | 0.75 | held-out gate |
| SkillOpt round-1 | — | **REJECT** | keep `skill_v0000`; history sealed |

## AW OOD (134/134; tail reeval running)

| Metric | Value |
|---|---|
| Coverage | **134 / 134** |
| SR (official merged) | **0.6791** (91/134) |
| Pre-reeval SR | 0.606 (hang stubs depressed score) |
| Promoted via stub batch | **9** tasks |
| vs Native 0.81 | below |
| vs Align 0.84 | below |
| Ops | `aw_tail_stubs_reeval.py` finishing ids 125–130 |

## Skill landed on USR

- `invalidate_perception_after_world_change`, `invalidate_stale_suffix`, `verify_grounded_object`
- `block_nonpickupable_take` (also blocks heat/clean/cool/slice of appliances)
- SkillOpt strict held-out SR gate (`training/skillopt_evolve.py`)
