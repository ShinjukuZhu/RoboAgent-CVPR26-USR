# Fallback results (EB + SkillOpt sealed; AW finishing)

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

## AW OOD (in progress)

| Metric | Value |
|---|---|
| Coverage | **133 / 134** (missing **114**; **113 SR=1** sealed) |
| SR so far | **0.609** (81/133) — depressed by ~35 hang/watchdog stubs |
| Path to ≥0.84 | need ~32 more successes; prioritize stub promote-only batch reeval |
| Ops | `aw_reeval_when_free.sh` → after 114, `aw_stub_batch_reeval.sh` (model loaded once per chunk) |

## Skill landed on USR

- `invalidate_perception_after_world_change`, `invalidate_stale_suffix`, `verify_grounded_object`
- `block_nonpickupable_take` (also blocks heat/clean/cool/slice of appliances)
- SkillOpt strict held-out SR gate (`training/skillopt_evolve.py`)

Do not treat AW as sealed until 134/134 and post-reeval SR are written by
`wait_and_finalize.sh` → `FALLBACK_FINAL_RESULTS.md`.
