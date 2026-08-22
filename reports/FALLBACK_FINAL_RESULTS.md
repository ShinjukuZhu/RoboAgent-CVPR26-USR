# Fallback final results

Protocol: RoboAgent official AW `eval_out_of_distribution` (0–133) and
EB-ALFRED `base` (0–49). Stack: Align+USR + effect-verified Skill
(`usr_fb_*` run dirs on the fallback branch).

Survey / reproduced-workload analysis: `reports/FALLBACK_MIN_STANDARD.md`.

## Sealed SR

| Split | n | SR | Reference to beat |
|---|---:|---:|---|
| AW OOD | 134 | 0.6791 | Align 0.84 / Native 0.81 |
| EB base | 50 | 0.8400 | Align+USR 0.78 / Align 0.80 |

AW vs Align: BELOW Align 0.84 — skillfix reeval running (loop-break + no step cap)
EB vs Align+USR: PASS (≥0.78)
EB vs Align: PASS (≥0.80)

## AW optimization (in flight)

Root cause on remaining failures:
- 37 zero-GCR episodes include hang stubs and futile action loops (e.g. repeated
  `clean` without GCR gain; task 125 hit the old 60-step cap).
- Fix deployed: skip confirmed clean/heat/cool/slice, GCR-stall replan
  (`071da2a`: partial-GCR stall limit base+4), remove 60-step reeval cap
  (`ROBOAGENT_MAX_AW_STEPS=0`, wall timeout 5400s).
- `training/aw_fail_skillfix_reeval.py` — serial single-GPU promote-only reeval
  of 43 OOD fail ids; THOR cleanup per task.

**skillfix progress (2026-08-22 ~19:58):** 27/43 done, **0 promote**, AW SR unchanged
at 91/134. Latest: task **96** running. Pattern: exploration loops (task 94) or
take-then-wander without put (tasks 93–95); all stall ~52 env steps.

| task | GCR | env_steps | wall_s | outcome |
|---:|---:|---:|---:|---|
| 93 | 0.00 | 52 | 372 | NO_PROMOTE |
| 94 | 0.00 | 52 | 366 | NO_PROMOTE |
| 95 | 0.00 | 52 | 304 | NO_PROMOTE |
| 96 | — | — | — | IN FLIGHT |

## Executor hotfixes (this session)

- `agents/agent.py`: skip malformed VLM `Query:` lines (missing `)`).
- `agents/eg_llm_backend.py`: fix `legal_objects` / add `exploration_exhausted`
  so EG stops burning VLM calls after all `in|on|target|near` directions tried
  (task 94 root cause). Synced to server; applies to new `run_aw.py` subprocesses.
- `training/restart_skillopt_r2_dsel.sh`: resume D_sel without re-running D_tr.

## SkillOpt

Gate: ACCEPT only on strict held-out selection SR increase (D_sel tasks 20–39).

| Round | Decision | Notes |
|---|---|---|
| 1 | **REJECT** | D_sel SR 0.75 → 0.50 (`repeated_effect_miss_limit` 2→3) |
| 2 | **IN FLIGHT** | D_tr **20/20** complete (SR=0.10); D_sel baseline **1/20**
  (task 20 SR=0); resumed task **21** after VLM malformed-query crash
  (`agents/agent.py` skip fix + `training/restart_skillopt_r2_dsel.sh`) |

Round-2 output is separate from sealed round-1 `skillopt/` tree. On ACCEPT:
deploy skill and rerun OOD fails; on REJECT: document gate and continue executor
fixes (reeval alone unlikely to close ~22-task AW gap).

## Artifacts

- `reports/partial_results/usr_fb_aw_ood.jsonl`
- `reports/partial_results/usr_fb_eb50.jsonl`
- `reports/partial_results/usr_fb_skillopt_dev.jsonl`
- `reports/partial_results/skillopt_history.jsonl`
