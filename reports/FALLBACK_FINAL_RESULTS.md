# Fallback final results

Protocol: RoboAgent official AW `eval_out_of_distribution` (0–133) and
EB-ALFRED `base` (0–49). Stack: Align+USR + effect-verified Skill
(`usr_fb_*` run dirs on the fallback branch).

Survey / reproduced-workload analysis: `reports/FALLBACK_MIN_STANDARD.md`.

## Sealed SR

| Split | n | SR | Reference to beat |
|---|---:|---:|---|
| AW OOD | 134 | 0.6642 | Align 0.84 / Native 0.81 |
| EB base | 50 | 0.8400 | Align+USR 0.78 / Align 0.80 |

AW vs Align: BELOW Align 0.84 — inspect failures
EB vs Align+USR: PASS (≥0.78)
EB vs Align: PASS (≥0.80)

## SkillOpt

Gate: ACCEPT only on strict held-out selection SR increase.

| Round | Decision | Reason |
|---|---|---|
| 1 | **REJECT** | selection SR did not strictly rise (0.7500 -> 0.5000) |
| 2 | **SKIP** | optimizer proposed no new bounded edit after round-1 reject |
| 3 | **SKIP** | optimizer proposed no new bounded edit after round-1 reject |

## Artifacts

- `reports/partial_results/usr_fb_aw_ood.jsonl`
- `reports/partial_results/usr_fb_eb50.jsonl`
- `reports/partial_results/usr_fb_skillopt_dev.jsonl`
- `reports/partial_results/skillopt_history.jsonl`
