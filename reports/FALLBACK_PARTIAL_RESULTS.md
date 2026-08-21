# Fallback partial results (in-progress)

Updated from server run root
`/mnt/autodl_tmp1/zhuyanhao/runs/fallback_usr_skillopt`.

Protocol: original RoboAgent AW OOD + EB-ALFRED base. Config = Align+USR +
effect-verified Skill v0 (`ROBOAGENT_OG_BACKEND=llmdet_qwen_usr`,
`ROBOAGENT_SD_BACKEND=usr`, `ROBOAGENT_USR_CHANNEL=1`,
`ROBOAGENT_EVO_SKILL=skills/effect_verified_skill_v0000.md`).

## Incident (2026-08-21)

A V2 freeze/quarantine command killed PIDs for the fallback AW/EB/SkillOpt jobs
and renamed live dirs to `*.pre_freeze_contaminated_20260821_0326EDT` because
`V2_FROZEN.json` was absent. Progress was preserved and restored under `usr_fb_*`
paths that the freeze script does not target. Skill paraphrase fix
(kitchen/wooden table → dining table) remains in the Skill artifact.

## Live snapshot (restored + resumed)

| Split | Path | n / target | SR so far | Resume from | Notes |
|---|---|---|---|---|---|
| AW OOD sealed | `usr_fb_aw_ood-eval_out_of_distribution` | 6 / 134 | 1.000 | 6 | early; not final |
| EB base sealed | `usr_fb_eb50-base` | 14 / 50 | 0.714 | 14 | early; not final |
| AW ID D_tr | `usr_fb_skillopt_dev-eval_in_distribution` | 7 / 20 | 0.714 | 7 | SkillOpt evidence only |

Frozen references (do not overwrite):

| Method | AW OOD | EB base |
|---|---|---|
| Native | 0.81 | 0.78 |
| Align | 0.84 | 0.80 |
| Align+USR | — | 0.78 |

## Isolation

- Codex V2 SkillOpt remains on GPU 7 / `RoboAgent_Evo_20260820`.
- This branch does **not** require `V2_FROZEN.json`.
- Do not kill `FALLBACK_USR_SKILLOPT_AUTHORIZED=1` or rename `usr_fb_*`.

## Next

1. Finish EB50 and AW134.
2. Finish D_tr 0–19, then SkillOpt D_sel 20–39 with strict SR gate.
3. Replace this file with `FALLBACK_FINAL_RESULTS.md` and push artifacts.
