# Fallback partial results (in-progress)

Updated from server run root
`/mnt/autodl_tmp1/zhuyanhao/runs/fallback_usr_skillopt`.

Protocol: original RoboAgent AW OOD + EB-ALFRED base. Config = Align+USR +
effect-verified Skill v0 (`ROBOAGENT_OG_BACKEND=llmdet_qwen_usr`,
`ROBOAGENT_SD_BACKEND=usr`, `ROBOAGENT_USR_CHANNEL=1`,
`ROBOAGENT_EVO_SKILL=skills/effect_verified_skill_v0000.md`).

## Live snapshot

| Split | Path | n / target | SR so far | Notes |
|---|---|---|---|---|
| AW OOD sealed | `official_aw_ood-eval_out_of_distribution` | 4 / 134 | 1.000 | early; not final |
| EB base sealed | `official_eb50-base` | 7 / 50 | 0.714 | early; not final |
| AW ID D_tr | `skillopt_dev-eval_in_distribution` | 5 / 20 | 0.800 | SkillOpt evidence only |

Frozen references (do not overwrite):

| Method | AW OOD | EB base |
|---|---|---|
| Native | 0.81 | 0.78 |
| Align | 0.84 | 0.80 |
| Align+USR | — | 0.78 |

## Isolation

- Codex V2 SkillOpt remains on GPU 7 / `RoboAgent_Evo_20260820`.
- This branch does **not** require `V2_FROZEN.json`.
- A prior contaminated launch was quarantined; current runs use
  `official_aw_ood` / `official_eb50` directories.

## Next

1. Finish EB50 and AW134.
2. Finish D_tr 0–19, then SkillOpt D_sel 20–39 with strict SR gate.
3. Replace this file with `FALLBACK_FINAL_RESULTS.md` and push artifacts.
