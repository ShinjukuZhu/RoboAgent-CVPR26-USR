# Fallback partial results (in-progress)

Updated from server run root
`/mnt/autodl_tmp1/zhuyanhao/runs/fallback_usr_skillopt`.

Protocol: original RoboAgent AW OOD + EB-ALFRED base. Config = Align+USR +
effect-verified Skill v0.

## Live snapshot

| Split | Path | n / target | SR so far | Notes |
|---|---|---|---|---|
| AW OOD sealed | `usr_fb_aw_ood-eval_out_of_distribution` | 6 / 134 | 1.000 | not final |
| EB base sealed | `usr_fb_eb50-base` | 17 / 50 | 0.765 | not final |
| AW ID D_tr | `usr_fb_skillopt_dev-eval_in_distribution` | 8 / 20 | 0.750 | SkillOpt evidence only |

Frozen references: Native AW 0.81 / EB 0.78; Align AW 0.84 / EB 0.80; Align+USR EB 0.78.

## Isolation

- Codex V2 remains untouched (GPU 7 SkillOpt + profile C).
- Fallback uses `usr_fb_*` + `FALLBACK_USR_SKILLOPT_AUTHORIZED=1`; must not be quarantined for missing `V2_FROZEN.json`.

## Next

1. Finish EB50 and AW134; require SR ≥ Align (AW) / Align+USR (EB) preferably higher.
2. Finish D_tr then SkillOpt D_sel with strict SR gate.
3. Write `FALLBACK_FINAL_RESULTS.md` and push artifacts.
