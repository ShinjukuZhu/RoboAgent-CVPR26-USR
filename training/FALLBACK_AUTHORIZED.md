# Fallback branch note: independent of V2 sealed-eval latch.

This directory (`RoboAgent_USR_SkillOpt`) and run root
`/mnt/autodl_tmp1/zhuyanhao/runs/fallback_usr_skillopt` are authorized to run
official RoboAgent AW OOD and EB-ALFRED base evaluations for the minimum-standard
branch. They do **not** consume or wait for `/mnt/autodl_tmp1/zhuyanhao/runs/V2_FROZEN.json`.

Active eval directories use the `usr_fb_*` prefix on purpose:
- `usr_fb_aw_ood-eval_out_of_distribution`
- `usr_fb_eb50-base`
- `usr_fb_skillopt_dev-eval_in_distribution`

V2 freeze / quarantine scripts must **not** kill processes with
`FALLBACK_USR_SKILLOPT_AUTHORIZED=1` and must **not** rename `usr_fb_*` dirs.
Those jobs are the fallback branch, not pre-freeze sealed contamination.

V2 jobs must continue to refuse sealed evaluation until their own freeze file exists.
Do not copy this authorization into RoboAgent_Evo_* trees.
