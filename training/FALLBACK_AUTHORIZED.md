# Minimum-standard fallback runs (independent of V2)

Code: `RoboAgent_USR_SkillOpt`
Run root: `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`

This tree does **not** wait for `V2_FROZEN.json`.

V2 freeze scripts previously killed jobs under `fallback_usr_skillopt` and
moved that directory. The run root was therefore renamed to `usr_minstd_skillopt`
so V2 quarantine commands that hard-code the old path do not hit this work.

Do **not** kill processes with `FALLBACK_USR_SKILLOPT_AUTHORIZED=1`.
Do **not** rename or delete `usr_minstd_skillopt` / `usr_fb_*` for missing V2 freeze.
