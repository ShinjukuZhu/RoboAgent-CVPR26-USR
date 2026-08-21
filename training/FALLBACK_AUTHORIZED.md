# Fallback branch note: independent of V2 sealed-eval latch.

This directory (`RoboAgent_USR_SkillOpt`) and run root
`/mnt/autodl_tmp1/zhuyanhao/runs/fallback_usr_skillopt` are authorized to run
official RoboAgent AW OOD and EB-ALFRED base evaluations for the minimum-standard
branch. They do **not** consume or wait for `/mnt/autodl_tmp1/zhuyanhao/runs/V2_FROZEN.json`.

V2 jobs must continue to refuse sealed evaluation until their own freeze file exists.
Do not copy this authorization into RoboAgent_Evo_* trees.
