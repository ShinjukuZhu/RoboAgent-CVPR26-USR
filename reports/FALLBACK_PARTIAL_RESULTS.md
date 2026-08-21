# Fallback partial results (in-progress)

Server: `/mnt/autodl_tmp1/zhuyanhao/runs/fallback_usr_skillopt` (`usr_fb_*`).

| Split | n / target | SR so far |
|---|---|---|
| AW OOD | 6 / 134 | 1.000 |
| EB base | 20 / 50 | 0.750 |
| SkillOpt D_tr | 10 / 20 | 0.700 |

References: Align AW 0.84 / EB 0.80; Align+USR EB 0.78; Native AW 0.81 / EB 0.78.

Early EB fails 1 & 7 were pre–table-paraphrase Skill; post-fix episodes continue under fixed aliases.
SkillOpt waiter + finalize hooks are live; V2 jobs untouched.
