#!/usr/bin/env bash
# Single-GPU AW skillfix reeval — main optimization path after loop-break deploy.
set -euo pipefail
ROOT=${ROOT:-/mnt/autodl_tmp1/zhuyanhao}
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
RUN=$ROOT/runs/usr_minstd_skillopt
LOG=$RUN/logs/aw_fail_skillfix.log
PY=$ROOT/envs/RoboAgent_AW/bin/python
GPU=${FORCE_GPU:-2}

bash "$CODE/training/stop_all_ours.sh" | tee -a "$LOG"
export FORCE_GPU=$GPU
export TASK_TIMEOUT_SEC=${TASK_TIMEOUT_SEC:-5400}
export ROBOAGENT_MAX_AW_STEPS=${ROBOAGENT_MAX_AW_STEPS:-0}
nohup "$PY" -u "$CODE/training/aw_fail_skillfix_reeval.py" >> "$LOG" 2>&1 &
echo "skillfix_pid=$! gpu=$GPU log=$LOG"
