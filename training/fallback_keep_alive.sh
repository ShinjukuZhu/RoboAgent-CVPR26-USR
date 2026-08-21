#!/bin/bash
# Keep usr_minstd fallback jobs alive without touching V2.
# Relaunches AW / EB / SkillOpt if they die before completion.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
EB_ROOT=$ROOT/code/EmbodiedBench
SKILL=$CODE/skills/effect_verified_skill_v0000.md
LOG=$RUN/logs/fallback_keep_alive.log
mkdir -p "$RUN/logs"

ensure_xvfb() {
  local d=$1
  if [ ! -S "/tmp/.X11-unix/X${d}" ]; then
    /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${d}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
    sleep 2
  fi
}

relaunch_aw() {
  local unique
  unique=$("$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
print(len(ids))
PY
)
  [ "$unique" -ge 134 ] && return 0
  pgrep -f "runs/usr_minstd_skillopt/usr_fb_aw_ood " >/dev/null && return 0
  pgrep -f "aw_fill_missing.sh" >/dev/null && return 0
  echo "$(date -Is) relaunch AW fill_missing (unique=$unique)" >>"$LOG"
  nohup bash "$CODE/training/aw_fill_missing.sh" >>"$RUN/logs/aw_fill_missing.log" 2>&1 &
}

relaunch_eb() {
  local n
  n=$(wc -l < "$RUN/usr_fb_eb50-base/results.jsonl" | tr -d ' ')
  # unique task count
  n=$("$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
print(len(ids))
PY
)
  [ "$n" -ge 50 ] && return 0
  pgrep -f "runs/usr_minstd_skillopt/usr_fb_eb50 " >/dev/null && return 0
  pgrep -f "eb_finish_missing.sh" >/dev/null && return 0
  pgrep -f "eb_skip44_finish.sh" >/dev/null && return 0
  echo "$(date -Is) relaunch EB finish_missing (unique=$n)" >>"$LOG"
  nohup bash "$CODE/training/eb_finish_missing.sh" >>"$RUN/logs/eb_finish_missing.log" 2>&1 &
}

relaunch_skillopt() {
  local n
  n=$(wc -l < "$RUN/skillopt/selection_current/run-eval_in_distribution/results.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
  if [ -f "$RUN/skillopt/runtime_state.json" ]; then
    return 0
  fi
  [ -f "$RUN/skillopt/history.jsonl" ] && [ "$(wc -l < "$RUN/skillopt/history.jsonl" | tr -d ' ')" -ge 1 ] && [ "$n" -ge 20 ] && return 0
  pgrep -f "skillopt_evolve.py .*usr_minstd_skillopt" >/dev/null && return 0
  pgrep -f "seal_skillopt_after_round1.sh" >/dev/null && return 0
  ensure_xvfb 98
  echo "$(date -Is) relaunch SkillOpt (sel_n=$n)" >>"$LOG"
  SEL_CMD="cd $CODE && env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=4 DISPLAY=:98 ALFWORLD_DATA=$ROOT/data/alfworld PATH=$ENV_BIN:\$PATH PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FALLBACK_USR_SKILLOPT_AUTHORIZED=1 ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 ROBOAGENT_EVO_SKILL={skill} $ENV_BIN/python -u $CODE/run_aw.py --qwen_path $ROOT/ckpt/RoboAgent_CVPR26 --save_path {output}/run --split eval_in_distribution --start 20 --end 40 --seed 42"
  nohup "$ENV_BIN/python" -u "$CODE/training/skillopt_evolve.py" \
    --initial-skill "$SKILL" \
    --development-run "$RUN/usr_fb_skillopt_dev-eval_in_distribution" \
    --output "$RUN/skillopt" --rounds 3 \
    --selection-start 20 --selection-end 40 \
    --selection-command "$SEL_CMD" \
    >> "$RUN/logs/skillopt.log" 2>&1 &
}

ensure_marker() {
  "$ENV_BIN/python" - <<'PY' >>"$LOG" 2>&1 || true
import json
from pathlib import Path
r=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
m=r/"USR_MINSTD_DO_NOT_TOUCH.json"
payload={
  "owner":"zhuyanhao",
  "branch":"research/fallback-usr-skillopt",
  "purpose":"fallback_min_standard_usr_skillopt",
  "do_not_move":True,
  "do_not_kill":True,
}
m.write_text(json.dumps(payload, indent=2)+"\n")
print("marker ok")
PY
}

echo "$(date -Is) keep_alive start" >>"$LOG"
while true; do
  sleep 180
  echo "$(date -Is) tick" >>"$LOG"
  # refuse to act if tree moved
  if [ ! -d "$RUN" ]; then
    echo "$(date -Is) RUN missing" >>"$LOG"
    sleep 60
    continue
  fi
  ensure_marker || true
  relaunch_aw || true
  relaunch_eb || true
  relaunch_skillopt || true
  ps -u zhuyanhao -o pid=,etime=,cmd= | grep usr_minstd_skillopt | grep -E 'run_aw|run_ebalf|skillopt_evolve' | grep -v grep >>"$LOG" || echo "NO_JOBS" >>"$LOG"
done
