#!/usr/bin/env bash
# Launch / refresh the four primary AW OOD range watchdogs.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
LOG=$RUN/logs/aw_launch_watchdogs.log
mkdir -p "$RUN/logs"

launch_one() {
  local gpu=$1 disp=$2 start=$3 end=$4 tag=$5 stuck=${6:-720}
  if pgrep -f "$tag" >/dev/null; then
    echo "$(date -Is) skip live $tag" | tee -a "$LOG"
    return 0
  fi
  if [ ! -S "/tmp/.X11-unix/X${disp}" ]; then
    /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${disp}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
    sleep 2
  fi
  echo "$(date -Is) start $tag gpu=$gpu $start-$end" | tee -a "$LOG"
  nohup env GPU="$gpu" DISPLAY_NUM="$disp" START="$start" END="$end" STUCK_SEC="$stuck" SAVE_TAG="$tag" \
    bash "$CODE/training/aw_range_watchdog.sh" \
    >> "$RUN/logs/${tag}.log" 2>&1 &
}

# Prefer covering holes; watchdogs themselves skip completed ids.
launch_one 1 96 54 62 usr_fb_aw_ood_wd_54_62
launch_one 4 94 62 70 usr_fb_aw_ood_wd_62_70
launch_one 7 95 74 90 usr_fb_aw_ood_wd_74_90
launch_one 6 97 91 134 usr_fb_aw_ood_wd_91_134
