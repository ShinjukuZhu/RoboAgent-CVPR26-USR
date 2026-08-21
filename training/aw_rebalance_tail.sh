#!/usr/bin/env bash
# When mid-gap / 74–90 watchdogs finish, kill the monolithic 91–134 worker and
# resplit the remaining tail across freed GPUs (no overlapping task ownership).
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
PY=$ROOT/envs/RoboAgent_AW/bin/python
LOG=$RUN/logs/aw_rebalance_tail.log
mkdir -p "$RUN/logs"

collect_state() {
  "$PY" - <<'PY'
import json
from pathlib import Path
root=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
ids=set()
for p in [root/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"]:
    if p.exists():
        ids|={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()}
for pat in ("usr_fb_aw_ood_shard_*/run-eval_out_of_distribution/results.jsonl",
            "usr_fb_aw_ood_wd_*/run-eval_out_of_distribution/results.jsonl"):
    for p in root.glob(pat):
        ids|={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()}
miss_tail=[i for i in range(91,134) if i not in ids]
mid_done=1 if all(i in ids for i in range(54,70)) else 0
s74_done=1 if all(i in ids for i in range(74,90)) else 0
print(len(miss_tail))
print(mid_done)
print(s74_done)
print(" ".join(str(i) for i in miss_tail))
PY
}

launch_tail() {
  local gpu=$1 disp=$2 start=$3 end=$4 tag=$5
  [ "$start" -ge "$end" ] && return 0
  if pgrep -f "$tag" >/dev/null; then
    return 0
  fi
  if [ ! -S "/tmp/.X11-unix/X${disp}" ]; then
    /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${disp}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
    sleep 2
  fi
  echo "$(date -Is) tail $tag gpu=$gpu $start-$end" | tee -a "$LOG"
  nohup env GPU="$gpu" DISPLAY_NUM="$disp" START="$start" END="$end" STUCK_SEC=480 SAVE_TAG="$tag" \
    bash "$CODE/training/aw_range_watchdog.sh" \
    >> "$RUN/logs/${tag}.log" 2>&1 &
}

stop_mono_tail() {
  "$PY" - <<'PY'
import os, signal, subprocess
out=subprocess.check_output(["ps","-u","zhuyanhao","-o","pid=,args="], text=True)
for line in out.splitlines():
    parts=line.strip().split(None,1)
    if len(parts)<2: continue
    pid, args=int(parts[0]), parts[1]
    if "usr_fb_aw_ood_wd_91_134" in args:
        try:
            os.kill(pid, signal.SIGTERM)
            print("stopped mono", pid)
        except ProcessLookupError:
            pass
PY
  sleep 3
}

split_and_launch() {
  local miss_csv=$1
  # miss_csv is space-separated task ids
  mapfile -t miss < <(echo "$miss_csv" | tr ' ' '\n' | grep -E '^[0-9]+$')
  local n=${#miss[@]}
  [ "$n" -eq 0 ] && return 0
  stop_mono_tail
  # Prefer up to 4 GPUs: 1,4,6,7
  local gpus=(1 4 6 7) disps=(96 94 97 95)
  local chunks=$(( n < 4 ? n : 4 ))
  local i=0
  while [ $i -lt $chunks ]; do
    local start_idx=$(( i * n / chunks ))
    local end_idx=$(( (i + 1) * n / chunks ))
    local a=${miss[start_idx]}
    local b=$(( miss[end_idx-1] + 1 ))
    launch_tail "${gpus[$i]}" "${disps[$i]}" "$a" "$b" "usr_fb_aw_ood_wd_tail_${a}_${b}"
    i=$((i+1))
  done
}

SPLIT_DONE=0
while true; do
  state=$(collect_state)
  miss_n=$(echo "$state" | sed -n '1p')
  mid_done=$(echo "$state" | sed -n '2p')
  s74_done=$(echo "$state" | sed -n '3p')
  miss_csv=$(echo "$state" | sed -n '4p')
  echo "$(date -Is) miss_tail=$miss_n mid_done=$mid_done s74_done=$s74_done split_done=$SPLIT_DONE" >>"$LOG"
  if [ "${miss_n:-1}" -eq 0 ]; then
    echo "$(date -Is) tail complete" | tee -a "$LOG"
    exit 0
  fi
  # Resplit once mid-gap is done (frees GPU1/4) or when 74-90 done and mono still alone.
  if [ "$SPLIT_DONE" -eq 0 ]; then
    if [ "${mid_done:-0}" -eq 1 ] || { [ "${s74_done:-0}" -eq 1 ] && [ "${miss_n:-0}" -gt 20 ]; }; then
      # Wait until mid watchdogs actually exited if mid_done
      if [ "${mid_done:-0}" -eq 1 ]; then
        if pgrep -f 'usr_fb_aw_ood_wd_54_62' >/dev/null || pgrep -f 'usr_fb_aw_ood_wd_62_70' >/dev/null; then
          sleep 60
          continue
        fi
      fi
      split_and_launch "$miss_csv"
      SPLIT_DONE=1
    fi
  fi
  sleep 120
done
