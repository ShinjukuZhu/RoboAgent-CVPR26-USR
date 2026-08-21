#!/usr/bin/env bash
# Run an AW OOD contiguous range with a hang watchdog (no per-task model reload).
# If results.jsonl does not gain a new unique id for STUCK_SEC, kill, stub the
# lowest missing id in-range, and resume from the next id.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${GPU:-1}
DISPLAY_NUM=${DISPLAY_NUM:-96}
START=${START:-52}
END=${END:-70}
STUCK_SEC=${STUCK_SEC:-480}
POLL_SEC=${POLL_SEC:-30}
SAVE_TAG=${SAVE_TAG:-usr_fb_aw_ood_wd_${START}_${END}}
SAVE_PATH=$RUN/$SAVE_TAG
LOG=$RUN/logs/${SAVE_TAG}.log
mkdir -p "$RUN/logs" "$SAVE_PATH"

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${DISPLAY_NUM}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
  sleep 2
fi

merge_into_main() {
  "$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
main = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
rows = {}
if main.exists():
    for line in main.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["task_idx"])] = r
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
for shard in list(root.glob("usr_fb_aw_ood_shard_*/run-eval_out_of_distribution/results.jsonl")) + \
             list(root.glob("usr_fb_aw_ood_wd_*/run-eval_out_of_distribution/results.jsonl")) + \
             list(root.glob("usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")):
    if not shard.exists():
        continue
    for line in shard.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = int(r["task_idx"])
        if tid not in rows:
            rows[tid] = r
ordered = [rows[i] for i in sorted(rows)]
main.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
print({"n": len(ordered), "SR": round(sum(int(r.get("SR") or 0) for r in ordered) / len(ordered), 4) if ordered else None})
PY
}

next_start() {
  "$ENV_BIN/python" - <<PY
import json
from pathlib import Path
start, end = int("$START"), int("$END")
ids=set()
paths=[
 Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"),
 Path("$SAVE_PATH/run-eval_out_of_distribution/results.jsonl"),
]
for p in paths:
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                ids.add(int(json.loads(line)["task_idx"]))
for i in range(start, end):
    if i not in ids:
        print(i)
        break
else:
    print(end)
PY
}

stub_task() {
  local tid=$1
  "$ENV_BIN/python" - <<PY
import json
from pathlib import Path
tid=int("$tid")
p=Path("$SAVE_PATH/run-eval_out_of_distribution/results.jsonl")
p.parent.mkdir(parents=True, exist_ok=True)
rows={}
if p.exists():
    for line in p.read_text().splitlines():
        if line.strip():
            r=json.loads(line); rows[int(r["task_idx"])]=r
if tid not in rows:
    rows[tid]={"task_idx": tid, "SR": 0, "success": False, "error": "timeout_or_hang", "note": "aw_range_watchdog stub"}
p.write_text("".join(json.dumps(rows[i], ensure_ascii=False)+"\n" for i in sorted(rows)))
print("stubbed", tid)
PY
}

echo "$(date -Is) range watchdog start=$START end=$END gpu=$GPU stuck=${STUCK_SEC}s" | tee -a "$LOG"

while true; do
  cur=$(next_start)
  if [ "$cur" -ge "$END" ]; then
    echo "$(date -Is) range complete" | tee -a "$LOG"
    merge_into_main
    exit 0
  fi
  echo "$(date -Is) launch from $cur" | tee -a "$LOG"
  cd "$CODE"
  set +e
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=$GPU DISPLAY=:$DISPLAY_NUM ALFWORLD_DATA=$ROOT/data/alfworld \
    PATH=$ENV_BIN:$PATH PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
    ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
    ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
    ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
    ROBOAGENT_EVO_SKILL=$SKILL \
    python -u run_aw.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
      --save_path "$SAVE_PATH/run" --split eval_out_of_distribution \
      --start "$cur" --end "$END" --seed 42 \
    >> "$LOG" 2>&1 &
  worker=$!
  set -e
  last_n=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("$SAVE_PATH/run-eval_out_of_distribution/results.jsonl")
print(len({int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()}) if p.exists() else 0)
PY
)
  last_change=$(date +%s)
  while kill -0 "$worker" 2>/dev/null; do
    sleep "$POLL_SEC"
    n=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("$SAVE_PATH/run-eval_out_of_distribution/results.jsonl")
print(len({int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()}) if p.exists() else 0)
PY
)
    now=$(date +%s)
    if [ "$n" -gt "$last_n" ]; then
      last_n=$n
      last_change=$now
      merge_into_main || true
      echo "$(date -Is) progress n=$n" | tee -a "$LOG"
    elif [ $((now - last_change)) -ge "$STUCK_SEC" ]; then
      echo "$(date -Is) STUCK ${STUCK_SEC}s at n=$n; kill worker $worker" | tee -a "$LOG"
      kill "$worker" 2>/dev/null || true
      sleep 2
      kill -9 "$worker" 2>/dev/null || true
      # stub the current missing id
      miss=$(next_start)
      if [ "$miss" -lt "$END" ]; then
        stub_task "$miss"
        merge_into_main || true
      fi
      break
    fi
  done
  wait "$worker" 2>/dev/null || true
  merge_into_main || true
  # if worker exited cleanly without covering range, loop resumes at next missing
done
