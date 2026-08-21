#!/bin/bash
# Skip stuck EB task 44: finish 45-49 first, then retry 44 alone.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
EB_ROOT=$ROOT/code/EmbodiedBench
SKILL=$CODE/skills/effect_verified_skill_v0000.md
LOG=$RUN/logs/eb50_skill.log
mkdir -p "$RUN/logs"

N=$(wc -l < "$RUN/usr_fb_eb50-base/results.jsonl" | tr -d ' ')
echo "current_n=$N"
if [ "$N" -ge 50 ]; then
  echo "EB already complete"
  exit 0
fi

# kill stuck EB on task 44
pkill -f "runs/usr_minstd_skillopt/usr_fb_eb50 " || true
sleep 3

[ -S /tmp/.X11-unix/X99 ] || \
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
sleep 2

launch() {
  local start=$1 end=$2
  echo "$(date -Is) launch EB start=$start end=$end" | tee -a "$LOG"
  cd "$CODE"
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=3 DISPLAY=:99 PATH=$ENV_BIN:$PATH \
    PYTHONPATH=$EB_ROOT:${PYTHONPATH:-} PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
    ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
    ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
    ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
    ROBOAGENT_EVO_SKILL=$SKILL \
    python -u run_ebalf.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
      --save_path "$RUN/usr_fb_eb50" \
      --data_path "$EB_ROOT/embodiedbench/envs/eb_alfred/data/splits/splits.json" \
      --split base --server-num 99 --start "$start" --end "$end" --seed 42 \
      >> "$LOG" 2>&1
}

# If still at 44 lines (0..43 done), skip 44 -> run 45-50, then 44-45.
HAVE44=$("$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()}
print(44 in ids)
print(max(ids) if ids else -1)
print(len(ids))
PY
)
echo "have44_info=$HAVE44"
HAS44=$(echo "$HAVE44" | sed -n '1p')
MAX=$(echo "$HAVE44" | sed -n '2p')

if [ "$HAS44" = "False" ] && [ "$MAX" -eq 43 ]; then
  echo "Skipping stuck task 44; running 45-50 first"
  launch 45 50
  echo "$(date -Is) 45-50 finished n=$(wc -l < "$RUN/usr_fb_eb50-base/results.jsonl")" | tee -a "$LOG"
  echo "Retrying task 44 alone"
  launch 44 45
  echo "$(date -Is) task44 finished n=$(wc -l < "$RUN/usr_fb_eb50-base/results.jsonl")" | tee -a "$LOG"
else
  # generic resume from next missing
  NEXT=$("$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()}
for i in range(50):
    if i not in ids:
        print(i); break
else:
    print(50)
PY
)
  echo "resume_next=$NEXT"
  if [ "$NEXT" -lt 50 ]; then
    launch "$NEXT" 50
  fi
fi

# sort/dedupe results by task_idx
"$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
main=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
rows={}
for line in main.read_text().splitlines():
    if not line.strip():
        continue
    r=json.loads(line)
    rows[int(r["task_idx"])]=r
ordered=[rows[i] for i in sorted(rows)]
main.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in ordered))
sr=sum(int(r.get("SR") or 0) for r in ordered)/len(ordered) if ordered else None
print(json.dumps({"n": len(ordered), "SR": sr, "ids": sorted(rows)}))
PY
