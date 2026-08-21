#!/bin/bash
# Finish remaining EB base tasks by missing-id order.
# Put chronically stuck ids last (Thor hang / put loops).
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
EB_ROOT=$ROOT/code/EmbodiedBench
SKILL=$CODE/skills/effect_verified_skill_v0000.md
LOG=$RUN/logs/eb50_skill.log
STUCK_LAST=${STUCK_LAST:-"44 47"}
mkdir -p "$RUN/logs"

pkill -f "runs/usr_minstd_skillopt/usr_fb_eb50 " || true
pkill -f "eb_skip44_finish.sh" || true
sleep 3

[ -S /tmp/.X11-unix/X99 ] || \
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
sleep 2

ORDER=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
missing=sorted(set(range(50))-ids)
stuck=set(int(x) for x in "$STUCK_LAST".split())
first=[i for i in missing if i not in stuck]
last=[i for i in missing if i in stuck]
print(" ".join(str(i) for i in first+last))
print("unique", len(ids), "missing", missing)
PY
)
echo "plan: $ORDER"
TASKS=$(echo "$ORDER" | head -1)

record_timeout_fail() {
  local tid=$1
  "$ENV_BIN/python" - <<PY
import json
from pathlib import Path
tid=int("$tid")
main=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
rows={}
if main.exists():
    for line in main.read_text().splitlines():
        if not line.strip():
            continue
        r=json.loads(line)
        rows[int(r["task_idx"])]=r
if tid in rows:
    print(f"task {tid} already present; skip fail stub")
else:
    stub={"task_idx": tid, "SR": 0, "success": False, "error": "timeout_or_hang", "note": "recorded by eb_finish_missing after per-task timeout"}
    with main.open("a") as f:
        f.write(json.dumps(stub, ensure_ascii=False)+"\n")
    print(f"wrote fail stub for task {tid}")
PY
}

launch_one() {
  local tid=$1
  local rc=0
  echo "$(date -Is) EB single task $tid" | tee -a "$LOG"
  cd "$CODE"
  set +e
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=3 DISPLAY=:99 PATH=$ENV_BIN:$PATH \
    PYTHONPATH=$EB_ROOT:${PYTHONPATH:-} PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
    ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
    ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
    ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
    ROBOAGENT_EVO_SKILL=$SKILL \
    timeout 1800 python -u run_ebalf.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
      --save_path "$RUN/usr_fb_eb50" \
      --data_path "$EB_ROOT/embodiedbench/envs/eb_alfred/data/splits/splits.json" \
      --split base --server-num 99 --start "$tid" --end $((tid+1)) --seed 42 \
      >> "$LOG" 2>&1
  rc=$?
  set -e
  # 124 = GNU timeout; also cover crash-without-row
  have=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
print(int($tid in ids))
PY
)
  if [ "$have" != "1" ]; then
    echo "$(date -Is) task $tid missing after exit=$rc; writing fail stub" | tee -a "$LOG"
    record_timeout_fail "$tid"
  elif [ "$rc" -ne 0 ]; then
    echo "$(date -Is) task $tid exit=$rc but row present" | tee -a "$LOG"
  fi
}

for tid in $TASKS; do
  # skip if already present (race)
  have=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
print(int($tid in ids))
PY
)
  if [ "$have" = "1" ]; then
    echo "skip existing $tid"
    continue
  fi
  launch_one "$tid"
done

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
miss=[i for i in range(50) if i not in rows]
print(json.dumps({"n": len(ordered), "SR": sr, "missing": miss}))
PY
