#!/usr/bin/env bash
# Fill an AW OOD range one task at a time into a shard dir (hang-safe).
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${GPU:-6}
DISPLAY_NUM=${DISPLAY_NUM:-97}
START=${START:-90}
END=${END:-134}
TIMEOUT_SEC=${TIMEOUT_SEC:-1200}
SHARD=$RUN/usr_fb_aw_ood_shard_${START}_${END}
LOG=$RUN/logs/aw_ood_shard_${START}_${END}.log
mkdir -p "$RUN/logs" "$SHARD"

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${DISPLAY_NUM}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
  sleep 2
fi

echo "$(date -Is) hang-safe shard start=$START end=$END gpu=$GPU" | tee -a "$LOG"

for ((tid=START; tid<END; tid++)); do
  have=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
paths=[
 Path("$SHARD/run-eval_out_of_distribution/results.jsonl"),
 Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"),
]
ids=set()
for p in paths:
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                ids.add(int(json.loads(line)["task_idx"]))
print(int($tid in ids))
PY
)
  if [ "$have" = "1" ]; then
    echo "skip $tid" | tee -a "$LOG"
    continue
  fi
  echo "$(date -Is) shard task $tid" | tee -a "$LOG"
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
    timeout "$TIMEOUT_SEC" python -u run_aw.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
      --save_path "$SHARD/run" --split eval_out_of_distribution \
      --start "$tid" --end $((tid+1)) --seed 42 \
    >> "$LOG" 2>&1
  rc=$?
  set -e
  have=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("$SHARD/run-eval_out_of_distribution/results.jsonl")
ids=set()
if p.exists():
    for line in p.read_text().splitlines():
        if line.strip():
            ids.add(int(json.loads(line)["task_idx"]))
print(int($tid in ids))
PY
)
  if [ "$have" != "1" ]; then
    echo "$(date -Is) stub fail task $tid rc=$rc" | tee -a "$LOG"
    "$ENV_BIN/python" - <<PY
import json
from pathlib import Path
tid=int("$tid")
p=Path("$SHARD/run-eval_out_of_distribution/results.jsonl")
p.parent.mkdir(parents=True, exist_ok=True)
rows={}
if p.exists():
    for line in p.read_text().splitlines():
        if line.strip():
            r=json.loads(line); rows[int(r["task_idx"])]=r
if tid not in rows:
    rows[tid]={"task_idx": tid, "SR": 0, "success": False, "error": "timeout_or_hang", "note": "aw_shard_fill stub"}
ordered=[rows[i] for i in sorted(rows)]
p.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in ordered))
print("wrote stub", tid)
PY
  fi
  # merge into main after each task
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
for shard in Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt").glob("usr_fb_aw_ood_shard_*/run-eval_out_of_distribution/results.jsonl"):
    for line in shard.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = int(r["task_idx"])
        if tid not in rows:
            rows[tid] = r
ordered = [rows[i] for i in sorted(rows)]
main.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
print({"n": len(ordered), "SR": sum(int(r.get("SR") or 0) for r in ordered) / len(ordered)})
PY
done
