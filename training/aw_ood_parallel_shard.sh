#!/usr/bin/env bash
# Parallel AW OOD shard on a free GPU; merges unique task rows into the main
# usr_fb_aw_ood results without touching V2.
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
SHARD=$RUN/usr_fb_aw_ood_shard_${START}_${END}
LOG=$RUN/logs/aw_ood_shard_${START}_${END}.log
mkdir -p "$RUN/logs" "$SHARD"

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${DISPLAY_NUM}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
  sleep 2
fi

echo "$(date -Is) AW shard start=$START end=$END gpu=$GPU" | tee -a "$LOG"
cd "$CODE"
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
    --save_path "$SHARD/run" --split eval_out_of_distribution \
    --start "$START" --end "$END" --seed 42 \
  >> "$LOG" 2>&1 || echo "$(date -Is) shard exited non-zero" | tee -a "$LOG"

"$ENV_BIN/python" - <<PY
import json
from pathlib import Path
main = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
shard = Path("$SHARD/run-eval_out_of_distribution/results.jsonl")
rows = {}
if main.exists():
    for line in main.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["task_idx"])] = r
added = []
if shard.exists():
    for line in shard.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = int(r["task_idx"])
        if tid not in rows:
            rows[tid] = r
            added.append(tid)
ordered = [rows[i] for i in sorted(rows)]
main.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
sr = sum(int(r.get("SR") or 0) for r in ordered) / len(ordered) if ordered else None
print(json.dumps({"n": len(ordered), "SR": sr, "added": added}))
PY
