#!/usr/bin/env bash
# Fill missing AW OOD task ids one-by-one (skip ranges covered by live shards).
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${GPU:-1}
DISPLAY_NUM=${DISPLAY_NUM:-96}
TIMEOUT_SEC=${TIMEOUT_SEC:-1200}
LOG=$RUN/logs/aw_fill_missing.log
mkdir -p "$RUN/logs"

# Do not pkill sibling fills; keep-alive already owns long-range restarts.

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${DISPLAY_NUM}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
  sleep 2
fi

ONLY_FROM=${ONLY_FROM:-52}
ONLY_TO=${ONLY_TO:-70}
TASKS=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
prefer=[i for i in range(int("$ONLY_FROM"), int("$ONLY_TO")) if i not in ids]
print(" ".join(str(i) for i in prefer))
print("unique", len(ids), "prefer", prefer)
PY
)
echo "plan: $TASKS" | tee -a "$LOG"
ORDER=$(echo "$TASKS" | head -1)

if [ -z "${ORDER// }" ]; then
  echo "no mid-gap tasks left" | tee -a "$LOG"
  exit 0
fi

for tid in $ORDER; do
  have=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
print(int($tid in ids))
PY
)
  if [ "$have" = "1" ]; then
    echo "skip $tid" | tee -a "$LOG"
    continue
  fi
  echo "$(date -Is) AW fill task $tid" | tee -a "$LOG"
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
      --save_path "$RUN/usr_fb_aw_ood" --split eval_out_of_distribution \
      --start "$tid" --end $((tid+1)) --seed 42 \
    >> "$LOG" 2>&1
  rc=$?
  set -e
  have=$("$ENV_BIN/python" - <<PY
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
print(int($tid in ids))
PY
)
  if [ "$have" != "1" ]; then
    echo "$(date -Is) task $tid missing after rc=$rc; write fail stub" | tee -a "$LOG"
    "$ENV_BIN/python" - <<PY
import json
from pathlib import Path
tid=int("$tid")
main=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
rows={}
if main.exists():
    for line in main.read_text().splitlines():
        if line.strip():
            r=json.loads(line); rows[int(r["task_idx"])]=r
if tid not in rows:
    stub={"task_idx": tid, "SR": 0, "success": False, "error": "timeout_or_hang", "note": "aw_fill_missing stub"}
    with main.open("a") as f:
        f.write(json.dumps(stub, ensure_ascii=False)+"\n")
    print("wrote stub", tid)
# keep sorted
rows={int(json.loads(x)["task_idx"]):json.loads(x) for x in main.read_text().splitlines() if x.strip()}
ordered=[rows[i] for i in sorted(rows)]
main.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in ordered))
PY
  fi
done

"$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
main=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
rows={int(json.loads(x)["task_idx"]):json.loads(x) for x in main.read_text().splitlines() if x.strip()}
miss=[i for i in range(134) if i not in rows]
sr=sum(int(r.get("SR") or 0) for r in rows.values())/len(rows) if rows else None
print(json.dumps({"n": len(rows), "SR": sr, "missing": miss}))
PY
