#!/usr/bin/env bash
# When AW134 + EB50 + SkillOpt history are present, finalize reports in-repo.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
PY=$ROOT/envs/RoboAgent_AW/bin/python
LOG=$RUN/logs/wait_and_finalize.log
mkdir -p "$RUN/logs"

while true; do
  ready=$("$PY" - <<'PY'
import json
from pathlib import Path
root=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main=root/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows={}
if main.exists():
    for line in main.read_text().splitlines():
        if line.strip():
            r=json.loads(line); rows[int(r["task_idx"])]=r
for pat in ("usr_fb_aw_ood_shard_*/run-eval_out_of_distribution/results.jsonl",
            "usr_fb_aw_ood_wd_*/run-eval_out_of_distribution/results.jsonl"):
    for p in root.glob(pat):
        for line in p.read_text().splitlines():
            if line.strip():
                r=json.loads(line); tid=int(r["task_idx"])
                if tid not in rows: rows[tid]=r
# merge back
if rows:
    ordered=[rows[i] for i in sorted(rows)]
    main.parent.mkdir(parents=True, exist_ok=True)
    main.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in ordered))
eb=root/"usr_fb_eb50-base/results.jsonl"
eb_ids={int(json.loads(x)["task_idx"]) for x in eb.read_text().splitlines() if x.strip()} if eb.exists() else set()
hist=(root/"skillopt/history.jsonl").exists()
aw_ok=set(rows)==set(range(134))
eb_ok=eb_ids==set(range(50))
print(1 if (aw_ok and eb_ok and hist) else 0)
print(len(rows))
print(len(eb_ids))
print(1 if hist else 0)
if rows:
    print(round(sum(int(r.get("SR") or 0) for r in rows.values())/len(rows),4))
else:
    print(0)
PY
)
  ready_flag=$(echo "$ready" | sed -n '1p')
  aw_n=$(echo "$ready" | sed -n '2p')
  eb_n=$(echo "$ready" | sed -n '3p')
  hist=$(echo "$ready" | sed -n '4p')
  aw_sr=$(echo "$ready" | sed -n '5p')
  echo "$(date -Is) aw_n=$aw_n eb_n=$eb_n hist=$hist aw_sr=$aw_sr ready=$ready_flag" | tee -a "$LOG"
  if [ "$ready_flag" = "1" ]; then
    if [ ! -f "$RUN/aw_fail_reeval_summary.json" ]; then
      echo "$(date -Is) running fail reeval before finalize" | tee -a "$LOG"
      # Prefer a free GPU; default GPU4/DISPLAY94
      GPU=${REEVAL_GPU:-4} DISPLAY_NUM=${REEVAL_DISPLAY:-94} \
        bash "$CODE/training/aw_fail_reeval.sh" | tee -a "$LOG" || true
    fi
    bash "$CODE/training/finalize_fallback_results.sh" | tee -a "$LOG"
    echo "$(date -Is) FINALIZED" | tee -a "$LOG"
    exit 0
  fi
  sleep 180
done
