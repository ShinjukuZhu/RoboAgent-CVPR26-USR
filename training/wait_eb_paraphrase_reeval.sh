#!/usr/bin/env bash
# When EB main (unique tasks 0-49) finishes, launch paraphrase re-eval.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
PY=$ROOT/envs/RoboAgent_AW/bin/python
while true; do
  n=$($PY - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
print(len(ids))
print(1 if ids == set(range(50)) else 0)
PY
)
  unique=$(echo "$n" | sed -n '1p')
  complete=$(echo "$n" | sed -n '2p')
  echo "$(date -Is) eb_unique=$unique complete=$complete" >> "$RUN/logs/eb_reeval_waiter.log"
  if [ "${complete:-0}" -eq 1 ]; then
    if pgrep -f "runs/usr_minstd_skillopt/usr_fb_eb50 " >/dev/null \
      || pgrep -f "eb_finish_missing.sh" >/dev/null \
      || pgrep -f "eb_skip44_finish.sh" >/dev/null; then
      echo "$(date -Is) waiting main EB exit" >> "$RUN/logs/eb_reeval_waiter.log"
      sleep 60
      continue
    fi
    if pgrep -f "reeval_eb_paraphrase_fails.sh" >/dev/null \
      || pgrep -f "eb_paraphrase_reeval/task_" >/dev/null; then
      echo "$(date -Is) paraphrase reeval already running" >> "$RUN/logs/eb_reeval_waiter.log"
      sleep 120
      continue
    fi
    if [ -f "$RUN/eb_paraphrase_reeval_summary.json" ]; then
      echo "$(date -Is) already done" >> "$RUN/logs/eb_reeval_waiter.log"
      exit 0
    fi
    echo "$(date -Is) launching paraphrase reeval" >> "$RUN/logs/eb_reeval_waiter.log"
    cd "$CODE" && bash training/reeval_eb_paraphrase_fails.sh >> "$RUN/logs/eb_paraphrase_reeval.log" 2>&1
    exit 0
  fi
  sleep 120
done
