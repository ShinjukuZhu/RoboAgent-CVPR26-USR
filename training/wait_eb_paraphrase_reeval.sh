#!/usr/bin/env bash
# When EB main (0-49) finishes, launch paraphrase re-eval on GPU 5 if free.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
while true; do
  n=$(wc -l < "$RUN/usr_fb_eb50-base/results.jsonl" 2>/dev/null | tr -d '[:space:]' || echo 0)
  echo "$(date -Is) eb_n=$n" >> "$RUN/logs/eb_reeval_waiter.log"
  if [ "${n:-0}" -ge 50 ]; then
    # wait until main EB process exits
    if pgrep -f 'usr_fb_eb50 .*run_ebalf' >/dev/null; then
      echo "$(date -Is) waiting main EB exit" >> "$RUN/logs/eb_reeval_waiter.log"
      sleep 60
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
