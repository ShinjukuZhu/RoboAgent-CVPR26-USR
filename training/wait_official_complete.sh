#!/usr/bin/env bash
# When official AW/EB finishes, write a final summary JSON. Does not touch V2.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/fallback_usr_skillopt
while true; do
  set +e
  /mnt/autodl_tmp1/zhuyanhao/envs/RoboAgent_AW/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/fallback_usr_skillopt")
aw = root / "usr_fb_aw_ood-eval_out_of_distribution" / "results.jsonl"
eb = root / "usr_fb_eb50-base" / "results.jsonl"
def load(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
aw_rows, eb_rows = load(aw), load(eb)
aw_ids = [int(r["task_idx"]) for r in aw_rows]
eb_ids = [int(r["task_idx"]) for r in eb_rows]
ready = aw_ids == list(range(134)) and eb_ids == list(range(50))
payload = {
    "ready": ready,
    "aw": {
        "n": len(aw_rows),
        "SR": (sum(int(r.get("SR") or 0) for r in aw_rows) / len(aw_rows)) if aw_rows else None,
    },
    "eb": {
        "n": len(eb_rows),
        "SR": (sum(int(r.get("SR") or 0) for r in eb_rows) / len(eb_rows)) if eb_rows else None,
    },
    "baselines": {
        "native_aw": 0.81, "align_aw": 0.84,
        "native_eb": 0.78, "align_eb": 0.80, "align_usr_eb": 0.78,
    },
}
(root / "final_ready.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
raise SystemExit(0 if ready else 1)
PY
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "FINAL_READY $(date -Is)" | tee -a "$RUN/logs/finalize.log"
    bash "$CODE/training/finalize_fallback_results.sh" | tee -a "$RUN/logs/finalize.log"
    exit 0
  fi
  sleep 300
done
