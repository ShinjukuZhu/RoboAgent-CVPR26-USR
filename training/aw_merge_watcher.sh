#!/usr/bin/env bash
# Periodically merge completed AW shard results into the main OOD jsonl.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
PY=$ROOT/envs/RoboAgent_AW/bin/python
LOG=$RUN/logs/aw_merge_watcher.log
mkdir -p "$RUN/logs"
while true; do
  $PY - <<'PY' >>"$LOG" 2>&1 || true
import json
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main = root / "usr_fb_aw_ood-eval_out_of_distribution" / "results.jsonl"
rows = {}
if main.exists():
    for line in main.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["task_idx"])] = r
added = []
for shard in sorted(root.glob("usr_fb_aw_ood_shard_*/run-eval_out_of_distribution/results.jsonl")):
    for line in shard.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = int(r["task_idx"])
        if tid not in rows:
            rows[tid] = r
            added.append(tid)
if added:
    ordered = [rows[i] for i in sorted(rows)]
    main.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    sr = sum(int(r.get("SR") or 0) for r in ordered) / len(ordered)
    print(json.dumps({"added": added, "n": len(ordered), "SR": sr}))
miss = [i for i in range(134) if i not in rows]
payload = {
    "n": len(rows),
    "SR": (sum(int(r.get("SR") or 0) for r in rows.values()) / len(rows)) if rows else None,
    "added": added,
    "missing_n": len(miss),
}
if not miss:
    payload["complete"] = True
    print(json.dumps(payload))
    raise SystemExit(0)
print(json.dumps(payload))
raise SystemExit(1)
PY
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "$(date -Is) AW merge complete" >>"$LOG"
    exit 0
  fi
  sleep 120
done
