#!/usr/bin/env bash
# Periodic status dump for the fallback branch. Does not touch V2 trees.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
while true; do
  /mnt/autodl_tmp1/zhuyanhao/envs/RoboAgent_AW/bin/python - <<'PY'
import json, time
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
payload = {"ts": time.time(), "runs": {}}
for name, need in {
    "usr_fb_aw_ood-eval_out_of_distribution": 134,
    "usr_fb_eb50-base": 50,
    "usr_fb_skillopt_dev-eval_in_distribution": 20,
}.items():
    path = root / name / "results.jsonl"
    rows = []
    if path.exists():
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ids = [int(r["task_idx"]) for r in rows]
    if "aw_ood" in name:
        expected = list(range(0, 134))
    elif name.endswith("base"):
        expected = list(range(0, 50))
    else:
        expected = list(range(0, 20))
    payload["runs"][name] = {
        "n": len(rows),
        "need": need,
        "SR": (sum(int(r.get("SR") or 0) for r in rows) / len(rows)) if rows else None,
        "task_ids": ids,
        "dups": len(ids) - len(set(ids)),
        "complete": ids == expected and len(ids) == need and len(ids) == len(set(ids)),
    }
(root / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({k: {"n": v["n"], "SR": v["SR"], "complete": v["complete"]} for k, v in payload["runs"].items()}))
PY
  sleep 180
done
