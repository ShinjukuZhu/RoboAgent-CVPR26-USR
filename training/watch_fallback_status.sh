#!/usr/bin/env bash
# Periodic status dump for the fallback branch. Does not touch V2 trees.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/fallback_usr_skillopt
OUT=$RUN/status.json
while true; do
  /mnt/autodl_tmp1/zhuyanhao/envs/RoboAgent_AW/bin/python - <<'PY'
import json, time
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/fallback_usr_skillopt")
payload = {"ts": time.time(), "runs": {}}
for name, need in {
    "official_aw_ood-eval_out_of_distribution": 134,
    "official_eb50-base": 50,
    "skillopt_dev-eval_in_distribution": 20,
}.items():
    path = root / name / "results.jsonl"
    rows = []
    if path.exists():
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ids = [int(r["task_idx"]) for r in rows]
    payload["runs"][name] = {
        "n": len(rows),
        "need": need,
        "SR": (sum(int(r.get("SR") or 0) for r in rows) / len(rows)) if rows else None,
        "task_ids": ids,
        "dups": len(ids) - len(set(ids)),
        "complete": len(rows) >= need and ids == list(range(need if name.endswith("base") or "ood" in name else 0, (0 if False else need))),
    }
# fix complete flags properly
for name, info in payload["runs"].items():
    need = info["need"]
    if "ood" in name:
        expected = list(range(0, 134))
    elif name.endswith("base"):
        expected = list(range(0, 50))
    else:
        expected = list(range(0, 20))
    info["complete"] = info["task_ids"] == expected and info["dups"] == 0
(root / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({k: {"n": v["n"], "SR": v["SR"], "complete": v["complete"]} for k, v in payload["runs"].items()}))
PY
  sleep 180
done
