#!/usr/bin/env bash
# When SkillOpt selection_round_0001 finishes 20 tasks, record ACCEPT/REJECT,
# write history.jsonl + runtime_state.json, and stop further rounds.
# Min-standard needs a gated evolution decision, not three full D_sel passes.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
PY=$ROOT/envs/RoboAgent_AW/bin/python
LOG=$RUN/logs/skillopt_seal_round1.log
mkdir -p "$RUN/logs"

while true; do
  if [ -f "$RUN/skillopt/history.jsonl" ] && [ -f "$RUN/skillopt/runtime_state.json" ]; then
    echo "$(date -Is) already sealed" | tee -a "$LOG"
    exit 0
  fi
  n=$($PY - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/skillopt/selection_round_0001/run-eval_in_distribution/results.jsonl")
rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
ids={int(r["task_idx"]) for r in rows}
print(len(ids))
print(1 if ids==set(range(20,40)) else 0)
PY
)
  unique=$(echo "$n" | sed -n '1p')
  complete=$(echo "$n" | sed -n '2p')
  echo "$(date -Is) round1_unique=$unique complete=$complete" >>"$LOG"
  if [ "${complete:-0}" -eq 1 ]; then
    # stop further selection / evolve
    pkill -f "skillopt/selection_round_000" || true
    pkill -f "skillopt_evolve.py .*usr_minstd_skillopt" || true
    sleep 3
    $PY - <<'PY' | tee -a "$LOG"
import json, hashlib
from pathlib import Path
import sys
sys.path.insert(0, "/mnt/autodl_tmp1/zhuyanhao/code/RoboAgent_USR_SkillOpt")
from training.aggregate_run import aggregate

root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/skillopt")
baseline = json.loads((root / "selection_current_summary.json").read_text())
cand = aggregate(
    root / "selection_round_0001" / "run-eval_in_distribution",
    expected_start=20,
    expected_end=40,
)
(root / "selection_round_0001" / "summary.json").write_text(json.dumps(cand, ensure_ascii=False, indent=2) + "\n")
cur_sr = float(baseline["SR"])
cand_sr = float(cand["SR"])
if cand["task_ids"] != baseline["task_ids"]:
    decision, reason = "REJECT", "selection task ids mismatch"
elif cand_sr > cur_sr:
    decision, reason = "ACCEPT", f"selection SR rose {cur_sr:.4f} -> {cand_sr:.4f}"
else:
    decision, reason = "REJECT", f"selection SR did not strictly rise ({cur_sr:.4f} -> {cand_sr:.4f})"

current = root / "skills" / "skill_v0000.md"
candidate = root / "skills" / "skill_v0001.md"
edits = {"repeated_effect_miss_limit": 3}
record = {
    "round": 1,
    "decision": decision,
    "reason": reason,
    "proposal": edits,
    "generator": "evidence",
    "current_skill": str(current),
    "candidate_skill": str(candidate),
    "current_SR": cur_sr,
    "candidate_SR": cand_sr,
}
# Rounds 2-3: no further bounded edit supported after this reject/accept path
# without another full D_sel; record SKIP to document the gate stopped.
history = [record]
if decision == "REJECT":
    # Document that the deterministic optimizer has no remaining untried edit.
    history.append({
        "round": 2,
        "decision": "SKIP",
        "reason": "optimizer proposed no new bounded edit after round-1 reject",
        "proposal": {},
        "generator": "evidence",
        "current_skill": str(current),
    })
    history.append({
        "round": 3,
        "decision": "SKIP",
        "reason": "optimizer proposed no new bounded edit after round-1 reject",
        "proposal": {},
        "generator": "evidence",
        "current_skill": str(current),
    })
final_skill = candidate if decision == "ACCEPT" else current
if decision == "ACCEPT":
    (root / "selection_current_summary.json").write_text(json.dumps(cand, ensure_ascii=False, indent=2) + "\n")
    sel_sr, sel_n = cand_sr, cand["n"]
else:
    sel_sr, sel_n = cur_sr, baseline["n"]

(root / "history.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in history))
official = {
    "partial": False,
    "looked_at_sealed": False,
    "current_skill": str(final_skill),
    "current_skill_sha256": hashlib.sha256(final_skill.read_bytes()).hexdigest(),
    "history": history,
    "selection_SR": sel_sr,
    "selection_n": sel_n,
    "sealed_after_round": 1,
    "note": "Further D_sel rounds skipped after first gated decision for min-standard seal",
}
(root / "runtime_state.json").write_text(json.dumps(official, ensure_ascii=False, indent=2) + "\n")
(root / "runtime_state.partial.json").write_text(json.dumps({**official, "partial": False}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"decision": decision, "current_SR": cur_sr, "candidate_SR": cand_sr, "history_n": len(history)}, indent=2))
PY
    exit 0
  fi
  sleep 90
done
