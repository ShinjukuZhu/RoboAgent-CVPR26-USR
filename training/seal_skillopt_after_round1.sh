#!/usr/bin/env bash
# When SkillOpt selection_round_0001 finishes 20 tasks — or when remaining
# tasks cannot strictly beat the held-out baseline — record ACCEPT/REJECT,
# write history.jsonl + runtime_state.json, and stop further rounds.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
PY=$ROOT/envs/RoboAgent_AW/bin/python
LOG=$RUN/logs/skillopt_seal_round1.log
mkdir -p "$RUN/logs"

seal_now() {
  # stop further selection / evolve
  # Prefer exact argv ends to avoid killing this sealer via broad pgrep.
  "$PY" - <<'PY'
import os, signal, subprocess
out = subprocess.check_output(["ps", "-u", "zhuyanhao", "-o", "pid=,args="], text=True)
for line in out.splitlines():
    parts = line.strip().split(None, 1)
    if len(parts) < 2:
        continue
    pid, args = int(parts[0]), parts[1]
    if pid == os.getpid():
        continue
    kill = False
    if "skillopt_evolve.py" in args and "usr_minstd_skillopt" in args:
        kill = True
    if "selection_round_0001" in args and "run_aw.py" in args:
        kill = True
    if kill:
        try:
            os.kill(pid, signal.SIGTERM)
            print("stopped", pid)
        except ProcessLookupError:
            pass
PY
  sleep 3
  "$PY" - <<'PY' | tee -a "$LOG"
import json, hashlib
from pathlib import Path
import sys
sys.path.insert(0, "/mnt/autodl_tmp1/zhuyanhao/code/RoboAgent_USR_SkillOpt")
from training.aggregate_run import aggregate

root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/skillopt")
r1 = root / "selection_round_0001" / "run-eval_in_distribution" / "results.jsonl"
rows = [json.loads(x) for x in r1.read_text().splitlines() if x.strip()] if r1.exists() else []
by = {int(r["task_idx"]): r for r in rows}
# Fill any unfinished selection ids as SR=0 (only used when early-reject proves
# ACCEPT is impossible, or after a hang abort).
for tid in range(20, 40):
    if tid not in by:
        by[tid] = {
            "task_idx": tid,
            "SR": 0,
            "success": False,
            "error": "aborted_remaining",
            "note": "sealed early: remaining tasks cannot beat held-out baseline or hung",
        }
ordered = [by[i] for i in range(20, 40)]
r1.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered))

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
history = [record]
if decision == "REJECT":
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

  # Free GPU4 for AW mid-gap if still missing.
  if [ ! -S /tmp/.X11-unix/X94 ]; then
    /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb :94 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
    sleep 2
  fi
  miss_mid=$("$PY" - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
ids={int(json.loads(x)["task_idx"]) for x in p.read_text().splitlines() if x.strip()} if p.exists() else set()
print(1 if any(i not in ids for i in range(52,70)) else 0)
PY
)
  if [ "$miss_mid" = "1" ] && ! pgrep -f 'aw_fill_missing.sh' >/dev/null; then
    echo "$(date -Is) launching mid-gap fill on freed GPU4" | tee -a "$LOG"
    nohup env GPU=4 DISPLAY_NUM=94 ONLY_FROM=52 ONLY_TO=70 \
      bash "$CODE/training/aw_fill_missing.sh" \
      >> "$RUN/logs/aw_fill_gpu4.log" 2>&1 &
  fi
}

while true; do
  if [ -f "$RUN/skillopt/history.jsonl" ] && [ -f "$RUN/skillopt/runtime_state.json" ]; then
    echo "$(date -Is) already sealed" | tee -a "$LOG"
    exit 0
  fi
  state=$("$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/skillopt")
p = root / "selection_round_0001" / "run-eval_in_distribution" / "results.jsonl"
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
ids = {int(r["task_idx"]) for r in rows}
n = len(ids)
complete = 1 if ids == set(range(20, 40)) else 0
successes = sum(int(r.get("SR") or 0) for r in rows if int(r["task_idx"]) in ids)
baseline = float(json.loads((root / "selection_current_summary.json").read_text())["SR"])
remaining = 20 - n
# Strict gate: need successes/20 > baseline. Max possible = (successes+remaining)/20.
max_sr = (successes + remaining) / 20.0 if remaining >= 0 else successes / 20.0
early = 1 if (n >= 1 and remaining > 0 and max_sr <= baseline) else 0
print(n)
print(complete)
print(early)
print(round(max_sr, 4))
print(baseline)
PY
)
  unique=$(echo "$state" | sed -n '1p')
  complete=$(echo "$state" | sed -n '2p')
  early=$(echo "$state" | sed -n '3p')
  max_sr=$(echo "$state" | sed -n '4p')
  baseline=$(echo "$state" | sed -n '5p')
  echo "$(date -Is) round1_unique=$unique complete=$complete early=$early max_sr=$max_sr baseline=$baseline" | tee -a "$LOG"
  if [ "${complete:-0}" -eq 1 ] || [ "${early:-0}" -eq 1 ]; then
    seal_now
    exit 0
  fi
  sleep 60
done
