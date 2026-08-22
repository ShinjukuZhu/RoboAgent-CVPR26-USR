#!/usr/bin/env bash
# One-shot status for the USR-SkillOpt goal (AW/EB SR, skillfix, SkillOpt r2).
set -uo pipefail
ROOT=${ROOT:-/mnt/autodl_tmp1/zhuyanhao}
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
PY=${PY:-$ROOT/envs/RoboAgent_AW/bin/python}

"$PY" - <<'PY'
import json, re
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
aw = root / "usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
eb = root / "usr_fb_eb50-base/results.jsonl"

def load(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

ar, er = load(aw), load(eb)
aws = sum(int(r.get("SR") or 0) for r in ar)
ebs = sum(int(r.get("SR") or 0) for r in er)
print(f"AW SR {aws}/{len(ar)} = {aws/len(ar):.4f}  (target 0.84)")
print(f"EB SR {ebs}/{len(er)} = {ebs/len(er):.4f}  (target 0.80)")

sf = root / "logs/aw_fail_skillfix.log"
if sf.exists():
    done = re.findall(r"^(PROMOTED|NO_PROMOTE) (\d+)", sf.read_text(), re.M)
    prom = [t for k, t in done if k == "PROMOTED"]
    print(f"skillfix done={len(done)} promoted={prom[-5:]}")

r2 = root / "skillopt_round2/dev-eval_in_distribution/results.jsonl"
nd = len(load(r2))
print(f"skillopt_r2_dev {nd}/20")
hist = root / "skillopt_round2/history.jsonl"
if hist.exists():
    for line in hist.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            print(f"skillopt_r2 round{row.get('round')} {row.get('decision')}: {row.get('reason','')[:120]}")
PY

echo "--- processes ---"
ps -u "$(whoami)" -o pid,pcpu,etime,args 2>/dev/null | grep -E "skillfix|skillopt_round2|run_aw" | grep -v grep | head -6 || echo "(none)"
echo "thor=$(pgrep -u "$(whoami)" -c -f 'thor-201909061227-Linux64' 2>/dev/null || echo 0)"
