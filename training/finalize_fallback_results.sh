#!/usr/bin/env bash
# Pull sealed-run summaries into the repo when AW134 + EB50 are complete.
set -euo pipefail
ROOT=${ROOT:-/mnt/autodl_tmp1/zhuyanhao}
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
OUT=${OUT:-$CODE/reports}
PY=$ROOT/envs/RoboAgent_AW/bin/python

$PY - <<'PY'
import json, shutil
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
out = Path("/mnt/autodl_tmp1/zhuyanhao/code/RoboAgent_USR_SkillOpt/reports")
partial = out / "partial_results"
partial.mkdir(parents=True, exist_ok=True)
aw = root / "usr_fb_aw_ood-eval_out_of_distribution" / "results.jsonl"
eb = root / "usr_fb_eb50-base" / "results.jsonl"
sk = root / "usr_fb_skillopt_dev-eval_in_distribution" / "results.jsonl"
skillopt = root / "skillopt"

def load(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

aw_rows, eb_rows, sk_rows = load(aw), load(eb), load(sk)

def dedupe(rows):
    by = {}
    for r in rows:
        by[int(r["task_idx"])] = r
    return [by[i] for i in sorted(by)]

aw_rows, eb_rows, sk_rows = dedupe(aw_rows), dedupe(eb_rows), dedupe(sk_rows)
aw_ok = [int(r["task_idx"]) for r in aw_rows] == list(range(134))
eb_ok = [int(r["task_idx"]) for r in eb_rows] == list(range(50))
def sr(rows):
    return (sum(int(r.get("SR") or 0) for r in rows) / len(rows)) if rows else None

payload = {
    "aw_complete": aw_ok,
    "eb_complete": eb_ok,
    "aw": {"n": len(aw_rows), "SR": sr(aw_rows)},
    "eb": {"n": len(eb_rows), "SR": sr(eb_rows)},
    "skillopt_dev": {"n": len(sk_rows), "SR": sr(sk_rows)},
    "baselines": {"native_aw": 0.81, "align_aw": 0.84, "native_eb": 0.78, "align_eb": 0.80, "align_usr_eb": 0.78},
}
for src, name in [(aw, "usr_fb_aw_ood.jsonl"), (eb, "usr_fb_eb50.jsonl"), (sk, "usr_fb_skillopt_dev.jsonl")]:
    if src.exists():
        shutil.copy2(src, partial / name)
hist = skillopt / "history.jsonl"
if not hist.exists():
    # Older runs stored decisions only inside runtime_state*.json.
    for cand in (skillopt / "runtime_state.json", skillopt / "runtime_state.partial.json"):
        if not cand.exists():
            continue
        payload = json.loads(cand.read_text())
        rows = list(payload.get("history") or [])
        if rows:
            hist.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            break
if hist.exists():
    shutil.copy2(hist, partial / "skillopt_history.jsonl")
    payload["skillopt_history"] = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
# Paraphrase reeval summary if present
reeval_sum = root / "eb_paraphrase_reeval_summary.json"
if reeval_sum.exists():
    shutil.copy2(reeval_sum, partial / "eb_paraphrase_reeval_summary.json")
    payload["eb_paraphrase_reeval"] = json.loads(reeval_sum.read_text())
(out / "final_ready.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if not (aw_ok and eb_ok):
    raise SystemExit(2)

aw_sr, eb_sr = payload["aw"]["SR"], payload["eb"]["SR"]
beats_aw = aw_sr is not None and aw_sr >= 0.84
beats_eb = eb_sr is not None and eb_sr >= 0.78
beats_align_eb = eb_sr is not None and eb_sr >= 0.80
hist_rows = payload.get("skillopt_history") or []
if hist_rows:
    lines = ["| Round | Decision | Reason |", "|---|---|---|"]
    for row in hist_rows:
        lines.append(
            f"| {row.get('round')} | **{row.get('decision')}** | {row.get('reason','')} |"
        )
    skillopt_block = "\n".join(lines)
else:
    skillopt_block = "See `reports/partial_results/skillopt_history.jsonl` when present."
md = f"""# Fallback final results

Protocol: RoboAgent official AW `eval_out_of_distribution` (0–133) and
EB-ALFRED `base` (0–49). Stack: Align+USR + effect-verified Skill
(`usr_fb_*` run dirs on the fallback branch).

Survey / reproduced-workload analysis: `reports/FALLBACK_MIN_STANDARD.md`.

## Sealed SR

| Split | n | SR | Reference to beat |
|---|---:|---:|---|
| AW OOD | 134 | {aw_sr:.4f} | Align 0.84 / Native 0.81 |
| EB base | 50 | {eb_sr:.4f} | Align+USR 0.78 / Align 0.80 |

AW vs Align: {"PASS (≥0.84)" if beats_aw else "BELOW Align 0.84 — inspect failures"}
EB vs Align+USR: {"PASS (≥0.78)" if beats_eb else "BELOW Align+USR 0.78 — inspect failures"}
EB vs Align: {"PASS (≥0.80)" if beats_align_eb else "BELOW Align 0.80"}

## SkillOpt

Gate: ACCEPT only on strict held-out selection SR increase.

{skillopt_block}

## Artifacts

- `reports/partial_results/usr_fb_aw_ood.jsonl`
- `reports/partial_results/usr_fb_eb50.jsonl`
- `reports/partial_results/usr_fb_skillopt_dev.jsonl`
- `reports/partial_results/skillopt_history.jsonl`
"""
(out / "FALLBACK_FINAL_RESULTS.md").write_text(md)
print("WROTE", out / "FALLBACK_FINAL_RESULTS.md")
PY
