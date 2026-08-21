#!/usr/bin/env bash
# Re-evaluate AW OOD failures with the current Skill; promote only SR=1 rows.
# Does not overwrite successes. Intended after code/skill hotfixes.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${GPU:-4}
DISPLAY_NUM=${DISPLAY_NUM:-94}
TIMEOUT_SEC=${TIMEOUT_SEC:-900}
OUT=$RUN/aw_fail_reeval
LOG=$RUN/logs/aw_fail_reeval.log
mkdir -p "$RUN/logs" "$OUT"

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${DISPLAY_NUM}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
  sleep 2
fi

FAILS=$("$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
root=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main=root/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows={int(json.loads(x)["task_idx"]):json.loads(x) for x in main.read_text().splitlines() if x.strip()} if main.exists() else {}
for pat in ("usr_fb_aw_ood_shard_*/run-eval_out_of_distribution/results.jsonl",
            "usr_fb_aw_ood_wd_*/run-eval_out_of_distribution/results.jsonl"):
    for p in root.glob(pat):
        for line in p.read_text().splitlines():
            if line.strip():
                r=json.loads(line); tid=int(r["task_idx"])
                if tid not in rows: rows[tid]=r
fails=sorted(tid for tid,r in rows.items() if int(r.get("SR") or 0)==0 and tid < 134)
print(" ".join(str(i) for i in fails))
print(len(fails))
PY
)
ORDER=$(echo "$FAILS" | head -1)
N=$(echo "$FAILS" | sed -n '2p')
echo "$(date -Is) reeval $N fails: $ORDER" | tee -a "$LOG"
[ -z "${ORDER// }" ] && exit 0

promoted=()
for tid in $ORDER; do
  echo "$(date -Is) reeval task $tid" | tee -a "$LOG"
  cd "$CODE"
  set +e
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=$GPU DISPLAY=:$DISPLAY_NUM ALFWORLD_DATA=$ROOT/data/alfworld \
    PATH=$ENV_BIN:$PATH PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
    ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
    ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
    ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
    ROBOAGENT_EVO_SKILL=$SKILL \
    timeout "$TIMEOUT_SEC" python -u run_aw.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
      --save_path "$OUT/task_${tid}/run" --split eval_out_of_distribution \
      --start "$tid" --end $((tid+1)) --seed 42 \
    >> "$LOG" 2>&1
  rc=$?
  set -e
  "$ENV_BIN/python" - <<PY
import json
from pathlib import Path
tid=int("$tid")
base=Path("$OUT/task_${tid}")
cands=[base/"run-eval_out_of_distribution"/"results.jsonl", Path(str(base)+"-eval_out_of_distribution")/"results.jsonl"]
main=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
summary=Path("$OUT/summary.jsonl")
row=None
for cand in cands:
    if not cand.exists():
        continue
    for line in cand.read_text().splitlines():
        if line.strip():
            r=json.loads(line)
            if int(r["task_idx"])==tid:
                row=r; break
    if row is not None:
        break
rec={"task_idx": tid, "rc": int("$rc"), "SR": None, "promoted": False}
if row is not None:
    rec["SR"]=int(row.get("SR") or 0)
    if rec["SR"]==1:
        rows={}
        if main.exists():
            for line in main.read_text().splitlines():
                if line.strip():
                    r=json.loads(line); rows[int(r["task_idx"])]=r
        row["note"]="aw_fail_reeval_promoted"
        rows[tid]=row
        main.write_text("".join(json.dumps(rows[i], ensure_ascii=False)+"\n" for i in sorted(rows)))
        rec["promoted"]=True
        print("PROMOTED", tid)
with summary.open("a") as f:
    f.write(json.dumps(rec, ensure_ascii=False)+"\n")
print(rec)
PY
done

"$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
main=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
rows=[json.loads(x) for x in main.read_text().splitlines() if x.strip()]
sr=sum(int(r.get("SR") or 0) for r in rows)/len(rows) if rows else None
summary=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/aw_fail_reeval/summary.jsonl")
prom=0
if summary.exists():
    for line in summary.read_text().splitlines():
        if line.strip() and json.loads(line).get("promoted"):
            prom+=1
out={
  "n": len(rows),
  "SR": sr,
  "promoted": prom,
}
Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/aw_fail_reeval_summary.json").write_text(
    json.dumps(out, indent=2)+"\n"
)
print(json.dumps(out, indent=2))
PY
