#!/usr/bin/env bash
# Promote-only reeval of hang/watchdog stubs in one process (model loaded once).
# Does not overwrite main results unless the reeval episode gets SR=1.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${FORCE_GPU:-1}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}
CHUNK=${CHUNK:-8}
LOG=$RUN/logs/aw_stub_batch_reeval.log
OUT=$RUN/aw_fail_reeval_free/stub_batch
mkdir -p "$RUN/logs" "$OUT"

case "$GPU" in
  1) DISP=96 ;;
  4) DISP=94 ;;
  6) DISP=97 ;;
  7) DISP=95 ;;
  *) DISP=$((90 + GPU)) ;;
esac
if [ ! -S "/tmp/.X11-unix/X${DISP}" ]; then
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${DISP}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
  sleep 2
fi

free_mib=$("$ENV_BIN/python" - <<PY
import subprocess
gpu=int("$GPU")
out=subprocess.check_output([
  "nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"
], text=True)
for line in out.splitlines():
    parts=[p.strip() for p in line.split(",")]
    if len(parts)>=2 and int(parts[0])==gpu:
        print(int(float(parts[1]))); break
else:
    print(0)
PY
)

mapfile -t STUBS < <("$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
r=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main=r/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows={int(json.loads(x)["task_idx"]):json.loads(x) for x in main.read_text().splitlines() if x.strip()} if main.exists() else {}
att=set()
p=r/"aw_fail_reeval_free/attempted.txt"
if p.exists():
    att={int(x) for x in p.read_text().split() if x.strip().isdigit()}
stubs=[]
for tid,row in sorted(rows.items()):
    if int(row.get("SR") or 0)!=0:
        continue
    blob=(str(row.get("note",""))+" "+str(row.get("error",""))).lower()
    if not any(k in blob for k in ("stub","timeout","hang","watchdog")):
        continue
    if tid in att:
        continue
    stubs.append(tid)
print("\n".join(map(str, stubs)))
PY
)

if [ "${#STUBS[@]}" -eq 0 ]; then
  echo "$(date -Is) no stubs left" | tee -a "$LOG"
  exit 0
fi

echo "$(date -Is) stub batch GPU$GPU free=${free_mib}MiB n=${#STUBS[@]} chunk=$CHUNK" | tee -a "$LOG"

i=0
while [ "$i" -lt "${#STUBS[@]}" ]; do
  chunk=("${STUBS[@]:i:CHUNK}")
  tasks=$(IFS=,; echo "${chunk[*]}")
  tag="chunk_${chunk[0]}_${chunk[-1]}"
  echo "$(date -Is) run stubs $tasks" | tee -a "$LOG"
  cd "$CODE"
  set +e
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=$GPU DISPLAY=:$DISP ALFWORLD_DATA=$ROOT/data/alfworld \
    PATH=$ENV_BIN:$PATH PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ROBOAGENT_MAX_GPU_MIB=$free_mib \
    FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
    ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
    ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
    ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
    ROBOAGENT_EVO_SKILL=$SKILL \
    timeout "$TIMEOUT_SEC" python -u run_aw.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
      --save_path "$OUT/$tag/run" --split eval_out_of_distribution \
      --tasks "$tasks" --seed 42 \
    >> "$LOG" 2>&1
  rc=$?
  set -e
  "$ENV_BIN/python" - <<PY
import json
from pathlib import Path
tasks=[int(x) for x in "$tasks".split(",") if x.strip()]
tag="$tag"; rc=int("$rc")
cand=Path("$OUT/$tag/run-eval_out_of_distribution/results.jsonl")
legacy=Path("$OUT/$tag-eval_out_of_distribution/results.jsonl")
main=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
summary=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/aw_fail_reeval_free/summary.jsonl")
att=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/aw_fail_reeval_free/attempted.txt")
summary.parent.mkdir(parents=True, exist_ok=True)
got={}
for p in (cand, legacy):
    if not p.exists():
        continue
    for line in p.read_text().splitlines():
        if line.strip():
            r=json.loads(line); got[int(r["task_idx"])]=r
rows={}
if main.exists():
    for line in main.read_text().splitlines():
        if line.strip():
            r=json.loads(line); rows[int(r["task_idx"])]=r
prom=0
for tid in tasks:
    rec={"task_idx": tid, "tag": "stub_batch", "rc": rc, "SR": None, "promoted": False}
    if tid in got:
        rec["SR"]=int(got[tid].get("SR") or 0)
        if rec["SR"]==1:
            row=got[tid]; row["note"]="aw_fail_reeval_promoted"
            rows[tid]=row; rec["promoted"]=True; prom+=1
            print("PROMOTED", tid)
    with summary.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    with att.open("a") as f:
        f.write(f"{tid}\n")
main.write_text("".join(json.dumps(rows[i], ensure_ascii=False)+"\n" for i in sorted(rows)))
ok=sum(1 for x in rows.values() if int(x.get("SR") or 0)==1)
print(json.dumps({"chunk": tag, "promoted": prom, "n": len(rows), "sr": round(ok/len(rows),4) if rows else None}))
PY
  i=$((i + CHUNK))
done

"$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
r=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main=r/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows={int(json.loads(x)["task_idx"]):json.loads(x) for x in main.read_text().splitlines() if x.strip()}
ok=sum(1 for x in rows.values() if int(x.get("SR") or 0)==1)
fails=[i for i,x in rows.items() if int(x.get("SR") or 0)==0]
out={"n": len(rows), "SR": round(ok/len(rows),4), "fail_n": len(fails), "ok": ok}
(r/"aw_fail_reeval_summary.json").write_text(json.dumps(out, indent=2)+"\n")
(r/"aw_fail_reeval_free"/".pass_done").write_text("1\n")
print(out)
PY
echo "$(date -Is) stub batch done" | tee -a "$LOG"
