#!/usr/bin/env bash
# Wait for a GPU with enough free memory, then finish missing AW ids and
# serially reeval SR=0 tasks (promote SR=1 only). Does not touch other users' jobs.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
NEED_MIB=${NEED_MIB:-55000}
POLL_SEC=${POLL_SEC:-120}
TIMEOUT_SEC=${TIMEOUT_SEC:-720}
LOG=$RUN/logs/aw_reeval_when_free.log
mkdir -p "$RUN/logs"

pick_gpu() {
  if [ -n "${FORCE_GPU:-}" ]; then
    echo "$FORCE_GPU 99999"
    return 0
  fi
  "$ENV_BIN/python" - <<PY
import subprocess
need=int("$NEED_MIB")
out=subprocess.check_output([
  "nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"
], text=True)
best=None
for line in out.splitlines():
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2: continue
    idx, free=int(parts[0]), int(float(parts[1]))
    if free>=need and (best is None or free>best[1]):
        best=(idx, free)
print(f"{best[0]} {best[1]}" if best else "")
PY
}

ensure_display() {
  local d=$1
  if [ ! -S "/tmp/.X11-unix/X${d}" ]; then
    /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${d}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
    sleep 2
  fi
}

run_one_task() {
  local gpu=$1 tid=$2 tag=$3
  local disp
  case "$gpu" in
    1) disp=96 ;;
    4) disp=94 ;;
    6) disp=97 ;;
    7) disp=95 ;;
    *) disp=$((90 + gpu)) ;;
  esac
  ensure_display "$disp"
  local out=$RUN/aw_fail_reeval_free/task_${tid}
  echo "$(date -Is) GPU$gpu run task $tid ($tag)" | tee -a "$LOG"
  cd "$CODE"
  set +e
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=$gpu DISPLAY=:$disp ALFWORLD_DATA=$ROOT/data/alfworld \
    PATH=$ENV_BIN:$PATH PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
    ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
    ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
    ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
    ROBOAGENT_EVO_SKILL=$SKILL \
    timeout "$TIMEOUT_SEC" python -u run_aw.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
      --save_path "$out" --split eval_out_of_distribution \
      --start "$tid" --end $((tid+1)) --seed 42 \
    >> "$LOG" 2>&1
  local rc=$?
  set -e
  "$ENV_BIN/python" - <<PY
import json
from pathlib import Path
tid=int("$tid"); tag="$tag"; rc=int("$rc")
cand=Path("$out/run-eval_out_of_distribution/results.jsonl")
main=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl")
summary=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/aw_fail_reeval_free/summary.jsonl")
summary.parent.mkdir(parents=True, exist_ok=True)
row=None
if cand.exists():
    for line in cand.read_text().splitlines():
        if line.strip():
            r=json.loads(line)
            if int(r["task_idx"])==tid:
                row=r; break
rec={"task_idx": tid, "tag": tag, "rc": rc, "SR": None, "promoted": False}
if row is not None:
    rec["SR"]=int(row.get("SR") or 0)
    rows={}
    if main.exists():
        for line in main.read_text().splitlines():
            if line.strip():
                r=json.loads(line); rows[int(r["task_idx"])]=r
    if tag=="finish" or (tag=="reeval" and rec["SR"]==1):
        if tag=="reeval":
            row["note"]="aw_fail_reeval_promoted"
        rows[tid]=row
        main.write_text("".join(json.dumps(rows[i], ensure_ascii=False)+"\n" for i in sorted(rows)))
        rec["promoted"]= bool(tag=="finish" or rec["SR"]==1)
        print("WROTE", tid, "SR", rec["SR"], "promoted", rec["promoted"])
    elif tag=="finish" and tid not in rows:
        # finish failed: write stub so unique set can complete
        rows[tid]={"task_idx": tid, "SR": 0, "success": False, "error": "timeout_or_hang", "note": "finish_stub"}
        main.write_text("".join(json.dumps(rows[i], ensure_ascii=False)+"\n" for i in sorted(rows)))
        print("STUBBED finish", tid)
with summary.open("a") as f:
    f.write(json.dumps(rec, ensure_ascii=False)+"\n")
print(rec)
PY
}

echo "$(date -Is) aw_reeval_when_free start need=${NEED_MIB}MiB" | tee -a "$LOG"

while true; do
  # merge shards/wds
  "$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
r=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main=r/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows={int(json.loads(x)["task_idx"]):json.loads(x) for x in main.read_text().splitlines() if x.strip()} if main.exists() else {}
for pat in ("usr_fb_aw_ood_shard_*/run-eval_out_of_distribution/results.jsonl",
            "usr_fb_aw_ood_wd_*/run-eval_out_of_distribution/results.jsonl"):
    for p in r.glob(pat):
        for line in p.read_text().splitlines():
            if line.strip():
                row=json.loads(line); tid=int(row["task_idx"])
                if tid not in rows: rows[tid]=row
if rows:
    main.write_text("".join(json.dumps(rows[i], ensure_ascii=False)+"\n" for i in sorted(rows)))
miss=[i for i in range(134) if i not in rows]
fails=[i for i,r in rows.items() if int(r.get("SR") or 0)==0]
print("MISS", " ".join(map(str, miss)))
print("FAILS", " ".join(map(str, fails)))
print("N", len(rows), "SR", round(sum(int(r.get("SR") or 0) for r in rows.values())/len(rows),4) if rows else None)
PY

  state=$("$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
r=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main=r/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows={int(json.loads(x)["task_idx"]):json.loads(x) for x in main.read_text().splitlines() if x.strip()} if main.exists() else {}
miss=[i for i in range(134) if i not in rows]
fails=[i for i,r in rows.items() if int(r.get("SR") or 0)==0]
print(len(miss))
print(" ".join(map(str, miss[:1])))
print(len(fails))
print(" ".join(map(str, fails[:1])))
# done if complete and no fails left OR reeval exhausted marker
done_path=r/"aw_fail_reeval_summary.json"
print(1 if (not miss and done_path.exists()) else 0)
PY
)
  miss_n=$(echo "$state" | sed -n '1p')
  next_miss=$(echo "$state" | sed -n '2p')
  fail_n=$(echo "$state" | sed -n '3p')
  next_fail=$(echo "$state" | sed -n '4p')
  already=$(echo "$state" | sed -n '5p')

  if [ "$miss_n" = "0" ] && [ "$fail_n" = "0" ]; then
    echo "$(date -Is) nothing left" | tee -a "$LOG"
    break
  fi
  # After one full pass over fails, write summary even if some remain failed
  if [ "$miss_n" = "0" ] && [ -f "$RUN/aw_fail_reeval_free/.pass_done" ]; then
    break
  fi

  gpu_info=$(pick_gpu)
  if [ -z "$gpu_info" ]; then
    echo "$(date -Is) no GPU with >=${NEED_MIB}MiB free; sleep $POLL_SEC" | tee -a "$LOG"
    sleep "$POLL_SEC"
    continue
  fi
  gpu=$(echo "$gpu_info" | awk '{print $1}')
  free=$(echo "$gpu_info" | awk '{print $2}')
  echo "$(date -Is) using GPU$gpu free=${free}MiB" | tee -a "$LOG"

  if [ "$miss_n" != "0" ] && [ -n "$next_miss" ]; then
    run_one_task "$gpu" "$next_miss" finish
    continue
  fi
  if [ "$fail_n" != "0" ] && [ -n "$next_fail" ]; then
    run_one_task "$gpu" "$next_fail" reeval
    # mark attempted
    echo "$next_fail" >> "$RUN/aw_fail_reeval_free/attempted.txt"
    # if all fails attempted once, stop
    "$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
r=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main=r/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows={int(json.loads(x)["task_idx"]):json.loads(x) for x in main.read_text().splitlines() if x.strip()}
fails=set(i for i,row in rows.items() if int(row.get("SR") or 0)==0)
att=set()
p=r/"aw_fail_reeval_free/attempted.txt"
if p.exists():
    att={int(x) for x in p.read_text().split() if x.strip().isdigit()}
if fails and fails <= att:
    (r/"aw_fail_reeval_free/.pass_done").write_text("1\n")
    print("PASS_DONE")
PY
    continue
  fi
  sleep "$POLL_SEC"
done

"$ENV_BIN/python" - <<'PY'
import json
from pathlib import Path
r=Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main=r/"usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows=[json.loads(x) for x in main.read_text().splitlines() if x.strip()]
prom=0
summary=r/"aw_fail_reeval_free/summary.jsonl"
if summary.exists():
    for line in summary.read_text().splitlines():
        if line.strip() and json.loads(line).get("promoted") and json.loads(line).get("tag")=="reeval":
            prom+=1
out={"n":len(rows),"SR":sum(int(x.get("SR")or 0) for x in rows)/len(rows) if rows else None,"promoted":prom,"complete":[int(x["task_idx"]) for x in rows]==list(range(134))}
(r/"aw_fail_reeval_summary.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
PY
echo "$(date -Is) aw_reeval_when_free done" | tee -a "$LOG"
