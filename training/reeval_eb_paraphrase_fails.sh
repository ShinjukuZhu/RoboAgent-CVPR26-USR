#!/usr/bin/env bash
# Re-evaluate EB tasks that failed from known Skill paraphrase bugs, then
# splice successes into usr_fb_eb50-base/results.jsonl.
# Does not touch Codex/V2.
# Parallel: GPUS="6 7" DISPLAYS="90 91" (one worker per GPU).
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
CKPT=$ROOT/ckpt/RoboAgent_CVPR26
RUN=$ROOT/runs/usr_minstd_skillopt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
EB_ROOT=$ROOT/code/EmbodiedBench
EB_DATA=$EB_ROOT/embodiedbench/envs/eb_alfred/data/splits/splits.json
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPUS=(${GPUS:-6 7})
DISPLAYS=(${DISPLAYS:-90 91})
TASKS=${TASKS:-"1 7 19 22 25 28 29 37 46"}
OUT=$RUN/eb_paraphrase_reeval
LOG=$RUN/logs/eb_paraphrase_reeval.log
mkdir -p "$OUT" "$RUN/logs"
cd "$CODE"
export FALLBACK_USR_SKILLOPT_AUTHORIZED=1

# Auto-extend TASKS with sealed fails that logged a hard grounding reject.
AUTO=$($ENV_BIN/python - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base")
p = root / "results.jsonl"
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
fails = [int(r["task_idx"]) for r in rows if not int(r.get("SR") or 0)]
extra = []
skip_errors = {"timeout_or_hang", "put_loop_closed_receptacle"}
fail_meta = {int(r["task_idx"]): r for r in rows if not int(r.get("SR") or 0)}
for tid in fails:
    meta = fail_meta.get(tid) or {}
    if str(meta.get("error") or "") in skip_errors:
        continue
    ep = root / f"episode_{tid}"
    if not ep.exists():
        continue
    for f in ep.rglob("*"):
        if not f.is_file() or f.stat().st_size > 2_000_000:
            continue
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        if "grounding_effect_check" in text and '"verified": false' in text:
            extra.append(tid)
            break
print(" ".join(str(i) for i in sorted(set(extra))))
PY
)
TASKS=$(echo "$TASKS $AUTO" | tr ' ' '\n' | awk 'NF' | sort -n | uniq | tr '\n' ' ')
echo "REEVAL_TASKS=$TASKS" | tee -a "$LOG"

# Skip tasks already successfully reevaluated
TASKS=$($ENV_BIN/python - <<PY
import json
from pathlib import Path
root = Path("$OUT")
want = [int(x) for x in "$TASKS".split() if x.strip()]
keep = []
for tid in want:
    p = root / f"task_{tid}-base" / "results.jsonl"
    ok = False
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("task_idx", -1)) == tid and int(r.get("SR") or 0) == 1:
                ok = True
                break
    if not ok:
        keep.append(tid)
print(" ".join(str(i) for i in keep))
PY
)
echo "REEVAL_REMAINING=$TASKS" | tee -a "$LOG"
[ -z "${TASKS// }" ] && echo "nothing left to reeval" | tee -a "$LOG"

run_one() {
  local tid=$1 gpu=$2 disp=$3
  if [ ! -S "/tmp/.X11-unix/X${disp}" ]; then
    /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${disp}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
    sleep 2
  fi
    echo "$(date -Is) REEVAL task $tid gpu=$gpu display=:$disp" | tee -a "$LOG"
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=$gpu DISPLAY=:$disp PATH=$ENV_BIN:$PATH \
    PYTHONPATH=$EB_ROOT:${PYTHONPATH:-} PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
    ROBOAGENT_OG_BACKEND=llmdet_qwen_usr \
    ROBOAGENT_EG_BACKEND=qwen \
    ROBOAGENT_SD_BACKEND=usr \
    ROBOAGENT_USR_CHANNEL=1 \
    ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large \
    ROBOAGENT_LLMDET_THRESHOLD=0.35 \
    ROBOAGENT_EVO_SKILL=$SKILL \
    python -u run_ebalf.py --qwen_path "$CKPT" \
      --save_path "$OUT/task_$tid" \
      --data_path "$EB_DATA" --split base --server-num "$disp" \
      --start "$tid" --end $((tid + 1)) --seed 42 \
    >> "$LOG" 2>&1 || echo "$(date -Is) task $tid failed/timeout" >>"$LOG"
  # Promote any new SR=1 results immediately so sealed EB SR rises during the run.
  "$ENV_BIN/python" - <<'PY' >>"$LOG" 2>&1 || true
import json, shutil
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main = root / "usr_fb_eb50-base" / "results.jsonl"
backup = root / "usr_fb_eb50-base" / "results.jsonl.pre_paraphrase_reeval"
if main.exists() and not backup.exists():
    shutil.copy2(main, backup)
rows = {int(json.loads(x)["task_idx"]): json.loads(x) for x in main.read_text().splitlines() if x.strip()}
promoted = []
for p in sorted((root / "eb_paraphrase_reeval").glob("task_*-base/results.jsonl")):
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = int(r["task_idx"])
        new_sr = int(r.get("SR") or 0)
        old_sr = int(rows.get(tid, {}).get("SR") or 0)
        if new_sr == 1 and old_sr < 1:
            rows[tid] = r
            promoted.append(tid)
        break
if promoted:
    ordered = [rows[i] for i in sorted(rows)]
    main.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    sr = sum(int(r.get("SR") or 0) for r in ordered) / len(ordered)
    print(json.dumps({"incremental_promoted": promoted, "SR": sr}))
PY
}

# Round-robin worker queues
n_workers=${#GPUS[@]}
declare -a queues
for ((i=0; i<n_workers; i++)); do queues[$i]=""; done
idx=0
for tid in $TASKS; do
  queues[$idx]="${queues[$idx]} $tid"
  idx=$(( (idx + 1) % n_workers ))
done

pids=()
for ((i=0; i<n_workers; i++)); do
  (
    for tid in ${queues[$i]}; do
      run_one "$tid" "${GPUS[$i]}" "${DISPLAYS[$i]}"
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]:-}"; do
  wait "$pid" || true
done

$ENV_BIN/python - <<'PY'
import json, shutil
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt")
main = root / "usr_fb_eb50-base" / "results.jsonl"
reeval_root = root / "eb_paraphrase_reeval"
backup = root / "usr_fb_eb50-base" / "results.jsonl.pre_paraphrase_reeval"
if main.exists() and not backup.exists():
    shutil.copy2(main, backup)
rows = {}
if main.exists():
    for line in main.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[int(r["task_idx"])] = r
replaced = []
for p in sorted(reeval_root.glob("task_*-base/results.jsonl")):
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = int(r["task_idx"])
        old = rows.get(tid)
        new_sr = int(r.get("SR") or 0)
        old_sr = None if old is None else int(old.get("SR") or 0)
        # Only promote successes so a failed reeval cannot clobber a later win.
        if new_sr == 1 and (old_sr or 0) < 1:
            rows[tid] = r
            replaced.append({"task_idx": tid, "old_SR": old_sr, "new_SR": new_sr})
        else:
            replaced.append({"task_idx": tid, "old_SR": old_sr, "new_SR": new_sr, "kept_old": True})
ordered = [rows[i] for i in sorted(rows)]
main.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
sr = sum(int(r.get("SR") or 0) for r in ordered) / len(ordered) if ordered else None
summary = {"n": len(ordered), "SR": sr, "replaced": replaced}
(root / "eb_paraphrase_reeval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
