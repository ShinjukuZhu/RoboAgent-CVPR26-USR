#!/usr/bin/env bash
# Re-evaluate EB tasks that failed from known Skill paraphrase bugs, then
# splice successes into usr_fb_eb50-base/results.jsonl.
# Does not touch Codex/V2. Requires GPU free (default GPU=5).
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
CKPT=$ROOT/ckpt/RoboAgent_CVPR26
RUN=$ROOT/runs/usr_minstd_skillopt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
EB_ROOT=$ROOT/code/EmbodiedBench
EB_DATA=$EB_ROOT/embodiedbench/envs/eb_alfred/data/splits/splits.json
SKILL=$CODE/skills/effect_verified_skill_v0000.md
# Prefer a free GPU; GPU5 is often taken by V2 probes.
GPU=${GPU:-6}
DISPLAY_NUM=${DISPLAY_NUM:-90}
# Default known paraphrase false-rejects; caller may override TASKS.
# Extra ids with grounding_effect_check verified=false are appended below.
TASKS=${TASKS:-"1 7 19 22 25 28 29 37 46"}
OUT=$RUN/eb_paraphrase_reeval
mkdir -p "$OUT" "$RUN/logs"
cd "$CODE"
export FALLBACK_USR_SKILLOPT_AUTHORIZED=1

# Ensure display
if [ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${DISPLAY_NUM}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
  sleep 2
fi

# Auto-extend TASKS with sealed fails that logged a hard grounding reject.
AUTO=$($ENV_BIN/python - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt/usr_fb_eb50-base")
rows = []
p = root / "results.jsonl"
if p.exists():
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
fails = [int(r["task_idx"]) for r in rows if not int(r.get("SR") or 0)]
extra = []
for tid in fails:
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
# merge unique
TASKS=$(echo "$TASKS $AUTO" | tr ' ' '\n' | awk 'NF' | sort -n | uniq | tr '\n' ' ')
echo "REEVAL_TASKS=$TASKS"
for tid in $TASKS; do
  start=$tid
  end=$((tid + 1))
  echo "REEVAL task $tid"
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=$GPU DISPLAY=:$DISPLAY_NUM PATH=$ENV_BIN:$PATH \
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
      --data_path "$EB_DATA" --split base --server-num "$DISPLAY_NUM" \
      --start "$start" --end "$end" --seed 42 \
    >> "$RUN/logs/eb_paraphrase_reeval.log" 2>&1 || true
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
        rows[tid] = r
        replaced.append({"task_idx": tid, "old_SR": None if old is None else int(old.get("SR") or 0), "new_SR": int(r.get("SR") or 0)})
ordered = [rows[i] for i in sorted(rows)]
main.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
sr = sum(int(r.get("SR") or 0) for r in ordered) / len(ordered) if ordered else None
summary = {"n": len(ordered), "SR": sr, "replaced": replaced}
(root / "eb_paraphrase_reeval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
