#!/usr/bin/env bash
# SkillOpt on AW ID: D_tr=0-19, D_sel=20-39. Does not read sealed OOD/EB.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
CKPT=$ROOT/ckpt/RoboAgent_CVPR26
RUN=$ROOT/runs/usr_minstd_skillopt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${GPU:-6}
DISPLAY_NUM=${DISPLAY_NUM:-98}
DEV_START=${DEV_START:-0}
ALFWORLD_DATA=${ALFWORLD_DATA:-$ROOT/data/alfworld}
mkdir -p "$RUN/logs" "$RUN/skillopt"
cd "$CODE"
export FALLBACK_USR_SKILLOPT_AUTHORIZED=1

DEV_DIR=$RUN/usr_fb_skillopt_dev-eval_in_distribution
DEV_LOG=$RUN/logs/aw_id_dev.log
touch "$DEV_LOG"
n_done=0
if [[ -f $DEV_DIR/results.jsonl ]]; then
  n_done=$(wc -l < "$DEV_DIR/results.jsonl" | tr -d ' ')
fi
if [[ "$n_done" -lt 20 ]]; then
  START=${DEV_START:-$n_done}
  env -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=$GPU DISPLAY=:$DISPLAY_NUM ALFWORLD_DATA=$ALFWORLD_DATA PATH=$ENV_BIN:$PATH \
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
    ROBOAGENT_OG_BACKEND=llmdet_qwen_usr \
    ROBOAGENT_EG_BACKEND=qwen \
    ROBOAGENT_SD_BACKEND=usr \
    ROBOAGENT_USR_CHANNEL=1 \
    ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large \
    ROBOAGENT_LLMDET_THRESHOLD=0.35 \
    ROBOAGENT_EVO_SKILL=$SKILL \
    python -u run_aw.py --qwen_path "$CKPT" \
      --save_path "$RUN/usr_fb_skillopt_dev" \
      --split eval_in_distribution --start "$START" --end 20 --seed 42 \
    >> "$DEV_LOG" 2>&1
fi

# Absolute paths: selection subprocess must not depend on caller cwd.
SEL_CMD='cd '"$CODE"' && env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES='$GPU' DISPLAY=:'$DISPLAY_NUM' ALFWORLD_DATA='$ALFWORLD_DATA' PATH='$ENV_BIN':$PATH PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FALLBACK_USR_SKILLOPT_AUTHORIZED=1 ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 ROBOAGENT_LLMDET_PATH='$ROOT'/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 ROBOAGENT_EVO_SKILL={skill} '"$ENV_BIN"'/python -u '"$CODE"'/run_aw.py --qwen_path '"$CKPT"' --save_path {output}/run --split eval_in_distribution --start 20 --end 40 --seed 42'

nohup env -u LD_LIBRARY_PATH PATH=$ENV_BIN:$PATH PYTHONUNBUFFERED=1 FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
  "$ENV_BIN/python" -u "$CODE/training/skillopt_evolve.py" \
    --initial-skill "$SKILL" \
    --development-run "$DEV_DIR" \
    --output "$RUN/skillopt" \
    --rounds 3 \
    --selection-start 20 \
    --selection-end 40 \
    --selection-command "$SEL_CMD" \
  >> "$RUN/logs/skillopt.log" 2>&1 &
echo $! > "$RUN/skillopt.pid"
echo "SKILLOPT_PID=$(cat $RUN/skillopt.pid) GPU=$GPU LOG=$RUN/logs/skillopt.log"
