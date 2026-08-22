#!/usr/bin/env bash
# Resume SkillOpt round-2 D_sel only (do not re-run D_tr). Uses skillopt_evolve resume.
set -euo pipefail
ROOT=${ROOT:-/mnt/autodl_tmp1/zhuyanhao}
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
CKPT=$ROOT/ckpt/RoboAgent_CVPR26
RUN=$ROOT/runs/usr_minstd_skillopt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${GPU:-4}
DISPLAY_NUM=${DISPLAY_NUM:-94}
ALFWORLD_DATA=${ALFWORLD_DATA:-$ROOT/data/alfworld}
OUT=$RUN/skillopt_round2
DEV_DIR=$OUT/dev-eval_in_distribution
SEL_CMD='cd '"$CODE"' && env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES='$GPU' DISPLAY=:'$DISPLAY_NUM' ALFWORLD_DATA='$ALFWORLD_DATA' PATH='$ENV_BIN':$PATH PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROBOAGENT_MAX_AW_STEPS=0 FALLBACK_USR_SKILLOPT_AUTHORIZED=1 ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 ROBOAGENT_LLMDET_PATH='$ROOT'/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 ROBOAGENT_EVO_SKILL={skill} '"$ENV_BIN"'/python -u '"$CODE"'/run_aw.py --qwen_path '"$CKPT"' --save_path {output}/run --split eval_in_distribution --start 20 --end 40 --seed 42'

nohup env -u LD_LIBRARY_PATH PATH=$ENV_BIN:$PATH PYTHONUNBUFFERED=1 FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
  "$ENV_BIN/python" -u "$CODE/training/skillopt_evolve.py" \
    --initial-skill "$SKILL" \
    --development-run "$DEV_DIR" \
    --output "$OUT" \
    --rounds 2 \
    --selection-start 20 \
    --selection-end 40 \
    --selection-command "$SEL_CMD" \
  >> "$RUN/logs/skillopt_round2.log" 2>&1 &
echo $! > "$OUT/pid"
echo "restarted pid=$(cat "$OUT/pid")"
