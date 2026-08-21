#!/usr/bin/env bash
# Official AW OOD (n=134), Align+USR + effect-verified Skill.
# Does NOT use the V2 RoboAgent_CVPR26 working tree.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
CKPT=$ROOT/ckpt/RoboAgent_CVPR26
RUN=$ROOT/runs/fallback_usr_skillopt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${GPU:-0}
DISPLAY_NUM=${DISPLAY_NUM:-96}
ALFWORLD_DATA=${ALFWORLD_DATA:-$ROOT/data/alfworld}
mkdir -p "$RUN/logs"
cd "$CODE"
LOG=$RUN/logs/aw_ood_skill.log
: > "$LOG"
nohup env -u LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=$GPU DISPLAY=:$DISPLAY_NUM ALFWORLD_DATA=$ALFWORLD_DATA PATH=$ENV_BIN:$PATH \
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ROBOAGENT_OG_BACKEND=llmdet_qwen_usr \
  ROBOAGENT_EG_BACKEND=qwen \
  ROBOAGENT_SD_BACKEND=usr \
  ROBOAGENT_USR_CHANNEL=1 \
  ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large \
  ROBOAGENT_LLMDET_THRESHOLD=0.35 \
  ROBOAGENT_EVO_SKILL=$SKILL \
  python -u run_aw.py --qwen_path "$CKPT" \
    --save_path "$RUN/sealed_aw_ood" \
    --split eval_out_of_distribution --start 0 --end 134 --seed 42 \
  > "$LOG" 2>&1 &
echo $! > "$RUN/aw_ood_skill.pid"
echo "AW_PID=$(cat $RUN/aw_ood_skill.pid) GPU=$GPU DISPLAY=:$DISPLAY_NUM LOG=$LOG"
