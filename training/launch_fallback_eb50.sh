#!/usr/bin/env bash
# Official EB-ALFRED base (n=50), Align+USR + effect-verified Skill.
# Independent of RoboAgent-Evo V2. Does NOT require V2_FROZEN.json.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
CODE=${CODE:-$ROOT/code/RoboAgent_USR_SkillOpt}
CKPT=$ROOT/ckpt/RoboAgent_CVPR26
RUN=$ROOT/runs/usr_minstd_skillopt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
EB_ROOT=$ROOT/code/EmbodiedBench
EB_DATA=$EB_ROOT/embodiedbench/envs/eb_alfred/data/splits/splits.json
SKILL=$CODE/skills/effect_verified_skill_v0000.md
GPU=${GPU:-5}
DISPLAY_NUM=${DISPLAY_NUM:-97}
START=${START:-0}
END=${END:-50}
mkdir -p "$RUN/logs"
cd "$CODE"
LOG=$RUN/logs/eb50_skill.log
touch "$LOG"
export FALLBACK_USR_SKILLOPT_AUTHORIZED=1
nohup env -u LD_LIBRARY_PATH \
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
    --save_path "$RUN/usr_fb_eb50" \
    --data_path "$EB_DATA" --split base --server-num "$DISPLAY_NUM" \
    --start "$START" --end "$END" --seed 42 \
  >> "$LOG" 2>&1 &
echo $! > "$RUN/eb50_skill.pid"
echo "EB_PID=$(cat $RUN/eb50_skill.pid) GPU=$GPU START=$START END=$END DISPLAY=:$DISPLAY_NUM LOG=$LOG"
