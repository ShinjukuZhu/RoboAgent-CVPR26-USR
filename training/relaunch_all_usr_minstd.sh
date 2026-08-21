#!/bin/bash
# Relaunch fallback AW/EB/SkillOpt under usr_minstd_skillopt. Does not touch V2.
set -euo pipefail
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
EB_ROOT=$ROOT/code/EmbodiedBench
SKILL=$CODE/skills/effect_verified_skill_v0000.md
mkdir -p "$RUN/logs"

pkill -f "runs/usr_minstd_skillopt/usr_fb_aw_ood " || true
pkill -f "runs/usr_minstd_skillopt/usr_fb_eb50 " || true
pkill -f "runs/usr_minstd_skillopt/skillopt" || true
pkill -f "skillopt_evolve.py .*usr_minstd_skillopt" || true
pkill -f "eb_hang_watchdog" || true
sleep 3

AW_N=$(wc -l < "$RUN/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl" | tr -d ' ')
EB_N=$(wc -l < "$RUN/usr_fb_eb50-base/results.jsonl" | tr -d ' ')
echo "AW_N=$AW_N EB_N=$EB_N"

for d in 96 97 98 99; do
  if [ ! -S "/tmp/.X11-unix/X${d}" ]; then
    /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb ":${d}" -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
  fi
done
sleep 2

cd "$CODE"

nohup env -u LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=1 DISPLAY=:96 ALFWORLD_DATA=$ROOT/data/alfworld \
  PATH=$ENV_BIN:$PATH PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
  ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
  ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
  ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
  ROBOAGENT_EVO_SKILL=$SKILL \
  python -u run_aw.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
    --save_path "$RUN/usr_fb_aw_ood" --split eval_out_of_distribution \
    --start "$AW_N" --end 134 --seed 42 \
  >> "$RUN/logs/aw_ood_skill.log" 2>&1 &
echo "AW_PID=$! start=$AW_N"

nohup env -u LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=3 DISPLAY=:99 PATH=$ENV_BIN:$PATH \
  PYTHONPATH=$EB_ROOT:${PYTHONPATH:-} PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
  ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
  ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
  ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
  ROBOAGENT_EVO_SKILL=$SKILL \
  python -u run_ebalf.py --qwen_path "$ROOT/ckpt/RoboAgent_CVPR26" \
    --save_path "$RUN/usr_fb_eb50" \
    --data_path "$EB_ROOT/embodiedbench/envs/eb_alfred/data/splits/splits.json" \
    --split base --server-num 99 --start "$EB_N" --end 50 --seed 42 \
  >> "$RUN/logs/eb50_skill.log" 2>&1 &
echo $! > "$RUN/eb50_skill.pid"
echo "EB_PID=$(cat "$RUN/eb50_skill.pid") start=$EB_N"

SEL_CMD="cd $CODE && env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=4 DISPLAY=:98 ALFWORLD_DATA=$ROOT/data/alfworld PATH=$ENV_BIN:\$PATH PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FALLBACK_USR_SKILLOPT_AUTHORIZED=1 ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 ROBOAGENT_EVO_SKILL={skill} $ENV_BIN/python -u $CODE/run_aw.py --qwen_path $ROOT/ckpt/RoboAgent_CVPR26 --save_path {output}/run --split eval_in_distribution --start 20 --end 40 --seed 42"
nohup "$ENV_BIN/python" -u "$CODE/training/skillopt_evolve.py" \
  --initial-skill "$SKILL" \
  --development-run "$RUN/usr_fb_skillopt_dev-eval_in_distribution" \
  --output "$RUN/skillopt" --rounds 3 \
  --selection-start 20 --selection-end 40 \
  --selection-command "$SEL_CMD" \
  >> "$RUN/logs/skillopt.log" 2>&1 &
echo "SKILLOPT_PID=$!"

pgrep -f wait_eb_paraphrase_reeval.sh >/dev/null || \
  nohup bash "$CODE/training/wait_eb_paraphrase_reeval.sh" >>"$RUN/logs/eb_reeval_waiter_stdout.log" 2>&1 &
pgrep -f wait_official_complete.sh >/dev/null || \
  nohup bash "$CODE/training/wait_official_complete.sh" >>"$RUN/logs/wait_official.log" 2>&1 &

cat > /tmp/relaunch_eb_minstd.sh <<'EOS'
#!/bin/bash
set -e
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
CODE=$ROOT/code/RoboAgent_USR_SkillOpt
ENV_BIN=$ROOT/envs/RoboAgent_AW/bin
EB_ROOT=$ROOT/code/EmbodiedBench
SKILL=$CODE/skills/effect_verified_skill_v0000.md
LOG=$RUN/logs/eb50_skill.log
EB_N=$(wc -l < "$RUN/usr_fb_eb50-base/results.jsonl" | tr -d ' ')
[ "${EB_N:-0}" -ge 50 ] && exit 0
pgrep -f "runs/usr_minstd_skillopt/usr_fb_eb50 " >/dev/null && exit 0
[ -S /tmp/.X11-unix/X99 ] || /mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
sleep 2
cd "$CODE"
nohup env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=3 DISPLAY=:99 PATH=$ENV_BIN:$PATH \
  PYTHONPATH=$EB_ROOT:${PYTHONPATH:-} PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FALLBACK_USR_SKILLOPT_AUTHORIZED=1 \
  ROBOAGENT_OG_BACKEND=llmdet_qwen_usr ROBOAGENT_EG_BACKEND=qwen \
  ROBOAGENT_SD_BACKEND=usr ROBOAGENT_USR_CHANNEL=1 \
  ROBOAGENT_LLMDET_PATH=$ROOT/ckpt/llmdet_large ROBOAGENT_LLMDET_THRESHOLD=0.35 \
  ROBOAGENT_EVO_SKILL=$SKILL \
  python -u run_ebalf.py --qwen_path $ROOT/ckpt/RoboAgent_CVPR26 \
    --save_path $RUN/usr_fb_eb50 \
    --data_path $EB_ROOT/embodiedbench/envs/eb_alfred/data/splits/splits.json \
    --split base --server-num 99 --start "$EB_N" --end 50 --seed 42 \
  >> "$LOG" 2>&1 &
echo $! > "$RUN/eb50_skill.pid"
echo "relaunched EB from $EB_N pid=$(cat $RUN/eb50_skill.pid)"
EOS
chmod +x /tmp/relaunch_eb_minstd.sh

nohup bash -c '
ROOT=/mnt/autodl_tmp1/zhuyanhao
RUN=$ROOT/runs/usr_minstd_skillopt
LOG=$RUN/logs/fallback_keep_alive.log
while true; do
  sleep 240
  date -Is >>"$LOG"
  n=$(wc -l < "$RUN/usr_fb_eb50-base/results.jsonl" 2>/dev/null | tr -d " ")
  if [ "${n:-0}" -lt 50 ] && ! pgrep -f "runs/usr_minstd_skillopt/usr_fb_eb50 " >/dev/null; then
    echo "EB dead; relaunch" >>"$LOG"
    bash /tmp/relaunch_eb_minstd.sh >>"$LOG" 2>&1 || true
  fi
  ps -u zhuyanhao -o pid=,etime=,cmd= | grep usr_minstd_skillopt | grep -E "run_aw|run_ebalf|skillopt_evolve" | grep -v grep >>"$LOG" || true
done
' >/dev/null 2>&1 &
echo "KEEPALIVE=$!"

sleep 18
echo "=== after relaunch ==="
ps -u zhuyanhao -o pid=,etime=,cmd= | grep usr_minstd_skillopt | grep -E 'run_aw|run_ebalf|skillopt' | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
wc -l "$RUN/usr_fb_aw_ood-eval_out_of_distribution/results.jsonl" "$RUN/usr_fb_eb50-base/results.jsonl"
tail -8 "$RUN/logs/eb50_skill.log" || true
