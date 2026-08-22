#!/usr/bin/env bash
# Stop all zhuyanhao jobs for USR-SkillOpt + Codex/V2 leftovers. Safe for shared server.
set -uo pipefail
USER_NAME=${USER_NAME:-zhuyanhao}

patterns=(
  'usr_minstd_skillopt'
  'RoboAgent_USR_SkillOpt/training/aw_'
  'run_aw.py'
  'run_ebalf.py'
  'openai_compat_qwen25vl_server'
  'planner_commitment_construct_unblock'
  'RoboAgent_Evo_V2'
  'RoboAgent_Evo_20260820'
  'fallback_keep_alive.sh'
  'watch_fallback_status.sh'
  'wait_and_finalize.sh'
  'skillopt_evolve.py'
)

for pat in "${patterns[@]}"; do
  pkill -u "$USER_NAME" -TERM -f "$pat" 2>/dev/null || true
done
sleep 3
for pat in "${patterns[@]}"; do
  pkill -u "$USER_NAME" -KILL -f "$pat" 2>/dev/null || true
done
pkill -u "$USER_NAME" -TERM -f 'thor-201909061227-Linux64' 2>/dev/null || true
sleep 2
pkill -u "$USER_NAME" -KILL -f 'thor-201909061227-Linux64' 2>/dev/null || true
pkill -u "$USER_NAME" -TERM -f 'xorg-prefix/usr/bin/Xvfb' 2>/dev/null || true
sleep 1
pkill -u "$USER_NAME" -KILL -f 'xorg-prefix/usr/bin/Xvfb' 2>/dev/null || true

echo "thor_left=$(pgrep -u "$USER_NAME" -c -f 'thor-201909061227' || true)"
echo "run_aw_left=$(pgrep -u "$USER_NAME" -c -f 'run_aw.py' || true)"
echo "python_left=$(pgrep -u "$USER_NAME" -c -f 'python' || true)"
