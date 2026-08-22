#!/usr/bin/env python3
"""Promote-only reeval for AW OOD failures after skill-loop fixes."""
import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/autodl_tmp1/zhuyanhao")
CODE = ROOT / "code/RoboAgent_USR_SkillOpt"
ENV = ROOT / "envs/RoboAgent_AW/bin"
RUN = ROOT / "runs/usr_minstd_skillopt"
LOG = RUN / "logs/aw_fail_skillfix.log"
SKILL = CODE / "skills/effect_verified_skill_v0000.md"
MAIN = RUN / "usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
OUT = RUN / "aw_fail_reeval_free/skillfix"
TIMEOUT = int(os.environ.get("TASK_TIMEOUT_SEC", "5400"))
XVFB = Path("/mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb")
sys.path.insert(0, str(CODE / "training"))
from thor_cleanup import cleanup_thor  # noqa: E402
EXCLUDE = {
    int(x)
    for x in os.environ.get("EXCLUDE_TASKS", "").split(",")
    if x.strip()
}
PRIORITY = [
    int(x)
    for x in os.environ.get(
        "PRIORITY_TASKS",
        "9,27,42,43,56,60,6,20,21,26,29,47,48,49,51,55",
    ).split(",")
    if x.strip()
]


def ensure_display(disp: int):
    sock = Path(f"/tmp/.X11-unix/X{disp}")
    if not sock.exists() and XVFB.exists():
        subprocess.Popen(
            [
                str(XVFB),
                f":{disp}",
                "-screen",
                "0",
                "1280x1024x24",
                "-ac",
                "+extension",
                "GLX",
                "+render",
                "-noreset",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)


def pick_gpu():
    forced = os.environ.get("FORCE_GPU", "").strip()
    if forced:
        return int(forced), 40000
    best = (None, -1)
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in out.splitlines():
        idx, free = [p.strip() for p in line.split(",")]
        idx, free = int(idx), int(float(free))
        if free > best[1]:
            best = (idx, free)
    return best


def fail_ids():
    rows = {
        int(json.loads(x)["task_idx"]): json.loads(x)
        for x in MAIN.read_text().splitlines()
        if x.strip()
    }
    fails = [tid for tid, row in sorted(rows.items()) if int(row.get("SR") or 0) != 1]
    ordered = [tid for tid in PRIORITY if tid in fails and tid not in EXCLUDE]
    ordered += [tid for tid in fails if tid not in ordered and tid not in EXCLUDE]
    return ordered


def merge_promote(tid: int, out: Path):
    cands = [
        out / "run-eval_out_of_distribution/results.jsonl",
        Path(str(out) + "-eval_out_of_distribution") / "results.jsonl",
    ]
    row = None
    for cand in cands:
        if not cand.exists():
            continue
        for line in cand.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r["task_idx"]) == tid:
                row = r
                break
        if row:
            break
    rows = {
        int(json.loads(x)["task_idx"]): json.loads(x)
        for x in MAIN.read_text().splitlines()
        if x.strip()
    }
    if row and int(row.get("SR") or 0) == 1:
        row["note"] = "aw_fail_skillfix_promoted"
        rows[tid] = row
        MAIN.write_text(
            "".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in sorted(rows))
        )
        print("PROMOTED", tid, flush=True)
    else:
        print("NO_PROMOTE", tid, row, flush=True)


def run_env(gpu: int, free: int, disp: int):
    return {
        **os.environ,
        "LD_LIBRARY_PATH": "",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "DISPLAY": f":{disp}",
        "ALFWORLD_DATA": str(ROOT / "data/alfworld"),
        "PATH": f"{ENV}:{os.environ.get('PATH', '')}",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "ROBOAGENT_MAX_GPU_MIB": str(free),
        "FALLBACK_USR_SKILLOPT_AUTHORIZED": "1",
        "ROBOAGENT_OG_BACKEND": "llmdet_qwen_usr",
        "ROBOAGENT_EG_BACKEND": "qwen",
        "ROBOAGENT_SD_BACKEND": "usr",
        "ROBOAGENT_USR_CHANNEL": "1",
        "ROBOAGENT_LLMDET_PATH": str(ROOT / "ckpt/llmdet_large"),
        "ROBOAGENT_LLMDET_THRESHOLD": "0.35",
        "ROBOAGENT_EVO_SKILL": str(SKILL),
        "ROBOAGENT_MAX_AW_STEPS": os.environ.get("ROBOAGENT_MAX_AW_STEPS", "0"),
    }


def main():
    atexit.register(lambda: cleanup_thor("KILL"))
    tasks = fail_ids()
    if not tasks:
        print("no failing tasks")
        return
    gpu, free = pick_gpu()
    disp = {1: 96, 4: 94, 6: 97, 7: 95, 2: 92}.get(gpu, 90 + gpu)
    ensure_display(disp)
    cleanup_thor("KILL")
    print(f"gpu={gpu} free={free}MiB disp=:{disp} tasks={tasks}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as logf:
        logf.write(f"\n=== skillfix batch n={len(tasks)} gpu={gpu} ===\n")
        logf.flush()
        for tid in tasks:
            out = OUT / f"task_{tid}"
            out.mkdir(parents=True, exist_ok=True)
            logf.write(f"\n=== TASK {tid} ===\n")
            logf.flush()
            rc = subprocess.call(
                [
                    "timeout",
                    str(TIMEOUT),
                    str(ENV / "python"),
                    "-u",
                    "run_aw.py",
                    "--qwen_path",
                    str(ROOT / "ckpt/RoboAgent_CVPR26"),
                    "--save_path",
                    str(out / "run"),
                    "--split",
                    "eval_out_of_distribution",
                    "--start",
                    str(tid),
                    "--end",
                    str(tid + 1),
                    "--seed",
                    "42",
                ],
                cwd=str(CODE),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=run_env(gpu, free, disp),
            )
            print("rc", tid, rc, flush=True)
            merge_promote(tid, out)
            cleanup_thor("TERM")
            time.sleep(2)
            cleanup_thor("KILL")
    subprocess.call(["bash", str(CODE / "training/finalize_fallback_results.sh")])


if __name__ == "__main__":
    main()
