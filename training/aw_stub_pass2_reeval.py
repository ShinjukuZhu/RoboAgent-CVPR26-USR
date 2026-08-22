#!/usr/bin/env python3
"""Second-pass promote-only reeval for hang stubs that failed first batch."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/autodl_tmp1/zhuyanhao")
CODE = ROOT / "code/RoboAgent_USR_SkillOpt"
ENV = ROOT / "envs/RoboAgent_AW/bin"
RUN = ROOT / "runs/usr_minstd_skillopt"
LOG = RUN / "logs/aw_stub_pass2.log"
SKILL = CODE / "skills/effect_verified_skill_v0000.md"
MAIN = RUN / "usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
OUT = RUN / "aw_fail_reeval_free/stub_pass2"
TIMEOUT = int(os.environ.get("TASK_TIMEOUT_SEC", "3600"))
CHUNK = int(os.environ.get("CHUNK", "5"))
EXCLUDE = {int(x) for x in os.environ.get("EXCLUDE_TASKS", "125,126,127,128,130").split(",") if x.strip()}
XVFB = Path("/mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb")


def ensure_display(disp: int):
    sock = Path(f"/tmp/.X11-unix/X{disp}")
    if not sock.exists():
        subprocess.Popen(
            [str(XVFB), f":{disp}", "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX", "+render", "-noreset"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
    rc = subprocess.call(
        "xdpyinfo >/dev/null 2>&1",
        shell=True,
        env={**os.environ, "DISPLAY": f":{disp}"},
    )
    if rc != 0:
        raise RuntimeError(f"DISPLAY :{disp} not ready (xdpyinfo rc={rc})")


def pick_gpu():
    forced = os.environ.get("FORCE_GPU", "").strip()
    if forced:
        idx = int(forced)
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if int(parts[0]) == idx:
                return idx, int(float(parts[1]))
        return idx, 40000
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


def stub_ids():
    rows = {
        int(json.loads(x)["task_idx"]): json.loads(x)
        for x in MAIN.read_text().splitlines()
        if x.strip()
    }
    stubs = []
    for tid, row in sorted(rows.items()):
        if tid in EXCLUDE or int(row.get("SR") or 0) != 0:
            continue
        blob = (str(row.get("note", "")) + " " + str(row.get("error", ""))).lower()
        if any(k in blob for k in ("stub", "timeout", "hang", "watchdog")):
            stubs.append(tid)
    return stubs


def merge_promote(tid: int, cand_dir: Path):
    cands = [
        cand_dir / "run-eval_out_of_distribution/results.jsonl",
        Path(str(cand_dir) + "-eval_out_of_distribution") / "results.jsonl",
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
        row["note"] = "aw_fail_reeval_promoted"
        rows[tid] = row
        MAIN.write_text(
            "".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in sorted(rows))
        )
        print("PROMOTED", tid)
        return True
    print("NO_PROMOTE", tid, row)
    return False


def run_chunk(gpu: int, free: int, tasks: list[int]):
    disp = {1: 96, 2: 92, 4: 94, 6: 97, 7: 95}.get(gpu, 90 + gpu)
    ensure_display(disp)
    tag = f"chunk_{tasks[0]}_{tasks[-1]}"
    cand_root = OUT / tag
    cand_root.mkdir(parents=True, exist_ok=True)
    tasks_s = ",".join(map(str, tasks))
    print(f"run {tasks_s} on GPU{gpu}", flush=True)
    with LOG.open("a") as logf:
        logf.write(f"\n=== {tag} tasks={tasks_s} ===\n")
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
                str(cand_root / "run"),
                "--split",
                "eval_out_of_distribution",
                "--tasks",
                tasks_s,
                "--seed",
                "42",
            ],
            cwd=str(CODE),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "LD_LIBRARY_PATH": "",
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "DISPLAY": f":{disp}",
                "ALFWORLD_DATA": str(ROOT / "data/alfworld"),
                "PATH": f"{ENV}:{os.environ.get('PATH', '')}",
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "ROBOAGENT_MAX_GPU_MIB": str(min(free, 45000)),
                "ROBOAGENT_MAX_AW_STEPS": os.environ.get("ROBOAGENT_MAX_AW_STEPS", "0"),
                "FALLBACK_USR_SKILLOPT_AUTHORIZED": "1",
                "ROBOAGENT_OG_BACKEND": "llmdet_qwen_usr",
                "ROBOAGENT_EG_BACKEND": "qwen",
                "ROBOAGENT_SD_BACKEND": "usr",
                "ROBOAGENT_USR_CHANNEL": "1",
                "ROBOAGENT_LLMDET_PATH": str(ROOT / "ckpt/llmdet_large"),
                "ROBOAGENT_LLMDET_THRESHOLD": "0.35",
                "ROBOAGENT_EVO_SKILL": str(SKILL),
            },
        )
    prom = 0
    for tid in tasks:
        if merge_promote(tid, cand_root):
            prom += 1
    print(json.dumps({"tag": tag, "rc": rc, "promoted": prom}))
    return prom


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stubs = stub_ids()
    if not stubs:
        print("no stubs")
        return
    gpu, free = pick_gpu()
    print(f"pass2 GPU{gpu} free={free} n={len(stubs)} chunk={CHUNK}")
    total = 0
    for i in range(0, len(stubs), CHUNK):
        total += run_chunk(gpu, free, stubs[i : i + CHUNK])
    rows = {
        int(json.loads(x)["task_idx"]): json.loads(x)
        for x in MAIN.read_text().splitlines()
        if x.strip()
    }
    ok = sum(1 for x in rows.values() if int(x.get("SR") or 0) == 1)
    summary = {"n": len(rows), "SR": round(ok / len(rows), 4), "ok": ok, "pass2_promoted": total}
    (RUN / "aw_stub_pass2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    subprocess.call(["bash", str(CODE / "training/finalize_fallback_results.sh")])
    print("DONE", summary)


if __name__ == "__main__":
    main()
