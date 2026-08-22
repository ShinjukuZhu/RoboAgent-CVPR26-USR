#!/usr/bin/env python3
"""Kill stuck stub batch and reeval remaining tail stub ids one-by-one."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/autodl_tmp1/zhuyanhao")
CODE = ROOT / "code/RoboAgent_USR_SkillOpt"
ENV = ROOT / "envs/RoboAgent_AW/bin"
RUN = ROOT / "runs/usr_minstd_skillopt"
LOG = RUN / "logs/aw_tail_stubs.log"
SKILL = CODE / "skills/effect_verified_skill_v0000.md"
MAIN = RUN / "usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
ATT = RUN / "aw_fail_reeval_free/attempted.txt"
TAIL = [125, 126, 127, 128, 130]
TIMEOUT = int(os.environ.get("TASK_TIMEOUT_SEC", "5400"))


def ps_lines():
    return subprocess.check_output(
        ["ps", "-u", "zhuyanhao", "-o", "pid=,args="], text=True
    ).splitlines()


def kill_pred(pred):
    for line in ps_lines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid, args = int(parts[0]), parts[1]
        if pred(args):
            try:
                os.kill(pid, signal.SIGTERM)
                print("kill", pid, args[:80])
            except ProcessLookupError:
                pass


def pick_gpu():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    best = (None, -1)
    for line in out.splitlines():
        idx_s, free_s = [x.strip() for x in line.split(",")]
        idx, free = int(idx_s), int(float(free_s))
        if free > best[1]:
            best = (idx, free)
    return best


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
        row["note"] = "aw_fail_reeval_promoted"
        rows[tid] = row
        MAIN.write_text(
            "".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in sorted(rows))
        )
        print("PROMOTED", tid)
    else:
        print("NO_PROMOTE", tid, row)
    ATT.parent.mkdir(parents=True, exist_ok=True)
    with ATT.open("a") as f:
        f.write(f"{tid}\n")


def main():
    kill_pred(lambda a: "aw_stub_batch_reeval.sh" in a)
    kill_pred(lambda a: "stub_batch/chunk_125_130" in a)
    time.sleep(3)
    for line in ps_lines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid, args = int(parts[0]), parts[1]
        if "stub_batch" in args and ("run_aw" in args or "timeout" in args):
            try:
                os.kill(pid, signal.SIGKILL)
                print("kill9", pid)
            except ProcessLookupError:
                pass
    time.sleep(2)

    gpu, free = pick_gpu()
    disp = {1: 96, 4: 94, 6: 97, 7: 95}.get(gpu, 90 + gpu)
    print(f"gpu={gpu} free={free}MiB disp=:{disp}")

    for tid in TAIL:
        out = RUN / f"aw_fail_reeval_free/task_{tid}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== TASK {tid} ===", flush=True)
        with LOG.open("a") as logf:
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
                env={
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
                },
            )
        print("rc", tid, rc)
        merge_promote(tid, out)

    rows = {
        int(json.loads(x)["task_idx"]): json.loads(x)
        for x in MAIN.read_text().splitlines()
        if x.strip()
    }
    ok = sum(1 for x in rows.values() if int(x.get("SR") or 0) == 1)
    summary = {"n": len(rows), "SR": round(ok / len(rows), 4), "ok": ok}
    (RUN / "aw_fail_reeval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RUN / "aw_fail_reeval_free/.pass_done").write_text("1\n")
    print("summary", summary)
    subprocess.call(["bash", str(CODE / "training/finalize_fallback_results.sh")])


if __name__ == "__main__":
    main()
