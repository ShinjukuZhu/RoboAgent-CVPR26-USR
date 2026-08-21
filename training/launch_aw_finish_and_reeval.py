#!/usr/bin/env python3
"""Kill overlapping AW workers, finish last missing ids, launch 3-way fail reeval."""
import json, os, signal, subprocess, time
from pathlib import Path

ROOT = Path("/mnt/autodl_tmp1/zhuyanhao")
RUN = ROOT / "runs/usr_minstd_skillopt"
CODE = ROOT / "code/RoboAgent_USR_SkillOpt"
LOG = RUN / "logs"
LOG.mkdir(parents=True, exist_ok=True)


def ps():
    return subprocess.check_output(["ps", "-u", "zhuyanhao", "-o", "pid=,args="], text=True)


def kill_pred(pred, why):
    for line in ps().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid, args = int(parts[0]), parts[1]
        if pred(args):
            try:
                os.kill(pid, signal.SIGTERM)
                print("killed", pid, why)
            except ProcessLookupError:
                pass


kill_pred(lambda a: "usr_fb_aw_ood_wd_" in a, "wd")
kill_pred(lambda a: "aw_range_watchdog.sh" in a, "wd_bash")
kill_pred(lambda a: "aw_fail_reeval" in a, "reeval")
time.sleep(4)
for line in ps().splitlines():
    parts = line.strip().split(None, 1)
    if len(parts) < 2:
        continue
    pid, args = int(parts[0]), parts[1]
    if "usr_fb_aw_ood_wd_" in args and "run_aw.py" in args:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
time.sleep(2)

main = RUN / "usr_fb_aw_ood-eval_out_of_distribution/results.jsonl"
rows = {}
if main.exists():
    for line in main.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["task_idx"])] = r
for pat in (
    "usr_fb_aw_ood_shard_*/run-eval_out_of_distribution/results.jsonl",
    "usr_fb_aw_ood_wd_*/run-eval_out_of_distribution/results.jsonl",
):
    for p in RUN.glob(pat):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                tid = int(r["task_idx"])
                if tid not in rows:
                    rows[tid] = r
main.write_text("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in sorted(rows)))
miss = [i for i in range(134) if i not in rows]
fails = sorted(tid for tid, r in rows.items() if int(r.get("SR") or 0) == 0)
sr = sum(int(r.get("SR") or 0) for r in rows.values()) / len(rows) if rows else None
print(json.dumps({"n": len(rows), "SR": sr, "miss": miss, "fail_n": len(fails)}))

for d in (94, 95, 96, 97):
    if not Path(f"/tmp/.X11-unix/X{d}").exists():
        subprocess.Popen(
            [
                "/mnt/autodl_tmp1/zhuyanhao/xorg-prefix/usr/bin/Xvfb",
                f":{d}",
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
time.sleep(1)

if miss:
    env = os.environ.copy()
    env.update(
        {
            "GPU": "6",
            "DISPLAY_NUM": "97",
            "START": str(min(miss)),
            "END": str(max(miss) + 1),
            "STUCK_SEC": "480",
            "SAVE_TAG": f"usr_fb_aw_ood_wd_finish_{min(miss)}_{max(miss)+1}",
        }
    )
    subprocess.Popen(
        ["bash", str(CODE / "training/aw_range_watchdog.sh")],
        stdout=open(LOG / "aw_finish_last.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    print("started finish", env["START"], env["END"])

chunks = [[], [], []]
for i, tid in enumerate(fails):
    chunks[i % 3].append(tid)

for gpu, disp, chunk, tag in [
    (1, 96, chunks[0], "g1"),
    (4, 94, chunks[1], "g4"),
    (7, 95, chunks[2], "g7"),
]:
    if not chunk:
        continue
    env = os.environ.copy()
    env.update(
        {
            "GPU": str(gpu),
            "DISPLAY_NUM": str(disp),
            "TIMEOUT_SEC": "720",
            "TAG": tag,
            "ONLY_TASKS": " ".join(str(x) for x in chunk),
        }
    )
    subprocess.Popen(
        ["bash", str(CODE / "training/aw_fail_reeval_subset.sh")],
        stdout=open(LOG / f"aw_fail_reeval_{tag}.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    print("started reeval", tag, "n", len(chunk), "head", chunk[:6])

time.sleep(3)
for line in ps().splitlines():
    if any(k in line for k in ["aw_fail_reeval", "aw_range_watchdog", "wd_finish", "usr_fb_aw_ood_wd_"]):
        print(line[:210])
