#!/usr/bin/env python3
"""Aggregate one official AW/EB run. Labels are the original SR/GCR fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def find_results(run_dir: Path) -> Path:
    direct = run_dir / "results.jsonl"
    if direct.is_file():
        return direct
    matches = sorted(run_dir.glob("*-eval_*/results.jsonl")) + sorted(run_dir.glob("*-base/results.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"multiple results.jsonl under {run_dir}: {matches}")
    raise FileNotFoundError(f"no results.jsonl in {run_dir}")


def load_rows(run_dir: Path):
    path = find_results(run_dir)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [int(row["task_idx"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task_idx in {path}")
    return sorted(rows, key=lambda row: int(row["task_idx"]))


def aggregate(run_dir: Path, expected_start=None, expected_end=None):
    rows = load_rows(run_dir)
    if expected_start is not None and expected_end is not None:
        expected = list(range(expected_start, expected_end))
        actual = [int(row["task_idx"]) for row in rows]
        if actual != expected:
            raise ValueError(f"task ids mismatch: expected {expected}, got {actual}")
    return {
        "n": len(rows),
        "task_ids": [int(row["task_idx"]) for row in rows],
        "successes": sum(int(row.get("SR", 0)) for row in rows),
        "SR": mean(float(row.get("SR", 0)) for row in rows) if rows else 0.0,
        "mean_GCR": mean(float(row.get("GCR", 0)) for row in rows) if rows else 0.0,
        "per_task": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = aggregate(args.run_dir, args.start, args.end)
    target = args.out or args.run_dir / "summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
