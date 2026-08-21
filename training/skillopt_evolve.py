#!/usr/bin/env python3
"""SkillOpt-style evolution of the USR effect-verified Skill.

State machine, matching SkillOpt (arXiv:2605.23904):

  development trajectories
    -> bounded candidate edit of the Skill document
    -> frozen selection-set comparison
    -> ACCEPT / REJECT / SKIP
    -> versioned history and rejected-edit buffer

The executor (agents/evo_skill.py) stays frozen. Only the Skill markdown JSON
is modified. Selection uses official RoboAgent SR. Ties are rejected.
Sealed AW OOD / EB paths are refused during training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.evo_skill import EDITABLE_FIELDS, EvoSkillSpec
from training.aggregate_run import aggregate


def _json_block(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("candidate generator returned no JSON object")
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        from json_repair import repair_json
        return json.loads(repair_json(raw))


def read_skill_payload(path: Path) -> Dict[str, Any]:
    return EvoSkillSpec.from_markdown(path).to_payload()


def write_candidate(current_path: Path, out_path: Path, edits: Dict[str, Any]) -> None:
    current = read_skill_payload(current_path)
    unknown = sorted(set(edits) - EDITABLE_FIELDS)
    if unknown:
        raise ValueError(f"candidate attempted forbidden fields: {unknown}")
    if not edits:
        raise ValueError("candidate contains no edits")
    candidate = dict(current)
    candidate.update(edits)
    candidate["version"] = int(current["version"]) + 1
    body = (
        f"# Effect-Verified Skill v{candidate['version']}\n\n"
        "Candidate produced by SkillOpt training on RoboAgent-USR.\n\n"
        "```json\n"
        + json.dumps(candidate, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    out_path.write_text(body)
    EvoSkillSpec.from_markdown(out_path)


def write_partial_state(output: Path, current: Path, history: List[Dict[str, Any]]) -> None:
    payload = {
        "partial": True,
        "not_final_runtime_state": True,
        "looked_at_sealed": False,
        "current_skill": str(current),
        "current_skill_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        "history": history,
    }
    (output / "runtime_state.partial.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def write_infrastructure_error(output: Path, round_idx: int, stage: str, exc: Exception) -> None:
    payload = {
        "round": round_idx,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "not_a_skillopt_decision": True,
        "retry_from_disk": True,
    }
    (output / "last_infrastructure_error.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def _refuse_sealed(path: Optional[Path]) -> None:
    if path is None:
        return
    text = str(path).lower()
    if "sealed" in text:
        raise ValueError(f"refusing sealed trajectory path: {path}")


def evaluate_trajectories(run_dir: Path, baseline_run: Optional[Path] = None) -> Dict[str, Any]:
    for path in (run_dir, baseline_run):
        _refuse_sealed(path)
        if path is not None and "sealed" in str(path).lower():
            raise ValueError(f"refusing sealed trajectory path: {path}")
    summary = aggregate(run_dir)
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}
    search_root = run_dir
    if not list(run_dir.glob("episode_*/trace.jsonl")):
        nested = list(run_dir.glob("*/episode_*/trace.jsonl"))
        if nested:
            search_root = nested[0].parents[1]
    for trace in sorted(search_root.glob("episode_*/trace.jsonl")):
        for line in trace.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(event.get("event", ""))
            if name not in {
                "action_effect_check",
                "grounding_effect_check",
                "evo_suffix_invalidated",
                "action_precondition_check",
                "perception_stale_after_world_change",
            }:
                continue
            if event.get("verified") is False or name in {
                "evo_suffix_invalidated",
                "perception_stale_after_world_change",
            }:
                counts[name] = counts.get(name, 0) + 1
                examples.setdefault(name, [])
                if len(examples[name]) < 8:
                    examples[name].append(str(event.get("reason") or event.get("action") or "")[:300])
    result = {
        "metrics": {k: v for k, v in summary.items() if k != "per_task"},
        "unverified_event_counts": counts,
        "examples": examples,
    }
    if baseline_run is not None:
        baseline = aggregate(baseline_run)
        candidate_by_id = {int(row["task_idx"]): row for row in summary["per_task"]}
        baseline_by_id = {int(row["task_idx"]): row for row in baseline["per_task"]}
        shared = sorted(set(candidate_by_id) & set(baseline_by_id))
        paired = []
        for task_idx in shared:
            base = baseline_by_id[task_idx]
            cand = candidate_by_id[task_idx]
            paired.append({
                "task_idx": task_idx,
                "baseline_SR": int(base.get("SR") or 0),
                "candidate_SR": int(cand.get("SR") or 0),
            })
        result["paired_baseline_evidence"] = {
            "shared_task_ids": shared,
            "baseline_success_candidate_fail": [
                row["task_idx"] for row in paired
                if row["baseline_SR"] == 1 and row["candidate_SR"] == 0
            ],
            "paired_tasks": paired,
        }
    return result


def rejected_edit_buffer(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buffer: List[Dict[str, Any]] = []
    seen = set()
    for record in history:
        if record.get("decision") != "REJECT":
            continue
        edits = dict(record.get("proposal") or {})
        if not edits:
            current_path = Path(str(record.get("current_skill") or ""))
            candidate_path = Path(str(record.get("candidate_skill") or ""))
            if current_path.is_file() and candidate_path.is_file():
                current = read_skill_payload(current_path)
                candidate = read_skill_payload(candidate_path)
                edits = {
                    key: candidate.get(key)
                    for key in EDITABLE_FIELDS
                    if candidate.get(key) != current.get(key)
                }
        edits.pop("decision", None)
        if not edits:
            continue
        signature = json.dumps(edits, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        buffer.append({
            "round": int(record.get("round") or 0),
            "edits": edits,
            "rejection_reason": str(record.get("reason") or ""),
        })
    return buffer[-12:]


def propose_from_evidence(
    current: Dict[str, Any],
    evidence: Dict[str, Any],
    rejected_edits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Deterministic SkillOpt optimizer when no LLM endpoint is available.

    It only reads development-set event counts and proposes one bounded field
    change. It does not inject task ids or object names.
    """
    rejected = {
        json.dumps(item.get("edits") or {}, sort_keys=True)
        for item in (rejected_edits or [])
    }
    counts = evidence.get("unverified_event_counts") or {}
    candidates: List[Dict[str, Any]] = []
    if counts.get("grounding_effect_check", 0) >= 2 and current.get("grounding_contract_mode") == "literal":
        candidates.append({"grounding_contract_mode": "referential_only"})
    if counts.get("action_effect_check", 0) >= 4 and int(current.get("repeated_effect_miss_limit") or 2) < 3:
        candidates.append({"repeated_effect_miss_limit": int(current["repeated_effect_miss_limit"]) + 1})
    if current.get("scheduler_context_mode") == "always":
        candidates.append({"scheduler_context_mode": "on_intervention"})
    if not current.get("invalidate_perception_after_world_change", True):
        candidates.append({"invalidate_perception_after_world_change": True})
    if current.get("skip_feedback_mode") == "silent":
        candidates.append({"skip_feedback_mode": "virtual_success"})
    if counts.get("evo_suffix_invalidated", 0) >= 3:
        candidates.append({
            "recovery_instruction": (
                "Re-observe, preserve confirmed progress, change viewpoint or "
                "target instance, then replan only the unfinished suffix."
            )
        })
    for edits in candidates:
        signature = json.dumps(edits, sort_keys=True)
        if signature not in rejected:
            return edits
    return {"decision": "skip"}


def propose_with_llm(
    endpoint: str,
    model: str,
    current: Dict[str, Any],
    evidence: Dict[str, Any],
    rejected_edits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    prompt = f"""You are optimizing an embodied robot Skill from real development trajectories.
The Skill runs inside RoboAgent. Modify only fields in this list:
{sorted(EDITABLE_FIELDS)}

Current Skill JSON:
{json.dumps(current, ensure_ascii=False)}

Development evidence:
{json.dumps(evidence, ensure_ascii=False)}

Rejected edit buffer from earlier held-out evaluations:
{json.dumps(rejected_edits or [], ensure_ascii=False)}

Propose one bounded modification that could improve official task success rate.
Do not add benchmark-specific task ids, object names, or answer labels.
Do not repeat an edit in the rejected buffer unless the new proposal changes an
additional causal field. Return ONLY a JSON object of changed fields.
If the evidence does not support a modification, return {{"decision":"skip"}}.
"""
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You propose conservative, testable robot Skill edits."},
            {"role": "user", "content": prompt},
        ],
    }
    request = Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urlopen(request, timeout=180) as response:
        result = json.loads(response.read())
    content = result["choices"][0]["message"]["content"]
    return _json_block(content)


def selection_decision(current: Dict[str, Any], candidate: Dict[str, Any]) -> tuple[str, str]:
    """SkillOpt gate: strict held-out SR improvement. Ties are rejected."""
    if current["task_ids"] != candidate["task_ids"] or not current["task_ids"]:
        return "SKIP", "selection task ids are missing or differ"
    sr_delta = candidate["SR"] - current["SR"]
    if sr_delta > 0:
        return "ACCEPT", f"selection SR improved by {sr_delta:.6f}"
    if sr_delta < 0:
        return "REJECT", f"selection SR regressed by {sr_delta:.6f}"
    return "REJECT", "selection tie does not justify a version change"


def with_resume_start(command: str, start: int) -> str:
    if re.search(r"--start\s+\d+", command):
        return re.sub(r"--start\s+\d+", f"--start {start}", command, count=1)
    return command + f" --start {start}"


def next_start(results_path: Path, start: int, end: int) -> tuple[str, int]:
    if not results_path.is_file():
        return "missing", start
    ids = []
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        ids.append(int(json.loads(line)["task_idx"]))
    expected = list(range(start, end))
    if ids == expected:
        return "complete", end
    if not ids:
        return "missing", start
    if ids != expected[: len(ids)]:
        raise ValueError(f"partial results are not a prefix of {expected}: {ids}")
    return "partial", ids[-1] + 1


def _actual_run_dir(output_dir: Path) -> Path:
    direct = output_dir / "results.jsonl"
    if direct.is_file():
        return output_dir
    matches = sorted(output_dir.glob("run-*"))
    if matches:
        return matches[0]
    return output_dir / "run-eval_in_distribution"


def run_selection(
    command_template: str,
    skill: Path,
    output_dir: Path,
    *,
    selection_start: int = 20,
    selection_end: int = 40,
) -> Dict[str, Any]:
    actual = _actual_run_dir(output_dir)
    status, resume = next_start(actual / "results.jsonl", selection_start, selection_end)
    if status == "complete":
        summary = aggregate(actual, expected_start=selection_start, expected_end=selection_end)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)
    command = with_resume_start(
        command_template.format(skill=str(skill.resolve()), output=str(output_dir.resolve())),
        resume,
    )
    log = output_dir / "command.stdout"
    env = os.environ.copy()
    env["ROBOAGENT_EVO_SKILL"] = str(skill.resolve())
    started = time.time()
    with log.open("w") as handle:
        proc = subprocess.run(command, shell=True, env=env, stdout=handle, stderr=subprocess.STDOUT)
    invocation = {
        "command": command,
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - started,
        "skill_sha256": hashlib.sha256(skill.read_bytes()).hexdigest(),
    }
    (output_dir / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"selection command failed ({proc.returncode}); see {log}")
    actual = _actual_run_dir(output_dir)
    summary = aggregate(actual, expected_start=selection_start, expected_end=selection_end)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--baseline-development-run", type=Path)
    parser.add_argument("--initial-selection-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--model", default="local-qwen25vl7b")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--selection-command", required=True)
    parser.add_argument("--selection-start", type=int, default=20)
    parser.add_argument("--selection-end", type=int, default=40)
    args = parser.parse_args()

    official = args.output / "runtime_state.json"
    if official.exists():
        print(official.read_text())
        return

    args.output.mkdir(parents=True, exist_ok=True)
    skills_dir = args.output / "skills"
    skills_dir.mkdir(exist_ok=True)
    current = skills_dir / "skill_v0000.md"
    if not current.exists():
        current.write_bytes(args.initial_skill.read_bytes())

    history: List[Dict[str, Any]] = []
    partial = args.output / "runtime_state.partial.json"
    if partial.exists():
        payload = json.loads(partial.read_text())
        history = list(payload.get("history") or [])
        restored = Path(str(payload.get("current_skill") or current))
        if restored.is_file():
            current = restored

    evidence = evaluate_trajectories(args.development_run, args.baseline_development_run)
    (args.output / "development_evaluation.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )

    if args.initial_selection_run and not (args.output / "selection_current_summary.json").exists():
        current_sel = aggregate(
            args.initial_selection_run,
            expected_start=args.selection_start,
            expected_end=args.selection_end,
        )
        (args.output / "selection_current_summary.json").write_text(
            json.dumps(current_sel, ensure_ascii=False, indent=2) + "\n"
        )
    elif not (args.output / "selection_current_summary.json").exists():
        current_sel = run_selection(
            args.selection_command,
            current,
            args.output / "selection_current",
            selection_start=args.selection_start,
            selection_end=args.selection_end,
        )
        (args.output / "selection_current_summary.json").write_text(
            json.dumps(current_sel, ensure_ascii=False, indent=2) + "\n"
        )
    else:
        current_sel = json.loads((args.output / "selection_current_summary.json").read_text())

    done_rounds = {int(item.get("round") or 0) for item in history}
    for round_idx in range(1, args.rounds + 1):
        if round_idx in done_rounds:
            continue
        try:
            rejected = rejected_edit_buffer(history)
            current_payload = read_skill_payload(current)
            proposal = {"decision": "skip"}
            raw_reply = ""
            if args.endpoint:
                try:
                    proposal = propose_with_llm(
                        args.endpoint, args.model, current_payload, evidence, rejected
                    )
                    raw_reply = "llm"
                except (URLError, TimeoutError, ValueError) as exc:
                    write_infrastructure_error(args.output, round_idx, "candidate_service", exc)
                    proposal = propose_from_evidence(current_payload, evidence, rejected)
                    raw_reply = f"evidence_fallback:{type(exc).__name__}"
            else:
                proposal = propose_from_evidence(current_payload, evidence, rejected)
                raw_reply = "evidence"

            if str(proposal.get("decision", "")).lower() == "skip" or set(proposal) <= {"decision"}:
                record = {
                    "round": round_idx,
                    "decision": "SKIP",
                    "reason": "optimizer proposed no bounded edit",
                    "proposal": {},
                    "generator": raw_reply,
                    "current_skill": str(current),
                }
                history.append(record)
                write_partial_state(args.output, current, history)
                continue

            candidate = skills_dir / f"skill_v{int(current_payload['version']) + 1:04d}.md"
            edits = {k: v for k, v in proposal.items() if k in EDITABLE_FIELDS}
            write_candidate(current, candidate, edits)
            cand_sel = run_selection(
                args.selection_command,
                candidate,
                args.output / f"selection_round_{round_idx:04d}",
                selection_start=args.selection_start,
                selection_end=args.selection_end,
            )
            decision, reason = selection_decision(current_sel, cand_sel)
            record = {
                "round": round_idx,
                "decision": decision,
                "reason": reason,
                "proposal": edits,
                "generator": raw_reply,
                "current_skill": str(current),
                "candidate_skill": str(candidate),
                "current_SR": current_sel["SR"],
                "candidate_SR": cand_sel["SR"],
            }
            history.append(record)
            if decision == "ACCEPT":
                current = candidate
                current_sel = cand_sel
                (args.output / "selection_current_summary.json").write_text(
                    json.dumps(current_sel, ensure_ascii=False, indent=2) + "\n"
                )
            write_partial_state(args.output, current, history)
        except Exception as exc:
            write_infrastructure_error(args.output, round_idx, "round", exc)
            raise

    official_state = {
        "partial": False,
        "looked_at_sealed": False,
        "current_skill": str(current),
        "current_skill_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        "history": history,
        "selection_SR": current_sel["SR"],
        "selection_n": current_sel["n"],
    }
    official.write_text(json.dumps(official_state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(official_state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
