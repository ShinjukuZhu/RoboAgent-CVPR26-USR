import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from training.skillopt_evolve import (
    evaluate_trajectories,
    propose_from_evidence,
    rejected_edit_buffer,
    run_selection,
    selection_decision,
    with_resume_start,
    write_infrastructure_error,
    write_partial_state,
)

ROOT = Path(__file__).resolve().parents[1]


def metrics(sr, task_ids=None):
    return {
        "task_ids": task_ids or [20, 21],
        "SR": sr,
    }


class SelectionGateTest(unittest.TestCase):
    def test_accepts_sr_gain(self):
        decision, _ = selection_decision(metrics(0.5), metrics(1.0))
        self.assertEqual(decision, "ACCEPT")

    def test_rejects_exact_tie(self):
        decision, _ = selection_decision(metrics(0.5), metrics(0.5))
        self.assertEqual(decision, "REJECT")

    def test_rejects_sr_regression(self):
        decision, _ = selection_decision(metrics(0.8), metrics(0.5))
        self.assertEqual(decision, "REJECT")

    def test_skips_mismatched_task_ids(self):
        candidate = metrics(0.8, [21, 22])
        decision, _ = selection_decision(metrics(0.5), candidate)
        self.assertEqual(decision, "SKIP")

    def test_file_execution_does_not_require_pre_set_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "training" / "skillopt_evolve.py"), "--help"],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("selection-command", proc.stdout)

    def test_paired_development_evidence_exposes_success_regression(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            cand = root / "candidate"
            base.mkdir()
            cand.mkdir()
            common = {"task_idx": 40, "GCR": 1.0, "valid_run": 1}
            (base / "results.jsonl").write_text(json.dumps({**common, "SR": 1}) + "\n")
            (cand / "results.jsonl").write_text(json.dumps({**common, "SR": 0}) + "\n")
            evidence = evaluate_trajectories(cand, base)
        self.assertEqual(
            evidence["paired_baseline_evidence"]["baseline_success_candidate_fail"],
            [40],
        )

    def test_refuses_sealed_paths(self):
        with self.assertRaises(ValueError):
            evaluate_trajectories(Path("/tmp/sealed_aw_ood"))

    def test_rejected_buffer_recovers_edit_from_disk_and_deduplicates(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "current.md"
            candidate = root / "candidate.md"
            base = {
                "schema_version": "roboagent_evo_skill_v1",
                "version": 0,
                "name": "x",
                "repeated_effect_miss_limit": 2,
                "verify_grounded_object": True,
                "invalidate_stale_suffix": True,
                "expose_progress_to_scheduler": True,
                "skip_confirmed_effects": True,
                "recovery_instruction": "replan unfinished suffix now.",
            }
            changed = dict(base)
            changed.update({"version": 1, "repeated_effect_miss_limit": 3})
            current.write_text("```json\n" + json.dumps(base) + "\n```\n")
            candidate.write_text("```json\n" + json.dumps(changed) + "\n```\n")
            record = {
                "round": 1, "decision": "REJECT", "reason": "validity",
                "current_skill": str(current), "candidate_skill": str(candidate),
            }
            buffer = rejected_edit_buffer([record, {**record, "round": 2}])
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer[0]["edits"], {"repeated_effect_miss_limit": 3})


class EvidenceOptimizerTest(unittest.TestCase):
    def test_proposes_referential_grounding_from_mismatch_counts(self):
        edits = propose_from_evidence(
            {
                "grounding_contract_mode": "literal",
                "repeated_effect_miss_limit": 2,
                "scheduler_context_mode": "on_intervention",
                "invalidate_perception_after_world_change": True,
                "skip_feedback_mode": "virtual_success",
            },
            {"unverified_event_counts": {"grounding_effect_check": 4}},
            [],
        )
        self.assertEqual(edits, {"grounding_contract_mode": "referential_only"})

    def test_skips_when_rejected_buffer_covers_all_edits(self):
        current = {
            "grounding_contract_mode": "referential_only",
            "repeated_effect_miss_limit": 3,
            "scheduler_context_mode": "on_intervention",
            "invalidate_perception_after_world_change": True,
            "skip_feedback_mode": "virtual_success",
        }
        edits = propose_from_evidence(current, {"unverified_event_counts": {}}, [])
        self.assertEqual(edits.get("decision"), "skip")


class SkillOptCheckpointTest(unittest.TestCase):
    def test_partial_state_is_not_official_runtime_state(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "skill.md"
            skill.write_text("# x\n")
            write_partial_state(root, skill, [{"round": 1, "decision": "SKIP"}])
            partial = json.loads((root / "runtime_state.partial.json").read_text())
            self.assertTrue(partial["partial"])
            self.assertTrue(partial["not_final_runtime_state"])
            self.assertFalse(partial["looked_at_sealed"])
            self.assertFalse((root / "runtime_state.json").exists())

    def test_infrastructure_error_is_explicitly_not_a_round_decision(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_infrastructure_error(out, 2, "candidate_service", ConnectionError("down"))
            payload = json.loads((out / "last_infrastructure_error.json").read_text())
            self.assertTrue(payload["not_a_skillopt_decision"])
            self.assertTrue(payload["retry_from_disk"])
            self.assertNotIn("decision", payload)

    def test_with_resume_start_replaces_existing_flag(self):
        self.assertEqual(
            with_resume_start("python run_aw.py --start 20 --end 40 --save_path out", 30),
            "python run_aw.py --start 30 --end 40 --save_path out",
        )

    def test_run_selection_skips_subprocess_when_complete(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "skill.md"
            skill.write_text("# skill\n")
            out = root / "selection_round_0001"
            actual = out / "run-eval_in_distribution"
            actual.mkdir(parents=True)
            rows = []
            for i in range(20, 40):
                rows.append({"task_idx": i, "SR": 0, "GCR": 0.0, "valid_run": 1})
            (actual / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = run_selection(
                "false",
                skill,
                out,
                selection_start=20,
                selection_end=40,
            )
            self.assertEqual(summary["n"], 20)
            self.assertEqual(summary["task_ids"], list(range(20, 40)))

    def test_partial_history_skips_completed_rounds_without_rerun(self):
        import tempfile
        from unittest.mock import patch
        from training import skillopt_evolve

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            skill = root / "skill.md"
            skill.write_text(
                "# s\n```json\n"
                '{"schema_version":"roboagent_evo_skill_v1","version":0,"name":"t",'
                '"repeated_effect_miss_limit":2,"verify_grounded_object":true,'
                '"invalidate_stale_suffix":true,"expose_progress_to_scheduler":true,'
                '"skip_confirmed_effects":true,"recovery_instruction":"replan unfinished suffix."}\n'
                "```\n"
            )
            (out / "runtime_state.partial.json").write_text(json.dumps({
                "partial": True,
                "current_skill": str(skill),
                "history": [{"round": 1, "decision": "REJECT", "reason": "from disk"}],
            }))
            (out / "development_run").mkdir()
            (out / "development_run" / "results.jsonl").write_text(
                json.dumps({"task_idx": 0, "SR": 0, "valid_run": 1}) + "\n"
            )
            (out / "selection_current_summary.json").write_text(json.dumps({
                "task_ids": [20, 21],
                "SR": 0.5,
                "n": 2,
            }))
            with patch.object(sys, "argv", [
                "skillopt_evolve.py",
                "--initial-skill", str(skill),
                "--development-run", str(out / "development_run"),
                "--output", str(out),
                "--rounds", "1",
                "--selection-command", "false",
            ]):
                skillopt_evolve.main()
            official = json.loads((out / "runtime_state.json").read_text())
            self.assertEqual(official["history"], [{"round": 1, "decision": "REJECT", "reason": "from disk"}])
            self.assertFalse(official["looked_at_sealed"])


if __name__ == "__main__":
    unittest.main()
