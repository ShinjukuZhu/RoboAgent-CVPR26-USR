import json
import os
import tempfile
import unittest
from pathlib import Path

from agents.evo_skill import EffectVerifiedSkill, EvoSkillSpec, EDITABLE_FIELDS


ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "skills" / "effect_verified_skill_v0000.md"


class EvoSkillSpecTest(unittest.TestCase):
    def test_compiles_frozen_skill(self):
        spec = EvoSkillSpec.from_markdown(V0)
        self.assertEqual(spec.version, 0)
        self.assertEqual(spec.schema_version, "roboagent_evo_skill_v1")
        self.assertEqual(len(spec.sha256), 64)
        self.assertTrue(spec.invalidate_perception_after_world_change)
        self.assertEqual(spec.scheduler_context_mode, "on_intervention")
        self.assertEqual(spec.grounding_contract_mode, "referential_only")

    def test_rejects_out_of_range_threshold(self):
        text = V0.read_text().replace(
            '"repeated_effect_miss_limit": 2',
            '"repeated_effect_miss_limit": 99',
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.md"
            path.write_text(text)
            with self.assertRaises(ValueError):
                EvoSkillSpec.from_markdown(path)

    def test_editable_fields_do_not_include_executor_code(self):
        self.assertIn("invalidate_perception_after_world_change", EDITABLE_FIELDS)
        self.assertNotIn("aliases", EDITABLE_FIELDS)
        self.assertNotIn("schema_version", EDITABLE_FIELDS)


class EffectVerifiedRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.runtime = EffectVerifiedSkill(
            EvoSkillSpec.from_markdown(V0), trace_fn=self.events.append
        )
        self.runtime.set_task("put two apples on the table")

    def test_matching_grounding_passes(self):
        result = [{"label": "Apple"}]
        checked, intervention = self.runtime.validate_grounding("Apple 2", result)
        self.assertIs(checked, result)
        self.assertIsNone(intervention)

    def test_wrong_grounding_is_blocked_and_requests_replan(self):
        checked, intervention = self.runtime.validate_grounding(
            "Spoon 1", [{"label": "Ladle"}]
        )
        self.assertFalse(checked)
        self.assertEqual(intervention.kind, "object_mismatch")
        self.assertIn("requested", self.runtime.consume_replan_request())

    def test_functional_grounding_contract_accepts_valid_role_resolution(self):
        result = [{"label": "Fridge"}]
        restored, intervention = self.runtime.validate_grounding(
            "some tool for cooling apple", result
        )
        self.assertEqual(restored, result)
        self.assertIsNone(intervention)

    def test_effect_miss_gate_and_suffix_invalidation(self):
        self.assertIsNone(self.runtime.observe_action_result("pick up the Apple 1", False))
        intervention = self.runtime.observe_action_result("pick up the Apple 1", False)
        self.assertEqual(intervention.kind, "effect_unverified")
        self.assertTrue(intervention.invalidate_suffix)

    def test_world_change_makes_last_goto_shortcut_stale(self):
        self.assertFalse(self.runtime.should_bypass_last_goto_shortcut())
        self.runtime.observe_action_result("take the Apple 1 from the table", True)
        self.assertTrue(self.runtime.should_bypass_last_goto_shortcut())
        self.runtime.validate_grounding("Apple 1", [{"label": "Apple"}])
        self.assertFalse(self.runtime.should_bypass_last_goto_shortcut())

    def test_scheduler_context_only_after_intervention(self):
        self.assertEqual(self.runtime.scheduler_context(), "")
        self.runtime.observe_action_result("open the Fridge 1", False)
        self.runtime.observe_action_result("open the Fridge 1", False)
        ctx = self.runtime.scheduler_context()
        self.assertIn("Confirmed progress", ctx)
        self.assertEqual(self.runtime.scheduler_context(), "")

    def test_skip_confirmed_open(self):
        self.runtime.observe_action_result("open the Fridge 1", True)
        intervention = self.runtime.precheck_action("open the Fridge 1")
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "effect_already_satisfied")


if __name__ == "__main__":
    unittest.main()
