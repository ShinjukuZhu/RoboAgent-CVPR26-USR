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
        self.assertIn("block_nonpickupable_take", EDITABLE_FIELDS)
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

    def test_table_paraphrases_are_compatible(self):
        skill = EffectVerifiedSkill(EvoSkillSpec.from_markdown(V0))
        restored, intervention = skill.validate_grounding(
            "kitchen table", [{"label": "DiningTable"}]
        )
        self.assertEqual(restored, [{"label": "DiningTable"}])
        self.assertIsNone(intervention)
        restored, intervention = skill.validate_grounding(
            "wooden table", [{"label": "dining table"}]
        )
        self.assertIsNone(intervention)

    def test_fridge_and_remote_paraphrases(self):
        skill = EffectVerifiedSkill(EvoSkillSpec.from_markdown(V0))
        _, intervention = skill.validate_grounding(
            "refrigerator", [{"label": "Fridge"}]
        )
        self.assertIsNone(intervention)
        _, intervention = skill.validate_grounding(
            "tv remote", [{"label": "RemoteControl"}]
        )
        self.assertIsNone(intervention)
        _, intervention = skill.validate_grounding(
            "bar of soap (hint: the shelves)", [{"label": "SoapBar"}]
        )
        self.assertIsNone(intervention)

    def test_location_phrase_abstains(self):
        skill = EffectVerifiedSkill(EvoSkillSpec.from_markdown(V0))
        restored, intervention = skill.validate_grounding(
            "on the table", [{"label": "DiningTable"}]
        )
        self.assertEqual(restored, [{"label": "DiningTable"}])
        self.assertIsNone(intervention)
        # Compact planner tokens like "ontable" must not hard-reject.
        restored, intervention = skill.validate_grounding(
            "ontable", [{"label": "DiningTable"}]
        )
        self.assertEqual(restored, [{"label": "DiningTable"}])
        self.assertIsNone(intervention)

    def test_align_receptacle_paraphrases(self):
        skill = EffectVerifiedSkill(EvoSkillSpec.from_markdown(V0))
        _, intervention = skill.validate_grounding(
            "kitchen island", [{"label": "CounterTop"}]
        )
        self.assertIsNone(intervention)
        _, intervention = skill.validate_grounding(
            "tvstand", [{"label": "Dresser"}]
        )
        self.assertIsNone(intervention)

    def test_effect_predicate_target_abstains(self):
        skill = EffectVerifiedSkill(EvoSkillSpec.from_markdown(V0))
        restored, intervention = skill.validate_grounding(
            "closed(Cabinet 1)", [{"label": "Cabinet"}]
        )
        self.assertEqual(restored, [{"label": "Cabinet"}])
        self.assertIsNone(intervention)

    def test_distinct_tables_still_conflict(self):
        skill = EffectVerifiedSkill(EvoSkillSpec.from_markdown(V0))
        restored, intervention = skill.validate_grounding(
            "CoffeeTable", [{"label": "DiningTable"}]
        )
        self.assertFalse(restored)
        self.assertEqual(intervention.kind, "object_mismatch")
        self.assertFalse(intervention.invalidate_suffix)

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

    def test_skip_confirmed_clean(self):
        self.runtime.observe_action_result("clean Cloth 1 with Sink 1", True)
        intervention = self.runtime.precheck_action("clean Cloth 1 with Sink 1")
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "effect_already_satisfied")

    def test_goal_progress_stall_triggers_replan(self):
        self.runtime._last_gcr = 0.0
        self.assertIsNone(self.runtime.observe_goal_progress(0.0, "clean Cloth 1 with Sink 1", True))
        intervention = self.runtime.observe_goal_progress(0.0, "clean Cloth 1 with Sink 1", True)
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "goal_progress_stall")

    def test_goal_progress_stall_allows_bridge_steps_at_partial_gcr(self):
        self.runtime._last_gcr = 0.33
        for _ in range(5):
            self.assertIsNone(
                self.runtime.observe_goal_progress(0.33, "go to Cabinet 1", True)
            )
        intervention = self.runtime.observe_goal_progress(0.33, "go to Cabinet 2", True)
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "goal_progress_stall")

    def test_blocks_nonpickupable_take(self):
        intervention = self.runtime.precheck_action("take Microwave 1 from Microwave 1")
        self.assertIsNotNone(intervention)
        self.assertEqual(intervention.kind, "nonpickupable_take")
        self.assertTrue(intervention.invalidate_suffix)
        intervention = self.runtime.precheck_action("take CounterTop 1 from Toaster 1")
        self.assertEqual(intervention.kind, "nonpickupable_take")
        intervention = self.runtime.precheck_action("heat Microwave 1 with Microwave 1")
        self.assertEqual(intervention.kind, "nonpickupable_take")
        intervention = self.runtime.precheck_action("clean Sink 1 with Sink 1")
        self.assertEqual(intervention.kind, "nonpickupable_take")
        # Portable objects / valid heat still allowed.
        self.assertIsNone(self.runtime.precheck_action("take Apple 1 from CounterTop 1"))
        self.assertIsNone(self.runtime.precheck_action("heat Apple 1 with Microwave 1"))


if __name__ == "__main__":
    unittest.main()
