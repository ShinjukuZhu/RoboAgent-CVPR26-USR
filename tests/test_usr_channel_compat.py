import unittest

from agents.usr_channel import SkillChannel, reset_channel


class SkillChannelCompatTest(unittest.TestCase):
    def setUp(self):
        reset_channel()
        self.ch = SkillChannel()

    def test_has_get_field_and_log_decision(self):
        ok = self.ch.publish(
            "og",
            {
                "schema_version": "2.0",
                "environment_facts": {"object": {"class": "Apple"}},
                "decision_signals": {"found": True},
            },
            producer="test",
        )
        self.assertTrue(ok)
        self.assertTrue(self.ch.has("og"))
        self.assertEqual(self.ch.get_field("og", "object.class"), "Apple")
        self.assertTrue(self.ch.get_field("og", "found"))
        self.ch.log_decision("og", "grounded")
        events = [row["event"] for row in self.ch.contract_audit()]
        self.assertIn("decision", events)


if __name__ == "__main__":
    unittest.main()
