import unittest

from agents.eg_llm_backend import exploration_exhausted, legal_objects


class EgLegalObjectsTest(unittest.TestCase):
    def test_legal_objects_keeps_object_with_remaining_relations(self):
        observed = ["Shelf 1", "Desk 1", "Drawer 1"]
        explored = ["on shelf 1", "in drawer 1"]
        allowed = legal_objects(observed, explored, "alfworld")
        self.assertEqual(allowed, {"shelf 1", "desk 1", "drawer 1"})

    def test_legal_objects_drops_fully_explored_object(self):
        observed = ["Shelf 1"]
        explored = [
            "on shelf 1",
            "in shelf 1",
            "target shelf 1",
            "near shelf 1",
        ]
        allowed = legal_objects(observed, explored, "alfworld")
        self.assertEqual(allowed, set())

    def test_exploration_exhausted_when_all_relations_tried(self):
        observed = ["Shelf 1"]
        explored = ["on shelf 1", "in shelf 1", "target shelf 1", "near shelf 1"]
        self.assertTrue(exploration_exhausted(observed, explored, "alfworld"))

    def test_exploration_not_exhausted_when_relation_remains(self):
        observed = ["Shelf 1"]
        explored = ["on shelf 1"]
        self.assertFalse(exploration_exhausted(observed, explored, "alfworld"))


if __name__ == "__main__":
    unittest.main()
