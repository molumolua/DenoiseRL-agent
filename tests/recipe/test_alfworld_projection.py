import importlib.util
from pathlib import Path
import unittest


_PROJECTION_PATH = (
    Path(__file__).resolve().parents[2]
    / "agent_system/environments/env_package/alfworld/projection.py"
)
_SPEC = importlib.util.spec_from_file_location("alfworld_projection", _PROJECTION_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
alfworld_projection = _MODULE.alfworld_projection


class AlfworldProjectionTest(unittest.TestCase):
    def test_accepts_action_from_matching_admissible_pool(self):
        actions, valids = alfworld_projection(
            ["<think>Move there.</think><action>  GO   TO COUNTERTOP 1  </action>"],
            [["look", "go to countertop 1"]],
        )

        self.assertEqual(actions, ["go to countertop 1"])
        self.assertEqual(valids, [1])

    def test_rejects_well_formed_action_outside_admissible_pool(self):
        actions, valids = alfworld_projection(
            ["<think>Try something.</think><action>fly to moon</action>"],
            [["look", "inventory"]],
        )

        self.assertEqual(actions, ["fly to moon"])
        self.assertEqual(valids, [0])

    def test_validity_is_checked_against_each_environment_pool(self):
        actions, valids = alfworld_projection(
            [
                "<think>Inspect.</think><action>look</action>",
                "<think>Inspect.</think><action>look</action>",
            ],
            [["inventory"], ["look"]],
        )

        self.assertEqual(actions, ["look", "look"])
        self.assertEqual(valids, [0, 1])

    def test_rejects_response_without_required_tags(self):
        _actions, valids = alfworld_projection(
            ["look"],
            [["look"]],
        )

        self.assertEqual(valids, [0])

    def test_rejects_mismatched_batch_lengths(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            alfworld_projection(["look"], [])


if __name__ == "__main__":
    unittest.main()
