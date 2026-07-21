import unittest
from types import SimpleNamespace

from agent_system.alfworld_evaluation import (
    build_gamefile_reset_kwargs,
    collapse_trajectory_rows,
    ordered_validation_gamefiles,
    validate_gamefile_coverage,
)


class ExhaustiveAlfworldValidationTest(unittest.TestCase):
    def test_step_rows_are_collapsed_in_environment_order(self):
        trajectory_ids, outputs, scores = collapse_trajectory_rows(
            ["trajectory-b", "trajectory-b", "trajectory-a"],
            ["look", "take apple", "open fridge"],
            [10.0, 10.0, 0.0],
        )

        self.assertEqual(trajectory_ids, ["trajectory-b", "trajectory-a"])
        self.assertEqual(outputs, ["look\ntake apple", "open fridge"])
        self.assertEqual(scores, [10.0, 0.0])

    def test_validation_gamefiles_are_complete_unique_and_deterministic(self):
        val_envs = SimpleNamespace(
            envs=SimpleNamespace(
                game_files=("game/c", "game/a", "game/b"),
                num_games=3,
            )
        )

        self.assertEqual(
            ordered_validation_gamefiles(val_envs),
            ("game/a", "game/b", "game/c"),
        )

    def test_reset_kwargs_follow_interleaved_repeat_order(self):
        kwargs = build_gamefile_reset_kwargs(
            ("game/a", "game/b", "game/c"),
            start=1,
            count=2,
            repeats=2,
        )

        self.assertEqual(
            [item["gamefile"] for item in kwargs],
            ["game/b", "game/b", "game/c", "game/c"],
        )

    def test_coverage_requires_every_gamefile_exactly_once(self):
        metrics = validate_gamefile_coverage(
            ("game/a", "game/b", "game/c"),
            ["game/a", "game/b", "game/c"],
            repeats=1,
        )

        self.assertEqual(
            metrics,
            {
                "gamefile_count": 3.0,
                "gamefile_episode_count": 3.0,
                "gamefile_unique_count": 3.0,
                "gamefile_coverage": 1.0,
            },
        )

    def test_coverage_rejects_duplicate_and_missing_gamefiles(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Incomplete ALFWorld validation coverage",
        ):
            validate_gamefile_coverage(
                ("game/a", "game/b", "game/c"),
                ["game/a", "game/a", "game/c"],
                repeats=1,
            )
