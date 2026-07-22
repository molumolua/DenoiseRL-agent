import json
import unittest

from recipe.denoise_v2.gamefile_curriculum import TaskTypePoolCurriculum


def _task_types(problem_ids):
    return {
        problem_id: (
            "pick_and_place_simple"
            if problem_id % 2 == 0
            else "pick_heat_then_place_in_recep"
        )
        for problem_id in problem_ids
    }


class TaskTypePoolCurriculumTest(unittest.TestCase):
    def test_initial_batch_is_shuffled_deterministically_without_duplicates(self):
        problem_ids = tuple(range(10))
        first = TaskTypePoolCurriculum(
            problem_ids,
            _task_types(problem_ids),
            batch_size=4,
            shuffle_seed=17,
        )
        second = TaskTypePoolCurriculum(
            problem_ids,
            _task_types(problem_ids),
            batch_size=4,
            shuffle_seed=17,
        )

        self.assertEqual(first.active_problem_ids, second.active_problem_ids)
        self.assertEqual(len(set(first.active_problem_ids)), 4)
        self.assertNotEqual(first.active_problem_ids, problem_ids[:4])

    def test_task_type_rho_uses_cross_gamefile_average(self):
        problem_ids = tuple(range(6))
        task_types = {
            0: "pick",
            1: "pick",
            2: "pick",
            3: "heat",
            4: "heat",
            5: "heat",
        }
        curriculum = TaskTypePoolCurriculum(
            problem_ids,
            task_types,
            batch_size=6,
            target_accuracy=0.5,
            alpha=0.2,
            shuffle_seed=3,
        )

        metrics = curriculum.update(
            {
                problem_id: 1.0 if task_types[problem_id] == "pick" else 0.0
                for problem_id in curriculum.active_problem_ids
            }
        )

        self.assertAlmostEqual(curriculum.rho_for_task_type("pick"), 0.1)
        self.assertEqual(curriculum.rho_for_task_type("heat"), 0.0)
        self.assertAlmostEqual(metrics["denoise/v2/task_type/pick/accuracy"], 1.0)
        self.assertAlmostEqual(metrics["denoise/v2/task_type/heat/accuracy"], 0.0)
        self.assertEqual(metrics["denoise/v2/task_type/pick/batch_gamefiles"], 3.0)

    def test_every_step_replaces_the_complete_batch_without_reuse_in_epoch(self):
        problem_ids = tuple(range(8))
        curriculum = TaskTypePoolCurriculum(
            problem_ids,
            _task_types(problem_ids),
            batch_size=4,
            shuffle_seed=9,
        )
        first_batch = curriculum.active_problem_ids

        metrics = curriculum.update({problem_id: 0.5 for problem_id in first_batch})
        second_batch = curriculum.active_problem_ids

        self.assertTrue(set(first_batch).isdisjoint(second_batch))
        self.assertEqual(set(first_batch) | set(second_batch), set(problem_ids))
        self.assertEqual(metrics["denoise/v2/replaced_this_step"], 4.0)
        self.assertEqual(metrics["denoise/v2/gamefiles_trained_total"], 4.0)

        second_metrics = curriculum.update(
            {problem_id: 0.5 for problem_id in second_batch}
        )
        third_batch = curriculum.active_problem_ids
        self.assertTrue(set(second_batch).isdisjoint(third_batch))
        self.assertEqual(curriculum.pool_epoch, 2)
        self.assertEqual(curriculum.epochs_completed, 1)
        self.assertEqual(second_metrics["denoise/v2/epoch_advanced_this_step"], 1.0)

    def test_partial_epoch_is_filled_from_next_epoch_without_batch_duplicates(self):
        problem_ids = tuple(range(5))
        curriculum = TaskTypePoolCurriculum(
            problem_ids,
            _task_types(problem_ids),
            batch_size=3,
            shuffle_seed=5,
        )
        first_batch = curriculum.active_problem_ids
        remaining_in_first_epoch = set(problem_ids) - set(first_batch)

        metrics = curriculum.update({problem_id: 0.5 for problem_id in first_batch})
        second_batch = curriculum.active_problem_ids

        self.assertTrue(remaining_in_first_epoch.issubset(second_batch))
        self.assertEqual(len(set(second_batch)), 3)
        self.assertEqual(metrics["denoise/v2/epoch_advanced_this_step"], 1.0)
        self.assertEqual(curriculum.epochs_completed, 1)

    def test_checkpoint_round_trip_preserves_shuffle_and_task_states(self):
        problem_ids = tuple(range(9))
        kwargs = dict(
            problem_ids=problem_ids,
            problem_id_to_task_type=_task_types(problem_ids),
            batch_size=4,
            target_accuracy=0.5,
            shuffle_seed=21,
        )
        curriculum = TaskTypePoolCurriculum(**kwargs)
        curriculum.update(
            {
                problem_id: float(problem_id % 2 == 0)
                for problem_id in curriculum.active_problem_ids
            }
        )
        state = json.loads(json.dumps(curriculum.state_dict()))

        restored = TaskTypePoolCurriculum(**kwargs)
        restored.load_state_dict(state)

        self.assertEqual(restored.state_dict(), state)
        accuracies = {problem_id: 0.5 for problem_id in curriculum.active_problem_ids}
        curriculum.update(accuracies)
        restored.update(accuracies)
        self.assertEqual(restored.state_dict(), curriculum.state_dict())

    def test_old_per_gamefile_checkpoint_is_rejected(self):
        problem_ids = (0, 1)
        curriculum = TaskTypePoolCurriculum(
            problem_ids,
            _task_types(problem_ids),
            batch_size=1,
        )

        with self.assertRaisesRegex(ValueError, "requires version 3"):
            curriculum.load_state_dict({"version": 2})


if __name__ == "__main__":
    unittest.main()
