from types import SimpleNamespace

import numpy as np

from recipe.denoise_v2.collector import DenoiseTrajectoryCollector


class ConfigDict(dict):
    __getattr__ = dict.__getitem__


GAMEFILE_TASK_TYPES = {
    "game/a": "pick_and_place_simple",
    "game/b": "pick_heat_then_place_in_recep",
    "game/c": "pick_and_place_simple",
    "game/d": "pick_heat_then_place_in_recep",
}


def _collector():
    collector = DenoiseTrajectoryCollector.__new__(DenoiseTrajectoryCollector)
    collector.v2_enabled = True
    collector.enabled = True
    collector.mode = "online"
    collector.online_prefix_strategy = "full_then_ratio"
    collector.main_rollout_n = 0
    collector.sub_rollout_k = 16
    collector.online_prefix_candidates_per_group = 1
    collector.v2_cfg = {
        "initial_rho": 0.0,
        "min_rho": 0.0,
        "max_rho": 0.5,
        "target_accuracy": 0.75,
        "alpha": 0.2,
        "shuffle_seed": 11,
    }
    collector.config = ConfigDict(
        env=ConfigDict(seed=11, rollout=ConfigDict(n=16)),
        data=ConfigDict(train_batch_size=2, gen_batch_size=2),
        algorithm=ConfigDict(filter_groups=ConfigDict(enable=False)),
    )
    collector.configure_v2(GAMEFILE_TASK_TYPES, GAMEFILE_TASK_TYPES)
    return collector


def _training_batch(step, gamefiles, task_types, successes):
    # Repeat each trajectory to emulate two action rows. Curriculum accuracy
    # must be calculated from 16 unique trajectories per active gamefile.
    traj_uids = np.repeat(
        np.array([f"traj-{step}-{i}" for i in range(len(gamefiles))], dtype=object),
        2,
    )
    return SimpleNamespace(
        non_tensor_batch={
            "traj_uid": traj_uids,
            "denoise_v2_problem_id": np.repeat(np.asarray(gamefiles, dtype=object), 2),
            "denoise_v2_task_type": np.repeat(np.asarray(task_types, dtype=object), 2),
            "episode_success": np.repeat(np.asarray(successes, dtype=float), 2),
        }
    )


def test_configure_shuffles_pool_and_pins_task_type_rho_per_group():
    collector = _collector()

    assert collector.v2_curriculum.problem_ids == (
        "game/a",
        "game/b",
        "game/c",
        "game/d",
    )
    active = collector.v2_curriculum.active_problem_ids
    assert len(active) == 2
    assert len(set(active)) == 2

    reset_kwargs = collector._build_env_kwargs(32)
    for group_idx, gamefile in enumerate(active):
        group = reset_kwargs[group_idx * 16 : (group_idx + 1) * 16]
        assert all(item["gamefile"] == gamefile for item in group)
        assert all(item["denoise_v2_problem_id"] == gamefile for item in group)
        assert all(
            item["denoise_v2_task_type"] == GAMEFILE_TASK_TYPES[gamefile]
            for item in group
        )
        assert all(item["denoise_v2_rho"] == 0.0 for item in group)


def test_after_one_training_step_updates_task_types_and_replaces_every_gamefile():
    collector = _collector()
    first_active = collector.v2_curriculum.active_problem_ids
    gamefiles = [gamefile for gamefile in first_active for _ in range(16)]
    task_types = [GAMEFILE_TASK_TYPES[gamefile] for gamefile in gamefiles]
    successes = [1.0] * len(gamefiles)

    metrics = collector.after_training_step(
        _training_batch(0, gamefiles, task_types, successes)
    )

    second_active = collector.v2_curriculum.active_problem_ids
    assert set(first_active).isdisjoint(second_active)
    assert metrics["denoise/v2/replaced_this_step"] == 2.0
    for task_type in set(task_types):
        assert np.isclose(collector.v2_curriculum.rho_for_task_type(task_type), 0.05)

    next_kwargs = collector._build_env_kwargs(32)
    for group_idx, gamefile in enumerate(second_active):
        expected_rho = collector.v2_curriculum.rho_for_problem(gamefile)
        group = next_kwargs[group_idx * 16 : (group_idx + 1) * 16]
        assert all(item["gamefile"] == gamefile for item in group)
        assert all(item["denoise_v2_rho"] == expected_rho for item in group)
