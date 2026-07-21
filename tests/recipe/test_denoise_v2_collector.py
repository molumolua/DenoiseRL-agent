from types import SimpleNamespace

import numpy as np

from recipe.denoise_v2.collector import DenoiseTrajectoryCollector


class ConfigDict(dict):
    __getattr__ = dict.__getitem__


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
        "history_window": 10,
        "min_history": 2,
        "slope_threshold": 0.0075,
    }
    collector.config = ConfigDict(
        env=ConfigDict(rollout=ConfigDict(n=16)),
        data=ConfigDict(train_batch_size=2, gen_batch_size=2),
        algorithm=ConfigDict(filter_groups=ConfigDict(enable=False)),
    )
    collector.configure_v2(["game/c", "game/a", "game/b"])
    return collector


def _training_batch(step, gamefiles, successes):
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
            "episode_success": np.repeat(np.asarray(successes, dtype=float), 2),
        }
    )


def test_configure_sorts_pool_and_pins_one_gamefile_per_group():
    collector = _collector()

    assert collector.v2_curriculum.problem_ids == ("game/a", "game/b", "game/c")
    assert collector.v2_curriculum.active_problem_ids == ("game/a", "game/b")

    reset_kwargs = collector._build_env_kwargs(32)
    assert all(item["gamefile"] == "game/a" for item in reset_kwargs[:16])
    assert all(item["gamefile"] == "game/b" for item in reset_kwargs[16:])
    assert all(item["denoise_v2_rho"] == 0.0 for item in reset_kwargs)


def test_after_training_step_retires_stable_gamefile_and_advances_pool():
    collector = _collector()
    gamefiles = ["game/a"] * 16 + ["game/b"] * 16
    successes = [0.75] * 32

    first = collector.after_training_step(_training_batch(0, gamefiles, successes))
    second = collector.after_training_step(_training_batch(1, gamefiles, successes))

    assert first["denoise/v2/replaced_this_step"] == 0.0
    assert second["denoise/v2/stable_candidates"] == 2.0
    assert second["denoise/v2/replaced_this_step"] == 1.0
    assert collector.v2_curriculum.active_problem_ids == ("game/c", "game/b")

    next_kwargs = collector._build_env_kwargs(32)
    assert all(item["gamefile"] == "game/c" for item in next_kwargs[:16])
    assert all(item["gamefile"] == "game/b" for item in next_kwargs[16:])
