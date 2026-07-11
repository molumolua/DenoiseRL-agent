from types import SimpleNamespace

import numpy as np

from recipe.alfworld_denoise.collector import DenoiseTrajectoryCollector
from recipe.alfworld_denoise.trajectory_prefix import PrefixPool, PrefixStep, TrajectoryPrefix


def _collector(mode="online", grouping="mixed", group_n=3):
    collector = DenoiseTrajectoryCollector.__new__(DenoiseTrajectoryCollector)
    collector.config = SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=group_n)))
    collector.enabled = True
    collector.mode = mode
    collector.advantage_grouping = grouping
    return collector


def test_prefix_metadata_only_updates_denoise_sub_rollouts():
    collector = _collector()
    reset_kwargs = [
        {"denoise_is_sub": False, "denoise_prefix_len": 0},
        {"denoise_is_sub": True, "denoise_prefix_len": 0, "denoise_prefix_source": "online"},
        {"denoise_is_sub": True, "denoise_prefix_len": 0, "denoise_prefix_source": "online"},
    ]

    collector._update_prefix_metadata(
        reset_kwargs,
        np.array([0, 2, 0], dtype=np.int32),
        "online_full_then_ratio",
    )

    assert "denoise_prefix_source" not in reset_kwargs[0]
    assert reset_kwargs[1]["denoise_prefix_len"] == 2
    assert reset_kwargs[1]["denoise_prefix_source"] == "online_full_then_ratio"
    assert reset_kwargs[2]["denoise_prefix_len"] == 0
    assert reset_kwargs[2]["denoise_prefix_source"] == "online_full_then_ratio"


def test_split_advantage_grouping_applies_to_prefix_pool():
    collector = _collector(mode="prefix_pool", grouping="split", group_n=3)
    reset_kwargs = np.array(
        [
            {"denoise_is_sub": False},
            {"denoise_is_sub": True},
            {"denoise_is_sub": True},
            {"denoise_is_sub": False},
            {"denoise_is_sub": False},
            {"denoise_is_sub": True},
        ],
        dtype=object,
    )

    uid_batch = collector._build_uid_batch(batch_size=6, reset_kwargs=reset_kwargs)

    assert uid_batch[0] != uid_batch[1]
    assert uid_batch[1] == uid_batch[2]
    assert uid_batch[3] == uid_batch[4]
    assert uid_batch[3] != uid_batch[5]


class _FakePrefixEnv:
    def __init__(self):
        self.gamefile = ["/data/game-a/game.tw-pddl", "/data/game-a/game.tw-pddl", "/data/game-b/game.tw-pddl"]
        self.replay_indices = None
        self.replay_actions = None

    def reset_selected_with_prefixes(self, indices, prefix_actions):
        self.replay_indices = list(indices)
        self.replay_actions = [list(actions) for actions in prefix_actions]
        return {"text": ["replayed-0", "replayed-1", "replayed-2"], "image": None, "anchor": []}, []


def test_prefix_pool_replays_only_task_matched_prefixes():
    collector = _collector(mode="prefix_pool", grouping="mixed", group_n=3)
    collector.prefix_pool = PrefixPool(None, seed=0)
    collector.prefix_pool.all_prefixes = [
        TrajectoryPrefix(
            task_key="/data/game-a/game.tw-pddl",
            steps=(PrefixStep(action="open fridge 1"),),
            source="small-model",
        )
    ]
    reset_kwargs = [
        {"denoise_is_sub": False, "denoise_prefix_len": 0},
        {"denoise_is_sub": True, "denoise_prefix_len": 0},
        {"denoise_is_sub": True, "denoise_prefix_len": 0},
    ]
    envs = _FakePrefixEnv()

    obs, metrics = collector._run_prefix_pool_prefixes(
        obs={"text": ["init-0", "init-1", "init-2"], "image": None, "anchor": []},
        infos=[],
        envs=envs,
        reset_kwargs=reset_kwargs,
    )

    assert envs.replay_indices == [1]
    assert envs.replay_actions == [["open fridge 1"]]
    assert obs["text"] == ["replayed-0", "replayed-1", "replayed-2"]
    assert reset_kwargs[1]["denoise_prefix_task_key"] == "/data/game-a/game.tw-pddl"
    assert reset_kwargs[2]["denoise_is_sub"] is False
    assert metrics["denoise_is_sub"].tolist() == [False, True, False]


class _FakeGenBatch:
    batch = [None, None, None]


class _FakeOnlineEnv:
    def __init__(self):
        self.memory = SimpleNamespace(_data=[[], [], []])
        self.pre_text_obs = ["obs-0", "obs-1", "obs-2"]
        self.step_counts = {1: 0, 2: 0}
        self.replay_indices = None
        self.replay_actions = None
        self.mixed_prefix_lens = []

    def step_selected(self, indices, text_actions):
        dones = []
        infos = []
        for idx, _action in zip(indices, text_actions):
            self.step_counts[idx] += 1
            projected_action = f"projected-{idx}-{self.step_counts[idx]}"
            self.memory._data[idx].append({"action": projected_action})
            self.pre_text_obs[idx] = f"obs-{idx}-{self.step_counts[idx]}"
            dones.append(idx == 1 and self.step_counts[idx] == 1)
            infos.append({"is_action_valid": True})
        return {"text": [], "image": None, "anchor": []}, np.zeros(len(indices)), np.array(dones), infos

    def build_mixed_text_obs_after_prefix(self, prefix_lens):
        self.mixed_prefix_lens.append(list(prefix_lens))
        return [f"text-{i}-{prefix_lens[i]}" for i in range(len(prefix_lens))]

    def reset_selected_with_prefixes(self, indices, prefix_actions):
        self.replay_indices = list(indices)
        self.replay_actions = [list(actions) for actions in prefix_actions]
        return {"text": ["reset-0", "reset-1", "reset-2"], "image": None, "anchor": self.pre_text_obs}, []


def test_step_budget_resets_and_drops_terminal_prefix_step():
    collector = _collector()
    collector.online_max_prefix_steps = 2
    collector.online_avoid_terminal_prefix = True
    collector._ensure_online_ready = lambda: None
    collector._generate_denoise_actions = (
        lambda _gen_batch, _current_obs, active_indices: [f"raw-{idx}" for idx in active_indices]
    )
    reset_kwargs = [
        {"denoise_is_sub": False, "denoise_prefix_len": 0},
        {"denoise_is_sub": True, "denoise_prefix_len": 0, "denoise_prefix_source": "online"},
        {"denoise_is_sub": True, "denoise_prefix_len": 0, "denoise_prefix_source": "online"},
    ]
    envs = _FakeOnlineEnv()

    replay_obs, metrics = collector._run_step_budget_prefixes(
        gen_batch=_FakeGenBatch(),
        obs={"text": ["init-0", "init-1", "init-2"], "image": None, "anchor": []},
        envs=envs,
        reset_kwargs=reset_kwargs,
    )

    assert metrics["denoise_prefix_len"].tolist() == [0.0, 0.0, 2.0]
    assert metrics["denoise_prefix_generated_len"].tolist() == [0.0, 1.0, 2.0]
    assert metrics["denoise_prefix_action_count"].tolist() == [0.0, 1.0, 2.0]
    assert metrics["denoise_prefix_invalid_rate"].tolist() == [0.0, 0.0, 0.0]
    assert metrics["denoise_prefix_terminal"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["denoise_prefix_terminal_dropped"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["denoise_prefix_empty"].tolist() == [0.0, 1.0, 0.0]
    assert envs.replay_indices == [1, 2]
    assert envs.replay_actions == [[], ["projected-2-1", "projected-2-2"]]
    assert envs.mixed_prefix_lens == [[0, 1, 1], [0, 1, 2]]
    assert replay_obs["text"] == ["reset-0", "reset-1", "reset-2"]
    assert "denoise_prefix_source" not in reset_kwargs[0]
    assert reset_kwargs[1]["denoise_prefix_len"] == 0
    assert reset_kwargs[1]["denoise_prefix_source"] == "online_step_budget"
    assert reset_kwargs[2]["denoise_prefix_len"] == 2
    assert reset_kwargs[2]["denoise_prefix_source"] == "online_step_budget"
