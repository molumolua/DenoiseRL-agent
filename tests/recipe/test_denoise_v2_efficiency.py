import ast
from collections import OrderedDict
import inspect
import logging
from pathlib import Path
import types
import unittest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_class_without_optional_dependencies(path: Path, class_name: str):
    """Load one class definition without importing its heavyweight module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


def _load_fsdp_enter(vllm_version: str):
    path = _REPO_ROOT / "verl/workers/sharding_manager/fsdp_vllm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FSDPVLLMShardingManager"
    )
    enter_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__enter__"
    )
    enter_node.decorator_list = []
    module = ast.Module(body=[enter_node], type_ignores=[])
    ast.fix_missing_locations(module)

    class _Device:
        def empty_cache(self):
            pass

    class _PeftModel:
        pass

    namespace = {
        "OrderedDict": OrderedDict,
        "PeftModel": _PeftModel,
        "get_torch_device": lambda: _Device(),
        "inspect": inspect,
        "load_fsdp_model_to_gpu": lambda _module: None,
        "log_gpu_memory_usage": lambda *_args, **_kwargs: None,
        "logger": logging.getLogger(__name__),
        "offload_fsdp_model_to_cpu": lambda _module: None,
        "vllm_version": vllm_version,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["__enter__"]


class _FakeEnv:
    def __init__(self, gamefile):
        self.gamefile = gamefile
        self.close_calls = 0
        self.reset_calls = 0

    def seed(self, _seed):
        pass

    def reset(self):
        self.reset_calls += 1
        return ["observation"], {"extra.gamefile": [self.gamefile]}

    def close(self):
        self.close_calls += 1


class _FakeBaseEnv:
    def __init__(self):
        self.created = []

    def init_env(self, batch_size, game_files=None):
        assert batch_size == 1
        gamefile = None if game_files is None else game_files[0]
        env = _FakeEnv(gamefile)
        self.created.append(env)
        return env


class AlfworldPinnedEnvironmentReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker_cls = _load_class_without_optional_dependencies(
            _REPO_ROOT
            / "agent_system/environments/env_package/alfworld/envs.py",
            "AlfworldWorker",
        )

    def test_same_gamefile_reuses_pinned_environment(self):
        base_env = _FakeBaseEnv()
        worker = self.worker_cls(config={}, seed=7, base_env=base_env)

        worker.reset(gamefile="game-a")
        pinned_env = worker.env
        worker.reset(trajectory_prefix=[], gamefile="game-a")

        self.assertEqual(len(base_env.created), 2)  # default + one pinned env
        self.assertIs(worker.env, pinned_env)
        self.assertEqual(pinned_env.close_calls, 0)
        self.assertEqual(pinned_env.reset_calls, 2)

    def test_different_gamefile_replaces_and_closes_pinned_environment(self):
        base_env = _FakeBaseEnv()
        worker = self.worker_cls(config={}, seed=7, base_env=base_env)

        worker.reset(gamefile="game-a")
        old_pinned_env = worker.env
        worker.reset(gamefile="game-b")

        self.assertEqual(len(base_env.created), 3)
        self.assertEqual(old_pinned_env.close_calls, 1)
        self.assertEqual(worker.current_gamefile, "game-b")


class _FakeModule:
    def __init__(self):
        self._fsdp_wrapped_module = object()
        self.state_dict_calls = 0

    def state_dict(self):
        self.state_dict_calls += 1
        return {"weight": object()}


class _FakeInferenceEngine:
    def __init__(self):
        self.wake_calls = []
        self.sync_calls = 0

    def wake_up(self, tags=None):
        self.wake_calls.append(tags)

    def sync_model_weights(self, _params, load_format):
        self.sync_calls += 1


def _fake_sharding_manager(*, sync_every_generation: bool, base_sync_done: bool):
    manager = types.SimpleNamespace(
        base_sync_done=base_sync_done,
        device_mesh=None,
        full_params=False,
        inference_engine=_FakeInferenceEngine(),
        module=_FakeModule(),
        offload_param=False,
        sync_weights_every_generation=sync_every_generation,
    )

    def update_params(self, _params, peft_config=None):
        self.base_sync_done = True

    manager.update_params = types.MethodType(update_params, manager)
    return manager


class FixedDenoiserWeightSyncTest(unittest.TestCase):
    def test_fixed_worker_syncs_once_then_only_wakes_vllm(self):
        enter = _load_fsdp_enter("0.7.3")
        manager = _fake_sharding_manager(
            sync_every_generation=False,
            base_sync_done=False,
        )

        enter(manager)
        manager.inference_engine.wake_calls.clear()
        enter(manager)

        self.assertEqual(manager.module.state_dict_calls, 1)
        self.assertEqual(manager.inference_engine.wake_calls, [["weights"], ["kv_cache"]])

    def test_trainable_worker_keeps_syncing_each_generation(self):
        enter = _load_fsdp_enter("0.7.3")
        manager = _fake_sharding_manager(
            sync_every_generation=True,
            base_sync_done=True,
        )

        enter(manager)
        enter(manager)

        self.assertEqual(manager.module.state_dict_calls, 2)

    def test_legacy_vllm_keeps_syncing_fixed_worker(self):
        enter = _load_fsdp_enter("0.6.3")
        manager = _fake_sharding_manager(
            sync_every_generation=False,
            base_sync_done=True,
        )

        enter(manager)
        enter(manager)

        self.assertEqual(manager.module.state_dict_calls, 2)
        self.assertEqual(manager.inference_engine.sync_calls, 2)


if __name__ == "__main__":
    unittest.main()
