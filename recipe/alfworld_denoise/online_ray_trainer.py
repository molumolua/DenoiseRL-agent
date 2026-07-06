import copy

from omegaconf import OmegaConf, open_dict

from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.workers.rollout.async_server import AsyncLLMServerManager


DENOISE_ROLLOUT = "denoise_rollout"
SOLVER_ROLLOUT = "actor_rollout"


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default)


class OnlineDenoisePPOTrainer(RayPPOTrainer):
    """PPO trainer that colocates a fixed denoiser rollout model with the solver."""

    def _online_cfg(self):
        denoise_cfg = self.config.env.get("denoise", {})
        return denoise_cfg.get("online", {})

    def _build_denoise_actor_rollout_cfg(self):
        online_cfg = self._online_cfg()
        model_path = _cfg_get(online_cfg, "model_path", None)
        if model_path is None or str(model_path).strip() == "":
            raise ValueError("env.denoise.online.model_path must be set for online DenoiseRL.")

        denoise_cfg = copy.deepcopy(self.config.actor_rollout_ref)
        with open_dict(denoise_cfg):
            denoise_cfg.model.path = str(model_path)
            denoise_cfg.rollout.n = 1

            overrides = {
                "name": "rollout_name",
                "tensor_model_parallel_size": "tensor_model_parallel_size",
                "max_model_len": "max_model_len",
                "max_num_batched_tokens": "max_num_batched_tokens",
                "max_num_seqs": "max_num_seqs",
                "prompt_length": "prompt_length",
                "response_length": "response_length",
                "temperature": "temperature",
                "top_p": "top_p",
                "top_k": "top_k",
                "do_sample": "do_sample",
                "enforce_eager": "enforce_eager",
                "free_cache_engine": "free_cache_engine",
                "enable_chunked_prefill": "enable_chunked_prefill",
            }
            for rollout_key, online_key in overrides.items():
                value = _cfg_get(online_cfg, online_key, None)
                if value is not None:
                    setattr(denoise_cfg.rollout, rollout_key, value)

            gpu_util = _cfg_get(online_cfg, "gpu_memory_utilization", None)
            if gpu_util is not None:
                denoise_cfg.rollout.gpu_memory_utilization = float(gpu_util)

            # The denoiser is inference-only. It is started with role="rollout",
            # so these actor knobs are only defensive if a backend inspects them.
            denoise_cfg.actor.fsdp_config.optimizer_offload = True
            denoise_cfg.actor.use_kl_loss = False

        return denoise_cfg

    def _apply_shared_gpu_utilization(self, solver_cfg, denoise_cfg):
        online_cfg = self._online_cfg()
        shared_util = _cfg_get(online_cfg, "shared_gpu_memory_utilization", None)
        if shared_util is None or str(shared_util).lower() in {"", "none", "null"}:
            return

        base_util = float(shared_util)
        cap = float(_cfg_get(online_cfg, "colocate_gpu_util_cap", 0.9))
        with open_dict(denoise_cfg):
            denoise_cfg.rollout.gpu_memory_utilization = base_util
        with open_dict(solver_cfg):
            solver_cfg.rollout.gpu_memory_utilization = min(2.0 * base_util, cap)
        print(
            "[online-denoise] colocated vLLM gpu_memory_utilization: "
            f"denoiser(1st)={denoise_cfg.rollout.gpu_memory_utilization:.3f}, "
            f"solver(2nd,cumulative)={solver_cfg.rollout.gpu_memory_utilization:.3f}."
        )

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if not self.hybrid_engine:
            raise NotImplementedError("Online DenoiseRL requires hybrid_engine actor-rollout workers.")

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_cls = self.role_worker_mapping[Role.ActorRollout]
        solver_cfg = copy.deepcopy(self.config.actor_rollout_ref)
        denoise_cfg = self._build_denoise_actor_rollout_cfg()
        self._apply_shared_gpu_utilization(solver_cfg, denoise_cfg)

        # Insert the denoiser before the solver and initialize in that order, matching
        # the shared-vLLM memory accounting pattern used by hint_learn_v3.
        self.resource_pool_to_cls[resource_pool][DENOISE_ROLLOUT] = RayClassWithInitArgs(
            cls=actor_cls,
            config=denoise_cfg,
            role="rollout",
        )
        self.resource_pool_to_cls[resource_pool][SOLVER_ROLLOUT] = RayClassWithInitArgs(
            cls=actor_cls,
            config=solver_cfg,
            role="actor_rollout",
        )

        if self.use_critic:
            critic_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            self.resource_pool_to_cls[critic_pool]["critic"] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.critic,
            )

        if self.use_reference_policy:
            ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            self.resource_pool_to_cls[ref_pool]["ref"] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=solver_cfg,
                role="ref",
            )

        if self.use_rm:
            rm_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            self.resource_pool_to_cls[rm_pool]["rm"] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )

        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.denoise_rollout_wg = all_wg[DENOISE_ROLLOUT]
        self.actor_rollout_wg = all_wg[SOLVER_ROLLOUT]
        self.denoise_rollout_wg.init_model()
        self.actor_rollout_wg.init_model()

        if hasattr(self.traj_collector, "set_denoise_rollout_wg"):
            self.traj_collector.set_denoise_rollout_wg(self.denoise_rollout_wg)

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )
