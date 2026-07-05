import numpy as np

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto

from recipe.alfworld_denoise.trajectory_prefix import PrefixPool


class DenoiseTrajectoryCollector(TrajectoryCollector):
    """Mixes clean ALFWorld rollouts with replayed-prefix denoise rollouts."""

    def __init__(self, config, tokenizer, processor=None):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor)
        denoise_cfg = config.env.get("denoise", {})
        self.enabled = bool(denoise_cfg.get("enable", False))
        self.main_rollout_n = int(denoise_cfg.get("main_rollout_n", 1))
        self.sub_rollout_k = int(denoise_cfg.get("sub_rollout_k", 0))
        self.prefix_pool = PrefixPool(
            denoise_cfg.get("prefix_pool_path", None),
            min_steps=int(denoise_cfg.get("min_prefix_steps", 1)),
            max_steps=denoise_cfg.get("max_prefix_steps", None),
            seed=int(denoise_cfg.get("prefix_seed", config.env.get("seed", 0))),
            require_failed=bool(denoise_cfg.get("require_failed_prefixes", True)),
        )

    def _build_env_kwargs(self, repeated_batch_size: int) -> np.ndarray:
        total_n = int(self.config.env.rollout.n)
        if total_n <= 0 or repeated_batch_size % total_n != 0:
            raise ValueError(
                f"Expected repeated batch size to be divisible by env.rollout.n={total_n}, "
                f"got {repeated_batch_size}."
            )
        if self.main_rollout_n + self.sub_rollout_k != total_n:
            raise ValueError(
                "For alfworld_denoise, env.rollout.n must equal "
                f"main_rollout_n + sub_rollout_k, got {total_n} vs "
                f"{self.main_rollout_n} + {self.sub_rollout_k}."
            )

        env_kwargs: list[dict] = []
        num_groups = repeated_batch_size // total_n
        for _ in range(num_groups):
            prefixes = self.prefix_pool.sample(self.sub_rollout_k)
            for _main_idx in range(self.main_rollout_n):
                env_kwargs.append({"denoise_is_sub": False, "denoise_prefix_len": 0})

            for sub_idx in range(self.sub_rollout_k):
                if sub_idx < len(prefixes):
                    prefix = prefixes[sub_idx]
                    item = prefix.to_env_kwargs()
                    item.update(
                        {
                            "denoise_is_sub": True,
                            "denoise_prefix_len": len(prefix.steps),
                            "denoise_prefix_source": prefix.source,
                            "denoise_prefix_task_key": prefix.task_key,
                        }
                    )
                else:
                    item = {"denoise_is_sub": False, "denoise_prefix_len": 0}
                env_kwargs.append(item)
        return np.array(env_kwargs, dtype=object)

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs,
        is_train: bool = True,
    ) -> DataProto:
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            if self.enabled:
                if not self.prefix_pool:
                    raise ValueError(
                        "env.denoise.enable=True but no prefixes were loaded. "
                        "Set env.denoise.prefix_pool_path to a JSONL prefix pool."
                    )
                gen_batch.non_tensor_batch["env_kwargs"] = self._build_env_kwargs(len(gen_batch.batch))

        if self.config.algorithm.filter_groups.enable and is_train:
            (
                total_batch_list,
                total_episode_rewards,
                total_episode_lengths,
                total_success,
                total_traj_uid,
                total_tool_callings,
            ) = self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            (
                total_batch_list,
                total_episode_rewards,
                total_episode_lengths,
                total_success,
                total_traj_uid,
                total_tool_callings,
            ) = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )

        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
        )
