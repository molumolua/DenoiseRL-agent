import uuid
from typing import Optional

import numpy as np

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from recipe.alfworld_denoise.trajectory_prefix import PrefixPool


class DenoiseTrajectoryCollector(TrajectoryCollector):
    """Mixes clean ALFWorld rollouts with denoised sub-rollouts."""

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}

    @staticmethod
    def _optional_int(value):
        if value is None or str(value).strip().lower() in {"", "none", "null"}:
            return None
        return int(value)

    def __init__(self, config, tokenizer, processor=None, denoise_tokenizer=None, denoise_processor=None):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor)
        denoise_cfg = config.env.get("denoise", {})
        self.enabled = bool(denoise_cfg.get("enable", False))
        self.mode = self._normalize_mode(denoise_cfg.get("mode", "prefix_pool"))
        self.main_rollout_n = int(denoise_cfg.get("main_rollout_n", 1))
        self.sub_rollout_k = int(denoise_cfg.get("sub_rollout_k", 0))
        self.denoise_rollout_wg = None
        self.denoise_tokenizer = denoise_tokenizer
        self.denoise_processor = denoise_processor
        online_cfg = denoise_cfg.get("online", {})
        self.online_prefix_strategy = self._normalize_prefix_strategy(
            online_cfg.get("prefix_strategy", online_cfg.get("strategy", "full_then_ratio"))
        )
        max_prefix_steps = online_cfg.get("max_prefix_steps", None)
        if max_prefix_steps is None and self.online_prefix_strategy == "step_budget":
            max_prefix_steps = denoise_cfg.get("max_prefix_steps", None)
        self.online_max_prefix_steps = self._optional_int(max_prefix_steps)
        self.online_full_rollout_max_steps = int(online_cfg.get("full_rollout_max_steps", config.env.max_steps))
        self.online_prefix_ratio = float(online_cfg.get("prefix_ratio", 0.3))
        self.online_prefix_candidates_per_group = int(
            online_cfg.get("prefix_candidates_per_group", self.sub_rollout_k)
        )
        self.online_prefix_sample_seed = int(online_cfg.get("prefix_sample_seed", config.env.get("seed", 0)))
        self.online_prefix_rng = np.random.default_rng(self.online_prefix_sample_seed)
        self.online_avoid_terminal_prefix = self._as_bool(online_cfg.get("avoid_terminal_prefix", True))
        self.online_prompt_length = self._optional_int(online_cfg.get("prompt_length", None))
        if self.online_prompt_length is None:
            self.online_prompt_length = int(self.config.data.max_prompt_length)
        self.online_do_sample = self._as_bool(online_cfg.get("do_sample", True))
        self.advantage_grouping = self._normalize_advantage_grouping(
            denoise_cfg.get("advantage_grouping", online_cfg.get("advantage_grouping", "mixed"))
        )
        self.prefix_pool = PrefixPool(
            None if self.mode == "online" else denoise_cfg.get("prefix_pool_path", None),
            min_steps=int(denoise_cfg.get("min_prefix_steps", 1)),
            max_steps=denoise_cfg.get("max_prefix_steps", None),
            seed=int(denoise_cfg.get("prefix_seed", config.env.get("seed", 0))),
            require_failed=bool(denoise_cfg.get("require_failed_prefixes", True)),
        )

    @staticmethod
    def _normalize_mode(raw_mode) -> str:
        mode = str(raw_mode or "prefix_pool").lower().replace("-", "_")
        aliases = {
            "offline": "prefix_pool",
            "pool": "prefix_pool",
            "replay": "prefix_pool",
            "online_model": "online",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"prefix_pool", "online"}:
            raise ValueError(f"env.denoise.mode must be 'prefix_pool' or 'online', got {raw_mode!r}.")
        return mode

    @staticmethod
    def _normalize_prefix_strategy(raw_strategy) -> str:
        strategy = str(raw_strategy or "full_then_ratio").lower().replace("-", "_")
        aliases = {
            "ratio": "full_then_ratio",
            "full_ratio": "full_then_ratio",
            "full_rollout_ratio": "full_then_ratio",
            "step": "step_budget",
            "steps": "step_budget",
            "online": "step_budget",
            "max_steps": "step_budget",
        }
        strategy = aliases.get(strategy, strategy)
        if strategy not in {"full_then_ratio", "step_budget"}:
            raise ValueError(
                "env.denoise.online.prefix_strategy must be 'full_then_ratio' or "
                f"'step_budget', got {raw_strategy!r}."
            )
        return strategy

    @staticmethod
    def _normalize_advantage_grouping(raw_grouping) -> str:
        grouping = str(raw_grouping or "mixed").lower().replace("-", "_")
        aliases = {
            "joint": "mixed",
            "together": "mixed",
            "same": "mixed",
            "separate": "split",
        }
        grouping = aliases.get(grouping, grouping)
        if grouping not in {"mixed", "split"}:
            raise ValueError(
                "env.denoise.advantage_grouping must be 'mixed' or 'split', "
                f"got {raw_grouping!r}."
            )
        return grouping

    def set_denoise_rollout_wg(self, denoise_rollout_wg, denoise_tokenizer=None, denoise_processor=None):
        self.denoise_rollout_wg = denoise_rollout_wg
        if denoise_tokenizer is not None:
            self.denoise_tokenizer = denoise_tokenizer
        if denoise_processor is not None:
            self.denoise_processor = denoise_processor

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
            prefixes = [] if self.mode == "online" else self.prefix_pool.sample(self.sub_rollout_k)
            for _main_idx in range(self.main_rollout_n):
                env_kwargs.append({"denoise_is_sub": False, "denoise_prefix_len": 0})

            for sub_idx in range(self.sub_rollout_k):
                if self.mode == "online":
                    item = {
                        "denoise_is_sub": True,
                        "denoise_prefix_len": 0,
                        "denoise_prefix_source": "online",
                    }
                elif sub_idx < len(prefixes):
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

    def _ensure_online_ready(self):
        if self.denoise_rollout_wg is None:
            raise ValueError(
                "env.denoise.mode=online requires a denoise rollout worker group. "
                "Use recipe.alfworld_denoise.main_online_denoise for online DenoiseRL."
            )
        if self.denoise_tokenizer is None:
            raise ValueError("env.denoise.mode=online requires a denoise tokenizer.")
        if self.online_prefix_strategy == "step_budget":
            if self.online_max_prefix_steps is None or self.online_max_prefix_steps <= 0:
                raise ValueError(
                    "env.denoise.online.max_prefix_steps must be > 0 when "
                    "prefix_strategy=step_budget."
                )
        else:
            if self.online_full_rollout_max_steps <= 0:
                raise ValueError(
                    "env.denoise.online.full_rollout_max_steps must be > 0 when "
                    "prefix_strategy=full_then_ratio."
                )
            if self.online_prefix_ratio < 0:
                raise ValueError("env.denoise.online.prefix_ratio must be >= 0.")
            if self.online_prefix_candidates_per_group <= 0:
                raise ValueError("env.denoise.online.prefix_candidates_per_group must be > 0.")
            if self.online_prefix_candidates_per_group > self.sub_rollout_k:
                raise ValueError(
                    "env.denoise.online.prefix_candidates_per_group cannot exceed "
                    f"env.denoise.sub_rollout_k ({self.sub_rollout_k}) because the "
                    "current implementation uses sub-rollout envs as shadow workers."
                )

    def _subset_obs(self, obs: dict, indices: list[int]) -> dict:
        def _take(value):
            if value is None:
                return None
            return [value[i] for i in indices]

        return {
            "text": _take(obs.get("text", None)),
            "image": _take(obs.get("image", None)),
            "anchor": _take(obs.get("anchor", None)),
        }

    def _preprocess_batch_with_tokenizer(self, gen_batch: DataProto, obs: dict, tokenizer) -> DataProto:
        batch_size = len(gen_batch.batch["input_ids"])
        obs_texts = obs.get("text", None)
        if obs_texts is None:
            raise ValueError("Online DenoiseRL currently expects text observations.")

        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        processed_samples = []
        for item in range(batch_size):
            obs_text = obs_texts[item]
            chat = np.array([{"content": obs_text, "role": "user"}])
            prompt_with_chat_template = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=tokenizer,
                max_length=self.online_prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            raw_prompt_ids = tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
            if len(raw_prompt_ids) > self.online_prompt_length:
                if self.config.data.truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.online_prompt_length:]
                elif self.config.data.truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.online_prompt_length]
                elif self.config.data.truncation == "middle":
                    left_half = self.online_prompt_length // 2
                    right_half = self.online_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif self.config.data.truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(raw_prompt_ids)} is longer than "
                        f"{self.online_prompt_length}."
                    )

            processed_samples.append(
                {
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "position_ids": position_ids[0],
                    "raw_prompt_ids": raw_prompt_ids,
                    "index": item,
                    "data_source": gen_batch.non_tensor_batch["data_source"][item],
                }
            )

        return DataProto.from_single_dict(data=collate_fn(processed_samples), meta_info=gen_batch.meta_info)

    def _build_denoise_generation_input(self, gen_batch: DataProto, obs: dict) -> DataProto:
        batch = self._preprocess_batch_with_tokenizer(gen_batch, obs, self.denoise_tokenizer)
        batch_input = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],
        )
        batch_input.meta_info = dict(gen_batch.meta_info)
        batch_input.meta_info.update(
            {
                "do_sample": self.online_do_sample,
                "validate": False,
                "recompute_log_prob": False,
            }
        )
        return batch_input

    def _current_obs_from_envs(self, envs, prefix_lens: np.ndarray) -> dict:
        if not hasattr(envs, "build_mixed_text_obs_after_prefix"):
            raise NotImplementedError("Online DenoiseRL currently requires AlfWorldEnvironmentManager.")
        text_obs = envs.build_mixed_text_obs_after_prefix(prefix_lens.tolist())
        anchor = list(envs.pre_text_obs)
        return {"text": text_obs, "image": None, "anchor": anchor}

    def _generate_denoise_actions(self, gen_batch: DataProto, current_obs: dict, active_indices: list[int]) -> list[str]:
        sub_batch = gen_batch.select_idxs(active_indices)
        sub_obs = self._subset_obs(current_obs, active_indices)
        batch_input = self._build_denoise_generation_input(sub_batch, sub_obs)
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, self.denoise_rollout_wg.world_size)
        batch_output_padded = self.denoise_rollout_wg.generate_sequences(batch_input_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        return self.denoise_tokenizer.batch_decode(
            batch_output.batch["responses"],
            skip_special_tokens=True,
        )

    def _select_replay_prefix_len(self, full_len: int, ended_terminal: bool) -> int:
        if full_len <= 0 or self.online_prefix_ratio <= 0:
            return 0
        prefix_len = int(np.ceil(full_len * self.online_prefix_ratio))
        prefix_len = max(1, min(prefix_len, full_len))
        if self.online_max_prefix_steps is not None:
            prefix_len = min(prefix_len, self.online_max_prefix_steps)
        if self.online_avoid_terminal_prefix and ended_terminal and prefix_len >= full_len:
            prefix_len = max(0, full_len - 1)
        return prefix_len

    def _sub_indices_by_group(self, reset_kwargs) -> list[list[int]]:
        total_n = int(self.config.env.rollout.n)
        groups = []
        for group_start in range(0, len(reset_kwargs), total_n):
            group_items = reset_kwargs[group_start: group_start + total_n]
            groups.append(
                [
                    group_start + local_idx
                    for local_idx, item in enumerate(group_items)
                    if isinstance(item, dict) and bool(item.get("denoise_is_sub", False))
                ]
            )
        return groups

    @staticmethod
    def _is_denoise_sub(item) -> bool:
        return isinstance(item, dict) and bool(item.get("denoise_is_sub", False))

    def _update_prefix_metadata(self, reset_kwargs, prefix_lens: np.ndarray, source: str) -> None:
        if reset_kwargs is None:
            return
        if isinstance(reset_kwargs, np.ndarray):
            reset_kwargs = reset_kwargs.tolist()
        for i, length in enumerate(prefix_lens):
            if i >= len(reset_kwargs):
                break
            item = reset_kwargs[i]
            if self._is_denoise_sub(item):
                item["denoise_prefix_len"] = int(length)
                item["denoise_prefix_source"] = source

    def _init_denoise_rollout_metrics(self, batch_size: int, reset_kwargs) -> dict[str, np.ndarray]:
        if isinstance(reset_kwargs, np.ndarray):
            reset_kwargs = reset_kwargs.tolist()
        reset_kwargs = reset_kwargs or [{} for _ in range(batch_size)]

        is_sub = np.zeros(batch_size, dtype=bool)
        prefix_lens = np.zeros(batch_size, dtype=np.float32)
        sources = np.array(["clean" for _ in range(batch_size)], dtype=object)
        for i in range(batch_size):
            item = reset_kwargs[i] if i < len(reset_kwargs) else {}
            if isinstance(item, dict):
                is_sub[i] = bool(item.get("denoise_is_sub", False))
                prefix_lens[i] = float(item.get("denoise_prefix_len", 0) or 0)
                sources[i] = str(item.get("denoise_prefix_source", "denoise" if is_sub[i] else "clean"))

        return {
            "denoise_is_sub": is_sub,
            "denoise_prefix_source": sources,
            "denoise_prefix_len": prefix_lens.copy(),
            "denoise_prefix_generated_len": prefix_lens.copy(),
            "denoise_prefix_action_count": prefix_lens.copy(),
            "denoise_prefix_invalid_count": np.zeros(batch_size, dtype=np.float32),
            "denoise_prefix_invalid_rate": np.zeros(batch_size, dtype=np.float32),
            "denoise_prefix_terminal": np.zeros(batch_size, dtype=np.float32),
            "denoise_prefix_terminal_dropped": np.zeros(batch_size, dtype=np.float32),
            "denoise_prefix_won": np.zeros(batch_size, dtype=np.float32),
            "denoise_prefix_empty": np.logical_and(is_sub, prefix_lens == 0).astype(np.float32),
        }

    def _build_online_prefix_metrics(
        self,
        batch_size: int,
        reset_kwargs,
        prefix_lens: np.ndarray,
        *,
        source: str,
        generated_lens: Optional[np.ndarray] = None,
        action_counts: Optional[np.ndarray] = None,
        invalid_counts: Optional[np.ndarray] = None,
        terminals: Optional[np.ndarray] = None,
        terminal_dropped: Optional[np.ndarray] = None,
        wins: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        metrics = self._init_denoise_rollout_metrics(batch_size, reset_kwargs)
        prefix_lens = prefix_lens.astype(np.float32)
        is_sub = metrics["denoise_is_sub"]
        generated_lens = prefix_lens if generated_lens is None else generated_lens.astype(np.float32)
        action_counts = generated_lens if action_counts is None else action_counts.astype(np.float32)
        invalid_counts = np.zeros(batch_size, dtype=np.float32) if invalid_counts is None else invalid_counts.astype(np.float32)
        terminals = np.zeros(batch_size, dtype=np.float32) if terminals is None else terminals.astype(np.float32)
        terminal_dropped = (
            np.zeros(batch_size, dtype=np.float32)
            if terminal_dropped is None
            else terminal_dropped.astype(np.float32)
        )
        wins = np.zeros(batch_size, dtype=np.float32) if wins is None else wins.astype(np.float32)

        metrics["denoise_prefix_source"][is_sub] = source
        metrics["denoise_prefix_len"] = prefix_lens
        metrics["denoise_prefix_generated_len"] = generated_lens
        metrics["denoise_prefix_action_count"] = action_counts
        metrics["denoise_prefix_invalid_count"] = invalid_counts
        metrics["denoise_prefix_invalid_rate"] = np.divide(
            invalid_counts,
            np.maximum(action_counts, 1.0),
            out=np.zeros_like(invalid_counts, dtype=np.float32),
            where=action_counts > 0,
        )
        metrics["denoise_prefix_terminal"] = terminals
        metrics["denoise_prefix_terminal_dropped"] = terminal_dropped
        metrics["denoise_prefix_won"] = wins
        metrics["denoise_prefix_empty"] = np.logical_and(is_sub, prefix_lens == 0).astype(np.float32)
        return metrics

    def _build_uid_batch(self, batch_size: int, reset_kwargs) -> np.ndarray:
        total_n = int(self.config.env.rollout.n)
        if total_n <= 0:
            uid = str(uuid.uuid4())
            return np.array([uid for _ in range(batch_size)], dtype=object)
        if (
            self.enabled
            and self.advantage_grouping == "split"
            and reset_kwargs is not None
        ):
            if isinstance(reset_kwargs, np.ndarray):
                reset_kwargs = reset_kwargs.tolist()
            uid_batch = []
            for group_start in range(0, batch_size, total_n):
                clean_uid = str(uuid.uuid4())
                denoise_uid = str(uuid.uuid4())
                for idx in range(group_start, min(group_start + total_n, batch_size)):
                    item = reset_kwargs[idx] if idx < len(reset_kwargs) else {}
                    uid_batch.append(denoise_uid if isinstance(item, dict) and item.get("denoise_is_sub") else clean_uid)
            return np.array(uid_batch, dtype=object)

        uid_batch = []
        for i in range(batch_size):
            if i % total_n == 0:
                uid = str(uuid.uuid4())
            uid_batch.append(uid)
        return np.array(uid_batch, dtype=object)

    def _run_step_budget_prefixes(self, gen_batch: DataProto, obs: dict, envs, reset_kwargs) -> tuple[dict, dict[str, np.ndarray]]:
        self._ensure_online_ready()
        if isinstance(reset_kwargs, np.ndarray):
            reset_kwargs = reset_kwargs.tolist()
        sub_groups = self._sub_indices_by_group(reset_kwargs or [])
        sub_indices = [idx for group in sub_groups for idx in group]
        prefix_lens = np.zeros(len(gen_batch.batch), dtype=np.int32)
        if not sub_indices:
            return obs, self._build_online_prefix_metrics(
                len(gen_batch.batch),
                reset_kwargs,
                prefix_lens,
                source="online_step_budget",
            )
        if not hasattr(envs, "reset_selected_with_prefixes"):
            raise NotImplementedError("step_budget online DenoiseRL requires AlfWorldEnvironmentManager.")

        current_obs = obs
        generated_lens = np.zeros(len(gen_batch.batch), dtype=np.int32)
        action_counts = np.zeros(len(gen_batch.batch), dtype=np.float32)
        invalid_counts = np.zeros(len(gen_batch.batch), dtype=np.float32)
        prefix_terminals = np.zeros(len(gen_batch.batch), dtype=np.float32)
        prefix_wins = np.zeros(len(gen_batch.batch), dtype=np.float32)
        prefix_actions = {idx: [] for idx in sub_indices}
        prefix_dones = {idx: False for idx in sub_indices}
        done_prefix = set()
        invalid_actions = 0
        total_actions = 0
        for _prefix_step in range(self.online_max_prefix_steps):
            active_indices = [idx for idx in sub_indices if idx not in done_prefix]
            if not active_indices:
                break

            text_actions = self._generate_denoise_actions(gen_batch, current_obs, active_indices)
            _next_obs, _rewards, dones, infos = envs.step_selected(active_indices, text_actions)
            total_actions += len(active_indices)
            invalid_actions += sum(1 for info in infos if not bool(info.get("is_action_valid", True)))
            for local_idx, env_idx in enumerate(active_indices):
                projected_action = text_actions[local_idx]
                if hasattr(envs, "memory") and envs.memory._data[env_idx]:
                    projected_action = envs.memory._data[env_idx][-1]["action"]
                prefix_actions[env_idx].append(projected_action)
                generated_lens[env_idx] = len(prefix_actions[env_idx])
                action_counts[env_idx] += 1.0
                if not bool(infos[local_idx].get("is_action_valid", True)):
                    invalid_counts[env_idx] += 1.0
                done = dones[local_idx]
                if bool(done):
                    prefix_dones[env_idx] = True
                    prefix_terminals[env_idx] = 1.0
                    prefix_wins[env_idx] = float(bool(infos[local_idx].get("won", False)))
                    done_prefix.add(env_idx)

            current_obs = self._current_obs_from_envs(envs, generated_lens)

        replay_actions = []
        terminal_dropped = np.zeros(len(gen_batch.batch), dtype=np.float32)
        for env_idx in sub_indices:
            actions = prefix_actions[env_idx]
            replay_len = len(actions)
            if self.online_avoid_terminal_prefix and prefix_dones[env_idx] and replay_len > 0:
                replay_len -= 1
                terminal_dropped[env_idx] = 1.0
            replay_prefix = actions[:replay_len]
            replay_actions.append(replay_prefix)
            prefix_lens[env_idx] = len(replay_prefix)

        replay_obs, _replay_infos = envs.reset_selected_with_prefixes(sub_indices, replay_actions)

        if total_actions:
            print(
                "Online DenoiseRL step_budget generated "
                f"{int(generated_lens.sum())} denoiser action(s); "
                f"replayed={int(prefix_lens.sum())}; "
                f"terminal_prefix_dropped={int(terminal_dropped.sum())}; "
                f"invalid={invalid_actions}/{total_actions}."
            )
        self._update_prefix_metadata(reset_kwargs, prefix_lens, "online_step_budget")
        prefix_metrics = self._build_online_prefix_metrics(
            len(gen_batch.batch),
            reset_kwargs,
            prefix_lens,
            source="online_step_budget",
            generated_lens=generated_lens,
            action_counts=action_counts,
            invalid_counts=invalid_counts,
            terminals=prefix_terminals,
            terminal_dropped=terminal_dropped,
            wins=prefix_wins,
        )
        return replay_obs, prefix_metrics

    def _run_full_then_ratio_prefixes(self, gen_batch: DataProto, obs: dict, envs, reset_kwargs) -> tuple[dict, dict[str, np.ndarray]]:
        self._ensure_online_ready()
        if isinstance(reset_kwargs, np.ndarray):
            reset_kwargs = reset_kwargs.tolist()
        sub_groups = self._sub_indices_by_group(reset_kwargs or [])
        sub_indices = [idx for group in sub_groups for idx in group]
        prefix_lens = np.zeros(len(gen_batch.batch), dtype=np.int32)
        if not sub_indices:
            return obs, self._build_online_prefix_metrics(
                len(gen_batch.batch),
                reset_kwargs,
                prefix_lens,
                source="online_full_then_ratio",
            )
        if not hasattr(envs, "reset_selected_with_prefixes"):
            raise NotImplementedError("full_then_ratio online DenoiseRL requires AlfWorldEnvironmentManager.")

        group_candidate_indices = []
        candidate_indices = []
        for group in sub_groups:
            candidates = group[: self.online_prefix_candidates_per_group]
            group_candidate_indices.append(candidates)
            candidate_indices.extend(candidates)

        current_obs = obs
        full_actions = {idx: [] for idx in candidate_indices}
        full_dones = {idx: False for idx in candidate_indices}
        full_wons = {idx: False for idx in candidate_indices}
        candidate_action_counts = {idx: 0.0 for idx in candidate_indices}
        candidate_invalid_counts = {idx: 0.0 for idx in candidate_indices}
        done_full = set()
        invalid_actions = 0
        total_actions = 0

        for _shadow_step in range(self.online_full_rollout_max_steps):
            active_indices = [idx for idx in candidate_indices if idx not in done_full]
            if not active_indices:
                break

            text_actions = self._generate_denoise_actions(gen_batch, current_obs, active_indices)
            _next_obs, _rewards, dones, infos = envs.step_selected(active_indices, text_actions)
            total_actions += len(active_indices)
            invalid_actions += sum(1 for info in infos if not bool(info.get("is_action_valid", True)))

            for local_idx, env_idx in enumerate(active_indices):
                candidate_action_counts[env_idx] += 1.0
                if not bool(infos[local_idx].get("is_action_valid", True)):
                    candidate_invalid_counts[env_idx] += 1.0
                projected_action = text_actions[local_idx]
                if hasattr(envs, "memory") and envs.memory._data[env_idx]:
                    projected_action = envs.memory._data[env_idx][-1]["action"]
                full_actions[env_idx].append(projected_action)
                if bool(dones[local_idx]):
                    full_dones[env_idx] = True
                    full_wons[env_idx] = bool(infos[local_idx].get("won", False))
                    done_full.add(env_idx)

            shadow_lens = np.array([len(full_actions.get(i, [])) for i in range(len(gen_batch.batch))])
            current_obs = self._current_obs_from_envs(envs, shadow_lens)

        replay_actions = []
        replay_generated_lens = np.zeros(len(gen_batch.batch), dtype=np.float32)
        replay_action_counts = np.zeros(len(gen_batch.batch), dtype=np.float32)
        replay_invalid_counts = np.zeros(len(gen_batch.batch), dtype=np.float32)
        replay_terminals = np.zeros(len(gen_batch.batch), dtype=np.float32)
        replay_terminal_dropped = np.zeros(len(gen_batch.batch), dtype=np.float32)
        replay_wins = np.zeros(len(gen_batch.batch), dtype=np.float32)
        terminal_steps_dropped = 0
        empty_prefix_choices = 0
        for group, candidates in zip(sub_groups, group_candidate_indices):
            candidate_prefixes = []
            for candidate_idx in candidates:
                actions = full_actions[candidate_idx]
                replay_len = self._select_replay_prefix_len(
                    full_len=len(actions),
                    ended_terminal=full_dones[candidate_idx],
                )
                dropped_terminal = full_dones[candidate_idx] and replay_len < len(actions)
                if dropped_terminal:
                    terminal_steps_dropped += 1
                candidate_prefixes.append(
                    {
                        "actions": actions[:replay_len],
                        "candidate_idx": candidate_idx,
                        "terminal_dropped": dropped_terminal,
                    }
                )

            for env_idx in group:
                if not candidate_prefixes:
                    chosen = None
                    chosen_prefix = []
                    empty_prefix_choices += 1
                else:
                    chosen_idx = int(self.online_prefix_rng.integers(len(candidate_prefixes)))
                    chosen = candidate_prefixes[chosen_idx]
                    chosen_prefix = list(chosen["actions"])
                    if not chosen_prefix:
                        empty_prefix_choices += 1
                replay_actions.append(chosen_prefix)
                prefix_lens[env_idx] = len(chosen_prefix)
                if chosen is not None:
                    candidate_idx = chosen["candidate_idx"]
                    replay_generated_lens[env_idx] = float(len(full_actions[candidate_idx]))
                    replay_action_counts[env_idx] = float(candidate_action_counts[candidate_idx])
                    replay_invalid_counts[env_idx] = float(candidate_invalid_counts[candidate_idx])
                    replay_terminals[env_idx] = float(full_dones[candidate_idx])
                    replay_terminal_dropped[env_idx] = float(chosen["terminal_dropped"])
                    replay_wins[env_idx] = float(full_wons[candidate_idx])

        replay_obs, _replay_infos = envs.reset_selected_with_prefixes(sub_indices, replay_actions)

        if total_actions:
            full_lengths = [len(full_actions[idx]) for idx in candidate_indices]
            replay_lengths = [len(actions) for actions in replay_actions]
            print(
                "Online DenoiseRL full_then_ratio generated "
                f"{sum(full_lengths)} shadow denoiser action(s); "
                f"replayed={sum(replay_lengths)}; "
                f"prefix_candidates/group={self.online_prefix_candidates_per_group}; "
                f"ratio={self.online_prefix_ratio:.3f}; "
                f"terminal_prefix_dropped={terminal_steps_dropped}; "
                f"empty_prefix_choices={empty_prefix_choices}; "
                f"invalid={invalid_actions}/{total_actions}; "
                f"won={sum(1 for idx in candidate_indices if full_wons[idx])}/{len(candidate_indices)}."
            )
        self._update_prefix_metadata(reset_kwargs, prefix_lens, "online_full_then_ratio")
        prefix_metrics = self._build_online_prefix_metrics(
            len(gen_batch.batch),
            reset_kwargs,
            prefix_lens,
            source="online_full_then_ratio",
            generated_lens=replay_generated_lens,
            action_counts=replay_action_counts,
            invalid_counts=replay_invalid_counts,
            terminals=replay_terminals,
            terminal_dropped=replay_terminal_dropped,
            wins=replay_wins,
        )
        return replay_obs, prefix_metrics

    def _run_online_prefixes(self, gen_batch: DataProto, obs: dict, envs, reset_kwargs) -> tuple[dict, dict[str, np.ndarray]]:
        if self.online_prefix_strategy == "step_budget":
            return self._run_step_budget_prefixes(gen_batch, obs, envs, reset_kwargs)
        return self._run_full_then_ratio_prefixes(gen_batch, obs, envs, reset_kwargs)

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs,
            ) -> DataProto:
        batch_size = len(gen_batch.batch)

        reset_kwargs = gen_batch.non_tensor_batch.get('env_kwargs', None)
        if reset_kwargs is None and "alfworld" in self.config.env.env_name.lower():
            reset_kwargs = np.array([{} for _ in range(len(gen_batch.batch))], dtype=object)
        obs, infos = envs.reset(kwargs=reset_kwargs)
        denoise_rollout_metrics = {}
        if self.enabled:
            denoise_rollout_metrics = self._init_denoise_rollout_metrics(
                batch_size=batch_size,
                reset_kwargs=reset_kwargs,
            )
        if self.enabled and self.mode == "online":
            obs, denoise_rollout_metrics = self._run_online_prefixes(
                gen_batch=gen_batch,
                obs=obs,
                envs=envs,
                reset_kwargs=reset_kwargs,
            )

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        uid_batch = self._build_uid_batch(batch_size=batch_size, reset_kwargs=reset_kwargs)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            for metric_key, metric_values in denoise_rollout_metrics.items():
                batch.non_tensor_batch[metric_key] = metric_values
            batch = batch.union(batch_output)
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            next_obs, rewards, dones, infos = envs.step(text_actions)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            batch_list: list[dict] = to_list_of_dict(batch)
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if is_done.all():
                break

        success = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )
        if self.enabled and "success_rate" in success:
            episode_success = success["success_rate"]
            for i in range(batch_size):
                for data in total_batch_list[i]:
                    data["episode_success"] = float(episode_success[i])

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

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
                if self.mode == "prefix_pool" and not self.prefix_pool:
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
