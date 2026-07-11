# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.seed = seed
        self.base_env = base_env
        self.default_env = self._make_env()
        self.env = self.default_env

    def _make_env(self, game_files=None):
        try:
            env = self.base_env.init_env(batch_size=1, game_files=game_files)
        except TypeError as exc:
            if game_files is not None:
                raise NotImplementedError(
                    "Resetting ALFWorld to a specific gamefile is currently supported "
                    "for AlfredTWEnv only."
                ) from exc
            env = self.base_env.init_env(batch_size=1)
        env.seed(self.seed)
        return env

    def _close_replay_env(self):
        if self.env is not self.default_env and hasattr(self.env, "close"):
            self.env.close()
        self.env = self.default_env

    def _extract_prefix_actions(self, trajectory_prefix):
        if not trajectory_prefix:
            return []
        if isinstance(trajectory_prefix, dict):
            if "actions" in trajectory_prefix:
                return list(trajectory_prefix["actions"])
            if "steps" in trajectory_prefix:
                return [step["action"] for step in trajectory_prefix["steps"] if step.get("action")]
        if isinstance(trajectory_prefix, (list, tuple)):
            actions = []
            for item in trajectory_prefix:
                if isinstance(item, str):
                    actions.append(item)
                elif isinstance(item, dict) and item.get("action"):
                    actions.append(item["action"])
            return actions
        return []
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def _reset_current_env(self, trajectory_prefix=None):
        obs, infos = self.env.reset()
        initial_obs = obs[0]
        prefix_history = []
        prefix_done = False

        for action in self._extract_prefix_actions(trajectory_prefix):
            prefix_history.append({"text_obs": obs[0], "action": action})
            obs, _, dones, infos = self.env.step([action])
            prefix_done = bool(dones[0])
            if prefix_done:
                break

        infos['observation_text'] = obs
        infos['initial_observation_text'] = [initial_obs]
        infos['prefix_history'] = [prefix_history]
        infos['prefix_len'] = [len(prefix_history)]
        infos['prefix_done'] = [prefix_done]
        return obs, infos

    def reset(self, trajectory_prefix=None, gamefile=None):
        """Reset the environment, optionally pinning it to a specific gamefile."""
        if gamefile:
            self._close_replay_env()
            self.env = self._make_env(game_files=[gamefile])
        else:
            self._close_replay_env()
        return self._reset_current_env(trajectory_prefix)
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_games = base_env.num_games
        env_num = min(env_num, self.num_games) if not is_train else env_num
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.active_processes = self.num_processes

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.active_processes, \
            "The num of actions must be equal to the num of active processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers[:self.active_processes]):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs(active_only=True)
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def step_selected(self, indices, actions):
        if len(indices) != len(actions):
            raise ValueError(f"indices/actions length mismatch: {len(indices)} vs {len(actions)}")
        if not indices:
            return [], None, [], [], []

        futures = []
        for idx, action in zip(indices, actions):
            if idx < 0 or idx >= self.active_processes:
                raise IndexError(f"selected env index {idx} outside active range [0, {self.active_processes})")
            futures.append(self.workers[idx].step.remote(action))

        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []
        results = ray.get(futures)
        for idx, (obs, scores, dones, info) in zip(indices, results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)
            self.prev_admissible_commands[idx] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        # Online DenoiseRL currently targets AlfredTWEnv. Keep the return shape
        # aligned with step(); multimodal selected stepping can be added when needed.
        image_obs_list = None
        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset_selected(self, indices, kwargs):
        if len(indices) != len(kwargs):
            raise ValueError(f"indices/kwargs length mismatch: {len(indices)} vs {len(kwargs)}")
        if not indices:
            return [], None, []

        futures = []
        for idx, item in zip(indices, kwargs):
            if idx < 0 or idx >= self.active_processes:
                raise IndexError(f"selected env index {idx} outside active range [0, {self.active_processes})")
            item = item or {}
            trajectory_prefix = (
                item.get("trajectory_prefix")
                or item.get("prefix_actions")
                or item.get("prefix_steps")
            )
            gamefile = item.get("gamefile") or item.get("extra.gamefile")
            futures.append(self.workers[idx].reset.remote(trajectory_prefix, gamefile))

        text_obs_list = []
        info_list = []
        results = ray.get(futures)
        for idx, (obs, info) in zip(indices, results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[idx] = info['admissible_commands']
            info_list.append(info)

        image_obs_list = None
        return text_obs_list, image_obs_list, info_list

    def _normalize_reset_kwargs(self, kwargs):
        if kwargs is None:
            return [{} for _ in range(self.num_processes)]
        if isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()
        if isinstance(kwargs, dict):
            return [dict(kwargs) for _ in range(self.num_processes)]
        if isinstance(kwargs, list):
            if len(kwargs) > self.num_processes:
                raise ValueError(
                    f"reset kwargs length ({len(kwargs)}) must be <= num_processes ({self.num_processes})."
                )
            return [dict(item or {}) for item in kwargs]
        raise TypeError(f"Unsupported reset kwargs type: {type(kwargs)}")

    def reset(self, kwargs=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        reset_kwargs = self._normalize_reset_kwargs(kwargs)
        self.active_processes = len(reset_kwargs)
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers[:self.active_processes]):
            trajectory_prefix = (
                reset_kwargs[i].get("trajectory_prefix")
                or reset_kwargs[i].get("prefix_actions")
                or reset_kwargs[i].get("prefix_steps")
            )
            gamefile = reset_kwargs[i].get("gamefile") or reset_kwargs[i].get("extra.gamefile")
            if isinstance(trajectory_prefix, dict):
                gamefile = gamefile or trajectory_prefix.get("gamefile")
            future = worker.reset.remote(trajectory_prefix, gamefile)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs(active_only=True)
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self, active_only=False):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        workers = self.workers[:self.active_processes] if active_only else self.workers
        for worker in workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands[:self.active_processes]

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
