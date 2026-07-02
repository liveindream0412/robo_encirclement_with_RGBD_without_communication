# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectMARLEnv


class SharedActorMARLVecEnvWrapper(VecEnv):
    """RSL-RL wrapper for homogeneous MARL with one shared per-agent actor.

    It exposes ``num_envs * num_agents`` samples to RSL-RL. Each sample is one robot's local policy
    observation and one robot's 3-D action. The critic observation is the centralized global state,
    repeated once per agent, so actor execution stays decentralized while training can use global information.
    """

    def __init__(self, env: gym.Env, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, DirectMARLEnv):
            raise ValueError(f"Expected a DirectMARLEnv, got {type(env.unwrapped)}")
        self.env = env
        self.clip_actions = clip_actions
        self.agent_names = list(self.unwrapped.possible_agents)
        self.num_base_envs = self.unwrapped.num_envs
        self.num_agents = len(self.agent_names)
        self.num_envs = self.num_base_envs * self.num_agents
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = gym.spaces.flatdim(self.unwrapped.action_spaces[self.agent_names[0]])
        self.single_observation_space = self.unwrapped.observation_spaces[self.agent_names[0]]
        self.single_action_space = self.unwrapped.action_spaces[self.agent_names[0]]
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.env.reset()

    @property
    def unwrapped(self) -> DirectMARLEnv:
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf.repeat(self.num_agents)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value.reshape(self.num_agents, self.num_base_envs)[0]

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        return self._to_rsl_obs(obs_dict), extras

    def get_observations(self) -> TensorDict:
        return self._to_rsl_obs(self.unwrapped._get_observations())

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        marl_actions = self._split_actions(actions)
        obs_dict, reward_dict, terminated_dict, truncated_dict, extras = self.env.step(marl_actions)
        obs = self._to_rsl_obs(obs_dict)
        rewards = torch.cat([reward_dict[agent].reshape(-1) for agent in self.agent_names], dim=0)
        dones = torch.cat(
            [(terminated_dict[agent] | truncated_dict[agent]).reshape(-1) for agent in self.agent_names], dim=0
        ).long()
        time_outs = torch.cat([truncated_dict[agent].reshape(-1) for agent in self.agent_names], dim=0)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = time_outs
        return obs, rewards, dones, extras

    def close(self):
        return self.env.close()

    def _to_rsl_obs(self, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
        policy_obs = torch.cat([obs_dict[agent].reshape(self.num_base_envs, -1) for agent in self.agent_names], dim=0)
        critic_state = self.unwrapped.state().reshape(self.num_base_envs, -1).repeat(self.num_agents, 1)
        return TensorDict({"policy": policy_obs, "critic": critic_state}, batch_size=[self.num_envs])

    def _split_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        actions = actions.reshape(self.num_agents, self.num_base_envs, self.num_actions)
        return {agent: actions[i] for i, agent in enumerate(self.agent_names)}
