# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation: {name}")


class ActorCriticRgbdMlp(nn.Module):
    """RGB-D actor with modality encoders and centralized critic."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        image_height: int = 64,
        image_width: int = 64,
        state_dim: int = 7,
        rgb_channels: int = 3,
        depth_channels: int = 1,
        cnn_feature_dim: int = 128,
        state_feature_dim: int = 128,
        fusion_hidden_dim: int = 256,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **_: object,
    ):
        super().__init__()
        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.image_height = image_height
        self.image_width = image_width
        self.state_dim = state_dim
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.rgb_dim = rgb_channels * image_height * image_width
        self.depth_dim = depth_channels * image_height * image_width
        self.expected_actor_obs_dim = self.rgb_dim + self.depth_dim + state_dim
        self.num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        self.noise_std_type = noise_std_type

        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        if num_actor_obs != self.expected_actor_obs_dim:
            raise ValueError(
                f"Actor observation dim mismatch: got {num_actor_obs}, expected {self.expected_actor_obs_dim}."
            )

        self.rgb_encoder = self._cnn_encoder(rgb_channels, image_height, image_width, cnn_feature_dim, activation)
        self.depth_encoder = self._cnn_encoder(depth_channels, image_height, image_width, cnn_feature_dim, activation)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim),
            _activation(activation),
            nn.Linear(state_feature_dim, state_feature_dim),
            _activation(activation),
        )
        self.actor_fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim * 2 + state_feature_dim, fusion_hidden_dim),
            _activation(activation),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            _activation(activation),
        )
        self.actor = self._mlp(fusion_hidden_dim, actor_hidden_dims or [256, 128], num_actions, activation)
        self.critic = self._mlp(self.num_critic_obs, critic_hidden_dims or [256, 128], 1, activation)

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _cnn_encoder(channels: int, image_height: int, image_width: int, output_dim: int, activation: str) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
            _activation(activation),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Flatten(),
            nn.Linear(64 * (image_height // 8) * (image_width // 8), output_dim),
            _activation(activation),
        )

    @staticmethod
    def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(_activation(activation))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def act(self, obs, masks: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        mean = self._actor_mean(self.get_actor_obs(obs))
        self._update_distribution(mean)
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs) if isinstance(obs, dict) or hasattr(obs, "keys") else obs
        return self._actor_mean(actor_obs)

    def evaluate(self, obs, masks: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(self.get_critic_obs(obs))

    def get_actor_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs) -> None:
        pass

    def _actor_mean(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self._encode_actor(observations))

    def _update_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def _encode_actor(self, observations: torch.Tensor) -> torch.Tensor:
        rgb = observations[:, : self.rgb_dim].reshape(-1, self.rgb_channels, self.image_height, self.image_width)
        depth_start = self.rgb_dim
        depth_end = depth_start + self.depth_dim
        depth = observations[:, depth_start:depth_end].reshape(
            -1, self.depth_channels, self.image_height, self.image_width
        )
        state = observations[:, depth_end : depth_end + self.state_dim]
        return self.actor_fusion(
            torch.cat([self.rgb_encoder(rgb), self.depth_encoder(depth), self.state_encoder(state)], dim=-1)
        )
