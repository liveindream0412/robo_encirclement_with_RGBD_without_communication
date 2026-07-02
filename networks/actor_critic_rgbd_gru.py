# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import unpad_trajectories


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


class ActorCriticRgbdGru(nn.Module):
    """RGB-D actor with modality encoders, GRU memory, and centralized critic."""

    is_recurrent = True

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
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
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
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.noise_std_type = noise_std_type

        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        if num_actor_obs != self.expected_actor_obs_dim:
            raise ValueError(
                f"Actor observation dim mismatch: got {num_actor_obs}, expected {self.expected_actor_obs_dim}."
            )

        act = _activation(activation)
        self.rgb_encoder = self._cnn_encoder(rgb_channels, image_height, image_width, cnn_feature_dim, activation)
        self.depth_encoder = self._cnn_encoder(depth_channels, image_height, image_width, cnn_feature_dim, activation)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim), act, nn.Linear(state_feature_dim, state_feature_dim), act
        )
        self.actor_fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim * 2 + state_feature_dim, fusion_hidden_dim),
            _activation(activation),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            _activation(activation),
        )
        self.actor_gru = nn.GRU(fusion_hidden_dim, rnn_hidden_dim, num_layers=rnn_num_layers)
        self.actor = self._mlp(rnn_hidden_dim, actor_hidden_dims or [256, 128], num_actions, activation)
        self.critic = self._mlp(self.num_critic_obs, critic_hidden_dims or [256, 128], 1, activation)

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}")

        self.distribution: Normal | None = None
        self.hidden_states: torch.Tensor | None = None
        self.critic_hidden_states: torch.Tensor | None = None
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
        if self.hidden_states is None:
            return
        if dones is None:
            self.hidden_states.zero_()
            if self.critic_hidden_states is not None:
                self.critic_hidden_states.zero_()
        else:
            done_ids = dones.bool()
            self.hidden_states[:, done_ids, :] = 0.0
            if self.critic_hidden_states is not None:
                self.critic_hidden_states[:, done_ids, :] = 0.0

    def get_hidden_states(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.hidden_states is None:
            return None, None
        if self.critic_hidden_states is None or self.critic_hidden_states.shape[1] != self.hidden_states.shape[1]:
            self.critic_hidden_states = torch.zeros(
                self.rnn_num_layers, self.hidden_states.shape[1], 1, device=self.hidden_states.device
            )
        return self.hidden_states, self.critic_hidden_states

    def act(self, obs, masks: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        mean = self._actor_mean(self.get_actor_obs(obs), masks=masks, hidden_states=hidden_states)
        self._update_distribution(mean)
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs) if isinstance(obs, dict) or hasattr(obs, "keys") else obs
        return self._actor_mean(actor_obs)

    def evaluate(self, obs, masks: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        values = self.critic(critic_obs)
        if critic_obs.dim() == 3 and masks is not None:
            values = unpad_trajectories(values, masks)
        return values

    def get_actor_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs) -> None:
        pass

    def _actor_mean(
        self, observations: torch.Tensor, masks: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self._forward_actor_core(observations, masks=masks, hidden_states=hidden_states)
        return self.actor(features)

    def _update_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def _forward_actor_core(
        self, observations: torch.Tensor, masks: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        if observations.dim() == 3:
            time_steps, batch_size, obs_dim = observations.shape
            flat_features = self._encode_actor(observations.reshape(time_steps * batch_size, obs_dim))
            seq_features = flat_features.reshape(time_steps, batch_size, -1)
            gru_hidden = hidden_states if hidden_states is not None else self._initial_hidden(batch_size, observations.device)
            if masks is not None:
                seq_features = seq_features * masks.reshape(time_steps, batch_size, 1)
            gru_out, next_hidden = self.actor_gru(seq_features, gru_hidden)
            if hidden_states is None:
                self.hidden_states = next_hidden.detach()
            if masks is not None:
                gru_out = unpad_trajectories(gru_out, masks)
            return gru_out

        batch_size = observations.shape[0]
        features = self._encode_actor(observations).unsqueeze(0)
        gru_hidden = hidden_states if hidden_states is not None else self._current_hidden(batch_size, observations.device)
        gru_out, next_hidden = self.actor_gru(features, gru_hidden)
        if hidden_states is None:
            self.hidden_states = next_hidden.detach()
        return gru_out.squeeze(0)

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

    def _initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)

    def _current_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.hidden_states is None or self.hidden_states.shape[1] != batch_size:
            self.hidden_states = self._initial_hidden(batch_size, device)
        return self.hidden_states
