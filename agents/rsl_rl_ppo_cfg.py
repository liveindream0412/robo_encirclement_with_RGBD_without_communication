# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg

from isaaclab_tasks.direct.robomaster_encirclement.networks import ActorCriticRgbdGru, ActorCriticRgbdMlp  # noqa: F401


@configclass
class RgbdGruActorCriticCfg(RslRlPpoActorCriticRecurrentCfg):
    class_name = "ActorCriticRgbdGru"
    init_noise_std = 0.05
    noise_std_type = "scalar"
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [256, 128]
    critic_hidden_dims = [256, 128]
    activation = "elu"
    rnn_type = "gru"
    rnn_hidden_dim = 256
    rnn_num_layers = 1
    image_height = 64
    image_width = 64
    state_dim = 7
    rgb_channels = 3
    depth_channels = 1
    cnn_feature_dim = 128
    state_feature_dim = 128
    fusion_hidden_dim = 256


@configclass
class RobomasterEncirclementPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "robomaster_encirclement_rgbd_gru"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RgbdGruActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=1.0,
    )
