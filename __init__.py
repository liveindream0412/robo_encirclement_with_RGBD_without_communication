# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB-D multi-agent Robomaster encirclement task."""

import gymnasium as gym

from . import agents
from . import networks  # noqa: F401


gym.register(
    id="Isaac-Robomaster-Encirclement-RGBD-Direct-v0",
    entry_point=f"{__name__}.robomaster_encirclement_env:RobomasterEncirclementEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robomaster_encirclement_env_cfg:RobomasterEncirclementEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobomasterEncirclementPPORunnerCfg",
    },
)
