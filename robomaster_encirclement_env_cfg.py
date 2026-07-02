# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.visual_navigation_finished.visual_navigation.assets.robots.gamebot import GAMEBOT_CFG


@configclass
class RobomasterEncirclementEnvCfg(DirectMARLEnvCfg):
    """Configuration for the RGB-D three-Robomaster encirclement task."""

    # env
    decimation = 4
    episode_length_s = 12.0
    possible_agents = ["robot_0", "robot_1", "robot_2"]
    action_spaces = {"robot_0": 3, "robot_1": 3, "robot_2": 3}
    observation_spaces = {"robot_0": 16391, "robot_1": 16391, "robot_2": 16391}
    state_space = 20

    # observation layout
    image_height = 64
    image_width = 64
    rgb_channels = 3
    depth_channels = 1
    state_dim = 7

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="multiply",
            #摩擦力
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=10.0, replicate_physics=True)
    viewer = ViewerCfg(eye=(8.0, 8.0, 6.0), lookat=(0.0, 0.0, 0.0))

    # robots
    robot_0_cfg: ArticulationCfg = GAMEBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_1_cfg: ArticulationCfg = GAMEBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_2_cfg: ArticulationCfg = GAMEBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot_2")
    wheel_joint_names = ["wheel_joint_.*"]
    wheel_base = 0.4
    axle_base = 0.414
    wheel_radius = 0.07
    action_scale = (1.5, 0.75, 2.4)
    action_offset = (0.0, 0.0, 0.0)
    #速度倍数
    policy_action_multiplier = 0.8
    action_rate_limit = (0.08, 0.08, 0.05)

    # target visible in RGB and depth
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 0.45, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.12), roughness=0.9, metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )

    # one RGB-D camera per robot
    camera_horizontal_fov = math.radians(82.0)

    camera_0_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot_0/chassis/camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.18, 0.0, 1.2), rot=(0.945, 0.0, 0.327, 0.0), convention="world"),
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.1, 10.0),
        ),
        width=image_width,
        height=image_height,
    )
    camera_1_cfg: TiledCameraCfg = camera_0_cfg.replace(prim_path="/World/envs/env_.*/Robot_1/chassis/camera")
    camera_2_cfg: TiledCameraCfg = camera_0_cfg.replace(prim_path="/World/envs/env_.*/Robot_2/chassis/camera")

    # task geometry
    target_height = 0.25
    robot_height = 0.03
    fixed_reset = False
    fixed_target_xy = (3.6, 3.6)
    fixed_robot_xy = ((0.9, 2.4), (0.9, 3.6), (0.9, 4.8))
    fixed_face_target = True
    initial_yaw_noise = 0.45
    spawn_radius_range = (2.0, 3.4)
    target_xy_range = (-0.8, 0.8)
    arena_half_size = 3.6
    curriculum_target_centered_prob = 0.2
    random_target_xy_range = (-1.8, 1.8)
    random_robot_xy_range = (-3.2, 3.2)
    random_min_target_distance = 1.3
    random_min_robot_distance = 0.8
    ideal_radius = 1.25
    ring_half_width = 0.25
    min_robot_distance = 0.55
    min_target_distance = 0.45
    hard_robot_collision_distance = 0.45
    hard_target_collision_distance = 0.6
    angle_tolerance = math.radians(24.0)
    max_gap_tolerance = math.radians(150.0)
    success_hold_time_s = 1.0

    # reward scales
    reward_approach_scale = 0.0
    reward_ring_scale = 1.5
    reward_angle_scale = 0.0
    reward_equilateral_scale = 6.0
    reward_equilateral_hold_scale = 14.0
    reward_side_balance_scale = 4.0
    reward_face_target_scale = 1.0
    reward_search_target_scale = 0.8
    reward_recover_target_scale = 3.0
    reward_recover_robot_scale = 4.0
    penalty_equilateral_error_scale = 0.6
    penalty_short_side_scale = 10.0
    penalty_side_spread_scale = 4.0
    reward_gap_scale = 0.0
    reward_small_gap_scale = 0.0
    reward_stable_scale = 0.0
    reward_soft_stable_scale = 0.0
    reward_look_at_target_scale = 0.0
    reward_angular_separation_scale = 0.0
    penalty_collision_scale = 36.0
    penalty_hard_collision_scale = 60.0
    penalty_crowd_scale = 12.0
    penalty_action_rate_scale = 0.04
    penalty_spin_scale = 0.15
    penalty_time_scale = 0.02

    # image preprocessing
    depth_min = 0.1
    depth_max = 10.0
