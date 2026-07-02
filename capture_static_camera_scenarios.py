# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture robot camera images from hand-placed diagnostic encirclement scenes."""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture static Robomaster camera diagnostic scenarios.")
parser.add_argument("--task", type=str, default="Isaac-Robomaster-Encirclement-RGBD-Direct-v0")
parser.add_argument("--output_dir", type=str, default="logs/robomaster_bc/static_camera_scenarios")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


SCENARIOS = {
    "success_ring": {
        "target_xy": (3.6, 3.6),
        "robot_xy": ((4.85, 3.6), (2.975, 4.6825), (2.975, 2.5175)),
    },
    "target_collision": {
        "target_xy": (3.6, 3.6),
        "robot_xy": ((3.05, 3.6), (2.975, 4.6825), (2.975, 2.5175)),
    },
}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    zero_actions = {
        agent: torch.zeros(env.unwrapped.num_envs, 3, device=env.unwrapped.device) for agent in env.unwrapped.agent_names
    }
    os.makedirs(args_cli.output_dir, exist_ok=True)

    for scenario_name, scenario in SCENARIOS.items():
        _set_scene(env.unwrapped, scenario["target_xy"], scenario["robot_xy"])
        for _ in range(8):
            obs, _, _, _, _ = env.step(zero_actions)
        scenario_dir = os.path.join(args_cli.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        _save_camera_images(env.unwrapped, scenario_dir)
        print(f"[INFO] Saved {scenario_name} camera images to: {scenario_dir}")

    env.close()


def _set_scene(env, target_xy_left_origin: tuple[float, float], robot_xy_left_origin: tuple[tuple[float, float], ...]) -> None:
    env_id = torch.tensor([0], device=env.device, dtype=torch.long)
    origin = env.scene.env_origins[0, :2]
    left_bottom = origin - env.cfg.arena_half_size

    target_xy = torch.tensor(target_xy_left_origin, device=env.device, dtype=torch.float32) + left_bottom
    env._target_pos_w[0, :2] = target_xy
    env._target_pos_w[0, 2] = env.cfg.target_height
    target_state = env.target.data.default_root_state[env_id].clone()
    target_state[:, :3] = env._target_pos_w[env_id]
    target_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    target_state[:, 7:] = 0.0
    env.target.write_root_pose_to_sim(target_state[:, :7], env_id)
    env.target.write_root_velocity_to_sim(target_state[:, 7:], env_id)

    robot_xy = torch.tensor(robot_xy_left_origin, device=env.device, dtype=torch.float32) + left_bottom
    target_xy_batch = target_xy.view(1, 2)
    yaws = torch.atan2(target_xy_batch[:, 1] - robot_xy[:, 1], target_xy_batch[:, 0] - robot_xy[:, 0])
    for robot_id, robot in enumerate(env.robots):
        root_state = robot.data.default_root_state[env_id].clone()
        root_state[:, 0:2] = robot_xy[robot_id]
        root_state[:, 2] = env.cfg.robot_height
        root_state[:, 3:7] = quat_from_euler_xyz(torch.zeros(1, device=env.device), torch.zeros(1, device=env.device), yaws[robot_id : robot_id + 1])
        root_state[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root_state[:, :7], env_id)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_id)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_id], robot.data.default_joint_vel[env_id], env_ids=env_id)
        env._actions[env.agent_names[robot_id]][env_id] = 0.0
        env._previous_actions[env.agent_names[robot_id]][env_id] = 0.0


def _save_camera_images(env, output_dir: str) -> None:
    for robot_id, camera in enumerate(env.cameras):
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype("uint8")
        Image.fromarray(rgb).save(os.path.join(output_dir, f"robot_{robot_id}_rgb.png"))

        depth = camera.data.output["distance_to_camera"][0, :, :, 0].detach().cpu()
        depth = torch.nan_to_num(depth, nan=env.cfg.depth_max, posinf=env.cfg.depth_max, neginf=env.cfg.depth_max)
        depth = depth.clamp(env.cfg.depth_min, env.cfg.depth_max)
        depth_norm = (depth - env.cfg.depth_min) / (env.cfg.depth_max - env.cfg.depth_min)
        depth_vis = ((1.0 - depth_norm) * 255.0).numpy().astype("uint8")
        Image.fromarray(depth_vis).save(os.path.join(output_dir, f"robot_{robot_id}_depth_near_bright.png"))


if __name__ == "__main__":
    main()
    simulation_app.close()
