# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect expert demonstrations from the geometric encirclement teacher."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Robomaster encirclement expert data.")
parser.add_argument("--task", type=str, default="Isaac-Robomaster-Encirclement-RGBD-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--num_steps", type=int, default=3000)
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--scenario_mix", action="store_true", default=False, help="Inject recovery/danger states during collection.")
parser.add_argument("--scenario_interval", type=int, default=420, help="Deprecated: fixed-interval injection is no longer used.")
parser.add_argument("--scenario_rollout_steps", type=int, default=80, help="Minimum teacher rollout steps before stillness can finish a recovery scenario.")
parser.add_argument("--scenario_fraction", type=float, default=0.25, help="Fraction of envs overwritten on each injection.")
parser.add_argument("--still_speed_threshold", type=float, default=0.04, help="Linear/angular speed threshold for considering a robot settled.")
parser.add_argument("--still_hold_steps", type=int, default=12, help="Consecutive settled steps required before injecting a new recovery scenario.")
parser.add_argument("--output", type=str, default="logs/robomaster_bc/expert_dataset.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.direct.robomaster_encirclement.teacher import GeometricEncirclementTeacher
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    teacher = GeometricEncirclementTeacher(env.unwrapped.cfg.ideal_radius, env.unwrapped.cfg.action_scale)

    obs, _ = env.reset()
    zero_actions = {
        agent: torch.zeros(env.unwrapped.num_envs, 3, device=env.unwrapped.device)
        for agent in env.unwrapped.agent_names
    }
    for _ in range(args_cli.warmup_steps):
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(zero_actions)
    policy_obs_chunks = []
    action_chunks = []
    aux_label_chunks = []

    active_recovery_env_ids = None
    recovery_start_step = -1
    still_hold_count = 0
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            if args_cli.scenario_mix and _should_inject_recovery(env.unwrapped, active_recovery_env_ids, recovery_start_step, still_hold_count, step):
                active_recovery_env_ids = _inject_recovery_scenarios(env.unwrapped, args_cli.scenario_fraction)
                obs = env.unwrapped._get_observations()
                recovery_start_step = step
                still_hold_count = 0
            actions = teacher(env.unwrapped)
            policy_obs_chunks.append(torch.cat([obs[agent] for agent in env.unwrapped.agent_names], dim=0).cpu())
            action_chunks.append(torch.cat([actions[agent] for agent in env.unwrapped.agent_names], dim=0).cpu())
            aux_label_chunks.append(_visual_aux_labels(env.unwrapped).cpu())
            obs, _, _, _, _ = env.step(actions)
            if args_cli.scenario_mix and active_recovery_env_ids is not None and step - recovery_start_step >= args_cli.scenario_rollout_steps:
                if _recovery_envs_settled(env.unwrapped, active_recovery_env_ids, args_cli.still_speed_threshold):
                    still_hold_count += 1
                else:
                    still_hold_count = 0
        if (step + 1) % 100 == 0:
            print(f"[INFO] Collected {step + 1}/{args_cli.num_steps} steps")

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "policy_obs": torch.cat(policy_obs_chunks, dim=0),
        "actions": torch.cat(action_chunks, dim=0),
        "aux_labels": torch.cat(aux_label_chunks, dim=0),
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "num_steps": args_cli.num_steps,
    }
    torch.save(dataset, output)
    print(f"[INFO] Saved expert dataset to: {output}")
    print(f"[INFO] policy_obs: {tuple(dataset['policy_obs'].shape)}")
    print(f"[INFO] actions: {tuple(dataset['actions'].shape)}")
    print(f"[INFO] aux_labels: {tuple(dataset['aux_labels'].shape)}")
    env.close()


def _inject_recovery_scenarios(env, fraction: float) -> torch.Tensor:
    num_inject = max(1, min(env.num_envs, int(env.num_envs * fraction)))
    env_ids = torch.randperm(env.num_envs, device=env.device)[:num_inject]
    origin = env.scene.env_origins[env_ids, :2]
    target_xy = origin + torch.empty(num_inject, 2, device=env.device).uniform_(-0.8, 0.8)
    env._target_pos_w[env_ids, :2] = target_xy
    env._target_pos_w[env_ids, 2] = env.cfg.target_height

    target_state = env.target.data.default_root_state[env_ids].clone()
    target_state[:, :3] = env._target_pos_w[env_ids]
    target_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    target_state[:, 7:] = 0.0
    env.target.write_root_pose_to_sim(target_state[:, :7], env_ids)
    env.target.write_root_velocity_to_sim(target_state[:, 7:], env_ids)

    robot_xy = _sample_recovery_robot_xy(env, target_xy, origin)
    yaws = torch.atan2(target_xy[:, None, 1] - robot_xy[:, :, 1], target_xy[:, None, 0] - robot_xy[:, :, 0])
    yaws = yaws + torch.empty_like(yaws).uniform_(-0.35, 0.35)
    for robot_id, robot in enumerate(env.robots):
        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = robot_xy[:, robot_id]
        root_state[:, 2] = env.cfg.robot_height
        root_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(num_inject, device=env.device), torch.zeros(num_inject, device=env.device), yaws[:, robot_id]
        )
        root_state[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids], robot.data.default_joint_vel[env_ids], env_ids=env_ids)
        env._actions[env.agent_names[robot_id]][env_ids] = 0.0
        env._previous_actions[env.agent_names[robot_id]][env_ids] = 0.0

    distances = torch.linalg.norm(env._robot_positions()[env_ids, :, :2] - env._target_pos_w[env_ids, None, :2], dim=-1)
    env._prev_target_distance[env_ids] = torch.abs(distances - env.cfg.ideal_radius)
    env._success_hold_buf[env_ids] = 0
    return env_ids


def _should_inject_recovery(env, active_env_ids, recovery_start_step: int, still_hold_count: int, step: int) -> bool:
    if active_env_ids is None:
        return True
    if step - recovery_start_step < args_cli.scenario_rollout_steps:
        return False
    return still_hold_count >= args_cli.still_hold_steps


def _recovery_envs_settled(env, env_ids: torch.Tensor, speed_threshold: float) -> bool:
    lin_speeds = []
    ang_speeds = []
    for robot in env.robots:
        lin_speeds.append(torch.linalg.norm(robot.data.root_lin_vel_w[env_ids, :2], dim=-1))
        ang_speeds.append(torch.abs(robot.data.root_ang_vel_w[env_ids, 2]))
    linear_ok = torch.stack(lin_speeds, dim=1).max() < speed_threshold
    angular_ok = torch.stack(ang_speeds, dim=1).max() < speed_threshold
    return bool(torch.logical_and(linear_ok, angular_ok).item())


def _sample_recovery_robot_xy(env, target_xy: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    num_envs = target_xy.shape[0]
    device = target_xy.device
    robot_xy = torch.zeros(num_envs, env.num_agents, 2, device=device)
    scenario_ids = torch.randint(0, 8, (num_envs,), device=device)
    base_angles = torch.empty(num_envs, device=device).uniform_(-torch.pi, torch.pi)

    for env_idx in range(num_envs):
        scenario_id = int(scenario_ids[env_idx].item())
        base = base_angles[env_idx]
        if scenario_id == 0:
            radii = torch.tensor([0.82, 1.25, 1.25], device=device)
            angles = base + torch.tensor([0.0, 2.1, -2.1], device=device)
        elif scenario_id == 1:
            radii = torch.tensor([1.12, 1.18, 1.35], device=device)
            angles = base + torch.tensor([0.0, 0.45, 2.5], device=device)
        elif scenario_id == 2:
            radii = torch.tensor([0.82, 0.95, 1.25], device=device)
            angles = base + torch.tensor([0.0, 0.55, -2.2], device=device)
        elif scenario_id == 3:
            radii = torch.tensor([1.35, 1.45, 1.55], device=device)
            angles = base + torch.tensor([-0.65, 0.0, 0.65], device=device)
        elif scenario_id == 4:
            radii = torch.tensor([1.25, 2.2, 2.5], device=device)
            angles = base + torch.tensor([0.0, 1.2, -1.3], device=device)
        elif scenario_id == 5:
            radii = torch.tensor([1.25, 1.25, 2.4], device=device)
            angles = base + torch.tensor([0.0, 0.75, -2.4], device=device)
        elif scenario_id == 6:
            radii = torch.tensor([1.25, 0.9, 2.2], device=device)
            angles = base + torch.tensor([0.0, 0.75, 2.5], device=device)
        else:
            radii = torch.empty(3, device=device).uniform_(0.85, 2.6)
            angles = base + torch.empty(3, device=device).uniform_(-1.0, 1.0)
        xy = target_xy[env_idx].unsqueeze(0) + radii.unsqueeze(-1) * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        xy = _separate_robots_xy(xy, min_distance=0.72)
        xy = torch.max(torch.min(xy, origin[env_idx].unsqueeze(0) + env.cfg.arena_half_size - 0.25), origin[env_idx].unsqueeze(0) - env.cfg.arena_half_size + 0.25)
        robot_xy[env_idx] = xy
    return robot_xy


def _separate_robots_xy(xy: torch.Tensor, min_distance: float) -> torch.Tensor:
    xy = xy.clone()
    for _ in range(4):
        for i in range(xy.shape[0]):
            for j in range(i + 1, xy.shape[0]):
                rel = xy[j] - xy[i]
                dist = torch.linalg.norm(rel).clamp(min=1.0e-6)
                if dist < min_distance:
                    direction = rel / dist
                    correction = 0.5 * (min_distance - dist) * direction
                    xy[i] -= correction
                    xy[j] += correction
    return xy


def _visual_aux_labels(env) -> torch.Tensor:
    robot_pos = env._robot_positions()
    robot_quat = env._robot_quats()
    target_pos = env._target_pos_w
    labels = []
    for robot_id in range(env.num_agents):
        observer_pos = robot_pos[:, robot_id]
        observer_quat = robot_quat[:, robot_id]
        camera_fov = torch.tensor(env.cfg.camera_horizontal_fov, device=observer_pos.device)
        entity_labels = [_entity_label(observer_pos, observer_quat, target_pos, camera_fov)]
        for other_id in range(env.num_agents):
            if other_id != robot_id:
                entity_labels.append(_entity_label(observer_pos, observer_quat, robot_pos[:, other_id], camera_fov))
        labels.append(torch.cat(entity_labels, dim=-1))
    return torch.cat(labels, dim=0)


def _entity_label(
    observer_pos: torch.Tensor, observer_quat: torch.Tensor, entity_pos: torch.Tensor, camera_fov: torch.Tensor
) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    max_distance = 6.0
    rel_w = entity_pos - observer_pos
    rel_b = quat_apply_inverse(yaw_quat(observer_quat), rel_w)
    dist = torch.linalg.norm(rel_b[:, :2], dim=-1).clamp(min=1.0e-6)
    bearing = torch.atan2(rel_b[:, 1], rel_b[:, 0])
    visible = ((rel_b[:, 0] > 0.0) & (dist < max_distance) & (torch.abs(bearing) < 0.5 * camera_fov)).float()
    return torch.stack(
        [visible, torch.sin(bearing) * visible, torch.cos(bearing) * visible, (dist / max_distance).clamp(max=1.0) * visible],
        dim=-1,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
