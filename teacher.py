# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific potential-field teacher policy for Robomaster encirclement."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat


class GeometricEncirclementTeacher:
    """Ordered equilateral ring controller used as an imitation teacher.

    The controller is a task-specific APF variant for three robots encircling a static target. It keeps the
    current angular order of robots, fits the nearest equilateral triangle on the target-centered ring, and applies
    radial and tangential potential-field commands. Close-range safety and front-blocking avoidance are retained.
    """

    def __init__(
        self,
        ideal_radius: float,
        action_scale: tuple[float, float, float],
        radial_gain: float = 1.8,
        angular_gain: float = 1.6,
        safety_gain: float = 3.0,
        target_keepout_gain: float = 2.0,
        obstacle_repulsion_gain: float = 1.0,
        damping_gain: float = 0.25,
        yaw_gain: float = 1.2,
        min_robot_distance: float = 1.05,
        target_keepout_margin: float = 0.25,
    ):
        self.ideal_radius = ideal_radius
        self.action_scale = action_scale
        self.radial_gain = radial_gain
        self.angular_gain = angular_gain
        self.safety_gain = safety_gain
        self.target_keepout_gain = target_keepout_gain
        self.obstacle_repulsion_gain = obstacle_repulsion_gain
        self.damping_gain = damping_gain
        self.yaw_gain = yaw_gain
        self.min_robot_distance = min_robot_distance
        self.target_keepout_margin = target_keepout_margin

    def __call__(self, env) -> dict[str, torch.Tensor]:
        robot_pos = env._robot_positions()
        robot_quat = env._robot_quats()
        target_pos = env._target_pos_w
        robot_vel_w = torch.stack([robot.data.root_lin_vel_w[:, :2] for robot in env.robots], dim=1)
        teacher_actions = self.compute_actions(robot_pos, robot_quat, target_pos, robot_vel_w)
        return {agent: teacher_actions[:, i] for i, agent in enumerate(env.agent_names)}

    def compute_actions(
        self,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
        target_pos: torch.Tensor,
        robot_vel_w: torch.Tensor | None = None,
        obstacle_points_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_envs, num_robots, _ = robot_pos.shape
        device = robot_pos.device
        robot_xy = robot_pos[:, :, :2]
        target_xy = target_pos[:, :2]
        rel = robot_xy - target_xy[:, None, :]
        radius = torch.linalg.norm(rel, dim=-1).clamp(min=1.0e-6)
        radial_dir = rel / radius.unsqueeze(-1)
        tangent_dir = torch.stack([-radial_dir[:, :, 1], radial_dir[:, :, 0]], dim=-1)
        angles = torch.atan2(rel[:, :, 1], rel[:, :, 0])
        assigned_angles = self._nearest_ordered_equilateral_angles(angles)

        radial_cmd = -self.radial_gain * (radius - self.ideal_radius).unsqueeze(-1) * radial_dir
        angle_error = wrap_to_pi(assigned_angles - angles)
        tangent_cmd = self.angular_gain * self.ideal_radius * angle_error.unsqueeze(-1) * tangent_dir
        safety_cmd = self._close_range_safety(robot_xy, robot_quat)
        target_cmd = self._target_keepout_force(rel, radius)
        obstacle_cmd = self._obstacle_repulsion(robot_xy, obstacle_points_w)
        damping_cmd = torch.zeros_like(radial_cmd) if robot_vel_w is None else -self.damping_gain * robot_vel_w

        cmd_w = radial_cmd + tangent_cmd + safety_cmd + target_cmd + obstacle_cmd + damping_cmd
        settled = (torch.abs(radius - self.ideal_radius) < 0.08) & (torch.abs(angle_error) < 0.12)
        cmd_w = torch.where(settled.unsqueeze(-1), 0.25 * cmd_w, cmd_w)
        speed_limit = self._speed_limit(radius, angle_error).unsqueeze(-1)
        cmd_norm = torch.linalg.norm(cmd_w, dim=-1, keepdim=True).clamp(min=1.0e-6)
        cmd_w = cmd_w * torch.clamp(speed_limit / cmd_norm, max=1.0)

        yaw_to_target = torch.atan2(target_xy[:, None, 1] - robot_xy[:, :, 1], target_xy[:, None, 0] - robot_xy[:, :, 0])
        yaw = euler_xyz_from_quat(robot_quat.reshape(-1, 4))[2].reshape(num_envs, num_robots)
        yaw_error = wrap_to_pi(yaw_to_target - yaw)
        cmd_b = quat_apply_inverse(
            yaw_quat(robot_quat.reshape(-1, 4)),
            torch.cat([cmd_w.reshape(-1, 2), torch.zeros(num_envs * num_robots, 1, device=device)], dim=-1),
        ).reshape(num_envs, num_robots, 3)

        command = torch.zeros(num_envs, num_robots, 3, device=device)
        command[:, :, 0] = cmd_b[:, :, 0] / max(self.action_scale[0], 1.0e-6)
        command[:, :, 1] = cmd_b[:, :, 1] / max(self.action_scale[1], 1.0e-6)
        command[:, :, 2] = self.yaw_gain * yaw_error / max(self.action_scale[2], 1.0e-6)
        return command.clamp(-1.0, 1.0)

    def _nearest_ordered_equilateral_angles(self, angles: torch.Tensor) -> torch.Tensor:
        num_envs, num_robots = angles.shape
        sorted_angles, sorted_ids = torch.sort(angles, dim=1)
        offsets = torch.linspace(0.0, 2.0 * math.pi, num_robots + 1, device=angles.device)[:-1]
        candidates = []
        costs = []
        for shift in range(num_robots):
            shifted_offsets = torch.roll(offsets, shifts=shift, dims=0)
            base = torch.atan2(
                torch.mean(torch.sin(sorted_angles - shifted_offsets[None, :]), dim=1),
                torch.mean(torch.cos(sorted_angles - shifted_offsets[None, :]), dim=1),
            )
            candidate = wrap_to_pi(base[:, None] + shifted_offsets[None, :])
            candidates.append(candidate)
            costs.append(torch.sum(torch.square(wrap_to_pi(candidate - sorted_angles)), dim=1))
        stacked_candidates = torch.stack(candidates, dim=1)
        stacked_costs = torch.stack(costs, dim=1)
        best = torch.argmin(stacked_costs, dim=1)
        selected_sorted = stacked_candidates[torch.arange(num_envs, device=angles.device), best]
        assigned = torch.zeros_like(angles)
        assigned.scatter_(1, sorted_ids, selected_sorted)
        return assigned

    def _close_range_safety(self, robot_xy: torch.Tensor, robot_quat: torch.Tensor) -> torch.Tensor:
        cmd = torch.zeros_like(robot_xy)
        num_envs, num_robots, _ = robot_xy.shape
        for i in range(num_robots):
            for j in range(num_robots):
                if i == j:
                    continue
                rel_w = robot_xy[:, j] - robot_xy[:, i]
                dist = torch.linalg.norm(rel_w, dim=-1, keepdim=True).clamp(min=1.0e-6)
                direction_to_j = rel_w / dist
                intrusion = torch.clamp(self.min_robot_distance - dist, min=0.0) / self.min_robot_distance
                force = -self.safety_gain * intrusion * direction_to_j

                rel_b = quat_apply_inverse(
                    yaw_quat(robot_quat[:, i]),
                    torch.cat([rel_w, torch.zeros(num_envs, 1, device=robot_xy.device)], dim=-1),
                )[:, :2]
                front_x_strength = torch.clamp((1.8 - rel_b[:, 0]) / 1.8, min=0.0, max=1.0)
                front_y_strength = torch.clamp((0.75 - torch.abs(rel_b[:, 1])) / 0.75, min=0.0, max=1.0)
                front_strength = torch.where(rel_b[:, 0] > 0.0, front_x_strength * front_y_strength, torch.zeros_like(front_x_strength))
                side_sign = torch.where(rel_b[:, 1] >= 0.0, -torch.ones_like(rel_b[:, 1]), torch.ones_like(rel_b[:, 1]))
                side_avoid_b = torch.stack([-0.25 * front_strength, 0.65 * front_strength * side_sign], dim=-1)
                force += self._body_to_world(side_avoid_b, robot_quat[:, i])
                cmd[:, i] += force
        return cmd

    def _target_keepout_force(self, rel_from_target: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        safe_radius = self.ideal_radius - self.target_keepout_margin
        intrusion = torch.clamp(safe_radius - radius, min=0.0).unsqueeze(-1) / max(safe_radius, 1.0e-6)
        direction = rel_from_target / radius.unsqueeze(-1).clamp(min=1.0e-6)
        return self.target_keepout_gain * intrusion * direction

    def _obstacle_repulsion(self, robot_xy: torch.Tensor, obstacle_points_w: torch.Tensor | None) -> torch.Tensor:
        if obstacle_points_w is None:
            return torch.zeros_like(robot_xy)
        obstacle_xy = obstacle_points_w[..., :2]
        rel = robot_xy[:, :, None, :] - obstacle_xy[:, None, :, :]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1.0e-6)
        influence = 1.0
        strength = torch.clamp(influence - dist, min=0.0) / influence
        return self.obstacle_repulsion_gain * torch.sum(strength * rel / dist, dim=2)

    def _speed_limit(self, radius: torch.Tensor, angle_error: torch.Tensor) -> torch.Tensor:
        radius_error = torch.abs(radius - self.ideal_radius)
        angular_error = torch.abs(angle_error)
        normalized_error = (radius_error + self.ideal_radius * angular_error).clamp(max=1.0)
        return 0.25 + 0.9 * normalized_error

    def _body_to_world(self, cmd_b: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        yaw = euler_xyz_from_quat(quat)[2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        return torch.stack(
            [cos_yaw * cmd_b[:, 0] - sin_yaw * cmd_b[:, 1], sin_yaw * cmd_b[:, 0] + cos_yaw * cmd_b[:, 1]],
            dim=-1,
        )
