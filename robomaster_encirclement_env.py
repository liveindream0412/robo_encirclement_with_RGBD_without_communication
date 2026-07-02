# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.shapes import spawn_cuboid
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, sample_uniform, wrap_to_pi, yaw_quat, quat_apply_inverse

from .robomaster_encirclement_env_cfg import RobomasterEncirclementEnvCfg


class RobomasterEncirclementEnv(DirectMARLEnv):
    """Three-agent RGB-D Robomaster encirclement environment."""

    cfg: RobomasterEncirclementEnvCfg

    def __init__(self, cfg: RobomasterEncirclementEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.agent_names = list(self.cfg.possible_agents)
        self.robots = [self.robot_0, self.robot_1, self.robot_2]
        self.cameras = [self.camera_0, self.camera_1, self.camera_2]
        self._num_robots = len(self.agent_names)

        self._joint_ids = []
        for robot in self.robots:
            joint_ids, _ = robot.find_joints(self.cfg.wheel_joint_names)
            self._joint_ids.append(joint_ids)

        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device).view(1, 3)
        self._action_offset = torch.tensor(self.cfg.action_offset, device=self.device).view(1, 3)
        self._action_rate_limit = torch.tensor(self.cfg.action_rate_limit, device=self.device).view(1, 3)
        self._actions = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.agent_names}
        self._previous_actions = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.agent_names}

        base = (self.cfg.wheel_base + self.cfg.axle_base) / 2.0
        self._mecanum_matrix = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [base, base, base, base]],
            device=self.device,
        ) / self.cfg.wheel_radius
        max_base_cmd = torch.tensor(self.cfg.action_scale, device=self.device)
        self._max_wheel_speed = torch.max(torch.abs(max_base_cmd)) / self.cfg.wheel_radius

        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_target_distance = torch.zeros(self.num_envs, self._num_robots, device=self.device)
        self._success_hold_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_hold_steps = max(1, int(self.cfg.success_hold_time_s / self.step_dt))

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "ring",
                "equilateral",
                "equilateral_error",
                "side_balance",
                "short_side",
                "side_spread",
                "equilateral_hold",
                "face_target",
                "search_target",
                "recover_target",
                "recover_robot",
                "collision",
                "hard_collision",
                "crowd",
                "action_rate",
                "time",
            ]
        }

    def _setup_scene(self):
        self.robot_0 = Articulation(self.cfg.robot_0_cfg)
        self.robot_1 = Articulation(self.cfg.robot_1_cfg)
        self.robot_2 = Articulation(self.cfg.robot_2_cfg)
        self.target = RigidObject(self.cfg.target_cfg)
        self.camera_0 = TiledCamera(self.cfg.camera_0_cfg)
        self.camera_1 = TiledCamera(self.cfg.camera_1_cfg)
        self.camera_2 = TiledCamera(self.cfg.camera_2_cfg)

        spawn_cuboid(
            prim_path="/World/ground",
            cfg=CuboidCfg(
                size=(200.0, 200.0, 0.02),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=self.cfg.sim.physics_material,
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                    project_uvw=True,
                    texture_scale=(0.25, 0.25),
                ),
            ),
            translation=(0.0, 0.0, -0.01),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot_0"] = self.robot_0
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2
        self.scene.rigid_objects["target"] = self.target
        self.scene.sensors["camera_0"] = self.camera_0
        self.scene.sensors["camera_1"] = self.camera_1
        self.scene.sensors["camera_2"] = self.camera_2

        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent in self.agent_names:
            self._previous_actions[agent][:] = self._actions[agent]
            target_action = (actions[agent].clone() * self.cfg.policy_action_multiplier).clamp(-1.0, 1.0)
            delta = (target_action - self._actions[agent]).clamp(-self._action_rate_limit, self._action_rate_limit)
            self._actions[agent][:] = (self._actions[agent] + delta).clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        for robot_id, robot in enumerate(self.robots):
            agent = self.agent_names[robot_id]
            base_cmd = self._actions[agent] * self._action_scale + self._action_offset
            wheel_cmd = torch.matmul(base_cmd, self._mecanum_matrix)
            max_wheel = torch.abs(wheel_cmd).amax(dim=-1, keepdim=True).clamp(min=1.0e-6)
            wheel_cmd = wheel_cmd * torch.clamp(self._max_wheel_speed / max_wheel, max=1.0)
            robot.set_joint_velocity_target(wheel_cmd, joint_ids=self._joint_ids[robot_id])

    def _get_observations(self) -> dict[str, torch.Tensor]:
        robot_pos = self._robot_positions()
        robot_quat = self._robot_quats()
        target_pos = self._target_pos_w
        observations = {}
        for robot_id, agent in enumerate(self.agent_names):
            rgb = self._camera_rgb(self.cameras[robot_id])
            depth = self._camera_depth(self.cameras[robot_id])
            state = self._agent_state(robot_id, robot_pos, robot_quat, target_pos)
            observations[agent] = torch.cat([rgb, depth, state], dim=-1)
        return observations

    def _get_states(self) -> torch.Tensor:
        robot_pos = self._robot_positions()
        robot_quat = self._robot_quats()
        robot_lin_vel = torch.stack([robot.data.root_lin_vel_b[:, :2] for robot in self.robots], dim=1)
        robot_ang_vel = torch.stack([robot.data.root_ang_vel_b[:, 2] for robot in self.robots], dim=1).unsqueeze(-1)
        yaw = euler_xyz_from_quat(robot_quat.reshape(-1, 4))[2].reshape(self.num_envs, self._num_robots, 1)
        return torch.cat(
            [
                (robot_pos[:, :, :2] - self.scene.env_origins[:, None, :2]).reshape(self.num_envs, -1),
                yaw.reshape(self.num_envs, -1),
                robot_lin_vel.reshape(self.num_envs, -1),
                robot_ang_vel.reshape(self.num_envs, -1),
                (self._target_pos_w[:, :2] - self.scene.env_origins[:, :2]),
            ],
            dim=-1,
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        robot_pos = self._robot_positions()
        target_xy = self._target_pos_w[:, None, :2]
        rel_target = robot_pos[:, :, :2] - target_xy
        distances = torch.linalg.norm(rel_target, dim=-1).clamp(min=1.0e-6)

        ring_error = torch.abs(distances - self.cfg.ideal_radius)
        ring_progress = (self._prev_target_distance - ring_error) / self.step_dt
        self._prev_target_distance[:] = ring_error
        ring = 1.0 - torch.tanh(ring_error / max(self.cfg.ring_half_width, 1.0e-6))
        ring_gate = ring.mean(dim=1).clamp(min=0.02)

        angles = torch.atan2(rel_target[:, :, 1], rel_target[:, :, 0])
        sorted_angles, _ = torch.sort(angles, dim=1)
        gaps = torch.diff(torch.cat([sorted_angles, sorted_angles[:, :1] + 2.0 * math.pi], dim=1), dim=1)
        ideal_gap = 2.0 * math.pi / self._num_robots
        gap_error = torch.abs(gaps - ideal_gap)
        angle_error = torch.mean(gap_error, dim=1)
        max_gap_error = torch.max(gap_error, dim=1).values
        gap_uniformity = 1.0 - torch.tanh(max_gap_error / ideal_gap)
        angle_reward = gap_uniformity * (0.35 + 0.65 * ring_gate)
        max_gap = torch.max(gaps, dim=1).values
        min_gap = torch.min(gaps, dim=1).values
        gap_gate = 0.35 + 0.65 * ring_gate
        gap_penalty = torch.clamp(max_gap - ideal_gap, min=0.0) / math.pi * gap_gate
        small_gap_penalty = torch.clamp(0.75 * ideal_gap - min_gap, min=0.0) / ideal_gap * gap_gate
        angular_separation = self._angular_separation_reward(angles) * gap_gate
        target_rel_robot_frame = self._target_relative_positions_b(robot_pos)
        target_bearing = torch.atan2(target_rel_robot_frame[:, :, 1], target_rel_robot_frame[:, :, 0])
        face_target_reward = torch.exp(-torch.abs(target_bearing) / 0.65).mean(dim=1)
        search_target_reward = torch.zeros(self.num_envs, device=self.device)

        pair_distances = self._pairwise_robot_distances(robot_pos)
        ideal_side_length = math.sqrt(3.0) * self.cfg.ideal_radius
        side_error = torch.abs(pair_distances - ideal_side_length)
        min_side = torch.min(pair_distances, dim=1).values
        max_side = torch.max(pair_distances, dim=1).values
        side_spread = max_side - min_side
        side_uniform_error = torch.std(pair_distances, dim=1)
        equilateral_error = side_error.mean(dim=1) + 0.5 * side_uniform_error
        equilateral_reward = torch.exp(-equilateral_error / 0.35) * ring.mean(dim=1)
        side_balance_reward = torch.exp(-side_spread / 0.35) * ring.mean(dim=1)
        equilateral_penalty = torch.tanh(equilateral_error / ideal_side_length)
        short_side_penalty = torch.square(torch.clamp(0.9 * ideal_side_length - min_side, min=0.0) / ideal_side_length)
        side_spread_penalty = torch.tanh(side_spread / ideal_side_length)
        crowd_penalty = torch.clamp(self.cfg.min_robot_distance - pair_distances, min=0.0).sum(dim=1)
        near_crowd_penalty = torch.clamp(0.85 - pair_distances, min=0.0).sum(dim=1)
        target_collision = torch.clamp(self.cfg.min_target_distance - distances, min=0.0).sum(dim=1)
        hard_robot_collision = (pair_distances < self.cfg.hard_robot_collision_distance).float().sum(dim=1)
        hard_target_collision = (distances < self.cfg.hard_target_collision_distance).float().sum(dim=1)
        hard_collision_penalty = hard_robot_collision + hard_target_collision
        collision_penalty = crowd_penalty + near_crowd_penalty + target_collision

        action_rate = torch.zeros(self.num_envs, device=self.device)
        spin_penalty = torch.zeros(self.num_envs, device=self.device)
        recover_target_reward = torch.zeros(self.num_envs, device=self.device)
        recover_robot_reward = torch.zeros(self.num_envs, device=self.device)
        for robot_id, agent in enumerate(self.agent_names):
            action_rate += torch.sum(torch.square(self._actions[agent] - self._previous_actions[agent]), dim=-1)
            linear_speed_cmd = torch.linalg.norm(self._actions[agent][:, :2], dim=-1)
            spin_penalty += torch.clamp(torch.abs(self._actions[agent][:, 2]) - 0.25, min=0.0) * torch.exp(-4.0 * linear_speed_cmd)
            search_target_reward += torch.clamp(-torch.sign(target_bearing[:, robot_id]) * self._actions[agent][:, 2], min=0.0) * (
                torch.abs(target_bearing[:, robot_id]) > 0.35
            ).float()
            action_world = self._action_world_xy(robot_id)
            away_from_target = rel_target[:, robot_id, :2] / distances[:, robot_id : robot_id + 1].clamp(min=1.0e-6)
            target_too_close = (distances[:, robot_id] < self.cfg.hard_target_collision_distance).float()
            recover_target_reward += target_too_close * torch.clamp(torch.sum(action_world * away_from_target, dim=-1), min=0.0)
            nearest_rel, nearest_dist = self._nearest_neighbor_vector(robot_id, robot_pos)
            away_from_neighbor = -nearest_rel / nearest_dist.unsqueeze(-1).clamp(min=1.0e-6)
            robot_too_close = (nearest_dist < 0.9 * ideal_side_length).float()
            recover_robot_reward += robot_too_close * torch.clamp(torch.sum(action_world * away_from_neighbor, dim=-1), min=0.0)

        in_ring = torch.all(ring_error < self.cfg.ring_half_width, dim=1)
        angle_ok = torch.all(torch.abs(gaps - ideal_gap) < self.cfg.angle_tolerance, dim=1)
        gap_ok = max_gap < self.cfg.max_gap_tolerance
        equilateral_ok = equilateral_error < 0.18
        collision_free = (collision_penalty <= 0.0) & (hard_collision_penalty <= 0.0)
        stable_now = in_ring & angle_ok & gap_ok & equilateral_ok & collision_free
        self._success_hold_buf = torch.where(stable_now, self._success_hold_buf + 1, torch.zeros_like(self._success_hold_buf))
        stable_reward = (self._success_hold_buf.float() / self._success_hold_steps).clamp(max=1.0)
        uniformity = 1.0 - torch.tanh(max_gap_error / self.cfg.angle_tolerance)
        soft_stable = ring.mean(dim=1) * uniformity.clamp(min=0.0) * collision_free.float()

        team_reward = (
            self.cfg.reward_ring_scale * ring.mean(dim=1)
            + self.cfg.reward_equilateral_scale * equilateral_reward
            + self.cfg.reward_side_balance_scale * side_balance_reward
            + self.cfg.reward_equilateral_hold_scale * stable_reward
            + self.cfg.reward_face_target_scale * face_target_reward
            + self.cfg.reward_search_target_scale * search_target_reward
            + self.cfg.reward_recover_target_scale * recover_target_reward
            + self.cfg.reward_recover_robot_scale * recover_robot_reward
            - self.cfg.penalty_equilateral_error_scale * equilateral_penalty
            - self.cfg.penalty_short_side_scale * short_side_penalty
            - self.cfg.penalty_side_spread_scale * side_spread_penalty
            - self.cfg.penalty_collision_scale * collision_penalty
            - self.cfg.penalty_hard_collision_scale * hard_collision_penalty
            - self.cfg.penalty_crowd_scale * near_crowd_penalty
            - self.cfg.penalty_action_rate_scale * action_rate
            - self.cfg.penalty_spin_scale * spin_penalty
            - self.cfg.penalty_time_scale
        ) * self.step_dt

        rewards = {}
        for robot_id, agent in enumerate(self.agent_names):
            individual = (
                -self.cfg.penalty_crowd_scale
                * torch.clamp(self.cfg.min_robot_distance - self._nearest_neighbor_distance(robot_id, robot_pos), min=0.0)
            ) * self.step_dt
            rewards[agent] = team_reward + individual

        self._episode_sums["ring"] += ring.mean(dim=1) * self.cfg.reward_ring_scale * self.step_dt
        self._episode_sums["equilateral"] += equilateral_reward * self.cfg.reward_equilateral_scale * self.step_dt
        self._episode_sums["equilateral_error"] += -equilateral_penalty * self.cfg.penalty_equilateral_error_scale * self.step_dt
        self._episode_sums["side_balance"] += side_balance_reward * self.cfg.reward_side_balance_scale * self.step_dt
        self._episode_sums["short_side"] += -short_side_penalty * self.cfg.penalty_short_side_scale * self.step_dt
        self._episode_sums["side_spread"] += -side_spread_penalty * self.cfg.penalty_side_spread_scale * self.step_dt
        self._episode_sums["equilateral_hold"] += stable_reward * self.cfg.reward_equilateral_hold_scale * self.step_dt
        self._episode_sums["face_target"] += face_target_reward * self.cfg.reward_face_target_scale * self.step_dt
        self._episode_sums["search_target"] += search_target_reward * self.cfg.reward_search_target_scale * self.step_dt
        self._episode_sums["recover_target"] += recover_target_reward * self.cfg.reward_recover_target_scale * self.step_dt
        self._episode_sums["recover_robot"] += recover_robot_reward * self.cfg.reward_recover_robot_scale * self.step_dt
        self._episode_sums["collision"] += -collision_penalty * self.cfg.penalty_collision_scale * self.step_dt
        self._episode_sums["hard_collision"] += -hard_collision_penalty * self.cfg.penalty_hard_collision_scale * self.step_dt
        self._episode_sums["crowd"] += -near_crowd_penalty * self.cfg.penalty_crowd_scale * self.step_dt
        self._episode_sums["action_rate"] += (
            -(action_rate * self.cfg.penalty_action_rate_scale + spin_penalty * self.cfg.penalty_spin_scale) * self.step_dt
        )
        self._episode_sums["time"] += -torch.ones_like(action_rate) * self.cfg.penalty_time_scale * self.step_dt
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        robot_pos = self._robot_positions()
        pair_distances = self._pairwise_robot_distances(robot_pos)
        distances = torch.linalg.norm(robot_pos[:, :, :2] - self._target_pos_w[:, None, :2], dim=-1)
        out_of_bounds = torch.any(torch.abs(robot_pos[:, :, :2] - self.scene.env_origins[:, None, :2]) > self.cfg.arena_half_size, dim=(1, 2))
        success = self._success_hold_buf >= self._success_hold_steps
        terminated_env = out_of_bounds | success
        time_out_env = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {agent: terminated_env for agent in self.agent_names}
        time_out = {agent: time_out_env for agent in self.agent_names}
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot_0._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._log_reset(env_ids)
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        if self.cfg.fixed_reset:
            fixed_target_xy = torch.tensor(self.cfg.fixed_target_xy, device=self.device, dtype=torch.float32).view(1, 2)
            target_xy = fixed_target_xy.expand(num_reset, -1) - self.cfg.arena_half_size
            use_target_centered = torch.ones(num_reset, dtype=torch.bool, device=self.device)
        else:
            use_target_centered = torch.rand(num_reset, device=self.device) < self.cfg.curriculum_target_centered_prob
            centered_target_xy = sample_uniform(
                self.cfg.target_xy_range[0],
                self.cfg.target_xy_range[1],
                (num_reset, 2),
                self.device,
            )
            random_target_xy = sample_uniform(
                self.cfg.random_target_xy_range[0],
                self.cfg.random_target_xy_range[1],
                (num_reset, 2),
                self.device,
            )
            target_xy = torch.where(use_target_centered.unsqueeze(-1), centered_target_xy, random_target_xy)
        self._target_pos_w[env_ids, :2] = self.scene.env_origins[env_ids, :2] + target_xy
        self._target_pos_w[env_ids, 2] = self.cfg.target_height

        target_state = self.target.data.default_root_state[env_ids].clone()
        target_state[:, :3] = self._target_pos_w[env_ids]
        target_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        target_state[:, 7:] = 0.0
        self.target.write_root_pose_to_sim(target_state[:, :7], env_ids)
        self.target.write_root_velocity_to_sim(target_state[:, 7:], env_ids)

        target_pos_reset = self._target_pos_w[env_ids]
        env_origins_reset = self.scene.env_origins[env_ids]
        if self.cfg.fixed_reset:
            fixed_robot_xy = torch.tensor(self.cfg.fixed_robot_xy, device=self.device, dtype=torch.float32)
            robot_xy = env_origins_reset[:, None, :2] + fixed_robot_xy.unsqueeze(0) - self.cfg.arena_half_size
        else:
            robot_xy = self._sample_mixed_robot_positions(num_reset, use_target_centered, target_pos_reset, env_origins_reset)
        yaw = self._sample_initial_yaws(robot_xy, use_target_centered, target_pos_reset)

        for robot_id, robot in enumerate(self.robots):
            root_state = robot.data.default_root_state[env_ids].clone()
            root_state[:, 0:2] = robot_xy[:, robot_id]
            root_state[:, 2] = self.cfg.robot_height
            root_state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw[:, robot_id]), torch.zeros_like(yaw[:, robot_id]), yaw[:, robot_id])
            root_state[:, 7:] = 0.0
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids], robot.data.default_joint_vel[env_ids], env_ids=env_ids)
            self._actions[self.agent_names[robot_id]][env_ids] = 0.0
            self._previous_actions[self.agent_names[robot_id]][env_ids] = 0.0

        distances = torch.linalg.norm(self._robot_positions()[env_ids, :, :2] - self._target_pos_w[env_ids, None, :2], dim=-1)
        self._prev_target_distance[env_ids] = torch.abs(distances - self.cfg.ideal_radius)
        self._success_hold_buf[env_ids] = 0

    def _sample_mixed_robot_positions(
        self,
        num_reset: int,
        use_target_centered: torch.Tensor,
        target_pos_w: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> torch.Tensor:
        base_angles = torch.linspace(0.0, 2.0 * math.pi, self._num_robots + 1, device=self.device)[:-1]
        random_offset = sample_uniform(-math.pi, math.pi, (num_reset, 1), self.device)
        curriculum_radii = sample_uniform(
            self.cfg.spawn_radius_range[0], self.cfg.spawn_radius_range[1], (num_reset, self._num_robots), self.device
        )
        curriculum_angles = (
            base_angles.unsqueeze(0)
            + random_offset
            + sample_uniform(-0.35, 0.35, (num_reset, self._num_robots), self.device)
        )
        curriculum_xy = target_pos_w[:, None, :2] + curriculum_radii.unsqueeze(-1) * torch.stack(
            [torch.cos(curriculum_angles), torch.sin(curriculum_angles)], dim=-1
        )

        random_xy = sample_uniform(
            self.cfg.random_robot_xy_range[0],
            self.cfg.random_robot_xy_range[1],
            (num_reset, self._num_robots, 2),
            self.device,
        )
        random_xy = random_xy + env_origins[:, None, :2]
        for _ in range(6):
            target_dist = torch.linalg.norm(random_xy - target_pos_w[:, None, :2], dim=-1)
            robot_dist = torch.cdist(random_xy, random_xy)
            robot_dist = robot_dist + torch.eye(self._num_robots, device=self.device).unsqueeze(0) * 100.0
            invalid = (target_dist < self.cfg.random_min_target_distance) | (
                robot_dist.min(dim=-1).values < self.cfg.random_min_robot_distance
            )
            if not torch.any(invalid):
                break
            resampled = sample_uniform(
                self.cfg.random_robot_xy_range[0],
                self.cfg.random_robot_xy_range[1],
                (num_reset, self._num_robots, 2),
                self.device,
            )
            resampled = resampled + env_origins[:, None, :2]
            random_xy = torch.where(invalid.unsqueeze(-1), resampled, random_xy)

        return torch.where(use_target_centered.view(num_reset, 1, 1), curriculum_xy, random_xy)

    def _sample_initial_yaws(
        self, robot_xy: torch.Tensor, use_target_centered: torch.Tensor, target_pos_w: torch.Tensor
    ) -> torch.Tensor:
        target_bearing = torch.atan2(
            target_pos_w[:, None, 1] - robot_xy[:, :, 1],
            target_pos_w[:, None, 0] - robot_xy[:, :, 0],
        )
        if self.cfg.fixed_face_target:
            yaw_noise = sample_uniform(-self.cfg.initial_yaw_noise, self.cfg.initial_yaw_noise, target_bearing.shape, self.device)
            return wrap_to_pi(target_bearing + yaw_noise)
        random_yaw = sample_uniform(-math.pi, math.pi, (robot_xy.shape[0], self._num_robots), self.device)
        target_facing_yaw = wrap_to_pi(
            target_bearing
            + sample_uniform(-self.cfg.initial_yaw_noise, self.cfg.initial_yaw_noise, target_bearing.shape, self.device)
        )
        return torch.where(use_target_centered.view(robot_xy.shape[0], 1), target_facing_yaw, random_yaw)

    def _camera_rgb(self, camera: TiledCamera) -> torch.Tensor:
        rgb = camera.data.output["rgb"].float() / 255.0
        return rgb.permute(0, 3, 1, 2).reshape(self.num_envs, -1)

    def _camera_depth(self, camera: TiledCamera) -> torch.Tensor:
        depth = camera.data.output["distance_to_camera"]
        depth = torch.nan_to_num(depth, nan=self.cfg.depth_max, posinf=self.cfg.depth_max, neginf=self.cfg.depth_max)
        depth = depth.clamp(self.cfg.depth_min, self.cfg.depth_max)
        depth = (depth - self.cfg.depth_min) / (self.cfg.depth_max - self.cfg.depth_min)
        return depth.permute(0, 3, 1, 2).reshape(self.num_envs, -1)

    def _target_relative_positions_b(self, robot_pos: torch.Tensor) -> torch.Tensor:
        robot_quat = self._robot_quats()
        rel_target_w = self._target_pos_w[:, None, :] - robot_pos
        return quat_apply_inverse(yaw_quat(robot_quat.reshape(-1, 4)), rel_target_w.reshape(-1, 3)).reshape(
            self.num_envs, self._num_robots, 3
        )

    def _angular_separation_reward(self, angles: torch.Tensor) -> torch.Tensor:
        pair_gaps = []
        for i in range(self._num_robots):
            for j in range(i + 1, self._num_robots):
                gap = torch.abs(wrap_to_pi(angles[:, i] - angles[:, j]))
                pair_gaps.append(gap)
        min_gap = torch.stack(pair_gaps, dim=1).min(dim=1).values
        return torch.tanh(min_gap / (2.0 * math.pi / self._num_robots))

    def _agent_state(self, robot_id: int, robot_pos: torch.Tensor, robot_quat: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        robot = self.robots[robot_id]
        pos_i = robot_pos[:, robot_id]
        quat_i = robot_quat[:, robot_id]
        yaw_i = euler_xyz_from_quat(quat_i)[2].unsqueeze(-1)
        state = torch.cat(
            [
                pos_i[:, :2] - (self.scene.env_origins[:, :2] - self.cfg.arena_half_size),
                torch.sin(yaw_i),
                torch.cos(yaw_i),
                robot.data.root_lin_vel_b[:, :2],
                robot.data.root_ang_vel_b[:, 2:3],
                self._actions[self.agent_names[robot_id]],
                self._previous_actions[self.agent_names[robot_id]],
            ],
            dim=-1,
        )
        if state.shape[1] < self.cfg.state_dim:
            pad = torch.zeros(self.num_envs, self.cfg.state_dim - state.shape[1], device=self.device)
            state = torch.cat([state, pad], dim=-1)
        return state[:, : self.cfg.state_dim]

    def _robot_positions(self) -> torch.Tensor:
        return torch.stack([robot.data.root_pos_w for robot in self.robots], dim=1)

    def _robot_quats(self) -> torch.Tensor:
        return torch.stack([robot.data.root_quat_w for robot in self.robots], dim=1)

    def _pairwise_robot_distances(self, robot_pos: torch.Tensor) -> torch.Tensor:
        pairs = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pairs.append(torch.linalg.norm(robot_pos[:, i, :2] - robot_pos[:, j, :2], dim=-1))
        return torch.stack(pairs, dim=1)

    def _nearest_neighbor_distance(self, robot_id: int, robot_pos: torch.Tensor) -> torch.Tensor:
        distances = []
        for other_id in range(self.num_agents):
            if other_id != robot_id:
                distances.append(torch.linalg.norm(robot_pos[:, robot_id, :2] - robot_pos[:, other_id, :2], dim=-1))
        return torch.stack(distances, dim=1).min(dim=1).values

    def _nearest_neighbor_vector(self, robot_id: int, robot_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = []
        distances = []
        for other_id in range(self.num_agents):
            if other_id != robot_id:
                rel = robot_pos[:, other_id, :2] - robot_pos[:, robot_id, :2]
                vectors.append(rel)
                distances.append(torch.linalg.norm(rel, dim=-1))
        stacked_vectors = torch.stack(vectors, dim=1)
        stacked_distances = torch.stack(distances, dim=1)
        nearest_ids = torch.argmin(stacked_distances, dim=1)
        batch_ids = torch.arange(self.num_envs, device=self.device)
        return stacked_vectors[batch_ids, nearest_ids], stacked_distances[batch_ids, nearest_ids]

    def _action_world_xy(self, robot_id: int) -> torch.Tensor:
        agent = self.agent_names[robot_id]
        cmd_b = self._actions[agent][:, :2] * self._action_scale[:, :2]
        yaw = euler_xyz_from_quat(self.robots[robot_id].data.root_quat_w)[2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        return torch.stack(
            [cos_yaw * cmd_b[:, 0] - sin_yaw * cmd_b[:, 1], sin_yaw * cmd_b[:, 0] + cos_yaw * cmd_b[:, 1]],
            dim=-1,
        )

    def _log_reset(self, env_ids: torch.Tensor) -> None:
        extras = {}
        for key, value in self._episode_sums.items():
            extras[f"Episode_Reward/{key}"] = torch.mean(value[env_ids]).item() / self.max_episode_length_s
            value[env_ids] = 0.0
        if env_ids.numel() > 0:
            extras["Metrics/success_hold_steps"] = torch.mean(self._success_hold_buf[env_ids].float()).item()
        self.extras["log"] = extras
