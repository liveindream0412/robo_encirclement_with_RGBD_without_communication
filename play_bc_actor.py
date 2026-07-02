# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a behavior-cloned shared Robomaster actor."""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a BC Robomaster encirclement actor.")
parser.add_argument("--task", type=str, default="Isaac-Robomaster-Encirclement-RGBD-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC actor checkpoint.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--save_camera_frames", action="store_true", default=False)
parser.add_argument("--camera_frame_interval", type=int, default=30)
parser.add_argument("--camera_frame_dir", type=str, default=None)
parser.add_argument("--save_camera_video", action="store_true", default=False)
parser.add_argument("--camera_video_length", type=int, default=800)
parser.add_argument("--camera_video_fps", type=int, default=30)
parser.add_argument("--camera_video_dir", type=str, default=None)
parser.add_argument("--print_coords", action="store_true", default=False)
parser.add_argument("--coord_interval", type=int, default=60)
parser.add_argument("--coord_env_id", type=int, default=0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import torch
from PIL import Image
from tensordict import TensorDict

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.direct.robomaster_encirclement.networks import ActorCriticRgbdGru, ActorCriticRgbdMlp
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    log_dir = os.path.dirname(retrieve_file_path(args_cli.checkpoint))

    camera_frame_dir = args_cli.camera_frame_dir or os.path.join(log_dir, "camera_frames", "bc_play")
    camera_video_dir = args_cli.camera_video_dir or os.path.join(log_dir, "camera_videos", "bc_play")
    camera_video_writers = None
    if args_cli.save_camera_frames:
        os.makedirs(camera_frame_dir, exist_ok=True)
        print(f"[INFO] Saving robot camera frames to: {camera_frame_dir}")
    if args_cli.save_camera_video:
        os.makedirs(camera_video_dir, exist_ok=True)
        camera_video_writers = [
            {
                "rgb": imageio.get_writer(os.path.join(camera_video_dir, f"robot_{robot_id}_rgb.mp4"), fps=args_cli.camera_video_fps),
                "depth": imageio.get_writer(os.path.join(camera_video_dir, f"robot_{robot_id}_depth.mp4"), fps=args_cli.camera_video_fps),
            }
            for robot_id in range(len(env.unwrapped.cameras))
        ]
        print(f"[INFO] Saving robot camera videos to: {camera_video_dir}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "bc_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording BC play video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    zero_actions = {
        agent: torch.zeros(env.unwrapped.num_envs, 3, device=env.unwrapped.device)
        for agent in env.unwrapped.agent_names
    }
    for _ in range(args_cli.warmup_steps):
        obs, _, _, _, _ = env.step(zero_actions)
    first_agent = env.unwrapped.agent_names[0]
    dummy_obs = TensorDict(
        {
            "policy": torch.zeros(1, obs[first_agent].shape[-1], device=env.unwrapped.device),
            "critic": torch.zeros(1, 1, device=env.unwrapped.device),
        },
        batch_size=[1],
    )
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading BC actor checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=env.unwrapped.device, weights_only=False)
    network = checkpoint.get("network", "gru") if isinstance(checkpoint, dict) else "gru"
    actor_cls = ActorCriticRgbdMlp if network == "mlp" else ActorCriticRgbdGru
    print(f"[INFO]: BC actor network: {network}")
    actor = actor_cls(dummy_obs, {"policy": ["policy"], "critic": ["critic"]}, 3).to(env.unwrapped.device)
    actor.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    actor.eval()

    dt = env.unwrapped.step_dt
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            agent_names = env.unwrapped.agent_names
            batched_obs = torch.cat([obs[agent] for agent in agent_names], dim=0)
            batched_actions = actor.act_inference(batched_obs).clamp(-1.0, 1.0)
            actions = {
                agent: batched_actions[i * env.unwrapped.num_envs : (i + 1) * env.unwrapped.num_envs]
                for i, agent in enumerate(agent_names)
            }
            obs, _, terminated, truncated, _ = env.step(actions)
            done = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
            for agent in agent_names:
                done |= terminated[agent] | truncated[agent]
            if torch.any(done):
                actor.reset(done.repeat(len(agent_names)))

        if args_cli.print_coords and timestep % args_cli.coord_interval == 0:
            _print_coordinate_debug(env.unwrapped, obs, args_cli.coord_env_id, timestep)
        if args_cli.save_camera_frames and timestep % args_cli.camera_frame_interval == 0:
            _save_camera_frames(env.unwrapped, camera_frame_dir, timestep)
        if args_cli.save_camera_video and camera_video_writers is not None:
            _append_camera_video_frames(env.unwrapped, camera_video_writers)

        if args_cli.video or args_cli.save_camera_video:
            timestep += 1
            max_length = max(args_cli.video_length if args_cli.video else 0, args_cli.camera_video_length if args_cli.save_camera_video else 0)
            if timestep >= max_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if camera_video_writers is not None:
        for writers in camera_video_writers:
            writers["rgb"].close()
            writers["depth"].close()
    env.close()


def _print_coordinate_debug(env, obs: dict[str, torch.Tensor], env_id: int, timestep: int) -> None:
    env_id = min(max(env_id, 0), env.num_envs - 1)
    origin = env.scene.env_origins[env_id, :2].detach().cpu()
    arena = float(env.cfg.arena_half_size)
    left_bottom = origin - arena
    right_top = origin + arena
    target_world = env._target_pos_w[env_id, :2].detach().cpu()
    target_center_local = target_world - origin
    target_left_origin = target_world - left_bottom
    print(f"[COORD step={timestep} env={env_id}] env_origin_xy={origin.tolist()}")
    print(f"[COORD step={timestep} env={env_id}] arena_left_bottom_world={left_bottom.tolist()} arena_right_top_world={right_top.tolist()}")
    print(
        f"[COORD step={timestep} env={env_id}] target_world_xy={target_world.tolist()} "
        f"target_center_origin_xy={target_center_local.tolist()} target_left_origin_xy={target_left_origin.tolist()}"
    )
    robot_pos = env._robot_positions()
    for robot_id, agent in enumerate(env.agent_names):
        robot_world = robot_pos[env_id, robot_id, :2].detach().cpu()
        robot_center_local = robot_world - origin
        robot_left_origin = robot_world - left_bottom
        policy_xy = obs[agent][env_id, -env.cfg.state_dim : -env.cfg.state_dim + 2].detach().cpu()
        error = policy_xy - robot_left_origin
        print(
            f"[COORD step={timestep} env={env_id}] {agent}: "
            f"world_xy={robot_world.tolist()} center_origin_xy={robot_center_local.tolist()} "
            f"left_origin_xy={robot_left_origin.tolist()} policy_self_xy={policy_xy.tolist()} "
            f"left_origin_error={error.tolist()}"
        )


def _save_camera_frames(env, output_dir: str, timestep: int) -> None:
    for robot_id, camera in enumerate(env.cameras):
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        image = Image.fromarray(rgb.astype("uint8"))
        image.save(os.path.join(output_dir, f"step_{timestep:06d}_robot_{robot_id}_rgb.png"))


def _append_camera_video_frames(env, writers) -> None:
    for robot_id, camera in enumerate(env.cameras):
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype("uint8")
        depth = camera.data.output["distance_to_camera"][0, :, :, 0].detach().cpu()
        depth = torch.nan_to_num(depth, nan=env.cfg.depth_max, posinf=env.cfg.depth_max, neginf=env.cfg.depth_max)
        depth = depth.clamp(env.cfg.depth_min, env.cfg.depth_max)
        depth_norm = (depth - env.cfg.depth_min) / (env.cfg.depth_max - env.cfg.depth_min)
        depth_vis = ((1.0 - depth_norm) * 255.0).numpy().astype("uint8")
        depth_rgb = torch.from_numpy(depth_vis).unsqueeze(-1).repeat(1, 1, 3).numpy()
        writers[robot_id]["rgb"].append_data(rgb)
        writers[robot_id]["depth"].append_data(depth_rgb)


if __name__ == "__main__":
    main()
    simulation_app.close()
