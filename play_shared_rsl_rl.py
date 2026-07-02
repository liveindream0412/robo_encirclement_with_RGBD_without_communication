# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a Robomaster encirclement checkpoint trained with the shared MARL RSL-RL wrapper."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RSL_RL_SCRIPT_DIR = _REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(_RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play Robomaster encirclement with a shared RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during play.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in steps.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of base multi-agent environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Robomaster-Encirclement-RGBD-Direct-v0", help="Task name.")
parser.add_argument("--network", type=str, default="gru", choices=["gru", "mlp"], help="Policy network used by RSL-RL.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import rsl_rl.runners.on_policy_runner as on_policy_runner_module
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.robomaster_encirclement.networks import ActorCriticRgbdGru, ActorCriticRgbdMlp
from isaaclab_tasks.direct.robomaster_encirclement.shared_marl_vecenv_wrapper import SharedActorMARLVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

on_policy_runner_module.ActorCriticRgbdGru = ActorCriticRgbdGru
on_policy_runner_module.ActorCriticRgbdMlp = ActorCriticRgbdMlp


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    if args_cli.network == "mlp":
        agent_cfg.policy.class_name = "ActorCriticRgbdMlp"
    else:
        agent_cfg.policy.class_name = "ActorCriticRgbdGru"
    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SharedActorMARLVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations().to(agent_cfg.device)
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions.to(env.device))
            obs = obs.to(agent_cfg.device)
            if torch.any(dones):
                runner.alg.policy.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
