# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Robomaster encirclement task with a shared decentralized actor and centralized critic."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RSL_RL_SCRIPT_DIR = _REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(_RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train Robomaster encirclement with shared RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recorded videos in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of base multi-agent environments.")
parser.add_argument("--task", type=str, default="Isaac-Robomaster-Encirclement-RGBD-Direct-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of training iterations.")
parser.add_argument("--network", type=str, default="gru", choices=["gru", "mlp"], help="Policy network used by RSL-RL.")
parser.add_argument("--pretrained_actor", type=str, default=None, help="Path to a BC actor checkpoint for PPO warm-start.")
parser.add_argument("--freeze_actor_backbone", action="store_true", default=False, help="Freeze RGB-D/state encoders and fusion after BC warm-start.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run distributed training.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Unused for direct environments.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import omni
import rsl_rl.runners.on_policy_runner as on_policy_runner_module
from isaaclab.utils.assets import retrieve_file_path
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.robomaster_encirclement.networks import ActorCriticRgbdGru, ActorCriticRgbdMlp
from isaaclab_tasks.direct.robomaster_encirclement.shared_marl_vecenv_wrapper import SharedActorMARLVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

on_policy_runner_module.ActorCriticRgbdGru = ActorCriticRgbdGru
on_policy_runner_module.ActorCriticRgbdMlp = ActorCriticRgbdMlp

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.network == "mlp":
        agent_cfg.policy.class_name = "ActorCriticRgbdMlp"
    else:
        agent_cfg.policy.class_name = "ActorCriticRgbdGru"
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn("IO descriptors are only supported for manager based RL environments.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = SharedActorMARLVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if args_cli.pretrained_actor is not None:
        pretrained_path = retrieve_file_path(args_cli.pretrained_actor)
        print(f"[INFO]: Loading BC actor warm-start from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=agent_cfg.device, weights_only=False)
        checkpoint_network = checkpoint.get("network") if isinstance(checkpoint, dict) else None
        if checkpoint_network is not None and checkpoint_network != args_cli.network:
            raise ValueError(
                f"BC checkpoint network mismatch: checkpoint has '{checkpoint_network}', "
                f"but PPO was started with --network {args_cli.network}."
            )
        pretrained_state = checkpoint.get("model_state_dict", checkpoint)
        policy_state = runner.alg.policy.state_dict()
        warm_start_state = {
            key: value
            for key, value in pretrained_state.items()
            if key in policy_state and policy_state[key].shape == value.shape and not key.startswith("critic.")
        }
        policy_state.update(warm_start_state)
        runner.alg.policy.load_state_dict(policy_state, strict=True)
        skipped = len(pretrained_state) - len(warm_start_state)
        print(f"[INFO]: BC warm-start loaded {len(warm_start_state)} tensors, skipped {skipped} tensors.")
        if hasattr(runner.alg.policy, "std"):
            runner.alg.policy.std.data.fill_(0.05)
            print("[INFO]: Set PPO action std to 0.05 for conservative BC fine-tuning.")
        if hasattr(runner.alg.policy, "log_std"):
            runner.alg.policy.log_std.data.fill_(torch.log(torch.tensor(0.05, device=runner.alg.policy.log_std.device)))
            print("[INFO]: Set PPO log action std to log(0.05) for conservative BC fine-tuning.")

    if args_cli.freeze_actor_backbone:
        _freeze_actor_backbone(runner.alg.policy)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    env.close()


def _freeze_actor_backbone(policy: torch.nn.Module) -> None:
    frozen_prefixes = ("rgb_encoder", "depth_encoder", "state_encoder", "actor_fusion")
    frozen_count = 0
    trainable_count = 0
    for name, param in policy.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad_(False)
            frozen_count += param.numel()
        else:
            trainable_count += param.numel()
    print(
        f"[INFO]: Frozen actor backbone parameters: {frozen_count}. "
        f"Remaining trainable parameters: {trainable_count}."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
