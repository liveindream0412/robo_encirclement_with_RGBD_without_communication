# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior cloning warm-start for the shared Robomaster RGB-D actor."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Train BC actor for Robomaster encirclement.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="logs/robomaster_bc/bc_actor.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--aux_weight", type=float, default=0.5)
    parser.add_argument("--network", type=str, default="gru", choices=["gru", "mlp"])
    parser.add_argument("--visual_pretrain_epochs", type=int, default=10)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=20)
    parser.add_argument("--finetune_lr_scale", type=float, default=0.25)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--windows_per_epoch", type=int, default=128)
    parser.add_argument("--sequence_batch_size", type=int, default=64)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from isaaclab_tasks.direct.robomaster_encirclement.networks import ActorCriticRgbdGru, ActorCriticRgbdMlp


def main():
    args = parse_args()
    data = torch.load(args_cli.dataset, map_location="cpu", weights_only=False)
    policy_obs = data["policy_obs"].float()
    actions = data["actions"].float().clamp(-1.0, 1.0)
    aux_labels = data.get("aux_labels")
    if aux_labels is not None:
        aux_labels = aux_labels.float()
    num_samples = policy_obs.shape[0]
    num_envs = int(data.get("num_envs", 1))
    num_steps = int(data.get("num_steps", num_samples // (num_envs * 3)))
    num_agents = 3
    batch_width = num_envs * num_agents
    if num_samples != num_steps * batch_width:
        raise ValueError(
            f"Dataset shape mismatch: num_samples={num_samples}, num_steps={num_steps}, "
            f"num_envs={num_envs}, num_agents={num_agents}."
        )
    policy_obs_seq = policy_obs.reshape(num_steps, batch_width, -1)
    actions_seq = actions.reshape(num_steps, batch_width, -1)
    aux_labels_seq = None if aux_labels is None else aux_labels.reshape(num_steps, batch_width, -1)

    dummy_obs = TensorDict(
        {
            "policy": torch.zeros(1, policy_obs.shape[-1], device=args_cli.device),
            "critic": torch.zeros(1, 1, device=args_cli.device),
        },
        batch_size=[1],
    )
    model_cls = ActorCriticRgbdMlp if args_cli.network == "mlp" else ActorCriticRgbdGru
    model = model_cls(dummy_obs, {"policy": ["policy"], "critic": ["critic"]}, actions.shape[-1]).to(args_cli.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr)

    if aux_labels_seq is not None and args_cli.visual_pretrain_epochs > 0:
        _set_backbone_trainable(model, True)
        _set_actor_sequence_trainable(model, False)
        optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=args_cli.lr)
        _train_visual_pretrain(model, optimizer, policy_obs_seq, aux_labels_seq, batch_width, args_cli)

    _set_actor_sequence_trainable(model, True)
    _set_backbone_trainable(model, args_cli.freeze_backbone_epochs <= 0)
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=args_cli.lr)

    sequence_length = min(args_cli.sequence_length, num_steps)
    max_start = max(num_steps - sequence_length + 1, 1)
    sequence_batch_size = max(1, min(args_cli.sequence_batch_size, batch_width))
    for epoch in range(args_cli.epochs):
        if epoch == args_cli.freeze_backbone_epochs and args_cli.freeze_backbone_epochs > 0:
            _set_backbone_trainable(model, True)
            optimizer = torch.optim.Adam(
                (param for param in model.parameters() if param.requires_grad),
                lr=args_cli.lr * args_cli.finetune_lr_scale,
            )
            print(
                f"[INFO] Unfroze visual backbone for joint fine-tuning at epoch {epoch + 1}; "
                f"lr={args_cli.lr * args_cli.finetune_lr_scale:.3g}"
            )
        running_loss = 0.0
        running_aux_loss = 0.0
        running_count = 0
        for _ in range(args_cli.windows_per_epoch):
            t0 = int(torch.randint(0, max_start, (1,)).item())
            batch_ids = torch.randperm(batch_width)[:sequence_batch_size]
            obs_window = policy_obs_seq[t0 : t0 + sequence_length, batch_ids].to(args_cli.device)
            action_window = actions_seq[t0 : t0 + sequence_length, batch_ids].to(args_cli.device)
            pred = model._actor_mean(
                obs_window,
                hidden_states=model._initial_hidden(sequence_batch_size, obs_window.device),
            )
            bc_loss = F.mse_loss(pred, action_window)
            loss = bc_loss
            if aux_labels_seq is not None:
                aux_window = aux_labels_seq[t0 : t0 + sequence_length, batch_ids].reshape(sequence_length * sequence_batch_size, -1).to(
                    args_cli.device
                )
                obs_flat = obs_window.reshape(sequence_length * sequence_batch_size, -1)
                visible_loss, reg_loss = _aux_loss(model, obs_flat, aux_window)
                aux_loss = visible_loss + reg_loss
                loss = bc_loss + args_cli.aux_weight * aux_loss
                running_aux_loss += aux_loss.item() * sequence_length * sequence_batch_size
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += bc_loss.item() * sequence_length * sequence_batch_size
            running_count += sequence_length * sequence_batch_size
        aux_msg = ""
        if aux_labels_seq is not None:
            aux_msg = f", aux_loss={running_aux_loss / max(running_count, 1):.6f}"
        print(f"[INFO] Epoch {epoch + 1}/{args_cli.epochs}: bc_loss={running_loss / max(running_count, 1):.6f}{aux_msg}")

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "dataset": args_cli.dataset, "network": args_cli.network}, output)
    print(f"[INFO] Saved BC actor checkpoint to: {output}")


def _set_backbone_trainable(model: ActorCriticRgbdGru | ActorCriticRgbdMlp, trainable: bool) -> None:
    modules = (model.rgb_encoder, model.depth_encoder, model.state_encoder, model.actor_fusion)
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(trainable)


def _set_actor_sequence_trainable(model: ActorCriticRgbdGru | ActorCriticRgbdMlp, trainable: bool) -> None:
    modules = []
    if hasattr(model, "actor_gru"):
        modules.append(model.actor_gru)
    modules.append(model.actor)
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(trainable)
    if hasattr(model, "std"):
        model.std.requires_grad_(trainable)
    if hasattr(model, "log_std"):
        model.log_std.requires_grad_(trainable)


def _aux_loss(model: ActorCriticRgbdGru | ActorCriticRgbdMlp, obs_flat: torch.Tensor, aux_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_aux = model.predict_aux(obs_flat).reshape(-1, 3, 4)
    true_aux = aux_flat.reshape(-1, 3, 4)
    visible_loss = F.binary_cross_entropy_with_logits(pred_aux[:, :, 0], true_aux[:, :, 0])
    visible_mask = true_aux[:, :, 0:1]
    reg_loss = torch.sum(torch.square(pred_aux[:, :, 1:] - true_aux[:, :, 1:]) * visible_mask) / visible_mask.sum().clamp(
        min=1.0
    )
    return visible_loss, reg_loss


def _train_visual_pretrain(
    model: ActorCriticRgbdGru,
    optimizer: torch.optim.Optimizer,
    policy_obs_seq: torch.Tensor,
    aux_labels_seq: torch.Tensor,
    batch_width: int,
    args: argparse.Namespace,
) -> None:
    num_steps = policy_obs_seq.shape[0]
    sequence_length = min(args.sequence_length, num_steps)
    max_start = max(num_steps - sequence_length + 1, 1)
    sequence_batch_size = max(1, min(args.sequence_batch_size, batch_width))
    print(
        f"[INFO] Visual pretrain: epochs={args.visual_pretrain_epochs}, "
        f"windows_per_epoch={args.windows_per_epoch}, sequence_length={sequence_length}, "
        f"sequence_batch_size={sequence_batch_size}"
    )
    for epoch in range(args.visual_pretrain_epochs):
        running_aux_loss = 0.0
        running_count = 0
        for _ in range(args.windows_per_epoch):
            t0 = int(torch.randint(0, max_start, (1,)).item())
            batch_ids = torch.randperm(batch_width)[:sequence_batch_size]
            obs_flat = policy_obs_seq[t0 : t0 + sequence_length, batch_ids].reshape(sequence_length * sequence_batch_size, -1).to(
                args.device
            )
            aux_flat = aux_labels_seq[t0 : t0 + sequence_length, batch_ids].reshape(sequence_length * sequence_batch_size, -1).to(
                args.device
            )
            visible_loss, reg_loss = _aux_loss(model, obs_flat, aux_flat)
            aux_loss = visible_loss + reg_loss
            optimizer.zero_grad()
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
            optimizer.step()
            running_aux_loss += aux_loss.item() * sequence_length * sequence_batch_size
            running_count += sequence_length * sequence_batch_size
        print(
            f"[INFO] Visual pretrain epoch {epoch + 1}/{args.visual_pretrain_epochs}: "
            f"aux_loss={running_aux_loss / max(running_count, 1):.6f}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
