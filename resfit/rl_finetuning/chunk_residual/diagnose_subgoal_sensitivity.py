from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from tensordict import TensorDict

from resfit.rl_finetuning.chunk_residual.subgoal_sensitivity import (
    measure_subgoal_gradient_sensitivity,
    measure_subgoal_sensitivity,
)
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import NUM_STAGES
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.rl.stage_utils import parse_stage_budget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure whether a trained residual policy is sensitive to observation.subgoal."
    )
    parser.add_argument("--run_dir", required=True, help="Directory containing best.pt")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to run_dir/best.pt")
    parser.add_argument("--offline_buffer_cache", required=True, help="Offline buffer cache directory with storage/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stddev", type=float, default=0.0)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["shuffle", "zero", "noise"],
        choices=["shuffle", "zero", "noise"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit metrics as one JSON object")
    return parser


def _image_keys(obs_td) -> list[str]:
    keys = [k for k in obs_td.keys() if isinstance(k, str) and k.startswith("observation.images.")]
    if not keys:
        keys = [k for k in obs_td.keys() if isinstance(k, str) and "image" in k]
    if not keys:
        raise ValueError("offline obs has no image keys")
    return list(keys)


def _plain_obs(obs_td) -> dict[str, torch.Tensor]:
    return {k: v for k, v in obs_td.items()}


def _load_cache(cache_dir: str) -> TensorDict:
    storage = Path(cache_dir) / "storage"
    if not storage.is_dir():
        raise FileNotFoundError(f"offline buffer storage not found: {storage}")
    return TensorDict.load_memmap(str(storage))


def _make_agent(ckpt: dict, first_obs, first_action: torch.Tensor, image_keys: list[str], device: str) -> QAgent:
    cfg_args = ckpt["config"]
    state_dim = int(first_obs["observation.state"].shape[-1])
    subgoal_dim = int(first_obs["observation.subgoal"].shape[-1]) if getattr(cfg_args, "subgoal_conditioned", False) else 0
    action_dim = int(first_action.reshape(first_action.shape[0], -1).shape[-1])
    img_shape = tuple(int(x) for x in first_obs[image_keys[0]].shape[-3:])

    cfg = ResidualTD3BoxCleanConfig()
    cfg.agent.actor_lr = getattr(cfg_args, "actor_lr", cfg.agent.actor_lr)
    cfg.agent.critic_lr = getattr(cfg_args, "critic_lr", cfg.agent.critic_lr)
    cfg.agent.actor.action_scale = getattr(cfg_args, "action_scale", cfg.agent.actor.action_scale)
    cfg.agent.bc_loss_coef = getattr(cfg_args, "demo_bc_coef", cfg.agent.bc_loss_coef)
    cfg.agent.bc_loss_dynamic = 0
    cfg.agent.device = device

    num_stages = NUM_STAGES.get(getattr(cfg_args, "task", ""), 1)
    stage_budget = parse_stage_budget(getattr(cfg_args, "stage_budget", None), num_stages)
    agent = QAgent(
        obs_shape=img_shape,
        prop_shape=(state_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        stage_conditioned=getattr(cfg_args, "stage_conditioned", False),
        num_stages=num_stages,
        stage_budget=stage_budget,
        subgoal_conditioned=getattr(cfg_args, "subgoal_conditioned", False),
        subgoal_dim=subgoal_dim,
    )
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.train(False)
    return agent


def _sample_batch(data: TensorDict, batch_size: int, device: str) -> TensorDict:
    n = int(data.batch_size[0])
    if n <= 0:
        raise ValueError("offline buffer cache is empty")
    take = min(batch_size, n)
    idx = torch.randperm(n)[:take]
    return data[idx].to(device, non_blocking=True)


def run(args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(args.seed)
    ckpt_path = args.checkpoint or os.path.join(args.run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    data = _load_cache(args.offline_buffer_cache)

    first = data[:1]
    image_keys = _image_keys(first["obs"])
    agent = _make_agent(ckpt, first["obs"], first["action"], image_keys, args.device)

    sums: dict[str, float] = defaultdict(float)
    for _ in range(args.num_batches):
        batch = _sample_batch(data, args.batch_size, args.device)
        obs = _plain_obs(batch["obs"])
        action = batch["action"].reshape(batch["action"].shape[0], -1)
        metrics = measure_subgoal_sensitivity(
            agent,
            obs,
            action,
            stddev=args.stddev,
            variants=args.variants,
            noise_scale=args.noise_scale,
        )
        metrics.update(measure_subgoal_gradient_sensitivity(agent, obs, action))
        for key, value in metrics.items():
            sums[key] += value

    return {key: value / args.num_batches for key, value in sorted(sums.items())}


def main() -> None:
    args = build_parser().parse_args()
    metrics = run(args)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        for key, value in metrics.items():
            print(f"{key}: {value:.6g}")


if __name__ == "__main__":
    main()
