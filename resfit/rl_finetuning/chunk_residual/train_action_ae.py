"""离线预训练动作 chunk 自编码器(AE),用 RL 同款 ActionScaler 归一化。

用法(从仓库根目录,conda env residual):
  python -m resfit.rl_finetuning.chunk_residual.train_action_ae \
      --dataset ankile/dexmg-two-arm-box-cleanup \
      --chunk_length 20 --action_scale 0.2 --min_range_per_dim 0.1 \
      --steps 20000 --batch_size 256 \
      --out resfit/rl_finetuning/chunk_residual/ckpt/ae_boxcleanup.pt
冻结后的权重供 M2 的 residual_flow actor 加载。
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.factory import resolve_delta_timestamps

from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.chunk_residual.action_autoencoder import (
    ActionAutoencoder, action_autoencoder_loss,
)


def build_chunk_dataset(repo_id: str, chunk_length: int):
    """构造带 delta_timestamps 的 LeRobotDataset,使 sample['action'] 为 [chunk_length, D]。"""
    policy_cfg = ACTConfig()
    policy_cfg.chunk_size = chunk_length
    policy_cfg.n_action_steps = chunk_length
    meta = LeRobotDataset(repo_id).meta            # 先拿 meta 解析 delta_timestamps
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    ds = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, download_videos=False)
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--conv_layers", type=int, default=2)
    p.add_argument("--conv_kernel", type=int, default=3)
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--velocity_loss_weight", type=float, default=0.1)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = build_chunk_dataset(args.dataset, args.chunk_length)
    action_stats = ds.meta.stats["action"]
    action_dim = len(action_stats["min"])
    scaler = ActionScaler.from_dataset_stats(
        action_stats=action_stats, action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device=args.device,
    )

    n_val = max(1, int(len(ds) * args.val_fraction))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(0, args.num_workers // 2))

    ae = ActionAutoencoder(action_dim=action_dim, chunk_length=args.chunk_length,
                           latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                           conv_layers=args.conv_layers, conv_kernel=args.conv_kernel).to(args.device)
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr)

    def to_flat(sample):
        a = sample["action"].float().to(args.device)        # [B, L, D]
        a = scaler.scale(a)                                  # [-1,1]
        return a.reshape(a.shape[0], -1)                     # [B, L*D]

    step = 0
    ae.train()
    while step < args.steps:
        for sample in train_loader:
            x = to_flat(sample)
            opt.zero_grad()
            loss, m = action_autoencoder_loss(ae, x, args.velocity_loss_weight)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"[step {step}] loss={m['loss']:.4f} recon_l1={m['recon_l1']:.4f} "
                      f"vel_l1={m['velocity_l1']:.4f}")
            if step >= args.steps:
                break

    # held-out 评估
    ae.eval()
    with torch.no_grad():
        rec_sum, vel_sum, n = 0.0, 0.0, 0
        for sample in val_loader:
            x = to_flat(sample)
            _, m = action_autoencoder_loss(ae, x, args.velocity_loss_weight)
            rec_sum += float(m["recon_l1"]) * x.shape[0]
            vel_sum += float(m["velocity_l1"]) * x.shape[0]
            n += x.shape[0]
        print(f"[VAL] recon_l1={rec_sum / n:.5f} velocity_l1={vel_sum / n:.5f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _to_list(x):
        return x.tolist() if hasattr(x, "tolist") else list(x)

    # 注:M2 的训练脚本会直接从 dataset stats 重建 ActionScaler(单一真源);
    # 这里保存仅为复现/核对用。stats 转成纯 list,确保整个 ckpt 对 torch.load(weights_only=True) 安全。
    torch.save({
        "state_dict": ae.state_dict(),
        "ae_config": {"action_dim": action_dim, "chunk_length": args.chunk_length,
                      "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
                      "conv_layers": args.conv_layers, "conv_kernel": args.conv_kernel,
                      "use_layer_norm": True},
        "action_scaler": {"action_stats": {"min": _to_list(action_stats["min"]),
                                           "max": _to_list(action_stats["max"])},
                          "action_scale": args.action_scale,
                          "min_range_per_dim": args.min_range_per_dim},
    }, args.out)
    print(f"saved AE to {args.out}")


if __name__ == "__main__":
    main()
