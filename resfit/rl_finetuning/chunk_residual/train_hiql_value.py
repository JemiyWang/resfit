"""离线训练 HIQL action-free value(模块 ③a)。从仓库根跑:

    conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
      --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly --output outputs_chunk/three_piece_value.pt

设计见 docs/superpowers/specs/2026-06-07-hiql-value-design.md。
"""
import argparse

import h5py
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from resfit.rl_finetuning.chunk_residual.hiql_value import (
    build_transitions, train_value, save_value,
)
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    STATE18_KEYS, assemble_state18, sorted_demo_keys,
)
from resfit.rl_finetuning.utils.normalization import StateStandardizer


def read_per_demo_states(hdf5_path, dataset_id, device="cpu"):
    """读每条 demo 的标准化 18 维 state 序列(与 RL 训练同源 mean/std)。

    返回 (list[np.ndarray (T,18)], standardizer)。复用 assemble_state18 + STATE18_KEYS。
    """
    meta = LeRobotDatasetMetadata(dataset_id)
    standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=device)
    seqs = []
    with h5py.File(hdf5_path, "r") as f:
        for ep in sorted_demo_keys(list(f["data"].keys())):
            grp = f[f"data/{ep}"]
            obs_arrays = {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS}
            state_raw = assemble_state18(obs_arrays)  # (T,18) np
            state_n = standardizer.standardize(
                torch.as_tensor(state_raw, dtype=torch.float32)).cpu().numpy()
            seqs.append(state_n)
    return seqs, standardizer


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 HIQL action-free value(③a)")
    p.add_argument("--hdf5", required=True, help="源 hdf5(含 data/demo_i/obs/<key>)")
    p.add_argument("--dataset", required=True, help="LeRobot dataset id(取 state norm stats)")
    p.add_argument("--output", default="value.pt")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--ema", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--value_hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    seqs, standardizer = read_per_demo_states(args.hdf5, args.dataset)
    s, s_next, done = build_transitions(seqs)
    print(f"[hiql_value] demos={len(seqs)} transitions={s.shape[0]} state_dim={s.shape[1]}")
    model, v_stats = train_value(
        s, s_next, done, gamma=args.gamma, expectile=args.expectile, ema=args.ema,
        lr=args.lr, batch_size=args.batch_size, steps=args.steps,
        hidden=args.value_hidden, seed=args.seed)
    save_value(args.output, model, v_stats=v_stats,
               mean=standardizer._mean.cpu(), std=standardizer._std.cpu(),
               dataset_id=args.dataset)
    print(f"[hiql_value] saved {args.output}; v_stats={v_stats}")


if __name__ == "__main__":
    main()
