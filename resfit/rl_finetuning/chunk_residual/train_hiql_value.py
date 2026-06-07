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


def read_per_demo_states(hdf5_path, dataset_id, state_mode="eef", num_demos=None, device="cpu"):
    """读每条 demo 的标准化 state 序列(与 RL 训练同源 mean/std)。

    state_mode=eef: (T,18) 纯 eef。eef_piece: (T,30)=[eef18 | 标准化 rel_piece12],
    rel_piece 由 replay set_state 从 sim 算(object-aware;要 replay 全 demo,慢)。
    返回 (list[np.ndarray], standardizer, rel_piece_stats 或 None)。
    """
    import numpy as np
    meta = LeRobotDatasetMetadata(dataset_id)
    standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=device)
    seqs, replay_meta = [], []
    with h5py.File(hdf5_path, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
        if num_demos is not None:
            eps = eps[:num_demos]
        for ep in eps:
            grp = f[f"data/{ep}"]
            obs_arrays = {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS}
            state_raw = assemble_state18(obs_arrays)  # (T,18) np
            state_n = standardizer.standardize(
                torch.as_tensor(state_raw, dtype=torch.float32)).cpu().numpy()
            seqs.append(state_n)
            if state_mode == "eef_piece":
                replay_meta.append((grp["states"][()], grp.attrs["model_file"],
                                    grp.attrs.get("ep_meta")))
    if state_mode == "eef":
        return seqs, standardizer, None

    # eef_piece: replay set_state 算 rel_piece(每 demo (T,12)),全量算 stats 后标准化拼进 state
    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
        make_replay_env, replay_eef_rel_piece)
    from resfit.rl_finetuning.chunk_residual.object_state import rel_piece_stats
    env, _ = make_replay_env(hdf5_path)
    rel_raws = []
    try:
        for states, model_file, ep_meta in replay_meta:
            rel_raws.append(replay_eef_rel_piece(
                env, states, model_file=model_file, ep_meta=ep_meta))
    finally:
        env.close()
    mean, std = rel_piece_stats(np.concatenate(rel_raws, axis=0))
    seqs30 = []
    for s18, rp in zip(seqs, rel_raws):
        t = min(len(s18), len(rp))
        rp_std = ((rp[:t] - mean) / std).astype(np.float32)
        seqs30.append(np.concatenate([s18[:t], rp_std], axis=1))
    return seqs30, standardizer, (mean, std)


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 HIQL action-free value(③a)")
    p.add_argument("--hdf5", required=True, help="源 hdf5(含 data/demo_i/obs/<key>)")
    p.add_argument("--dataset", required=True, help="LeRobot dataset id(取 state norm stats)")
    p.add_argument("--output", default="value.pt")
    p.add_argument("--state_mode", choices=["eef", "eef_piece"], default="eef",
                   help="eef(18,默认)|eef_piece(30,加双臂 eef-rel-piece object-aware;要 replay 全 demo)")
    p.add_argument("--num_demos", type=int, default=None, help="只用前 N 条 demo(冒烟用;默认全部)")
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
    seqs, standardizer, rel_stats = read_per_demo_states(
        args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos)
    s, s_next, done = build_transitions(seqs)
    print(f"[hiql_value] state_mode={args.state_mode} demos={len(seqs)} "
          f"transitions={s.shape[0]} state_dim={s.shape[1]}")
    model, v_stats = train_value(
        s, s_next, done, gamma=args.gamma, expectile=args.expectile, ema=args.ema,
        lr=args.lr, batch_size=args.batch_size, steps=args.steps,
        hidden=args.value_hidden, seed=args.seed)
    save_value(args.output, model, v_stats=v_stats,
               mean=standardizer._mean.cpu(), std=standardizer._std.cpu(),
               dataset_id=args.dataset, state_mode=args.state_mode, rel_piece_stats=rel_stats)
    print(f"[hiql_value] saved {args.output}; state_mode={args.state_mode} v_stats={v_stats}")


if __name__ == "__main__":
    main()
