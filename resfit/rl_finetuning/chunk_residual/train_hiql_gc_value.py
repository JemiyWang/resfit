"""离线训练 goal-conditioned HIQL value(分层路 Phase 1)。从仓库根跑:

    conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
      --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly \
      --stage_cache outputs_chunk/three_piece_stages.npz \
      --output outputs_chunk/three_piece_gc_value.pt

设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import argparse

import h5py
import numpy as np

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
    build_gc_data, train_gc_value, save_gc_value, stage_entries_from_instant,
)
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    load_stage_cache, sorted_demo_keys,
)
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states


def stage_entries_aligned(hdf5_path, stage_cache, num_demos, seq_lens):
    """按 read_per_demo_states 的同款 demo 顺序读逐帧 stage,算 within-demo 入口下标,
    并裁剪到对应 seq 长度(eef_piece 路会把 seq 截到 min(len(s18),len(rel)))。"""
    stages_by_demo = load_stage_cache(stage_cache)
    with h5py.File(hdf5_path, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
    if num_demos is not None:
        eps = eps[:num_demos]
    out = []
    for ep, T in zip(eps, seq_lens):
        if ep not in stages_by_demo:
            raise KeyError(f"demo '{ep}' 不在 stage 缓存 '{stage_cache}' 里(缓存过期?重跑 precompute_stage_cache)")
        inst = np.asarray(stages_by_demo[ep])[:T]
        ent = stage_entries_from_instant(inst)
        out.append(ent)
    return out


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 goal-conditioned HIQL value(Phase 1)")
    p.add_argument("--hdf5", required=True, help="源 hdf5(含 data/demo_i/obs/<key>)")
    p.add_argument("--dataset", required=True, help="LeRobot dataset id(取 state norm stats)")
    p.add_argument("--stage_cache", required=True, help="逐帧 stage 缓存 npz(precompute_stage_cache 产)")
    p.add_argument("--output", default="gc_value.pt")
    p.add_argument("--num_demos", type=int, default=None, help="只用前 N 条 demo(冒烟用;默认全部)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--ema", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--rep_dim", type=int, default=10)
    p.add_argument("--value_hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal_future_mode", choices=["stage_entry", "geometric"], default="geometric",
                   help="未来目标采样:geometric(默认,HIQL 口径,几何分布取任意未来帧,覆盖中间态、填洞)| "
                        "stage_entry(旧口径,只锚 stage 入口)")
    p.add_argument("--use_layer_norm", type=int, choices=[0, 1], default=1,
                   help="value/rep 用 LN+GELU(1,默认,对齐 HIQL LayerNormMLP)| 裸 ReLU(0,旧口径)")
    p.add_argument("--value_loss_mode", choices=["shared_min", "hiql"], default="hiql",
                   help="value 损失:hiql(默认,对齐参考,per-critic 目标不取 min + adv 门控两参 expectile)| "
                        "shared_min(旧口径,两 critic 钉 min 共享目标 + 残差 expectile)")
    p.add_argument("--value_mask_mode", choices=["done_aware", "hiql"], default="hiql",
                   help="value TD mask:done_aware=(1-success)(1-done)旧行为;hiql=1-success(对齐HIQL,默认)")
    p.add_argument("--value_rep_mode", choices=["concat", "goal_only"], default="goal_only",
                   help="goal 编码器输入:concat=[g,s]旧行为;goal_only=只吃 g(对齐HIQL,默认)")
    return p


def main():
    args = build_parser().parse_args()
    # state_mode 固定 eef_piece(分层路必须含物体 pose;③a' 假设就绪)
    seqs, standardizer, rel_stats = read_per_demo_states(
        args.hdf5, args.dataset, "eef_piece", num_demos=args.num_demos)
    seq_lens = [len(s) for s in seqs]
    stage_entries = stage_entries_aligned(args.hdf5, args.stage_cache, args.num_demos, seq_lens)
    assert len(seqs) == len(stage_entries), \
        f"seqs/stage_entries 长度不一致: {len(seqs)} vs {len(stage_entries)}"
    data = build_gc_data(seqs, stage_entries)
    print(f"[hiql_gc] demos={len(seqs)} transitions={len(data['s_idx'])} "
          f"state_dim={data['states'].shape[1]} rep_dim={args.rep_dim} "
          f"goal_future_mode={args.goal_future_mode} use_layer_norm={bool(args.use_layer_norm)} "
          f"value_loss_mode={args.value_loss_mode} value_mask_mode={args.value_mask_mode} "
          f"value_rep_mode={args.value_rep_mode}")
    model, v_stats = train_gc_value(
        data, gamma=args.gamma, expectile=args.expectile, ema=args.ema, lr=args.lr,
        batch_size=args.batch_size, steps=args.steps, rep_dim=args.rep_dim,
        hidden=args.value_hidden, seed=args.seed, future_mode=args.goal_future_mode,
        use_layer_norm=bool(args.use_layer_norm), value_loss_mode=args.value_loss_mode,
        value_mask_mode=args.value_mask_mode, value_rep_mode=args.value_rep_mode)
    save_gc_value(args.output, model, v_stats=v_stats,
                  mean=standardizer._mean.cpu(), std=standardizer._std.cpu(),
                  dataset_id=args.dataset, state_mode="eef_piece", rel_piece_stats=rel_stats,
                  value_loss_mode=args.value_loss_mode, value_mask_mode=args.value_mask_mode)
    print(f"[hiql_gc] saved {args.output}; v_stats={v_stats}")


if __name__ == "__main__":
    main()
