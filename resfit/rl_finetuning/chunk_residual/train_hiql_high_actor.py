"""离线训练 HIQL 高层 π^h(分层路 Phase 2)。从仓库根跑:

    conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
      --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly \
      --stage_cache outputs_chunk/three_piece_stages.npz \
      --state30_cache outputs_chunk/three_piece_state30.npz \
      --gc_value_ckpt outputs_chunk/three_piece_gc_value.pt \
      --way_steps 25 --output outputs_chunk/three_piece_high_actor.pt

state30_cache 命中则秒读回放后的 30 维 state(否则首次 MuJoCo 回放全 demo 并落盘)。
设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import argparse

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, load_gc_value
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor, save_high_actor
from resfit.rl_finetuning.chunk_residual.state30_cache import load_or_build_state30
from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import stage_entries_aligned


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 HIQL 高层 π^h(Phase 2)")
    p.add_argument("--hdf5", required=True, help="源 hdf5(含 data/demo_i/obs/<key>)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage_cache", required=True)
    p.add_argument("--state30_cache", default=None,
                   help="回放后 30 维 state 缓存 npz;命中秒读,否则回放并落盘(强烈建议设)")
    p.add_argument("--gc_value_ckpt", required=True, help="Phase 1 产的冻结 gc_value.pt")
    p.add_argument("--output", default="high_actor.pt")
    p.add_argument("--num_demos", type=int, default=None)
    p.add_argument("--way_steps", type=int, default=25)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target_mode", choices=["fixed_waypoint", "clamp_to_goal"],
                   default="clamp_to_goal",
                   help="高层 AWR 航点:clamp_to_goal(默认,HIQL,近 goal 塌到 goal)| fixed_waypoint(旧口径,恒 +way)")
    p.add_argument("--high_p_randomgoal", type=float, default=0.3,
                   help="clamp_to_goal 下高层 goal 取 random 的概率(HIQL 默认 0.3;fixed_waypoint 下须为 0)")
    p.add_argument("--adv_agg", choices=["min", "mean"], default="mean",
                   help="高层优势双 critic 聚合:min=min(vw)-min(vs)旧行为;mean=均值(对齐HIQL,默认)")
    return p


def main():
    args = build_parser().parse_args()
    if args.state30_cache is None:
        import warnings
        warnings.warn("--state30_cache 未设置,将触发完整 MuJoCo 回放(可能耗时数小时);建议指向 state30 缓存 npz", stacklevel=2)
    seqs = load_or_build_state30(args.hdf5, args.dataset, args.num_demos, args.state30_cache)
    seq_lens = [len(s) for s in seqs]
    stage_entries = stage_entries_aligned(args.hdf5, args.stage_cache, args.num_demos, seq_lens)
    assert len(seqs) == len(stage_entries), \
        f"seqs/stage_entries 长度不一致: {len(seqs)} vs {len(stage_entries)}"
    data = build_gc_data(seqs, stage_entries)
    vf, info = load_gc_value(args.gc_value_ckpt)
    assert info["dataset_id"] == args.dataset, \
        f"gc_value 训练集 {info['dataset_id']!r} 与当前 --dataset {args.dataset!r} 不一致(异源 ckpt)"
    assert data["states"].shape[1] == vf.state_dim, \
        f"state_dim {data['states'].shape[1]} != gc_value {vf.state_dim}(gc_value state_mode={info['state_mode']},须同源)"
    print(f"[hiql_high] demos={len(seqs)} transitions={len(data['s_idx'])} "
          f"state_dim={vf.state_dim} rep_dim={vf.rep_dim} way_steps={args.way_steps} "
          f"target_mode={args.target_mode} high_p_randomgoal={args.high_p_randomgoal} "
          f"adv_agg={args.adv_agg}")
    ha = train_high_actor(data, vf, way_steps=args.way_steps, beta=args.beta, lr=args.lr,
                          batch_size=args.batch_size, steps=args.steps, hidden=args.hidden,
                          seed=args.seed, target_mode=args.target_mode,
                          high_p_randomgoal=args.high_p_randomgoal,
                          adv_agg=args.adv_agg)
    save_high_actor(args.output, ha, gc_value_ckpt=args.gc_value_ckpt,
                    way_steps=args.way_steps, beta=args.beta,
                    target_mode=args.target_mode, high_p_randomgoal=args.high_p_randomgoal,
                    adv_agg=args.adv_agg)
    print(f"[hiql_high] saved {args.output}")


if __name__ == "__main__":
    main()
