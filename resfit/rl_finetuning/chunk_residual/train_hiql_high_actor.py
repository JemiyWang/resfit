"""离线训练 HIQL 高层 π^h(分层路 Phase 2)。从仓库根跑:

    conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
      --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly \
      --stage_cache outputs_chunk/three_piece_stages.npz \
      --state30_cache outputs_chunk/three_piece_state30.npz \
      --gc_value_ckpt outputs_chunk/three_piece_gc_value.pt \
      --way_steps 25 --output outputs_chunk/three_piece_high_actor.pt

state30_cache 命中则秒读回放后的 30 维 state(否则首次 MuJoCo 回放全 demo 并落盘)。
支持 --state_mode act_feat(配 --act_feat_cache/--act_base_ckpt,走 read_per_demo_states 取冻结 ACT 特征)。
也支持 --state_mode pi0_feat(配 --pi0_feat_cache/--pi0_serve_ckpt_id 等,走缓存 pi0 prefix 特征当 state)。
设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import argparse
import warnings

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, load_gc_value
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor, save_high_actor
from resfit.rl_finetuning.chunk_residual.state30_cache import load_or_build_state30
from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import (
    stage_entries_aligned, validate_stage_cache,
)
from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    add_act_feat_args, add_pi0_feat_args, assert_act_feat_pair_consistent,
    validate_pi0_feat_cfg,
)


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 HIQL 高层 π^h(Phase 2)")
    p.add_argument("--hdf5", required=True, help="源 hdf5(含 data/demo_i/obs/<key>)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage_cache", default=None,
                   help="逐帧 stage 缓存 npz。仅 --target_mode fixed_waypoint 需要;"
                        "默认 clamp_to_goal 不读 stage,可省略")
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
    p.add_argument("--device", default="cpu", help="训练设备(cpu/cuda);默认 cpu 保持零回归,GPU 训练传 cuda")
    p.add_argument("--target_mode", choices=["fixed_waypoint", "clamp_to_goal"],
                   default="clamp_to_goal",
                   help="高层 AWR 航点:clamp_to_goal(默认,HIQL,近 goal 塌到 goal)| fixed_waypoint(旧口径,恒 +way)")
    p.add_argument("--high_p_randomgoal", type=float, default=0.3,
                   help="clamp_to_goal 下高层 goal 取 random 的概率(HIQL 默认 0.3;fixed_waypoint 下须为 0)")
    p.add_argument("--adv_agg", choices=["min", "mean"], default="mean",
                   help="高层优势双 critic 聚合:min=min(vw)-min(vs)旧行为;mean=均值(对齐HIQL,默认)")
    p.add_argument("--state_mode", choices=["eef_piece", "act_feat", "pi0_feat"], default="eef_piece",
                   help="state 来源:eef_piece(默认,30 维 sim 特权)|act_feat(冻结 ACT encoder 池化 ⊕ 本体)"
                        "|pi0_feat(冻结 pi0 prefix 特征,须已存在 --pi0_feat_cache)")
    add_act_feat_args(p)
    add_pi0_feat_args(p)
    p.add_argument("--data_source", choices=["hdf5", "lerobot"], default="hdf5",
                   help="act_feat 数据源:hdf5(默认)|lerobot(no-stage)")
    p.add_argument("--lerobot_root", default=None, help="--data_source lerobot 本地数据根目录")
    return p


def high_actor_needs_stage(args):
    """high_actor 仅 --target_mode fixed_waypoint 口径读 stage 入口(走 sample_gc_goals
    默认 future_mode=stage_entry);默认 clamp_to_goal 走 sample_high_goal_target 不读。
    main 与守卫共用此谓词,避免接线散落。"""
    return args.target_mode == "fixed_waypoint"


def main():
    args = build_parser().parse_args()
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
        validate_act_feat_cfg, setup_act_feat, read_per_demo_states,
        validate_data_source_cfg)
    validate_act_feat_cfg(args)
    validate_pi0_feat_cfg(args)
    validate_data_source_cfg(args)
    if args.state_mode not in ("act_feat", "pi0_feat") and args.state30_cache is None:
        warnings.warn("--state30_cache 未设置,将触发完整 MuJoCo 回放(可能耗时数小时);建议指向 state30 缓存 npz", stacklevel=2)
    validate_stage_cache(args.stage_cache, needs_stage=high_actor_needs_stage(args))
    if args.state_mode == "pi0_feat":
        from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
        _, _, _cache_sig = load_pi0_feat_cache(args.pi0_feat_cache)
        # 防张冠李戴:缓存签名核心字段须与 CLI 传入一致
        for k, v in (("serve_ckpt_id", args.pi0_serve_ckpt_id), ("image_keys", args.pi0_image_keys),
                     ("proprio_key", args.pi0_proprio_key), ("pooling", args.pi0_pooling),
                     ("prompt", args.pi0_prompt)):
            if _cache_sig.get(k) != v:
                raise ValueError(
                    f"缓存签名 {k}={_cache_sig.get(k)!r} 与 CLI {v!r} 不符(指向了错误的缓存?)")
        seqs, _, _ = read_per_demo_states(
            args.hdf5, args.dataset, "pi0_feat", num_demos=args.num_demos,
            pi0_feat_cache=args.pi0_feat_cache, pi0_feat_signature=_cache_sig)
        pi0_sig = _cache_sig
        act_sig = None
        effective_num_demos = _cache_sig.get("num_demos")
    elif args.state_mode == "act_feat":
        extractor, act_ckpt_id, image_keys, act_sig = setup_act_feat(args)
        seqs, _, _ = read_per_demo_states(
            args.hdf5, args.dataset, "act_feat", num_demos=args.num_demos,
            act_feat_cache=args.act_feat_cache, act_extractor=extractor,
            act_image_keys=image_keys, act_ckpt_id=act_ckpt_id,
            act_proprio_key=args.act_proprio_key, pooling=args.pooling,
            data_source=args.data_source, lerobot_root=args.lerobot_root)
        pi0_sig = None
        effective_num_demos = args.num_demos
    else:
        seqs = load_or_build_state30(args.hdf5, args.dataset, args.num_demos, args.state30_cache)
        act_sig = None
        pi0_sig = None
        effective_num_demos = args.num_demos
    seq_lens = [len(s) for s in seqs]
    # pi0_feat 时用缓存签名的 num_demos(与 seqs 长度自洽);其余用 CLI args.num_demos
    stage_entries = stage_entries_aligned(args.hdf5, args.stage_cache, effective_num_demos, seq_lens)
    assert len(seqs) == len(stage_entries), \
        f"seqs/stage_entries 长度不一致: {len(seqs)} vs {len(stage_entries)}"
    data = build_gc_data(seqs, stage_entries)
    vf, info = load_gc_value(args.gc_value_ckpt)
    assert info["dataset_id"] == args.dataset, \
        f"gc_value 训练集 {info['dataset_id']!r} 与当前 --dataset {args.dataset!r} 不一致(异源 ckpt)"
    assert data["states"].shape[1] == vf.state_dim, \
        f"state_dim {data['states'].shape[1]} != gc_value {vf.state_dim}(gc_value state_mode={info['state_mode']},须同源)"
    assert_act_feat_pair_consistent(info, act_sig, args.state_mode, pi0_sig=pi0_sig)
    print(f"[hiql_high] demos={len(seqs)} transitions={len(data['s_idx'])} "
          f"state_dim={vf.state_dim} rep_dim={vf.rep_dim} way_steps={args.way_steps} "
          f"target_mode={args.target_mode} high_p_randomgoal={args.high_p_randomgoal} "
          f"adv_agg={args.adv_agg}")
    ha = train_high_actor(data, vf, way_steps=args.way_steps, beta=args.beta, lr=args.lr,
                          batch_size=args.batch_size, steps=args.steps, hidden=args.hidden,
                          seed=args.seed, target_mode=args.target_mode,
                          high_p_randomgoal=args.high_p_randomgoal,
                          adv_agg=args.adv_agg, device=args.device)
    save_high_actor(args.output, ha, gc_value_ckpt=args.gc_value_ckpt,
                    way_steps=args.way_steps, beta=args.beta,
                    target_mode=args.target_mode, high_p_randomgoal=args.high_p_randomgoal,
                    adv_agg=args.adv_agg, act_feat_signature=act_sig,
                    pi0_feat_signature=pi0_sig,
                    state_mode=args.state_mode)
    print(f"[hiql_high] saved {args.output}")


if __name__ == "__main__":
    main()
