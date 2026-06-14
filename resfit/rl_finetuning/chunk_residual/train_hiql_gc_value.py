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
from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    read_per_demo_states, validate_act_feat_cfg, setup_act_feat, add_act_feat_args,
    unpack_state_aux, validate_data_source_cfg,
    add_pi0_feat_args, validate_pi0_feat_cfg,
)


def validate_stage_cache(stage_cache, *, needs_stage):
    """按口径校验 --stage_cache 是否必需。

    需要 stage 的口径(gc_value 的 --goal_future_mode stage_entry / high_actor 的
    --target_mode fixed_waypoint)必须提供 stage_cache;默认 geometric/clamp_to_goal 口径
    不读 stage,可省略。返回 stage_cache 原样(None 表示无 stage)。"""
    if needs_stage and stage_cache is None:
        raise ValueError(
            "该口径需要 --stage_cache(由 precompute_stage_cache 生成);"
            "若用默认 geometric(gc_value)/clamp_to_goal(high_actor)口径则可省略")
    return stage_cache


def stage_entries_aligned(hdf5_path, stage_cache, num_demos, seq_lens):
    """按 read_per_demo_states 的同款 demo 顺序读逐帧 stage,算 within-demo 入口下标,
    并裁剪到对应 seq 长度(eef_piece 路会把 seq 截到 min(len(s18),len(rel)))。

    stage_cache=None(默认 geometric 口径,不读 stage)时不起 hdf5/缓存,直接按 demo 数
    返回空入口数组——build_gc_data 会让 stage_entries_of 回退到末态;几何采样根本不读它,
    故与喂一份全 0 dummy cache 逐位等价。"""
    if stage_cache is None:
        return [np.empty(0, dtype=np.int64) for _ in seq_lens]
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
    p.add_argument("--stage_cache", default=None,
                   help="逐帧 stage 缓存 npz(precompute_stage_cache 产)。仅 --goal_future_mode "
                        "stage_entry 需要;默认 geometric 不读 stage,可省略")
    p.add_argument("--output", default="gc_value.pt")
    p.add_argument("--num_demos", type=int, default=None, help="只用前 N 条 demo(冒烟用;默认全部)")
    p.add_argument("--state30_cache", default=None,
                   help="state30 v2 缓存路径(eef_piece+全量时命中跳过 replay;不传=每次 replay)")
    p.add_argument("--state_mode", choices=["eef_piece", "act_feat", "pi0_feat"], default="eef_piece",
                   help="eef_piece(默认,sim 特权)|act_feat(冻结 ACT encoder ⊕ 本体)|pi0_feat(冻结 pi0 prefix 池化特征)")
    add_act_feat_args(p)
    add_pi0_feat_args(p)
    p.add_argument("--data_source", choices=["hdf5", "lerobot"], default="hdf5",
                   help="act_feat 数据源:hdf5(默认)|lerobot(no-stage)")
    p.add_argument("--lerobot_root", default=None, help="--data_source lerobot 本地数据根目录")
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


def gc_value_needs_stage(args):
    """gc_value 仅 --goal_future_mode stage_entry 口径读 stage 入口(见 sample_gc_goals
    的 stage_entry 分支);默认 geometric 不读。main 与守卫共用此谓词,避免接线散落。"""
    return args.goal_future_mode == "stage_entry"


def main():
    args = build_parser().parse_args()
    # 配置守卫先行(在重活 read_per_demo_states 之前 fail-fast)
    validate_stage_cache(args.stage_cache, needs_stage=gc_value_needs_stage(args))
    validate_act_feat_cfg(args)
    validate_pi0_feat_cfg(args)
    validate_data_source_cfg(args)

    # --- pi0_feat dispatch:读缓存签名做自洽断言,然后走 read_per_demo_states ---
    if args.state_mode == "pi0_feat":
        from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
        _, _, _cache_sig = load_pi0_feat_cache(args.pi0_feat_cache)   # 缓存内签名(含 build 的 num_demos)
        # 防张冠李戴:缓存签名核心字段须与 CLI 传入一致(serve_metadata 仅诊断、不断言)
        for k, v in (("serve_ckpt_id", args.pi0_serve_ckpt_id), ("image_keys", args.pi0_image_keys),
                     ("proprio_key", args.pi0_proprio_key), ("pooling", args.pi0_pooling),
                     ("prompt", args.pi0_prompt)):
            if _cache_sig.get(k) != v:
                raise ValueError(
                    f"缓存签名 {k}={_cache_sig.get(k)!r} 与 CLI {v!r} 不符(指向了错误的缓存?)")
        seqs, _standardizer, aux_stats = read_per_demo_states(
            args.hdf5, args.dataset, "pi0_feat", num_demos=args.num_demos,
            pi0_feat_cache=args.pi0_feat_cache, pi0_feat_signature=_cache_sig)   # 用缓存签名→自洽命中
        import torch
        mean, std = torch.as_tensor(aux_stats[0]), torch.as_tensor(aux_stats[1])
        rel_stats = None
        pi0_sig = _cache_sig
        act_sig = None
    else:
        # --- 现有 eef_piece/act_feat dispatch 原样不动 ---
        extractor, act_ckpt_id, image_keys, act_sig = setup_act_feat(args)
        seqs, standardizer, aux = read_per_demo_states(
            args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos,
            cache_path=args.state30_cache, act_feat_cache=args.act_feat_cache,
            act_extractor=extractor, act_image_keys=image_keys, act_ckpt_id=act_ckpt_id,
            act_proprio_key=args.act_proprio_key, pooling=args.pooling,
            data_source=args.data_source, lerobot_root=args.lerobot_root)
        mean, std, rel_stats = unpack_state_aux(standardizer, aux, args.state_mode)
        pi0_sig = None

    seq_lens = [len(s) for s in seqs]
    # pi0_feat 时 seqs 数量由缓存签名的 num_demos 决定(而非 CLI args.num_demos);
    # 用同款 effective_num_demos 传给 stage_entries_aligned 保持长度一致。
    effective_num_demos = (_cache_sig.get("num_demos") if args.state_mode == "pi0_feat"
                          else args.num_demos)
    stage_entries = stage_entries_aligned(args.hdf5, args.stage_cache, effective_num_demos, seq_lens)
    assert len(seqs) == len(stage_entries), \
        f"seqs/stage_entries 长度不一致: {len(seqs)} vs {len(stage_entries)}"
    data = build_gc_data(seqs, stage_entries)
    print(f"[hiql_gc] state_mode={args.state_mode} demos={len(seqs)} transitions={len(data['s_idx'])} "
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
                  mean=mean, std=std,
                  dataset_id=args.dataset, state_mode=args.state_mode, rel_piece_stats=rel_stats,
                  value_loss_mode=args.value_loss_mode, value_mask_mode=args.value_mask_mode,
                  act_feat_signature=act_sig, pi0_feat_signature=pi0_sig)
    print(f"[hiql_gc] saved {args.output}; v_stats={v_stats}")


if __name__ == "__main__":
    main()
