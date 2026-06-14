"""离线训练 HIQL action-free value(模块 ③a)。从仓库根跑:

    conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
      --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly --output outputs_chunk/three_piece_value.pt

设计见 docs/superpowers/specs/2026-06-07-hiql-value-design.md。
"""
import argparse
import os
import warnings

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


def read_per_demo_states(hdf5_path, dataset_id, state_mode="eef", num_demos=None,
                         device="cpu", cache_path=None,
                         act_feat_cache=None, act_extractor=None,
                         act_image_keys=None, act_ckpt_id=None, act_proprio_key="observation.state",
                         pooling="mean", state_standardizer=None,
                         data_source="hdf5", lerobot_root=None, _lerobot_ds=None,
                         _raw_obs_seqs=None,
                         pi0_feat_cache=None, pi0_feat_signature=None):
    """读每条 demo 的标准化 state 序列(与 RL 训练同源 mean/std)。

    state_mode=eef: (T,18) 纯 eef。eef_piece: (T,30)=[eef18 | 标准化 rel_piece12]。
    state_mode=act_feat: (T,D_emb+D_proprio) 冻结 ACT encoder 池化 ⊕ 本体(归一化)。
    state_mode=pi0_feat: (T,D_emb) 冻结 pi0 prefix 池化特征(cache-required,不连 serve)。
    cache_path(仅 eef_piece+num_demos=None 时):命中完整 v2 缓存则跳过 MuJoCo replay。
    act_feat_cache(仅 act_feat+num_demos=None 时):命中缓存则跳过 embed_batch。
    pi0_feat_cache(仅 pi0_feat):必须命中;缺失直接 raise(全程不连 serve/不 import openpi)。
    返回 (list[np.ndarray], standardizer_or_None, rel_piece_stats_or_emb_stats_or_None)。
    """
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.state30_cache import (
        state30_cache_reuse, save_state30_cache)

    # --- pi0_feat:cache-required,最优先短路 ---
    if state_mode == "pi0_feat":
        from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import pi0_feat_cache_reuse
        assert pi0_feat_signature is not None, "pi0_feat 需 pi0_feat_signature(build 时写的同款签名)"
        nd = pi0_feat_signature.get("num_demos", num_demos)  # num_demos 取自缓存签名(gc_value 传缓存内签名→自洽命中);不用调用侧 num_demos
        hit = pi0_feat_cache_reuse(pi0_feat_cache, signature=pi0_feat_signature, num_demos=nd)
        if hit is None:
            raise RuntimeError(
                f"pi0_feat 缓存未命中/缺失: {pi0_feat_cache}。请先在 residual 环境跑 "
                "build_pi0_feat_cache_via_serve.py(serve 须先起)生成缓存。")
        seqs, feat_stats = hit
        return seqs, None, feat_stats        # 缓存里 seqs 已标准化;standardizer 占位 None

    # --- act_feat:在加载 dataset 元信息/StateStandardizer 之前短路 ---
    if state_mode == "act_feat":
        from resfit.rl_finetuning.chunk_residual.act_feature import act_feat_signature
        from resfit.rl_finetuning.chunk_residual.act_feat_cache import (
            save_act_feat_cache, act_feat_cache_reuse)
        sig = act_feat_signature(act_ckpt_id, act_image_keys, act_proprio_key, pooling)
        sig = dict(sig, dataset_id=str(dataset_id), num_demos=num_demos)
        hit = act_feat_cache_reuse(act_feat_cache, signature=sig, num_demos=num_demos)
        if hit is not None:
            seqs_std, stats = hit
            print(f"[read_per_demo_states] act_feat 缓存命中 {act_feat_cache}")
            return seqs_std, None, (np.asarray(stats[0]), np.asarray(stats[1]))
        assert act_extractor is not None, "act_feat build 需 act_extractor(真 ACT 或 stub)"
        if _raw_obs_seqs is not None:
            raw_seqs = _raw_obs_seqs
        elif data_source == "lerobot":
            from resfit.rl_finetuning.chunk_residual.lerobot_demo_source import (
                open_lerobot, lerobot_episode_count, lerobot_episode_frames)
            ds = _lerobot_ds if _lerobot_ds is not None else open_lerobot(dataset_id, lerobot_root)
            n = lerobot_episode_count(ds)
            if num_demos is not None:
                n = min(n, num_demos)
            raw_seqs = []
            for ep in range(n):
                fr = lerobot_episode_frames(ds, ep, act_image_keys, act_proprio_key)
                raw_seqs.append({**fr["images"], act_proprio_key: fr["state"]})
        else:
            raw_seqs = _build_raw_obs_seqs(hdf5_path, act_image_keys, act_proprio_key, num_demos)
        # 命门 B:proprio 全栈 dataset-标准化(与在线 obs.state 同款)。
        proprio_std = state_standardizer
        if proprio_std is None and _raw_obs_seqs is None:   # 真 build 且未显式传 → 从 dataset stats 建
            proprio_std = StateStandardizer.from_dataset_stats(
                LeRobotDatasetMetadata(
                    dataset_id, root=lerobot_root if data_source == "lerobot" else None
                ).stats["observation.state"], device="cpu")
        if proprio_std is not None:
            for ro in raw_seqs:
                ro[act_proprio_key] = proprio_std.standardize(
                    torch.as_tensor(ro[act_proprio_key], dtype=torch.float32))
        raw_feat = [act_extractor.embed_batch(ro).cpu().numpy().astype(np.float32) for ro in raw_seqs]
        allf = np.concatenate(raw_feat, axis=0)
        mean = allf.mean(axis=0).astype(np.float32)
        std = np.maximum(allf.std(axis=0), 1e-6).astype(np.float32)
        seqs_std = [((s - mean) / std).astype(np.float32) for s in raw_feat]
        if act_feat_cache and num_demos is None:
            save_act_feat_cache(act_feat_cache, seqs_std, (mean, std), signature=sig)
            print(f"[read_per_demo_states] 已写 act_feat 缓存 {act_feat_cache}")
        return seqs_std, None, (mean, std)

    # --- 非 act_feat:以下为原有 eef / eef_piece 逻辑,原样保留 ---
    meta = LeRobotDatasetMetadata(dataset_id)
    standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=device)
    if state_mode == "eef_piece" and cache_path is not None:
        hit = state30_cache_reuse(cache_path, dataset_id=dataset_id, num_demos=num_demos)
        if hit is not None:
            seqs30, rel_stats = hit
            print(f"[read_per_demo_states] 缓存命中 {cache_path} → 跳过 replay({len(seqs30)} demo)")
            return seqs30, standardizer, rel_stats
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
    if cache_path is not None and num_demos is None:
        save_state30_cache(cache_path, seqs30, rel_stats=(mean, std), dataset_id=dataset_id)
        print(f"[read_per_demo_states] 已写 v2 缓存 {cache_path}")
    return seqs30, standardizer, (mean, std)


def _build_raw_obs_seqs(hdf5_path, image_keys, proprio_key, num_demos):
    """每条 demo -> 一个 raw_obs dict(整段 T 帧):ACT image_features 键 + proprio_key。"""
    out = []
    with h5py.File(hdf5_path, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
        if num_demos is not None:
            eps = eps[:num_demos]
        for ep in eps:
            grp = f[f"data/{ep}"]
            ro = {}
            for k in image_keys:
                name = k.replace("observation.images.", "")
                img = torch.as_tensor(grp[f"obs/{name}_image"][()])
                ro[k] = img.float().div(255.0).permute(0, 3, 1, 2)   # (T,H,W,C) uint8 -> (T,C,H,W) float [0,1]
            obs_arrays = {kk: grp[f"obs/{kk}"][()] for kk, _ in STATE18_KEYS}
            ro[proprio_key] = torch.as_tensor(assemble_state18(obs_arrays), dtype=torch.float32)
            out.append(ro)
    return out


def add_pi0_feat_args(p):
    p.add_argument("--pi0_feat_cache", default=None, help="pi0_feat 嵌入缓存 npz(仅 pi0_feat,须已存在)")
    p.add_argument("--pi0_serve_ckpt_id", default=None, help="serve 权重身份锚(仅 pi0_feat,同源签名)")
    p.add_argument("--pi0_image_keys", type=lambda s: s.split(","), default=None, help="逗号分隔图像键")
    p.add_argument("--pi0_proprio_key", default=None, help="本体 obs 键(仅 pi0_feat)")
    p.add_argument("--pi0_prompt", default="", help="pi05 prompt(仅 pi0_feat)")
    p.add_argument("--pi0_pooling", choices=["last", "mean"], default="last")


def validate_pi0_feat_cfg(args):
    """pi0_feat 须给已存在的 --pi0_feat_cache + serve_ckpt_id/image_keys/proprio_key;否则 ValueError。
    非 pi0_feat 传 --pi0_* 忽略。"""
    if getattr(args, "state_mode", None) != "pi0_feat":
        if getattr(args, "pi0_feat_cache", None):
            warnings.warn("非 pi0_feat 模式,--pi0_* 被忽略", stacklevel=2)
        return
    missing = [k for k in ("pi0_serve_ckpt_id", "pi0_image_keys", "pi0_proprio_key")
               if not getattr(args, k, None)]
    if missing:
        raise ValueError(f"--state_mode pi0_feat 需提供 {missing}")
    if not (args.pi0_feat_cache and os.path.exists(args.pi0_feat_cache)):
        raise ValueError("--state_mode pi0_feat 需 --pi0_feat_cache(已存在);请先跑 build_pi0_feat_cache_via_serve.py")


def validate_act_feat_cfg(args):
    """act_feat 须能命中缓存或能 build(给了 base ckpt);否则 ValueError。非 act_feat 传 act_* 忽略。"""
    if args.state_mode != "act_feat":
        if getattr(args, "act_feat_cache", None) or getattr(args, "act_base_ckpt", None):
            warnings.warn("非 act_feat 模式,--act_* 被忽略", stacklevel=2)
        return
    cache_ok = bool(args.act_feat_cache) and os.path.exists(args.act_feat_cache)
    if not cache_ok and not args.act_base_ckpt:
        raise ValueError("state_mode=act_feat 需 --act_feat_cache(已存在)或 --act_base_ckpt 以 build")


def validate_data_source_cfg(args):
    """--data_source lerobot 仅支持 act_feat 且需存在的本地 --lerobot_root;默认 hdf5 直接放行。"""
    import os
    if getattr(args, "data_source", "hdf5") != "lerobot":
        return
    assert getattr(args, "state_mode", None) == "act_feat", "--data_source lerobot 仅支持 --state_mode act_feat"
    assert args.lerobot_root and os.path.isdir(args.lerobot_root), \
        "--data_source lerobot 需 --lerobot_root(存在的本地数据目录)"


def setup_act_feat(args):
    """为 act_feat 准备 (extractor, act_ckpt_id, image_keys, signature_or_None)。
    非 act_feat → 全 None。缓存已存在 → 不建 extractor,从缓存签名取回 image_keys/ckpt(免重复传 flag);
    否则加载冻结 ACT 建 extractor。"""
    from pathlib import Path
    if getattr(args, "state_mode", None) != "act_feat":
        return None, None, None, None
    cache_ready = bool(args.act_feat_cache) and Path(args.act_feat_cache).exists()
    if cache_ready:
        from resfit.rl_finetuning.chunk_residual.act_feat_cache import load_act_feat_cache
        _, _, sig = load_act_feat_cache(args.act_feat_cache)
        ckpt = str(args.act_base_ckpt) if args.act_base_ckpt else sig.get("act_ckpt_id")
        image_keys = args.act_image_keys if args.act_image_keys is not None else sig.get("image_keys")
        return None, ckpt, image_keys, sig
    from resfit.lerobot.utils.load_policy import load_policy
    from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor
    cand = Path(args.act_base_ckpt) / "policy"
    act = load_policy(cand if cand.is_dir() else Path(args.act_base_ckpt))
    image_keys = args.act_image_keys or list(act.config.image_features.keys())
    ext = ActFeatureExtractor(act, image_keys, args.act_proprio_key, args.pooling)
    return ext, str(args.act_base_ckpt), image_keys, ext.signature(str(args.act_base_ckpt))


def unpack_state_aux(standardizer, aux, state_mode):
    """从 read_per_demo_states 的返回拆出 (mean, std, rel_stats)。
    act_feat: aux=(mean,std) → (tensor mean, tensor std, None);
    eef/eef_piece: 用 standardizer 的 mean/std,rel_stats=aux。"""
    if state_mode == "act_feat":
        return torch.as_tensor(aux[0]), torch.as_tensor(aux[1]), None
    return standardizer._mean.cpu(), standardizer._std.cpu(), aux


def assert_act_feat_pair_consistent(gc_info, act_sig, state_mode):
    """high_actor 加载 gc_value 时的同源校验:state_mode 必须一致;
    act_feat 时再比对特征签名的核心字段(act_ckpt_id/image_keys/proprio_key/pooling),
    容忍一侧多带 dataset_id/num_demos(build 路是 4 字段,cache-hit 路是 6 字段)。"""
    assert gc_info.get("state_mode") == state_mode, (
        f"high_actor state_mode={state_mode} 与 gc_value state_mode={gc_info.get('state_mode')} 不一致(异源)")
    if state_mode == "act_feat":
        core = ("act_ckpt_id", "image_keys", "proprio_key", "pooling")
        gv = gc_info.get("act_feat_signature") or {}
        ha = act_sig or {}
        bad = [k for k in core if gv.get(k) != ha.get(k)]
        assert not bad, (
            f"act_feat gc_value/high_actor 特征签名核心字段不一致 {bad}: gc_value={gv} vs high_actor={ha}")


def add_act_feat_args(p):
    """给 parser 加 act_feat 公共 flags(state_mode 各脚本自定义,不在此处)。"""
    p.add_argument("--act_feat_cache", default=None, help="act_feat 嵌入缓存 npz(cache-or-build)")
    p.add_argument("--act_base_ckpt", default=None, help="act_feat build 用的 ACT base 目录(同 run base)")
    p.add_argument("--act_image_keys", nargs="*", default=None, help="ACT image_features 键(默认取 base config)")
    p.add_argument("--act_proprio_key", default="observation.state")
    p.add_argument("--pooling", choices=["mean"], default="mean")


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 HIQL action-free value(③a)")
    p.add_argument("--hdf5", required=True, help="源 hdf5(含 data/demo_i/obs/<key>)")
    p.add_argument("--dataset", required=True, help="LeRobot dataset id(取 state norm stats)")
    p.add_argument("--output", default="value.pt")
    p.add_argument("--state_mode", choices=["eef", "eef_piece", "act_feat"], default="eef",
                   help="eef(18)|eef_piece(30,sim 特权)|act_feat(冻结 ACT encoder 池化 ⊕ 本体)。"
                        "pi0_feat 仅用于 train_hiql_gc_value,本脚本不支持。")
    p.add_argument("--num_demos", type=int, default=None, help="只用前 N 条 demo(冒烟用;默认全部)")
    p.add_argument("--state30_cache", default=None,
                   help="state30 v2 缓存路径(eef_piece+全量时命中跳过 replay;不传=每次 replay)")
    add_act_feat_args(p)
    p.add_argument("--data_source", choices=["hdf5", "lerobot"], default="hdf5",
                   help="act_feat 数据源:hdf5(默认)|lerobot(从 LeRobot 数据集读,no-stage)")
    p.add_argument("--lerobot_root", default=None, help="--data_source lerobot 的本地数据根目录")
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
    validate_act_feat_cfg(args)
    validate_data_source_cfg(args)
    extractor, act_ckpt_id, image_keys, _ = setup_act_feat(args)
    seqs, standardizer, aux = read_per_demo_states(
        args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos,
        cache_path=args.state30_cache, act_feat_cache=args.act_feat_cache,
        act_extractor=extractor, act_image_keys=image_keys, act_ckpt_id=act_ckpt_id,
        act_proprio_key=args.act_proprio_key, pooling=args.pooling,
        data_source=args.data_source, lerobot_root=args.lerobot_root)
    s, s_next, done = build_transitions(seqs)
    print(f"[hiql_value] state_mode={args.state_mode} demos={len(seqs)} "
          f"transitions={s.shape[0]} state_dim={s.shape[1]}")
    mean, std, rel_stats = unpack_state_aux(standardizer, aux, args.state_mode)
    model, v_stats = train_value(
        s, s_next, done, gamma=args.gamma, expectile=args.expectile, ema=args.ema,
        lr=args.lr, batch_size=args.batch_size, steps=args.steps,
        hidden=args.value_hidden, seed=args.seed)
    save_value(args.output, model, v_stats=v_stats, mean=mean, std=std,
               dataset_id=args.dataset, state_mode=args.state_mode, rel_piece_stats=rel_stats)
    print(f"[hiql_value] saved {args.output}; state_mode={args.state_mode} v_stats={v_stats}")


if __name__ == "__main__":
    main()
