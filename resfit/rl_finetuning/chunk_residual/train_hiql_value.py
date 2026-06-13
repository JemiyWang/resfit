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
                         pooling="mean", _raw_obs_seqs=None):
    """读每条 demo 的标准化 state 序列(与 RL 训练同源 mean/std)。

    state_mode=eef: (T,18) 纯 eef。eef_piece: (T,30)=[eef18 | 标准化 rel_piece12]。
    state_mode=act_feat: (T,D_emb+D_proprio) 冻结 ACT encoder 池化 ⊕ 本体(归一化)。
    cache_path(仅 eef_piece+num_demos=None 时):命中完整 v2 缓存则跳过 MuJoCo replay。
    act_feat_cache(仅 act_feat+num_demos=None 时):命中缓存则跳过 embed_batch。
    返回 (list[np.ndarray], standardizer_or_None, rel_piece_stats_or_emb_stats_or_None)。
    """
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.state30_cache import (
        state30_cache_reuse, save_state30_cache)

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
        raw_seqs = _raw_obs_seqs if _raw_obs_seqs is not None else _build_raw_obs_seqs(
            hdf5_path, act_image_keys, act_proprio_key, num_demos)
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
                ro[k] = torch.as_tensor(grp[f"obs/{name}_image"][()])
            obs_arrays = {kk: grp[f"obs/{kk}"][()] for kk, _ in STATE18_KEYS}
            ro[proprio_key] = torch.as_tensor(assemble_state18(obs_arrays), dtype=torch.float32)
            out.append(ro)
    return out


def validate_act_feat_cfg(args):
    """act_feat 须能命中缓存或能 build(给了 base ckpt);否则 ValueError。非 act_feat 传 act_* 忽略。"""
    if args.state_mode != "act_feat":
        if getattr(args, "act_feat_cache", None) or getattr(args, "act_base_ckpt", None):
            warnings.warn("非 act_feat 模式,--act_* 被忽略", stacklevel=2)
        return
    cache_ok = bool(args.act_feat_cache) and os.path.exists(args.act_feat_cache)
    if not cache_ok and not args.act_base_ckpt:
        raise ValueError("state_mode=act_feat 需 --act_feat_cache(已存在)或 --act_base_ckpt 以 build")


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
                   help="eef(18)|eef_piece(30,sim 特权)|act_feat(冻结 ACT encoder 池化 ⊕ 本体)")
    p.add_argument("--num_demos", type=int, default=None, help="只用前 N 条 demo(冒烟用;默认全部)")
    p.add_argument("--state30_cache", default=None,
                   help="state30 v2 缓存路径(eef_piece+全量时命中跳过 replay;不传=每次 replay)")
    add_act_feat_args(p)
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
    extractor, act_ckpt_id, image_keys, _ = setup_act_feat(args)
    seqs, standardizer, aux = read_per_demo_states(
        args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos,
        cache_path=args.state30_cache, act_feat_cache=args.act_feat_cache,
        act_extractor=extractor, act_image_keys=image_keys, act_ckpt_id=act_ckpt_id,
        act_proprio_key=args.act_proprio_key, pooling=args.pooling)
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
