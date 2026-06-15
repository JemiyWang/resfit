"""遍历源 hdf5,经 serve(openpi_client)取 pi05 prefix 特征 ⊕ proprio → 标准化 → pi0_feat 缓存。

residual 环境跑(只 openpi_client,不 import openpi);发原始 obs,预处理全在 serve 端(同源命门)。
用法(dexmg_hdf5):先在 openpi 环境起 pi0_serve/serve_with_feat.py,再:
  conda run -n residual python -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host H --port P --hdf5 X --dataset DS --image_keys agentview_image,... --proprio_key state18 \
    --prompt "..." --pooling last --serve_ckpt_id pi05_base --out_cache OUT [--num_demos N]

用法(libero):
  conda run -n residual python -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host H --port P --data_source libero \
    --lerobot_root /data/lerobot --suite libero_object --task_id 0 \
    --pooling last --serve_ckpt_id pi0_libero --out_cache OUT [--num_demos N]
"""
from __future__ import annotations

import argparse

import numpy as np

from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import sorted_demo_keys
from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import save_pi0_feat_cache


def pi0_feat_signature(dataset_id, num_demos, *, image_keys, proprio_key, pooling, prompt,
                       serve_ckpt_id, serve_metadata):
    """缓存/同源校验签名(可 JSON 序列化、稳定排序)。"""
    return {
        "dataset_id": str(dataset_id), "num_demos": num_demos,
        "image_keys": list(image_keys), "proprio_key": str(proprio_key),
        "pooling": str(pooling), "prompt": str(prompt),
        "serve_ckpt_id": str(serve_ckpt_id), "serve_metadata": serve_metadata or {},
    }


def assemble_pi0_feat_seqs(raw_feats, proprio_seqs):
    """list[(T,D_emb)] prefix 特征 + list[(T,D_p)] proprio → (seqs_std, mean, std)。

    逐 demo 尾拼 proprio → 全量算一组 (mean,std) → 标准化各 demo。
    """
    cat = [np.concatenate([np.asarray(f, np.float32), np.asarray(p, np.float32)], axis=1)
           for f, p in zip(raw_feats, proprio_seqs)]
    allcat = np.concatenate(cat, axis=0)
    mean = allcat.mean(0).astype(np.float32)
    std = np.maximum(allcat.std(0), 1e-6).astype(np.float32)
    seqs = [((c - mean) / std).astype(np.float32) for c in cat]
    return seqs, mean, std


def _server_metadata(client):
    """尽力取 serve 握手 metadata(辅助签名);取不到则空 dict。"""
    for attr in ("get_server_metadata", "metadata"):
        m = getattr(client, attr, None)
        if callable(m):
            return dict(m() or {})
        if isinstance(m, dict):
            return dict(m)
    return {}


def build_main(client, *, hdf5, dataset_id, image_keys, proprio_key, prompt, pooling,
               serve_ckpt_id, out_cache, num_demos=None):
    """核心流程(client 注入,便于 stub 测试)。

    注意:pooling 仅作**声明值**写入签名——真正的池化由 serve 端(serve_with_feat.py 起服务时的
    --pooling)决定,本端每帧只取 client.infer(obs)["prefix_feat"](serve 已池化好),不做聚合。
    故 pooling 必须与起 serve 时的 --pooling 一致(与 serve_ckpt_id 同理,人工保证同源);它写入
    签名是为了让缓存对 pooling 变更失效(换 pooling 重起 serve 后须重 build)。

    ⚠️ obs 格式:本函数发给 serve 的是扁平 obs {image_key: ndarray, "prompt": str}。
    serve(serve_with_feat.py)所用 --config 的 input_transform 必须接受这个扁平格式;若该
    config 期望嵌套 {"state":..., "images":{...}} 或需要 "state" key,则真 serve 会
    KeyError——届时需按所用 serve config 调整本函数的 obs 构造(image_keys/加 state)。
    单测走 stub client 不覆盖此,首次真 serve smoke 前务必核对。
    """
    import h5py
    raw_feats, proprios = [], []
    with h5py.File(hdf5, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
        if num_demos is not None:
            eps = eps[:num_demos]
        for ep in eps:
            grp = f[f"data/{ep}"]
            imgs = {k: grp[f"obs/{k}"][()] for k in image_keys}
            proprio = np.asarray(grp[f"obs/{proprio_key}"][()], np.float32)
            T = proprio.shape[0]
            assert T > 0, f"demo {ep} has 0 frames"
            feats = []
            for t in range(T):
                obs = {k: imgs[k][t] for k in image_keys}
                obs["prompt"] = prompt
                feats.append(np.asarray(client.infer(obs)["prefix_feat"], np.float32))
            raw_feats.append(np.stack(feats, axis=0))
            proprios.append(proprio)
    seqs, mean, std = assemble_pi0_feat_seqs(raw_feats, proprios)
    sig = pi0_feat_signature(dataset_id, num_demos, image_keys=image_keys, proprio_key=proprio_key,
                             pooling=pooling, prompt=prompt, serve_ckpt_id=serve_ckpt_id,
                             serve_metadata=_server_metadata(client))
    save_pi0_feat_cache(out_cache, seqs, (mean, std), signature=sig)
    print(f"[build_pi0_feat] wrote {out_cache}: {len(seqs)} demos, dim={seqs[0].shape[1]}")


def build_main_libero(client, *, lerobot_root, suite, task_id, pooling, serve_ckpt_id,
                      out_cache, num_demos=None):
    """LIBERO 数据源:从 LeRobot 读 demo,逐帧经 serve 取 prefix_feat ⊕ state(8) → 缓存。

    obs 用 libero_obs.build_libero_serve_obs(与残差 rollout 同款 client 侧预处理→同源)。
    """
    from resfit.rl_finetuning.chunk_residual.libero_offline import (
        libero_task_language, find_demo_episodes, read_libero_demo)
    from resfit.rl_finetuning.chunk_residual.libero_obs import build_libero_serve_obs
    BASE = "observation.images.agentview"
    WRIST = "observation.images.robot0_eye_in_hand"
    STATE = "observation.state"
    lang = libero_task_language(suite, task_id)
    eps = find_demo_episodes(lerobot_root, lang)
    if num_demos is not None:
        eps = eps[:num_demos]
    raw_feats, proprios = [], []
    for pq in eps:
        demo = read_libero_demo(pq)
        T = demo["state"].shape[0]
        assert T > 0, f"demo {pq} has 0 frames"
        feats = []
        for t in range(T):
            raw = {BASE: demo["agentview"][t], WRIST: demo["wrist"][t], STATE: demo["state"][t]}
            obs = build_libero_serve_obs(raw, base_key=BASE, wrist_key=WRIST, state_key=STATE,
                                        prompt=lang)
            feats.append(np.asarray(client.infer(obs)["prefix_feat"], np.float32))
        raw_feats.append(np.stack(feats, axis=0))
        proprios.append(np.asarray(demo["state"], np.float32))
    seqs, mean, std = assemble_pi0_feat_seqs(raw_feats, proprios)
    sig = pi0_feat_signature(f"libero_{suite}_{task_id}", num_demos,
                             image_keys=[BASE, WRIST], proprio_key=STATE, pooling=pooling,
                             prompt=lang, serve_ckpt_id=serve_ckpt_id,
                             serve_metadata=_server_metadata(client))
    save_pi0_feat_cache(out_cache, seqs, (mean, std), signature=sig)
    print(f"[build_pi0_feat libero] wrote {out_cache}: {len(seqs)} demos, dim={seqs[0].shape[1]}")


def _connect(host, port):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import websockets.sync.client as _wsc
    if not getattr(_wsc.connect, "_no_ping_patched", False):
        _orig = _wsc.connect

        def _no_ping(*a, **k):
            k.setdefault("ping_interval", None)
            return _orig(*a, **k)

        _no_ping._no_ping_patched = True
        _wsc.connect = _no_ping
    return WebsocketClientPolicy(host=host, port=port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--data_source", choices=["dexmg_hdf5", "libero"], default="dexmg_hdf5",
                    help="数据来源:dexmg_hdf5(旧行为)或 libero(LeRobot demo)")
    # dexmg_hdf5 专用参数(data_source=dexmg_hdf5 时必须,否则可省)
    ap.add_argument("--hdf5", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--image_keys", type=lambda s: s.split(","), default=None)
    ap.add_argument("--proprio_key", default=None)
    ap.add_argument("--prompt", default="")
    # libero 专用参数
    ap.add_argument("--lerobot_root", default=None, help="LeRobot 数据集根目录")
    ap.add_argument("--suite", default=None, help="LIBERO suite 名,如 libero_object")
    ap.add_argument("--task_id", type=int, default=None, help="LIBERO 任务 id")
    # 公共参数
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--serve_ckpt_id", required=True)
    ap.add_argument("--out_cache", required=True)
    ap.add_argument("--num_demos", type=int, default=None)
    args = ap.parse_args()

    if args.data_source == "dexmg_hdf5":
        missing = [f for f, v in [("--hdf5", args.hdf5), ("--dataset", args.dataset),
                                   ("--image_keys", args.image_keys),
                                   ("--proprio_key", args.proprio_key)] if v is None]
        if missing:
            ap.error(f"data_source=dexmg_hdf5 时以下参数必须提供: {', '.join(missing)}")
    else:  # libero
        missing = [f for f, v in [("--lerobot_root", args.lerobot_root),
                                   ("--suite", args.suite),
                                   ("--task_id", args.task_id)] if v is None]
        if missing:
            ap.error(f"data_source=libero 时以下参数必须提供: {', '.join(missing)}")

    client = _connect(args.host, args.port)
    if args.data_source == "libero":
        build_main_libero(client, lerobot_root=args.lerobot_root, suite=args.suite,
                          task_id=args.task_id, pooling=args.pooling,
                          serve_ckpt_id=args.serve_ckpt_id, out_cache=args.out_cache,
                          num_demos=args.num_demos)
    else:
        build_main(client, hdf5=args.hdf5, dataset_id=args.dataset, image_keys=args.image_keys,
                   proprio_key=args.proprio_key, prompt=args.prompt, pooling=args.pooling,
                   serve_ckpt_id=args.serve_ckpt_id, out_cache=args.out_cache,
                   num_demos=args.num_demos)


if __name__ == "__main__":
    main()
