"""遍历源 hdf5,经 serve(openpi_client)取 pi05 prefix 特征 ⊕ proprio → 标准化 → pi0_feat 缓存。

residual 环境跑(只 openpi_client,不 import openpi);发原始 obs,预处理全在 serve 端(同源命门)。
用法:先在 openpi 环境起 pi0_serve/serve_with_feat.py,再:
  conda run -n residual python -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host H --port P --hdf5 X --dataset DS --image_keys agentview_image,... --proprio_key state18 \
    --prompt "..." --pooling last --serve_ckpt_id pi05_base --out_cache OUT [--num_demos N]
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
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--image_keys", type=lambda s: s.split(","), required=True)
    ap.add_argument("--proprio_key", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--serve_ckpt_id", required=True)
    ap.add_argument("--out_cache", required=True)
    ap.add_argument("--num_demos", type=int, default=None)
    args = ap.parse_args()
    client = _connect(args.host, args.port)
    build_main(client, hdf5=args.hdf5, dataset_id=args.dataset, image_keys=args.image_keys,
               proprio_key=args.proprio_key, prompt=args.prompt, pooling=args.pooling,
               serve_ckpt_id=args.serve_ckpt_id, out_cache=args.out_cache, num_demos=args.num_demos)


if __name__ == "__main__":
    main()
