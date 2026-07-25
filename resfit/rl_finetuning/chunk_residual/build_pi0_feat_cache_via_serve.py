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
from resfit.rl_finetuning.wm_bridge.teleavatar_policy_state import (
    map_teleavatar_policy_state,
)


def pi0_feat_signature(dataset_id, num_demos, *, image_keys, proprio_key, pooling, prompt,
                       serve_ckpt_id, serve_metadata, policy_state_dim=None):
    """缓存/同源校验签名(可 JSON 序列化、稳定排序)。"""
    signature = {
        "dataset_id": str(dataset_id), "num_demos": num_demos,
        "image_keys": list(image_keys), "proprio_key": str(proprio_key),
        "pooling": str(pooling), "prompt": str(prompt),
        "serve_ckpt_id": str(serve_ckpt_id), "serve_metadata": serve_metadata or {},
    }
    if policy_state_dim is not None:
        signature["policy_state_dim"] = int(policy_state_dim)
    return signature


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


def build_main_libero(client, *, lerobot_root, language, pooling, serve_ckpt_id,
                      out_cache, num_demos=None, dataset_id=None):
    """LIBERO 数据源:从 LeRobot 读 demo,逐帧经 serve 取 prefix_feat ⊕ state(8) → 缓存。

    obs 用 libero_obs.build_libero_serve_obs(与残差 rollout 同款 client 侧预处理→同源)。
    language=任务语言串(直接传:数据集 episodes.jsonl 本身即任务语言;绕过 libero_task_language→
    不依赖 LIBERO 库,residual 环境无 libero 也能跑)。find_demo_episodes 按 language 匹配 episode。
    """
    from resfit.rl_finetuning.chunk_residual.libero_offline import (
        find_demo_episodes, read_libero_demo)
    from resfit.rl_finetuning.chunk_residual.libero_obs import build_libero_serve_obs
    BASE = "observation.images.agentview"
    WRIST = "observation.images.robot0_eye_in_hand"
    STATE = "observation.state"
    eps = find_demo_episodes(lerobot_root, language)
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
                                        prompt=language)
            feats.append(np.asarray(client.infer(obs)["prefix_feat"], np.float32))
        raw_feats.append(np.stack(feats, axis=0))
        proprios.append(np.asarray(demo["state"], np.float32))
    seqs, mean, std = assemble_pi0_feat_seqs(raw_feats, proprios)
    sig = pi0_feat_signature(dataset_id or f"libero:{language}", num_demos,
                             image_keys=[BASE, WRIST], proprio_key=STATE, pooling=pooling,
                             prompt=language, serve_ckpt_id=serve_ckpt_id,
                             serve_metadata=_server_metadata(client))
    save_pi0_feat_cache(out_cache, seqs, (mean, std), signature=sig)
    print(f"[build_pi0_feat libero] wrote {out_cache}: {len(seqs)} demos, dim={seqs[0].shape[1]}")


# LeRobot 键(数据集里) → serve 裸键(kai0 期望,见 piper_deploy.py 的 payload["images"])。
TELEAVATAR_CAM_MAP = {
    "observation.images.top_head": "top_head",
    "observation.images.hand_left": "hand_left",
    "observation.images.hand_right": "hand_right",
}

# monkeypatch 锚点:测试对本模块属性打桩;函数内引用这些名字须走模块级(见 build_main_teleavatar)。
list_teleavatar_episodes = None
read_teleavatar_episode_batched = None


def build_teleavatar_serve_obs(images, state, prompt, *, size=224):
    """Teleavatar 的 kai0 serve obs(嵌套 schema,对齐 piper_deploy.py)。

    images: {serve裸键: CHW-or-HWC 图};state: (Dp,) proprio。
    每图 resize_with_pad 到 size×size 并转 CHW(与 libero 路同款客户端预处理 → 与部署同源)。
    prompt 原样透传(serve 侧 tokenize)。

    ⚠️ 首次连真 serve 时须核对:①嵌套 vs 扁平 ②prompt 是文本还是预编码 embedding
       ③图是否要 uint8/CHW(piper_deploy 是 CHW uint8)。三者以真 serve config 为准,
       与 spec §13 的"接数据当天核对项"一致。
    """
    from resfit.rl_finetuning.chunk_residual.libero_obs import _to_hwc_uint8, resize_with_pad
    imgs = {}
    for cam, im in images.items():
        hwc = resize_with_pad(_to_hwc_uint8(im), size, size)   # HWC uint8
        imgs[cam] = np.transpose(hwc, (2, 0, 1))               # CHW uint8(对齐 piper_deploy)
    return {
        "state": np.asarray(state, np.float32).reshape(-1),
        "images": imgs,
        "prompt": prompt,
    }


def build_main_teleavatar(client, *, lerobot_root, repo_id, prompt, pooling,
                          serve_ckpt_id, out_cache, num_demos=None,
                          proprio_key="observation.state",
                          num_shards=1, shard_index=0,
                          policy_state_dim=16):
    """Teleavatar 数据源:从 LeRobot 读三相机 demo,逐帧经 serve 取 prefix_feat ⊕ state → 缓存。

    与 libero 路同构,差别仅在 obs schema(嵌套三相机)与 cam 键映射。
    """
    global list_teleavatar_episodes, read_teleavatar_episode_batched
    if list_teleavatar_episodes is None:    # 真实运行时懒加载;测试已 monkeypatch 则跳过
        from resfit.rl_finetuning.chunk_residual.teleavatar_batch_source import (
            list_teleavatar_episodes as _le, read_teleavatar_episode_batched as _re)
        list_teleavatar_episodes, read_teleavatar_episode_batched = _le, _re

    import os as _os
    lerobot_keys = list(TELEAVATAR_CAM_MAP.keys())
    # root 指向数据集目录本身(含 data/、videos/)。--lerobot_root 是父目录、--repo_id 是子目录名;
    # 若已传完整路径(join 后无 data/)则回退用 lerobot_root。
    dataset_root = _os.path.join(lerobot_root, repo_id)
    if not _os.path.isdir(_os.path.join(dataset_root, "data")):
        dataset_root = lerobot_root
    eps = list_teleavatar_episodes(dataset_root)
    # ★ 先分片再截 num_demos:反过来的话(先截前N再分片)会让高 index 分片拿到空集。
    #   分片(多卡并行):跨步取 eps[shard::num_shards],各分片均衡且不重叠。
    assert 0 <= shard_index < num_shards, f"shard_index {shard_index} 须在 [0,{num_shards})"
    if num_shards > 1:
        eps = eps[shard_index::num_shards]
    if num_demos is not None:            # 每分片各取前 num_demos(冒烟用)
        eps = eps[:num_demos]
    assert eps, f"没从 {dataset_root} 读到任何 episode(shard {shard_index}/{num_shards})"
    raw_feats, proprios = [], []
    # ★ 整段 torchcodec 批量解码(read_teleavatar_episode_batched),~17x 快于逐帧 ds[i]。
    for ep in eps:
        fr = read_teleavatar_episode_batched(dataset_root, ep, lerobot_keys, proprio_key)
        T = fr["state"].shape[0]
        assert T > 0, f"episode {ep} has 0 frames"
        feats = []
        for t in range(T):
            images = {TELEAVATAR_CAM_MAP[lk]: np.asarray(fr["images"][lk][t])
                      for lk in lerobot_keys}
            policy_state = map_teleavatar_policy_state(
                fr["state"][t],
                policy_state_dim,
            )
            obs = build_teleavatar_serve_obs(images, policy_state, prompt)
            feats.append(np.asarray(client.infer(obs)["prefix_feat"], np.float32))
        raw_feats.append(np.stack(feats, axis=0))
        proprios.append(np.asarray(fr["state"], np.float32))
    seqs, mean, std = assemble_pi0_feat_seqs(raw_feats, proprios)
    sig = pi0_feat_signature(repo_id, num_demos,
                             image_keys=list(TELEAVATAR_CAM_MAP.values()),
                             proprio_key=proprio_key, pooling=pooling, prompt=prompt,
                             serve_ckpt_id=serve_ckpt_id,
                             serve_metadata=_server_metadata(client),
                             policy_state_dim=policy_state_dim)
    save_pi0_feat_cache(out_cache, seqs, (mean, std), signature=sig)
    print(f"[build_pi0_feat teleavatar] wrote {out_cache}: {len(seqs)} demos, dim={seqs[0].shape[1]}")


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


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--data_source", choices=["dexmg_hdf5", "libero", "teleavatar"],
                    default="dexmg_hdf5",
                    help="数据来源:dexmg_hdf5(旧行为)|libero(LeRobot demo)|teleavatar(三相机 LeRobot demo)")
    # dexmg_hdf5 专用参数(data_source=dexmg_hdf5 时必须,否则可省)
    ap.add_argument("--hdf5", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--image_keys", type=lambda s: s.split(","), default=None)
    ap.add_argument("--proprio_key", default="observation.state")
    ap.add_argument("--prompt", default="")
    # libero / teleavatar 专用参数
    ap.add_argument("--lerobot_root", default=None, help="LeRobot 数据集根目录")
    ap.add_argument("--language", default=None,
                    help="LIBERO 任务语言串(直接传,数据集 episodes.jsonl 即任务语言;不依赖 LIBERO 库)")
    ap.add_argument("--repo_id", default=None,
                    help="teleavatar:LeRobot 数据集名/目录(如 task_success),与 --lerobot_root 组合")
    # 公共参数
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--serve_ckpt_id", required=True)
    ap.add_argument("--out_cache", required=True)
    ap.add_argument("--num_demos", type=int, default=None)
    # teleavatar 分片(多卡并行):各分片跨步取 eps[shard_index::num_shards]
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument(
        "--policy_state_dim",
        type=int,
        choices=(14, 16),
        default=16,
        help="TeleAvatar base-policy state dimension; cached proprio remains 16D",
    )
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.data_source == "dexmg_hdf5":
        missing = [f for f, v in [("--hdf5", args.hdf5), ("--dataset", args.dataset),
                                   ("--image_keys", args.image_keys),
                                   ("--proprio_key", args.proprio_key)] if v is None]
        if missing:
            ap.error(f"data_source=dexmg_hdf5 时以下参数必须提供: {', '.join(missing)}")
    elif args.data_source == "libero":
        missing = [f for f, v in [("--lerobot_root", args.lerobot_root),
                                   ("--language", args.language)] if v is None]
        if missing:
            ap.error(f"data_source=libero 时以下参数必须提供: {', '.join(missing)}")
    else:  # teleavatar
        missing = [f for f, v in [("--lerobot_root", args.lerobot_root),
                                   ("--repo_id", args.repo_id)] if v is None]
        if missing:
            ap.error(f"data_source=teleavatar 时以下参数必须提供: {', '.join(missing)}")

    client = _connect(args.host, args.port)
    if args.data_source == "libero":
        build_main_libero(client, lerobot_root=args.lerobot_root, language=args.language,
                          pooling=args.pooling, serve_ckpt_id=args.serve_ckpt_id,
                          out_cache=args.out_cache, num_demos=args.num_demos)
    elif args.data_source == "teleavatar":
        build_main_teleavatar(client, lerobot_root=args.lerobot_root, repo_id=args.repo_id,
                              prompt=args.prompt, pooling=args.pooling,
                              serve_ckpt_id=args.serve_ckpt_id, out_cache=args.out_cache,
                              num_demos=args.num_demos, proprio_key=args.proprio_key,
                              num_shards=args.num_shards, shard_index=args.shard_index,
                              policy_state_dim=args.policy_state_dim)
    else:
        build_main(client, hdf5=args.hdf5, dataset_id=args.dataset, image_keys=args.image_keys,
                   proprio_key=args.proprio_key, prompt=args.prompt, pooling=args.pooling,
                   serve_ckpt_id=args.serve_ckpt_id, out_cache=args.out_cache,
                   num_demos=args.num_demos)


if __name__ == "__main__":
    main()
