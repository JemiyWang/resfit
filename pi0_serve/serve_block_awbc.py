"""起 block(pi05_block_awbc)的 kai0 serve,并透出 prefix_feat(供 pi0_feat 缓存/wm_bridge)。

背景:pi05_block_awbc 是 run_awbc.py 通过 pick_cup_configs.make_awbc_config(...) **动态构造**的
TrainConfig,**从不注册进 openpi 的 _CONFIGS**。所以 serve_with_feat.py 的
`_config.get_config("pi05_block_awbc")` 会 KeyError。本 launcher 直接构造同款 config 对象,
其余(FeaturePolicy 包装 + WebsocketPolicyServer)完全复用 serve_with_feat.py 的路径。

运行环境(实测):chj 的 openpi venv + PYTHONPATH 指向 kai0 源码,即可加载与 49999 checkpoint
匹配的 openpi(定义了 LerobotTeleAvatarDataConfig)。

  PYTHONPATH=/mnt/mnt/data/data2/kai0/src:/mnt/mnt/data/data2/kai0 \
  CUDA_VISIBLE_DEVICES=0 \
  /mnt/mnt/data/chj/openpi/.venv/bin/python /mnt/mnt/data/resfit/pi0_serve/serve_block_awbc.py \
    --dir /mnt/mnt/data/data2/kai0/checkpoints/block/49999 --port 8001 --pooling mean

★ --pooling 必须与建缓存/训 V 侧一致(我们用 mean)。serve 与下游 pooling 不一致 → ψ 不同源。
★ 建缓存客户端会自己发 prompt="build block";此处 default_prompt 只是 fallback
  (config prompt_from_task=True,在线以 obs 里的 prompt 为准)。
"""
from __future__ import annotations

import dataclasses
import os
import sys

# ★ 在 import openpi 之前把路径就位:①本目录(feature_policy) ②kai0 仓(pick_cup_configs)。
#   openpi 本身由 PYTHONPATH=/data2/kai0/src 提供(运行命令见文件头),此处不动它。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, "/mnt/mnt/data/data2/kai0"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tyro  # noqa: E402

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402


def _ensure_paths():
    for p in (_HERE, "/mnt/mnt/data/data2/kai0"):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclasses.dataclass
class Args:
    dir: str                              # checkpoint 目录(如 .../checkpoints/block/49999)
    port: int = 8001
    host: str = "0.0.0.0"
    pooling: str = "mean"                 # ★ 与下游一致
    default_prompt: str = "build_block"
    config_name: str = "pi05_block_awbc"  # 只用于 assets/<name> 归属;此处 norm 从 checkpoint 读
    repo_id: str = "/mnt/mnt/data/domains_rise/block/block_success"  # 仅建 transform,不读数据
    asset_id: str = "inference"
    serve_ckpt_id: str = "pi05_block_awbc_49999"
    policy_state_dim: int = 16


def _create_policy(args: "Args"):
    _ensure_paths()
    import pick_cup_configs as cfgs

    # 直接构造与训练同款的 TrainConfig(不经 _config.get_config)
    train_config = cfgs.make_awbc_config(
        repo_id=args.repo_id,
        name=args.config_name,
        default_prompt=args.default_prompt,
    )
    assets = dataclasses.replace(
        train_config.data.assets,
        asset_id=args.asset_id,
    )
    train_config = dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, assets=assets),
    )
    # norm_stats 从 checkpoint 的 assets/<asset_id> 读;
    # data.create() 只建 transform,不访问 repo_id 数据集 → repo_id 不存在也不影响 serve。
    return _policy_config.create_trained_policy(
        train_config, args.dir, default_prompt=args.default_prompt)


def _serve(policy, host, port):
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=host, port=port, metadata=policy.metadata,
    ).serve_forever()


def run(args: "Args"):
    _ensure_paths()
    from feature_policy import wrap_with_feature

    inner = _create_policy(args)
    policy = wrap_with_feature(
        inner,
        pooling=args.pooling,
        metadata_overrides={
            "serve_ckpt_id": args.serve_ckpt_id,
            "pooling": args.pooling,
            "asset_id": args.asset_id,
            "policy_state_dim": args.policy_state_dim,
        },
    )
    print(f"[serve_block_awbc] up on {args.host}:{args.port} | pooling={args.pooling} | "
          f"dir={args.dir}", flush=True)
    _serve(policy, args.host, args.port)


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
