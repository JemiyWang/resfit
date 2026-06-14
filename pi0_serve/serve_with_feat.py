"""起 pi05 serve,但用 FeaturePolicy 包原 policy → infer 返回额外带 prefix_feat。

并列入口,不改 openpi 的 serve_policy.py。文件在 resfit/pi0_serve/,用 openpi 的 .venv 跑:
  /mnt/mnt/data/chj/openpi/.venv/bin/python /mnt/mnt/data/resfit/pi0_serve/serve_with_feat.py \
    --config <cfg> --dir <ckpt> --port 8000 --pooling last

WebsocketPolicyServer 签名对齐 openpi/scripts/serve_policy.py:
  WebsocketPolicyServer(policy=policy, host=host, port=port, metadata=policy.metadata)
"""
from __future__ import annotations

import dataclasses

import tyro

from feature_policy import wrap_with_feature
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str
    dir: str
    port: int = 8000
    host: str = "0.0.0.0"
    pooling: str = "last"
    default_prompt: str | None = None


def _create_policy(args: "Args"):
    return _policy_config.create_trained_policy(
        _config.get_config(args.config), args.dir, default_prompt=args.default_prompt)


def _serve(policy, host, port):
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy.metadata,
    ).serve_forever()


def run(args: "Args"):
    inner = _create_policy(args)
    policy = wrap_with_feature(inner, pooling=args.pooling)
    _serve(policy, args.host, args.port)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
