"""Load pi05 base policy: connect to a pi05 websocket server, wrap via Pi05PolicyAdapter.from_policy as a step-level base policy."""
from __future__ import annotations

import sys


def _import_ws_client():
    """openpi-client's WebsocketClientPolicy (standalone fn for test monkeypatch)."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: WPS433
    return WebsocketClientPolicy


def _import_pi05_adapter():
    """Pi05PolicyAdapter from kai0 (standalone fn for test monkeypatch)."""
    from resfit_pi05.pi05_policy_adapter import Pi05PolicyAdapter  # noqa: WPS433
    return Pi05PolicyAdapter


def load_pi05_base_policy(cfg, device, schema="dexmg"):
    """Connect to the pi05 websocket server and wrap it as a step-level base_policy.

    cfg must have: host, port, prompt, action_dim, execute_horizon, image_key_map, kai0_paths.
    Returned object provides select_action / reset / config.image_features.

    schema:
    - "dexmg" (default / others): kai0 resfit_pi05.Pi05PolicyAdapter (unchanged).
    - "libero": self-contained LiberoPi05Adapter (single-arm flat schema, no kai0 dependency).
    """
    for p in getattr(cfg, "kai0_paths", []):
        if p not in sys.path:
            sys.path.insert(0, p)

    WebsocketClientPolicy = _import_ws_client()
    # 禁掉 websocket keepalive ping:pi0 serve 首次推理触发冷启动 JIT 编译(可 >20s),
    # 期间不响应 keepalive ping 会被 websockets 默认 20s 超时断连(ConnectionClosedError)。
    # ping_interval=None 关掉心跳,长推理不掉线。
    import websockets.sync.client as _wsc
    if not getattr(_wsc.connect, "_no_ping_patched", False):
        _orig_connect = _wsc.connect

        def _connect_no_ping(*a, **k):
            k.setdefault("ping_interval", None)
            return _orig_connect(*a, **k)

        _connect_no_ping._no_ping_patched = True
        _wsc.connect = _connect_no_ping

    client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)

    if schema == "libero":
        from resfit.rl_finetuning.chunk_residual.libero_pi05_adapter import LiberoPi05Adapter
        return LiberoPi05Adapter.from_policy(
            client,
            prompt=cfg.prompt,
            action_dim=cfg.action_dim,
            device=str(device),
            execute_horizon=cfg.execute_horizon,
            image_key_map=dict(cfg.image_key_map),
        )

    Pi05PolicyAdapter = _import_pi05_adapter()
    return Pi05PolicyAdapter.from_policy(
        client,
        prompt=cfg.prompt,
        action_dim=cfg.action_dim,
        device=str(device),
        execute_horizon=cfg.execute_horizon,
        image_key_map=dict(cfg.image_key_map),
    )
