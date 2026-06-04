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


def load_pi05_base_policy(cfg, device):
    """Connect to the pi05 websocket server and wrap it as a step-level base_policy.

    cfg must have: host, port, prompt, action_dim, execute_horizon, image_key_map, kai0_paths.
    Returned object provides select_action / reset / config.image_features.
    """
    for p in getattr(cfg, "kai0_paths", []):
        if p not in sys.path:
            sys.path.insert(0, p)

    WebsocketClientPolicy = _import_ws_client()
    Pi05PolicyAdapter = _import_pi05_adapter()

    client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)
    return Pi05PolicyAdapter.from_policy(
        client,
        prompt=cfg.prompt,
        action_dim=cfg.action_dim,
        device=str(device),
        execute_horizon=cfg.execute_horizon,
        image_key_map=dict(cfg.image_key_map),
    )
