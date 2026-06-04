# tests/policies/pi05/test_load_pi05.py
import types

from resfit.lerobot.policies.pi05 import load_pi05


class _FakeClient:
    created_with = None

    def __init__(self, host, port):
        type(self).created_with = {"host": host, "port": port}


class _FakeAdapter:
    created_with = None

    @classmethod
    def from_policy(cls, policy, **kwargs):
        cls.created_with = {"policy": policy, **kwargs}
        inst = cls()
        inst.config = types.SimpleNamespace(
            image_features=dict.fromkeys(kwargs["image_key_map"])
        )
        return inst


def test_load_pi05_builds_ws_client_and_adapter(monkeypatch):
    monkeypatch.setattr(load_pi05, "_import_ws_client", lambda: _FakeClient)
    monkeypatch.setattr(load_pi05, "_import_pi05_adapter", lambda: _FakeAdapter)

    cfg = types.SimpleNamespace(
        type="pi05",
        host="127.0.0.1",
        port=8000,
        prompt="pick up the cube",
        action_dim=7,
        execute_horizon=30,
        image_key_map={
            "observation.images.agentview": "base",
            "observation.images.robot0_eye_in_hand": "left_wrist",
        },
        kai0_paths=["/data2/kai0"],
    )
    policy = load_pi05.load_pi05_base_policy(cfg, device="cpu")

    assert _FakeClient.created_with == {"host": "127.0.0.1", "port": 8000}
    assert isinstance(_FakeAdapter.created_with["policy"], _FakeClient)
    assert _FakeAdapter.created_with["action_dim"] == 7
    assert _FakeAdapter.created_with["prompt"] == "pick up the cube"
    assert _FakeAdapter.created_with["image_key_map"] == cfg.image_key_map
    assert list(policy.config.image_features.keys()) == list(cfg.image_key_map.keys())
