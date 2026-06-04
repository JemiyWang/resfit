# tests/rl_finetuning/test_base_policy_switch.py
from resfit.rl_finetuning.config.residual_td3 import BasePolicyConfig


def test_default_type_is_act():
    cfg = BasePolicyConfig()
    assert cfg.type == "act"


def test_pi05_fields_exist_with_defaults():
    cfg = BasePolicyConfig(type="pi05")
    assert cfg.type == "pi05"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.action_dim == 7
    assert cfg.execute_horizon == 30
    assert cfg.prompt == "pick up the cube"
    assert "observation.images.agentview" in cfg.image_key_map
    assert cfg.kai0_paths == ["/data2/kai0"]


# ---------------------------------------------------------------------------
# Integration smoke test: Pi05PolicyAdapter with a fake inner policy
# ---------------------------------------------------------------------------
import sys
import numpy as np

if "/data2/kai0" not in sys.path:  # provide package root to import resfit_pi05.*
    sys.path.insert(0, "/data2/kai0")
from resfit_pi05.pi05_policy_adapter import Pi05PolicyAdapter


class _FakePi05Policy:
    def infer(self, obs):
        # fixed 50x32 chunk; first 7 dims are a recognizable value
        chunk = np.zeros((50, 32), dtype=np.float32)
        chunk[:, :7] = 0.5
        return {"actions": chunk}


def test_pi05_adapter_drives_step_loop_shapes():
    adapter = Pi05PolicyAdapter.from_policy(
        _FakePi05Policy(), prompt="pick up the cube", action_dim=7, device="cpu",
        execute_horizon=30,
        image_key_map={
            "observation.images.agentview": "base",
            "observation.images.robot0_eye_in_hand": "left_wrist",
        },
    )
    # keys the env wrapper uses to fetch images
    assert list(adapter.config.image_features.keys()) == [
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
    ]

    B = 2
    raw_obs = {
        "observation.state": np.zeros((B, 9), dtype=np.float32),
        "observation.images.agentview": np.zeros((B, 224, 224, 3), dtype=np.uint8),
        "observation.images.robot0_eye_in_hand": np.zeros((B, 224, 224, 3), dtype=np.uint8),
    }
    a1 = adapter.select_action(raw_obs)
    assert tuple(a1.shape) == (B, 7)
    assert np.allclose(a1.cpu().numpy(), 0.5)

    # subsequent calls drain the queue; reset clears without error
    for _ in range(5):
        adapter.select_action(raw_obs)
    adapter.reset()
    a2 = adapter.select_action(raw_obs)
    assert tuple(a2.shape) == (B, 7)
