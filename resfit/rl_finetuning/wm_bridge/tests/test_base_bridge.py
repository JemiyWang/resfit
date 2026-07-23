import numpy as np
import pytest
import torch

from resfit.rl_finetuning.wm_bridge.base_bridge import Kai0ImaginationBase
from resfit.rl_finetuning.wm_bridge.wm_driver import CAMERA_KEYS


class _StubClient:
    def __init__(self, psi=None):
        self.n = 0
        self._psi = np.ones(8, np.float32) if psi is None else psi

    def infer(self, obs):
        self.n += 1
        acts = np.full((50, 16), float(self.n), dtype=np.float32)
        return {"actions": acts, "prefix_feat": self._psi}


def _obs(token):
    o = {"_wm_window_token": token,
         "_wm_native_frames": np.zeros((3, 3, 192, 256), np.float32),
         "observation.state": np.zeros((1, 16), np.float32)}
    for k in CAMERA_KEYS:
        o[k] = torch.zeros(1, 3, 84, 84)
    return o


def test_same_window_token_hits_cache_only_one_serve_call():
    c = _StubClient()
    b = Kai0ImaginationBase(c, prompt="build block")
    a1, p1 = b.query(_obs(7))
    a2, p2 = b.query(_obs(7))
    assert c.n == 1                        # ★ 一个 chunk 内只调一次 serve
    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(p1, p2)


def test_new_window_token_triggers_new_serve_call():
    c = _StubClient()
    b = Kai0ImaginationBase(c, prompt="build block")
    b.query(_obs(1))
    b.query(_obs(2))
    assert c.n == 2


def test_get_action_chunk_returns_batched_tensor():
    b = Kai0ImaginationBase(_StubClient(), prompt="build block")
    out = b.get_action_chunk(_obs(0), 50)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 50, 16)


def test_get_action_chunk_shares_cache_with_query():
    c = _StubClient()
    b = Kai0ImaginationBase(c, prompt="build block")
    b.query(_obs(3))
    b.get_action_chunk(_obs(3), 50)
    assert c.n == 1                        # ★ query 与 wrapper 的取动作共用同一次调用


def test_get_action_chunk_follows_observation_device():
    """base_action 会直接拼进 GPU obs，不能把 numpy serve 结果留在 CPU。"""
    b = Kai0ImaginationBase(_StubClient(), prompt="build block")
    b._cache_token = 9
    b._cache = (np.zeros((50, 16), np.float32), np.zeros(8, np.float32))
    obs = {"_wm_window_token": 9,
           "observation.state": torch.empty(1, 16, device="meta")}
    assert b.get_action_chunk(obs, 50).device.type == "meta"


def test_missing_prefix_feat_raises():
    class _NoFeat(_StubClient):
        def infer(self, obs):
            return {"actions": np.zeros((50, 16), np.float32), "prefix_feat": None}

    b = Kai0ImaginationBase(_NoFeat(), prompt="build block")
    with pytest.raises(RuntimeError, match="prefix_feat"):
        b.query(_obs(0))


def test_config_exposes_camera_image_features():
    b = Kai0ImaginationBase(_StubClient(), prompt="build block")
    assert set(b.config.image_features.keys()) == set(CAMERA_KEYS)


def test_wrong_chunk_length_raises():
    b = Kai0ImaginationBase(_StubClient(), prompt="build block")
    with pytest.raises(AssertionError):
        b.get_action_chunk(_obs(0), 25)


def test_serve_obs_is_nested_teleavatar_schema():
    """★ S1 验证过的 block kai0 嵌套 schema:{"state","images":{裸cam},"prompt"},
    图 [-1,1]→[0,1]→224 CHW。锁死,防回退到早先猜的扁平 schema。"""
    import numpy as np
    b = Kai0ImaginationBase(_StubClient(), prompt="build block")
    raw = {"_wm_native_frames": np.zeros((3, 3, 192, 256), np.float32),
           "observation.state": np.arange(16, dtype=np.float32)}
    obs = b._serve_obs(raw)
    assert set(obs.keys()) == {"state", "images", "prompt"}
    assert set(obs["images"].keys()) == {"top_head", "hand_left", "hand_right"}
    assert obs["images"]["top_head"].shape == (3, 224, 224)   # CHW 224
    assert obs["prompt"] == "build block"
    np.testing.assert_allclose(obs["state"], np.arange(16))
