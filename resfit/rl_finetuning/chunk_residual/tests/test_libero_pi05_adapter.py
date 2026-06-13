import numpy as np
import pytest
import torch
from collections import deque
from resfit.rl_finetuning.chunk_residual.libero_pi05_adapter import LiberoPi05Adapter


class _StubPolicy:
    """假 pi0_libero serve client:记录收到的 obs,infer 返回固定 10 步 chunk(每步 8 维,>action_dim)。"""
    def __init__(self): self.calls = 0; self.last_obs = None
    def infer(self, obs):
        self.calls += 1; self.last_obs = obs
        return {"actions": np.arange(10 * 8, dtype=np.float32).reshape(10, 8)}


def _raw_obs(B=1):
    return {"observation.images.agentview": np.zeros((B, 3, 8, 8), np.float32),
            "observation.images.robot0_eye_in_hand": np.zeros((B, 3, 8, 8), np.float32),
            "observation.state": np.zeros((B, 8), np.float32)}


def test_select_action_returns_BxD_and_slices_action_dim():
    p = _StubPolicy()
    ad = LiberoPi05Adapter(p, prompt="x", action_dim=7, execute_horizon=5)
    a = ad.select_action(_raw_obs(1))
    assert isinstance(a, torch.Tensor) and a.shape == (1, 7)      # 切到 action_dim=7
    assert np.allclose(a[0].cpu().numpy(), np.arange(7))          # chunk 第 0 步前 7 维


def test_queue_refills_after_execute_horizon():
    p = _StubPolicy()
    ad = LiberoPi05Adapter(p, prompt="x", action_dim=7, execute_horizon=3)
    for _ in range(3):
        ad.select_action(_raw_obs(1))
    assert p.calls == 1                                          # 前 3 步用同一 chunk
    ad.select_action(_raw_obs(1))
    assert p.calls == 2                                          # execute_horizon=3 用完,重新 infer


def test_to_openpi_obs_emits_flat_schema():
    p = _StubPolicy()
    ad = LiberoPi05Adapter(p, prompt="pp", action_dim=7)
    ad.select_action(_raw_obs(1))
    assert set(p.last_obs) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    assert p.last_obs["prompt"] == "pp" and p.last_obs["observation/state"].shape == (8,)


def test_reset_clears_queue():
    p = _StubPolicy()
    ad = LiberoPi05Adapter(p, prompt="x", action_dim=7, execute_horizon=5)
    ad.select_action(_raw_obs(1)); ad.reset()
    ad.select_action(_raw_obs(1))
    assert p.calls == 2                                          # reset 后队列空,重新 infer


def test_config_image_features_keys():
    ad = LiberoPi05Adapter(_StubPolicy(), prompt="x", action_dim=7)
    # train 接线读 base_policy.config.image_features.keys() 取图像键
    keys = set(ad.config.image_features.keys())
    assert keys == {"observation.images.agentview", "observation.images.robot0_eye_in_hand"}


def test_from_policy_classmethod():
    ad = LiberoPi05Adapter.from_policy(_StubPolicy(), prompt="x", action_dim=7,
                                       device="cpu", execute_horizon=5, image_key_map=None)
    assert isinstance(ad, LiberoPi05Adapter) and ad.action_dim == 7


def test_select_action_multi_env_independent_queues():
    # B=2:两 env 各自独立队列,stack 顺序与 env 对应
    p = _StubPolicy()
    ad = LiberoPi05Adapter(p, prompt="x", action_dim=7, execute_horizon=5)
    a = ad.select_action(_raw_obs(2))
    assert a.shape == (2, 7)
    assert np.allclose(a[0].cpu().numpy(), np.arange(7))      # 两 env 同 stub chunk,首步一致
    assert np.allclose(a[1].cpu().numpy(), np.arange(7))


def test_infer_empty_chunk_raises():
    class _EmptyPolicy:
        def infer(self, obs): return {"actions": np.zeros((0, 8), np.float32)}
    ad = LiberoPi05Adapter(_EmptyPolicy(), prompt="x", action_dim=7)
    with pytest.raises(ValueError):
        ad.select_action(_raw_obs(1))


def test_infer_missing_actions_key_raises():
    class _NoActionsPolicy:
        def infer(self, obs): return {"foo": 1}
    ad = LiberoPi05Adapter(_NoActionsPolicy(), prompt="x", action_dim=7)
    with pytest.raises(ValueError):
        ad.select_action(_raw_obs(1))
