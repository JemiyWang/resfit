"""base-policy-as-base 离线模式:base_action 由冻结 base policy 现算(非 GT-as-base)。"""
import h5py
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual import offline_stage_replay as osr
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import STATE18_KEYS, save_stage_cache


class _IdScaler:
    def scale(self, a): return torch.as_tensor(a, dtype=torch.float32)
    def unscale(self, a): return a


# _IdStd / _FakeRb(下方)与 save_stage_cache import 供后续 Task 2 的 build_offline_buffer 测试复用
class _IdStd:
    def standardize(self, s): return torch.as_tensor(s, dtype=torch.float32)


class _FakeRb:
    def __init__(self): self.items = []
    def add(self, td): self.items.append(td)


class _FakeBase:
    """假 base policy:reset 计数;select_action 第 i 次返回全 (i+1) 的 (1,7) 动作,记录见到的 key。"""
    def __init__(self, action_dim=7):
        self.reset_calls = 0
        self.select_calls = 0
        self.seen_keys = []
        self.seen_obs = []
        self.action_dim = action_dim
    def reset(self): self.reset_calls += 1
    def select_action(self, raw_obs):
        self.seen_keys.append(set(raw_obs.keys()))
        self.seen_obs.append({k: v.clone() for k, v in raw_obs.items()})
        i = self.select_calls
        self.select_calls += 1
        return torch.full((1, self.action_dim), float(i + 1))


def _make_tiny_hdf5(path, T=4):
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("states", data=np.zeros((T, 5), dtype=np.float32))
        g.create_dataset("actions", data=np.zeros((T, 7), dtype=np.float32))
        for k, d in STATE18_KEYS:
            g.create_dataset(f"obs/{k}", data=np.arange(T * d, dtype=np.float32).reshape(T, d))
        g.create_dataset("obs/agentview_image", data=np.zeros((T, 4, 4, 3), dtype=np.uint8))
        g.attrs["model_file"] = "dummy_model"
    return T


def test_demo_base_actions_order_reset_format_scale(tmp_path):
    hdf5 = str(tmp_path / "tiny.hdf5")
    T = _make_tiny_hdf5(hdf5)
    fake = _FakeBase(action_dim=7)
    with h5py.File(hdf5, "r") as f:
        grp = f["data/demo_0"]
        base_n = osr._demo_base_actions(
            fake, grp, image_keys=["observation.images.agentview"],
            action_scaler=_IdScaler(), device="cpu")
    assert fake.reset_calls == 1                      # 每 demo reset 一次
    assert fake.select_calls == T                     # 逐帧顺序调 T 次
    assert base_n.shape == (T, 7)
    # 每帧 select_action 返回 (i+1) → IdScaler 透传 → 第 t 行全 (t+1)
    for t in range(T):
        assert torch.allclose(base_n[t], torch.full((7,), float(t + 1)))
    # raw_obs 必含原始 state + 图像 key
    for keys in fake.seen_keys:
        assert "observation.state" in keys
        assert "observation.images.agentview" in keys
    # obs 格式:state=(1,18) float32 原始;图像=(1,3,H,W) float32 ∈[0,1]
    obs0 = fake.seen_obs[0]
    assert obs0["observation.state"].shape == (1, 18)
    assert obs0["observation.state"].dtype == torch.float32
    img = obs0["observation.images.agentview"]
    assert img.shape == (1, 3, 4, 4)              # tiny hdf5 图像是 4x4
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
