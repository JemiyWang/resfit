"""base-policy-as-base 离线模式:base_action 由冻结 base policy 现算(非 GT-as-base)。"""
import pytest
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


def _build(tmp_path, base_mode, base_policy):
    hdf5 = str(tmp_path / "tiny.hdf5")
    T = _make_tiny_hdf5(hdf5)
    stage_cache = str(tmp_path / "stages.npz")
    save_stage_cache(stage_cache, {"demo_0": np.arange(T, dtype=np.int8)})
    rb = _FakeRb()
    osr.build_offline_buffer(
        rb, hdf5, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
        image_keys=["observation.images.agentview"], bonus=1.0, mode="staged",
        gamma=0.99, stage_cache=stage_cache, potential=None,
        base_policy=base_policy, base_mode=base_mode, base_device="cpu")
    return rb, T


def test_base_policy_mode_anchors_to_base_not_gt(tmp_path):
    fake = _FakeBase(action_dim=7)
    rb, T = _build(tmp_path, base_mode="base_policy", base_policy=fake)
    # GT actions 全 0 → act(=GT)=0;base_action 应为 fake 的 (t+1),≠ GT
    for t in range(T - 1):
        ba = rb.items[t]["obs"]["observation.base_action"]
        act = rb.items[t]["action"]
        assert torch.allclose(act, torch.zeros(7))                  # action 仍存 GT(0)
        assert torch.allclose(ba, torch.full((7,), float(t + 1)))   # base_action = base policy 现算
        bc_target = act - ba                                        # 隐含残差目标 ≠ 0
        assert not torch.allclose(bc_target, torch.zeros(7))


def test_gt_mode_byte_equivalent_base_equals_action(tmp_path):
    rb, T = _build(tmp_path, base_mode="gt", base_policy=None)
    for t in range(T - 1):
        ba = rb.items[t]["obs"]["observation.base_action"]
        act = rb.items[t]["action"]
        assert torch.allclose(ba, act)                              # GT-as-base:base==action(逐位等价)


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


def test_base_policy_mode_requires_base_policy(tmp_path):
    with pytest.raises(ValueError, match="需传 base_policy"):
        _build(tmp_path, base_mode="base_policy", base_policy=None)


def test_unknown_base_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="未知 base_mode"):
        _build(tmp_path, base_mode="bogus", base_policy=None)


# ── Task 3:CLI flag + 缓存签名条件键 ────────────────────────────────────────
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    build_parser, _offline_buffer_signature)


def _sig_args(extra):
    base = ["--task", "TwoArmThreePieceAssembly", "--offline_dataset_path", "/tmp/x.hdf5"]
    return build_parser().parse_args(base + extra)


def test_cli_offline_base_mode_default_gt():
    assert build_parser().parse_args([]).offline_base_mode == "gt"


def test_cli_offline_base_mode_parses():
    assert build_parser().parse_args(["--offline_base_mode", "base_policy"]).offline_base_mode \
        == "base_policy"


def test_signature_gt_has_no_base_mode_key():
    img = ["observation.images.agentview"]
    s = _offline_buffer_signature(_sig_args([]), img, 100, "staged")
    assert "offline_base_mode" not in s          # gt 默认 → 不加键 → 旧缓存向后兼容


def test_signature_base_policy_keys_and_distinguish_base():
    img = ["observation.images.agentview"]
    sa = _offline_buffer_signature(
        _sig_args(["--offline_base_mode", "base_policy", "--base_wandb_id", "/tmp/baseA"]),
        img, 100, "staged")
    sb = _offline_buffer_signature(
        _sig_args(["--offline_base_mode", "base_policy", "--base_wandb_id", "/tmp/baseB"]),
        img, 100, "staged")
    assert sa.get("offline_base_mode") == "base_policy"
    assert "base_wandb_id" in sa
    assert sa != sb                              # 换 base → 不同签名 → 强制重建
