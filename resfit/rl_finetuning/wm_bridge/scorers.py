"""Φ 打分器。

PBRS 端点差只需要"给一个 ψ 打一个 Φ",所以接口是单点的,不是逐帧的。
ψ 由 base_bridge 从 kai0 serve 的 prefix_feat 取得,本模块不自己编码。

安全线:shaping 只用单状态 V;gc value 的 V(s,z) 绝不进 reward。
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class Scorer(Protocol):
    def phi(self, psi: np.ndarray, proprio: np.ndarray) -> float:
        ...


class DummyScorer:
    """恒 0。仅供单元测试;真实训练须显式 --allow_dummy_scorer 才可用(见 launcher)。"""

    expected_psi_anchor = None

    def phi(self, psi, proprio) -> float:
        return 0.0


class Kai0HiqlScorer:
    """ψ ⊕ proprio → 标准化 → ValueMLP → Φ。"""

    def __init__(self, value_model, *, mean, std, expected_psi_anchor=None):
        self.model = value_model
        self.model.eval()
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        assert self.mean.shape == self.std.shape, "mean/std 维度须一致"
        assert np.all(self.std > 0), "std 须逐维为正"
        self.state_dim = int(self.mean.shape[0])
        self.expected_psi_anchor = expected_psi_anchor

    @classmethod
    def from_value_ckpt(cls, path, device="cpu"):
        from resfit.rl_finetuning.chunk_residual.hiql_value import load_value
        model, info = load_value(path, map_location=device)
        # 同源锚 = value.pt 的 pi0_feat_signature.serve_ckpt_id(Task 14 落地机制)。
        # base 统一用 kai0/pi05,不涉及 ACT;不再用 act_weight_sha。
        sig = info.get("pi0_feat_signature") or {}
        return cls(model, mean=info["mean"], std=info["std"],
                   expected_psi_anchor=sig.get("serve_ckpt_id"))

    def phi(self, psi, proprio) -> float:
        p = np.asarray(psi, dtype=np.float32).reshape(-1)
        q = np.asarray(proprio, dtype=np.float32).reshape(-1)
        state = np.concatenate([p, q])
        assert state.shape[0] == self.state_dim, \
            f"ψ⊕proprio 维度 {state.shape[0]} != value.pt 的 state_dim {self.state_dim}"
        z = (state - self.mean) / self.std
        with torch.no_grad():
            v = self.model(torch.from_numpy(z.astype(np.float32)))
        return float(v.reshape(-1)[0])
