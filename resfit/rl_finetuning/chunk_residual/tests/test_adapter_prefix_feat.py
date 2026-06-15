"""测试 LiberoPi05Adapter 暴露 last_prefix_feat (Task 1: pi0_feat online)。"""
import numpy as np
from resfit.rl_finetuning.chunk_residual.libero_pi05_adapter import LiberoPi05Adapter


class _StubPolicy:
    def __init__(self, with_feat=True):
        self.with_feat = with_feat

    def infer(self, obs):
        out = {"actions": np.zeros((5, 7), np.float32)}
        if self.with_feat:
            out["prefix_feat"] = np.arange(2048, dtype=np.float32)
        return out


def test_adapter_stores_and_exposes_prefix_feat():
    a = LiberoPi05Adapter.from_policy(
        _StubPolicy(with_feat=True), prompt="x", action_dim=7,
        device="cpu", execute_horizon=5
    )
    assert a.last_prefix_feat() is None                        # 未 infer 前应为 None
    a._infer_chunk({"observation/image": np.zeros((224, 224, 3), np.uint8)})
    pf = a.last_prefix_feat()
    assert pf is not None and np.asarray(pf).shape == (2048,)


def test_adapter_prefix_feat_none_when_serve_omits_it():
    a = LiberoPi05Adapter.from_policy(
        _StubPolicy(with_feat=False), prompt="x", action_dim=7,
        device="cpu", execute_horizon=5
    )
    a._infer_chunk({"observation/image": np.zeros((224, 224, 3), np.uint8)})
    assert a.last_prefix_feat() is None                        # serve 没透特征→None,不崩
