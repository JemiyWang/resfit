import numpy as np
import pytest
import torch
from torch import nn

from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer, Kai0HiqlScorer


class _ConstValue(nn.Module):
    """把标准化后的 state 求和当 V,便于精确断言。"""

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.grad_enabled_during_forward = None

    def forward(self, s):
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        return s.sum(dim=-1, keepdim=True)


def test_dummy_scorer_returns_zero():
    s = DummyScorer()
    assert s.phi(np.ones(8, np.float32), np.ones(16, np.float32)) == 0.0


def test_kai0_scorer_concats_psi_and_proprio_then_standardizes():
    psi = np.array([1.0, 2.0], np.float32)
    proprio = np.array([3.0], np.float32)
    mean = np.array([1.0, 1.0, 1.0], np.float32)
    std = np.array([1.0, 2.0, 3.0], np.float32)
    sc = Kai0HiqlScorer(_ConstValue(3), mean=mean, std=std)
    # 标准化后 = [(1-1)/1, (2-1)/2, (3-1)/3] = [0, 0.5, 0.6667];求和 ≈ 1.1667
    assert sc.phi(psi, proprio) == pytest.approx(1.0 / 2 + 2.0 / 3, abs=1e-5)


def test_kai0_scorer_rejects_dim_mismatch():
    sc = Kai0HiqlScorer(_ConstValue(3),
                        mean=np.zeros(3, np.float32), std=np.ones(3, np.float32))
    with pytest.raises(AssertionError):
        sc.phi(np.ones(5, np.float32), np.ones(1, np.float32))


def test_kai0_scorer_returns_python_float():
    sc = Kai0HiqlScorer(_ConstValue(2),
                        mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    out = sc.phi(np.ones(1, np.float32), np.ones(1, np.float32))
    assert isinstance(out, float)


def test_kai0_scorer_does_not_track_grad():
    # model.parameters() 为空(_ConstValue 无 nn.Parameter),所以不能靠 "p.grad is
    # None" 断言(那样即使 phi 里的 torch.no_grad() 被删掉,循环也不会执行,测试恒真)。
    # 改为让 stub 在 forward 时直接记录 torch.is_grad_enabled(),精确验证 phi 是否
    # 真的在 no_grad 上下文里跑前向。
    model = _ConstValue(2)
    sc = Kai0HiqlScorer(model, mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    sc.phi(np.ones(1, np.float32), np.ones(1, np.float32))
    assert model.grad_enabled_during_forward is False
