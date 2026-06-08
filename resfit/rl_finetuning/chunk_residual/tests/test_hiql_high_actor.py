import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    """多核机上 torch 对微小张量开满线程会颠簸;本模块测试限 1 线程,跑完恢复(防泄漏到其它测试文件)。"""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def test_high_actor_forward_dist():
    ha = HighActor(state_dim=30, rep_dim=10, hidden=64)
    s = torch.randn(6, 30)
    g = torch.randn(6, 30)
    dist = ha(s, g)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (6, 10)
    z = dist.rsample()
    assert z.shape == (6, 10)
    lp = dist.log_prob(z).sum(-1)
    assert lp.shape == (6,)
    # std 与输入无关(log_std 是 nn.Parameter,不是 MLP 输出)
    assert torch.allclose(dist.scale[0], dist.scale[1])


def test_awr_weight():
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import awr_weight
    adv = torch.tensor([-1.0, 0.0, 1.0, 100.0])
    w = awr_weight(adv, beta=1.0, clip=100.0)
    assert torch.allclose(w[:3], torch.exp(torch.tensor([-1.0, 0.0, 1.0])), atol=1e-5)
    assert float(w[3]) == 100.0
    w2 = awr_weight(adv, beta=0.5, clip=100.0)
    assert torch.allclose(w2[2], torch.exp(torch.tensor(0.5)), atol=1e-5)
