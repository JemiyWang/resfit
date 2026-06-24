import copy
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
    GoalConditionedVF, value_update_step, expectile_loss_weighted)
from resfit.rl_finetuning.chunk_residual.hiql_value import expectile_loss


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


# 此函数逐字复刻 train_gc_value 旧内层(shared_min+done_aware),原实现若变更须同步。
def _manual_shared_min_step(model, target, opt, s, s_next, g, success, done,
                            *, gamma, expectile, ema):
    """逐字复刻 train_gc_value 旧内层(shared_min + done_aware)的一步,作为对照基线。"""
    reward = success - 1.0
    mask = (1.0 - success) * (1.0 - done)
    with torch.no_grad():
        nv1, nv2 = target(s_next, g)
        nv = torch.minimum(nv1, nv2)
        y = reward + gamma * mask * nv
    v1, v2 = model(s, g)
    loss = expectile_loss(y - v1, expectile) + expectile_loss(y - v2, expectile)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        for tp, mp in zip(target.parameters(), model.parameters()):
            tp.mul_(1.0 - ema).add_(ema * mp)
    return float(loss)


# 此函数逐字复刻 value_update_step 的 hiql 分支(value_loss_mode=hiql/value_mask_mode=hiql),原实现若变更须同步。
def _manual_hiql_step(model, target, opt, s, s_next, g, success, done,
                      *, gamma, expectile, ema):
    """逐字复刻 value_update_step hiql 分支(mask_mode='hiql')的一步,作为对照基线。"""
    reward = success - 1.0
    mask = 1.0 - success                      # value_mask_mode == "hiql"
    with torch.no_grad():
        nv1, nv2 = target(s_next, g)
        nv = torch.minimum(nv1, nv2)
        q = reward + gamma * mask * nv
        v1t, v2t = target(s, g)
        v_t = 0.5 * (v1t + v2t)
        adv = q - v_t
        q1 = reward + gamma * mask * nv1
        q2 = reward + gamma * mask * nv2
    v1, v2 = model(s, g)
    loss = (expectile_loss_weighted(adv, q1 - v1, expectile)
            + expectile_loss_weighted(adv, q2 - v2, expectile))
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        for tp, mp in zip(target.parameters(), model.parameters()):
            tp.mul_(1.0 - ema).add_(ema * mp)
    return float(loss)


def test_value_update_step_matches_manual_inline():
    torch.manual_seed(0)
    s = torch.randn(7, 6)
    s_next = torch.randn(7, 6)
    g = torch.randn(7, 6)
    success = torch.tensor([1., 0., 0., 1., 0., 0., 0.])
    done = torch.tensor([0., 0., 1., 0., 0., 1., 0.])

    # 两套模型从同一初始权重出发
    m_a = GoalConditionedVF(6, rep_dim=4, hidden=16)
    m_b = copy.deepcopy(m_a)
    t_a = copy.deepcopy(m_a)
    t_b = copy.deepcopy(m_a)
    for p in t_a.parameters():
        p.requires_grad_(False)
    for p in t_b.parameters():
        p.requires_grad_(False)
    opt_a = torch.optim.Adam(m_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(m_b.parameters(), lr=1e-3)

    la = _manual_shared_min_step(m_a, t_a, opt_a, s, s_next, g, success, done,
                                 gamma=0.99, expectile=0.7, ema=0.005)
    lb = value_update_step(m_b, t_b, opt_b, s, s_next, g, success, done,
                           gamma=0.99, expectile=0.7, ema=0.005)
    assert abs(la - lb) < 1e-9
    for ka, va in m_a.state_dict().items():
        assert torch.equal(va, m_b.state_dict()[ka]), f"model param 不等: {ka}"
    for ka, va in t_a.state_dict().items():
        assert torch.equal(va, t_b.state_dict()[ka]), f"target param 不等: {ka}"


def test_value_update_step_hiql_matches_manual_inline():
    torch.manual_seed(2)
    s = torch.randn(7, 6)
    s_next = torch.randn(7, 6)
    g = torch.randn(7, 6)
    success = torch.tensor([1., 0., 0., 1., 0., 0., 0.])
    done = torch.tensor([0., 0., 1., 0., 0., 1., 0.])

    # 两套模型从同一初始权重出发
    m_a = GoalConditionedVF(6, rep_dim=4, hidden=16)
    m_b = copy.deepcopy(m_a)
    t_a = copy.deepcopy(m_a)
    t_b = copy.deepcopy(m_a)
    for p in t_a.parameters():
        p.requires_grad_(False)
    for p in t_b.parameters():
        p.requires_grad_(False)
    opt_a = torch.optim.Adam(m_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(m_b.parameters(), lr=1e-3)

    la = _manual_hiql_step(m_a, t_a, opt_a, s, s_next, g, success, done,
                           gamma=0.99, expectile=0.7, ema=0.005)
    lb = value_update_step(m_b, t_b, opt_b, s, s_next, g, success, done,
                           gamma=0.99, expectile=0.7, ema=0.005,
                           value_loss_mode="hiql", value_mask_mode="hiql")
    assert abs(la - lb) < 1e-9
    for ka, va in m_a.state_dict().items():
        assert torch.equal(va, m_b.state_dict()[ka]), f"model param 不等: {ka}"
    for ka, va in t_a.state_dict().items():
        assert torch.equal(va, t_b.state_dict()[ka]), f"target param 不等: {ka}"


def test_value_update_step_hiql_mode_runs_and_changes_params():
    torch.manual_seed(1)
    s, s_next, g = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    success = torch.zeros(5)
    done = torch.zeros(5)
    m = GoalConditionedVF(4, rep_dim=4, hidden=16)
    before = copy.deepcopy(m.state_dict())
    t = copy.deepcopy(m)
    for p in t.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    loss = value_update_step(m, t, opt, s, s_next, g, success, done,
                             gamma=0.99, expectile=0.7, ema=0.005,
                             value_loss_mode="hiql", value_mask_mode="hiql")
    assert np.isfinite(loss)
    changed = any(not torch.equal(before[k], m.state_dict()[k]) for k in before)
    assert changed
