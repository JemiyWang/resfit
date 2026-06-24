import copy
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
    HighActor, awr_weight, high_actor_update_step)


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _manual_min_step(ha, vf, opt, s, sw, g, *, beta):
    with torch.no_grad():
        vs1, vs2 = vf(s, g)
        vw1, vw2 = vf(sw, g)
        adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
        w = awr_weight(adv, beta)
        z_tgt = vf.phi(s, sw)
    dist = ha(s, g)
    logp = dist.log_prob(z_tgt).sum(-1)
    loss = -(w * logp).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss)


def test_high_actor_update_step_matches_manual_inline():
    torch.manual_seed(0)
    vf = GoalConditionedVF(6, rep_dim=4, hidden=16)
    for p in vf.parameters():
        p.requires_grad_(False)
    s, sw, g = torch.randn(8, 6), torch.randn(8, 6), torch.randn(8, 6)

    ha_a = HighActor(6, rep_dim=4, hidden=16)
    ha_b = copy.deepcopy(ha_a)
    opt_a = torch.optim.Adam(ha_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(ha_b.parameters(), lr=1e-3)

    la = _manual_min_step(ha_a, vf, opt_a, s, sw, g, beta=1.0)
    lb = high_actor_update_step(ha_b, vf, opt_b, s, sw, g, beta=1.0, adv_agg="min")
    assert abs(la - lb) < 1e-9
    for k, v in ha_a.state_dict().items():
        assert torch.equal(v, ha_b.state_dict()[k]), f"high_actor param 不等: {k}"


def test_high_actor_update_step_mean_agg_runs():
    torch.manual_seed(2)
    vf = GoalConditionedVF(4, rep_dim=4, hidden=16)
    for p in vf.parameters():
        p.requires_grad_(False)
    ha = HighActor(4, rep_dim=4, hidden=16)
    before = copy.deepcopy(ha.state_dict())
    opt = torch.optim.Adam(ha.parameters(), lr=1e-2)
    s, sw, g = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    loss = high_actor_update_step(ha, vf, opt, s, sw, g, beta=1.0, adv_agg="mean")
    assert np.isfinite(loss)
    assert any(not torch.equal(before[k], ha.state_dict()[k]) for k in before)
    with pytest.raises(ValueError):
        high_actor_update_step(ha, vf, opt, s, sw, g, beta=1.0, adv_agg="bogus")
