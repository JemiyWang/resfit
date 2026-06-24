import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _make_eef_subgoal():
    gc = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=32)
    ha = HighActor(state_dim=30, rep_dim=10, hidden=32)
    goal = np.zeros(30, dtype=np.float32)
    return HiqlSubgoal(gc, ha, goal, device="cpu", renorm_subgoal=True,
                       state_mode="eef_piece",
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)))


def test_encode_state_eef_matches_build_state30():
    sg = _make_eef_subgoal()
    state18 = torch.randn(3, 18)
    rel12 = np.random.RandomState(0).randn(3, 12).astype(np.float32)
    s = sg.encode_state(state18, rel12)
    assert s.shape == (3, 30)
    assert torch.allclose(s, sg.build_state30(state18, rel12), atol=1e-6)


def test_subgoal_online_unchanged_by_refactor():
    """encode_state 抽取后,subgoal_online 的 z 与"自己用 encode_state 重算"逐位一致。"""
    sg = _make_eef_subgoal()
    state18 = torch.randn(4, 18)
    rel12 = np.random.RandomState(1).randn(4, 12).astype(np.float32)
    z = sg.subgoal_online(state18, rel12)
    s = sg.encode_state(state18, rel12)
    g = sg.goal.unsqueeze(0).expand(s.shape[0], -1)
    z_ref = sg.ha(s, g).mean
    z_ref = z_ref / (z_ref.norm(dim=-1, keepdim=True) + 1e-8) * (sg.rep_dim ** 0.5)
    assert torch.allclose(z, z_ref, atol=1e-6)
