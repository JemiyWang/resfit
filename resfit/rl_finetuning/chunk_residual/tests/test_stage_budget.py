import torch

from resfit.rl_finetuning.off_policy.rl.stage_utils import (
    stage_budget_factor, parse_stage_budget,
)

NUM_STAGES = 5
BUDGET = [1.0, 1.0, 1.0, 0.3, 0.1]


def test_stage_budget_factor_mixed():
    budget = torch.tensor(BUDGET)
    sid = torch.tensor([[0.], [3.], [4.], [1.]])
    f = stage_budget_factor(sid, budget, NUM_STAGES)
    assert f.shape == (4, 1)
    assert torch.allclose(f.reshape(-1), torch.tensor([1.0, 0.3, 0.1, 1.0]))


def test_stage_budget_factor_clamps_out_of_range():
    budget = torch.tensor([1.0, 0.5, 0.2])
    sid = torch.tensor([[5.], [-1.]])          # 越界 -> clamp 到 idx 2 和 0
    f = stage_budget_factor(sid, budget, 3)
    assert torch.allclose(f.reshape(-1), torch.tensor([0.2, 1.0]))


def test_parse_stage_budget_none():
    assert parse_stage_budget(None, NUM_STAGES) is None


def test_parse_stage_budget_ok():
    assert parse_stage_budget("1,1,1,0.3,0.1", NUM_STAGES) == [1.0, 1.0, 1.0, 0.3, 0.1]


def test_parse_stage_budget_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        parse_stage_budget("1,1,0.1", NUM_STAGES)   # 长度 3 != 5
