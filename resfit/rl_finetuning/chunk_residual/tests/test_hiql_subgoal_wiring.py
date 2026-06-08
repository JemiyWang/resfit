import numpy as np
import torch


def test_append_subgoal():
    from resfit.rl_finetuning.off_policy.rl.stage_utils import append_subgoal
    prop = torch.zeros(4, 18)
    z = torch.ones(4, 10)
    out = append_subgoal(prop, z)
    assert out.shape == (4, 28)
    assert torch.equal(out[:, 18:], z)
    assert out.device == prop.device
