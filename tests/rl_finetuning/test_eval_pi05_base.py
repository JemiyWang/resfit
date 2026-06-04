import numpy as np
import pytest

from resfit.rl_finetuning.scripts.eval_pi05_base import check_action


def test_check_action_accepts_valid():
    a = np.zeros((1, 14), dtype=np.float32)
    check_action(a, action_dim=14)  # should not raise


def test_check_action_rejects_wrong_dim():
    a = np.zeros((1, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="last dim"):
        check_action(a, action_dim=14)


def test_check_action_rejects_non_2d():
    a = np.zeros((14,), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        check_action(a, action_dim=14)


def test_check_action_rejects_nan():
    a = np.zeros((1, 14), dtype=np.float32)
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        check_action(a, action_dim=14)


def test_check_action_rejects_inf():
    a = np.zeros((1, 14), dtype=np.float32)
    a[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        check_action(a, action_dim=14)


def test_check_action_rejects_out_of_range():
    a = np.full((1, 14), 99.0, dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds limit"):
        check_action(a, action_dim=14)


def test_check_action_accepts_torch_tensor():
    import torch
    a = torch.zeros((1, 14), dtype=torch.float32)
    check_action(a, action_dim=14)  # should not raise
