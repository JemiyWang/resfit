import numpy as np
import pytest
import torch

from resfit.rl_finetuning.scripts.eval_pi05_base import check_action, run_smoke


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


class _FakeEnv:
    """Vec env returning fixed obs; done after `done_after` steps."""
    def __init__(self, done_after=3):
        self._done_after = done_after
        self._t = 0
        self._obs = {
            "observation.state": torch.zeros((1, 18)),
            "observation.images.agentview": torch.zeros((1, 3, 84, 84)),
        }

    def reset(self, **kwargs):
        self._t = 0
        return self._obs, {}

    def step(self, action):
        self._t += 1
        term = torch.tensor([self._t >= self._done_after])
        trunc = torch.tensor([False])
        return self._obs, torch.tensor([0.0]), term, trunc, {}


class _FakePolicy:
    def __init__(self, action):
        self._a = action
        self.reset_calls = 0

    def reset(self, env_ids=None):
        self.reset_calls += 1

    def select_action(self, obs):
        return self._a


def test_run_smoke_runs_all_episodes():
    pol = _FakePolicy(torch.zeros((1, 14)))
    env = _FakeEnv(done_after=3)
    report = run_smoke(env, pol, n_episodes=2, max_steps=10, action_dim=14)
    assert len(report["episodes"]) == 2
    assert report["episodes"] == [3, 3]            # ends on terminated, not max_steps
    assert len(report["infer_times"]) == 6         # 2 episodes * 3 steps
    assert report["action_min"] == 0.0
    assert report["action_max"] == 0.0
    assert pol.reset_calls >= 2                     # reset once per episode


def test_run_smoke_propagates_bad_action():
    pol = _FakePolicy(torch.full((1, 14), 99.0))   # out of range
    env = _FakeEnv(done_after=3)
    with pytest.raises(ValueError, match="exceeds limit"):
        run_smoke(env, pol, n_episodes=1, max_steps=10, action_dim=14)


def test_run_smoke_stops_at_max_steps():
    pol = _FakePolicy(torch.zeros((1, 14)))
    env = _FakeEnv(done_after=1000)                # never done
    report = run_smoke(env, pol, n_episodes=1, max_steps=5, action_dim=14)
    assert report["episodes"] == [5]
