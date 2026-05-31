import gymnasium as gym
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper

D, L = 4, 5  # 小尺寸便于测试


class _FakeVecEnv:
    """单环境、确定性、第 3 步后 terminate 的假向量环境。"""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(D,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-np.inf, np.inf, (1, 3), dtype=np.float32),
            "observation.images.cam": gym.spaces.Box(0, 1, (1, 3, 4, 4), dtype=np.float32),
        })
        self._t = 0
        self.last_actions = []

    def _obs(self):
        return {
            "observation.state": torch.zeros(1, 3),
            "observation.images.cam": torch.zeros(1, 3, 4, 4),
        }

    def reset(self, **kw):
        self._t = 0
        self.last_actions = []
        return self._obs(), {}

    def step(self, action):
        self.last_actions.append(action.clone())
        self._t += 1
        terminated = torch.tensor([self._t >= 3])
        truncated = torch.tensor([False])
        reward = torch.tensor([1.0 if self._t == 3 else 0.0])
        info = {}
        return self._obs(), reward, terminated, truncated, info


class _FakeBase:
    """返回固定 chunk(原始尺度)的假 ACT。"""
    class _Cfg:
        image_features = {"observation.images.cam": None}
    config = _Cfg()

    def reset(self, env_ids=None):
        pass

    def get_action_chunk(self, raw_obs, chunk_length):
        b = raw_obs["observation.state"].shape[0]
        # 原始尺度 chunk:每步全 0.5
        return torch.full((b, chunk_length, D), 0.5)


class _IdentityScaler:
    def scale(self, a):       # 原样(测试里不验证归一化数值)
        return torch.clamp(a, -1, 1)
    def unscale(self, a):
        return a


class _IdentityStd:
    def standardize(self, s):
        return s


def test_step_executes_chunk_and_accumulates_reward():
    env = _FakeVecEnv()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=L)
    obs, _ = w.reset()
    assert obs["observation.base_action"].shape == (1, L * D)   # 展平 chunk
    residual = torch.zeros(1, L * D)
    next_obs, reward, terminated, truncated, info = w.step(residual)
    # fake env 第 3 步 terminate → 只执行了 3 步
    assert len(env.last_actions) == 3
    assert terminated.item() is True
    assert reward.item() == 1.0                                 # 累积到成功
    assert info["scaled_action"].shape == (1, L * D)            # combined chunk
    assert next_obs["observation.base_action"].shape == (1, L * D)


def test_residual_is_added_to_base_before_unscale():
    env = _FakeVecEnv()
    scaler = _IdentityScaler()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), scaler, _IdentityStd(), chunk_length=L)
    w.reset()
    residual = torch.full((1, L * D), 0.1)
    w.step(residual)
    # base=0.5(scale 后仍 0.5),+0.1 → 0.6,unscale identity → env 收到 ~0.6
    assert torch.allclose(env.last_actions[0], torch.full((1, D), 0.6), atol=1e-5)


class _FakeVecEnvNoTerm:
    """永不终止的单环境,用于测试整段 chunk 全执行。"""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(D,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-np.inf, np.inf, (1, 3), dtype=np.float32),
            "observation.images.cam": gym.spaces.Box(0, 1, (1, 3, 4, 4), dtype=np.float32),
        })
        self.steps = 0

    def _obs(self):
        return {"observation.state": torch.zeros(1, 3),
                "observation.images.cam": torch.zeros(1, 3, 4, 4)}

    def reset(self, **kw):
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        return self._obs(), torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {}


class _ResetTrackingBase(_FakeBase):
    def __init__(self):
        self.reset_calls = 0

    def reset(self, env_ids=None):
        self.reset_calls += 1


def test_full_chunk_executes_all_steps_without_termination():
    env = _FakeVecEnvNoTerm()
    base = _ResetTrackingBase()
    w = ChunkResidualEnvWrapper(env, base, _IdentityScaler(), _IdentityStd(), chunk_length=L)
    w.reset()
    reset_after_reset = base.reset_calls
    _, reward, terminated, truncated, info = w.step(torch.zeros(1, L * D))
    assert env.steps == L                          # 全部 L 步执行
    assert terminated.item() is False and truncated.item() is False
    assert reward.item() == 0.0
    assert "final_obs" not in info                 # 不再透出 final_obs
    assert base.reset_calls == reset_after_reset   # 未终止 → 不额外 reset base


def test_step_before_reset_raises():
    env = _FakeVecEnv()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(), chunk_length=L)
    import pytest
    with pytest.raises(RuntimeError):
        w.step(torch.zeros(1, L * D))
