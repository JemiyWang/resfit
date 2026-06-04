from collections import deque

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
        self.last_actions = []

    def _obs(self):
        return {"observation.state": torch.zeros(1, 3),
                "observation.images.cam": torch.zeros(1, 3, 4, 4)}

    def reset(self, **kw):
        self.steps = 0
        self.last_actions = []
        return self._obs(), {}

    def step(self, action):
        self.last_actions.append(action.clone())
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


# ---------------------------------------------------------------------------
# Queue mode fixtures and tests
# ---------------------------------------------------------------------------


class _FakeQueueBase(_FakeBase):
    """模拟 ACT select_action 的内部 queue:每 PLAN_LEN 步重规划一次,每次 pop 一个。
    每次重规划用递增的 plan_id 填充(整段值=plan_id),便于断言'同一规划顺序取'。
    继承 _FakeBase 拿到 config / get_action_chunk(queue 模式不会用到后者)。"""
    PLAN_LEN = 2

    def __init__(self):
        self.queue = deque()
        self.model_calls = 0     # 重规划次数(= 跑底层 model 的次数)
        self.reset_calls = 0
        self.plan_id = 0

    def reset(self, env_ids=None):
        self.queue.clear()
        self.reset_calls += 1

    def select_action(self, raw_obs):
        b = raw_obs["observation.state"].shape[0]
        if len(self.queue) == 0:
            self.model_calls += 1
            self.plan_id += 1
            for _ in range(self.PLAN_LEN):
                self.queue.append(torch.full((b, D), self.plan_id * 0.3))
        return self.queue.popleft()


class _FakeVecEnvTermAt1:
    """第 1 步即 terminate 的单环境(配 chunk_length=1 测 done 清队)。"""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(D,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-np.inf, np.inf, (1, 3), dtype=np.float32),
            "observation.images.cam": gym.spaces.Box(0, 1, (1, 3, 4, 4), dtype=np.float32),
        })

    def _obs(self):
        return {"observation.state": torch.zeros(1, 3),
                "observation.images.cam": torch.zeros(1, 3, 4, 4)}

    def reset(self, **kw):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), torch.tensor([1.0]), torch.tensor([True]), torch.tensor([False]), {}


# --- construction validation ---

def test_queue_mode_requires_chunk_length_one():
    import pytest
    env = _FakeVecEnv()
    with pytest.raises(AssertionError):
        ChunkResidualEnvWrapper(env, _FakeQueueBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=2, base_action_mode="queue")


def test_queue_mode_constructs_at_cl1():
    env = _FakeVecEnv()
    w = ChunkResidualEnvWrapper(env, _FakeQueueBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, base_action_mode="queue")
    assert w.base_action_mode == "queue"
    assert w.flat_dim == D  # cl=1: flat_dim = chunk_length * action_dim = 1 * D


# --- queue consumption & model reuse ---

def test_queue_mode_uses_select_action_and_sequential_pop():
    env = _FakeVecEnvNoTerm()
    base = _FakeQueueBase()
    w = ChunkResidualEnvWrapper(env, base, _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, base_action_mode="queue")
    obs, _ = w.reset()
    assert obs["observation.base_action"].shape == (1, D)
    assert torch.allclose(obs["observation.base_action"], torch.full((1, D), 0.3))
    assert base.model_calls == 1

    next_obs, *_ = w.step(torch.zeros(1, D))
    assert torch.allclose(next_obs["observation.base_action"], torch.full((1, D), 0.3))
    assert base.model_calls == 1

    next_obs2, *_ = w.step(torch.zeros(1, D))
    assert torch.allclose(next_obs2["observation.base_action"], torch.full((1, D), 0.6))
    assert base.model_calls == 2


def test_queue_mode_clears_queue_on_done():
    env = _FakeVecEnvTermAt1()
    base = _FakeQueueBase()
    w = ChunkResidualEnvWrapper(env, base, _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, base_action_mode="queue")
    w.reset()
    resets_before = base.reset_calls
    _, _, terminated, _, _ = w.step(torch.zeros(1, D))
    assert terminated.item() is True
    assert base.reset_calls == resets_before + 1


# --- clamp behaviour ---

def test_queue_mode_does_not_clamp_combined():
    env = _FakeVecEnvNoTerm()
    w = ChunkResidualEnvWrapper(env, _FakeQueueBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, base_action_mode="queue")
    w.reset()
    # base raw = 0.3 (plan_id=1 * 0.3) → scaled 0.3; residual 0.8; combined = 1.1 (no clamp in queue mode)
    w.step(torch.full((1, D), 0.8))
    assert torch.allclose(env.last_actions[0], torch.full((1, D), 1.1), atol=1e-5)


def test_replan_mode_still_clamps_combined():
    env = _FakeVecEnvNoTerm()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=L)
    w.reset()
    w.step(torch.full((1, L * D), 0.8))
    assert torch.allclose(env.last_actions[0], torch.full((1, D), 1.0), atol=1e-5)


# --- stage_id 解耦(handoff §2):stage_id 用瞬时,reward/闩锁不动 ---

class _StageScriptEnv(_FakeVecEnvNoTerm):
    """每步经 info 透出脚本化瞬时 stage,用于测 stage_id 解耦。"""
    def __init__(self, stage_seq):
        super().__init__()
        self._seq = stage_seq
        self._i = 0

    def reset(self, **kw):
        self._i = 0
        return super().reset(**kw)

    def step(self, action):
        obs, r, term, trunc, _ = super().step(action)
        s = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return obs, r, term, trunc, {"stage_id": np.array([s])}


def test_emitted_stage_id_is_instantaneous_not_latch():
    # 瞬时先到 3(闩锁=3),再掉回 2(闩锁仍 3)。stage_id 应跟瞬时,闩锁另存供 reward。
    env = _StageScriptEnv([3, 2])
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=1)
    w.reset()
    obs1, *_ = w.step(torch.zeros(1, D))
    assert obs1["observation.stage_id"].item() == 3.0      # 瞬时 3
    obs2, *_ = w.step(torch.zeros(1, D))
    assert obs2["observation.stage_id"].item() == 2.0      # 解耦:用瞬时 2,而非闩锁 3
    assert w._stage == 3                                    # 闩锁保持(reward 用)
