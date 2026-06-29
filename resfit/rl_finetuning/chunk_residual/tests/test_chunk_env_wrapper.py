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


class _StateRampEnv:
    """obs.state 每步递增(不 terminate),便于区分 phi(start)/phi(next) 并验证 start 推进。"""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(D,), dtype=np.float32)
        self._k = 0

    def _obs(self):
        return {"observation.state": torch.full((1, 3), float(self._k)),
                "observation.images.cam": torch.zeros(1, 3, 4, 4)}

    def reset(self, **kw):
        self._k = 0
        return self._obs(), {}

    def step(self, action):
        self._k += 1
        return self._obs(), torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {}


class _StubPotential:
    """phi(state[B,3]) = 每行第一元素(= ramp 的 _k),模拟 V(state)。eef 模式 rel 被忽略。"""
    def phi(self, state_std, rel_piece_raw=None):
        return state_std[:, 0]


def test_step_hiql_potential_uses_v_and_advances_start():
    from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping
    env = _StateRampEnv()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, reward_shaping_mode="potential",
                                stage_reward_bonus=1.0, gamma=0.99, potential=_StubPotential())
    w.reset()                                  # start state _k=0 -> phi 0
    _, r1, *_ = w.step(torch.zeros(1, D))      # end _k=1 -> phi 1;env reward=0
    assert abs(float(r1) - potential_shaping(0.0, 1.0, bonus=1.0, gamma=0.99, done=False)) < 1e-5
    _, r2, *_ = w.step(torch.zeros(1, D))      # start 应推进到上次 end(_k=1),end=_k=2
    assert abs(float(r2) - potential_shaping(1.0, 2.0, bonus=1.0, gamma=0.99, done=False)) < 1e-5


class _StateRelRampEnv:
    """obs.state 与 info['rel_piece'] 同步递增。rel 经 info 透出(模拟 AsyncVectorEnv 批后 (1,12))。"""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(D,), dtype=np.float32)
        self._k = 0

    def _obs(self):
        return {"observation.state": torch.full((1, 3), float(self._k)),
                "observation.images.cam": torch.zeros(1, 3, 4, 4)}

    def _info(self):
        rel = np.zeros((1, 12), dtype=np.float32)   # (num_envs=1, 12)
        rel[0, 0] = 10.0 * self._k                  # rel 第一列 = 10*_k,便于与 state 区分
        return {"rel_piece": rel}

    def reset(self, **kw):
        self._k = 0
        return self._obs(), self._info()

    def step(self, action):
        self._k += 1
        return (self._obs(), torch.tensor([0.0]), torch.tensor([False]),
                torch.tensor([False]), self._info())


class _RelAwareStubPot:
    """phi = state第一列 + rel第一列;state_mode=eef_piece。rel 不透传则对不上。"""
    state_mode = "eef_piece"

    def phi(self, state_std, rel_piece_raw=None):
        base = state_std[:, 0]
        if rel_piece_raw is not None:
            base = base + float(np.asarray(rel_piece_raw).reshape(-1)[0])
        return base


def test_step_hiql_potential_eef_piece_threads_rel_from_info():
    """eef_piece:online 从 info['rel_piece'] 取 raw rel 喂 Φ,且 start_rel 跨 chunk 携带。"""
    from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping
    env = _StateRelRampEnv()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, reward_shaping_mode="potential",
                                stage_reward_bonus=1.0, gamma=0.99, potential=_RelAwareStubPot())
    w.reset()                                  # start: _k=0 -> phi = 0(state)+0(rel) = 0
    _, r1, *_ = w.step(torch.zeros(1, D))      # end: _k=1 -> phi = 1 + 10 = 11
    assert abs(float(r1) - potential_shaping(0.0, 11.0, bonus=1.0, gamma=0.99, done=False)) < 1e-5
    _, r2, *_ = w.step(torch.zeros(1, D))      # start 推进到 _k=1(phi 11),end _k=2(phi 2+20=22)
    assert abs(float(r2) - potential_shaping(11.0, 22.0, bonus=1.0, gamma=0.99, done=False)) < 1e-5


class _ActFeatPotential:
    state_mode = "act_feat"

    def __init__(self):
        self.inputs = []

    def phi(self, state_std, rel_piece_raw=None):
        assert rel_piece_raw is None
        x = torch.as_tensor(state_std, dtype=torch.float32)
        self.inputs.append(x.clone())
        return x[:, 0]


class _SequentialFeatureEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, raw_obs):
        self.calls += 1
        return torch.tensor([[float(self.calls), 99.0]])


def test_actfeat_potential_uses_feature_encoder_not_lowdim_state():
    env = _FakeVecEnvNoTerm()
    pot = _ActFeatPotential()
    enc = _SequentialFeatureEncoder()
    w = ChunkResidualEnvWrapper(
        env,
        _FakeBase(),
        _IdentityScaler(),
        _IdentityStd(),
        chunk_length=1,
        stage_reward_bonus=1.0,
        reward_shaping_mode="potential",
        gamma=0.5,
        potential=pot,
        potential_feature_encoder=enc,
    )
    w.reset()
    _, reward, _, _, _ = w.step(torch.zeros(1, D))

    assert enc.calls == 3
    assert torch.allclose(pot.inputs[0], torch.tensor([[1.0, 99.0]]))
    assert torch.allclose(pot.inputs[1], torch.tensor([[2.0, 99.0]]))
    assert reward.item() == 0.0
