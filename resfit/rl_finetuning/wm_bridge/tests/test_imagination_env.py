import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper
from resfit.rl_finetuning.wm_bridge.imagination_env import ImaginationVecEnv
from resfit.rl_finetuning.wm_bridge.init_states import InitStateSampler
from resfit.rl_finetuning.wm_bridge.wm_driver import (
    ACTION_DIM, CAMERA_KEYS, CHUNK_LENGTH, ActionNormalizer,
)


class _StubEpisode:
    n_frames = 50

    def read(self, frame_idx):
        return (np.zeros((3, 3, 192, 256), np.float32),
                np.zeros(16, np.float32))


class _StubWM:
    def __init__(self):
        self.n = 0

    def infer(self, obs=None, act_tokens=None, prompt="", **kw):
        self.n += 1
        assert act_tokens is not None, "永远不可走 act_tokens=None 兜底路径(坑4)"
        assert tuple(act_tokens.shape) == (1, 25, 30)
        return {"video": torch.zeros(3, 3, 29, 192, 256)}


class _StubBase:
    def __init__(self):
        self.n = 0

    def reset(self):
        pass

    def query(self, raw_obs):
        self.n += 1
        return (np.zeros((CHUNK_LENGTH, ACTION_DIM), np.float32),
                np.full(8, float(self.n), np.float32))

    def get_action_chunk(self, raw_obs, chunk_length):
        acts, _ = self.query(raw_obs)
        return torch.from_numpy(acts).unsqueeze(0)


class _CountingScorer:
    """Φ = ψ 的第 0 维,便于精确验算 PBRS。"""

    expected_psi_anchor = None

    def phi(self, psi, proprio):
        return float(np.asarray(psi).reshape(-1)[0])


def _env(**kw):
    return ImaginationVecEnv(
        wm=kw.pop("wm", _StubWM()),
        base=kw.pop("base", _StubBase()),
        scorer=kw.pop("scorer", _CountingScorer()),
        sampler=InitStateSampler([_StubEpisode()], rng=np.random.default_rng(0)),
        normalizer=ActionNormalizer(-np.ones(16, np.float32), np.ones(16, np.float32)),
        **kw)


def _act():
    return torch.zeros(1, ACTION_DIM)


def test_reset_returns_required_obs_keys():
    env = _env()
    obs, info = env.reset()
    for k in CAMERA_KEYS:
        assert obs[k].shape == (1, 3, 84, 84)
    assert obs["observation.state"].shape == (1, 16)
    assert "_wm_native_frames" in obs and "_wm_window_token" in obs


def test_first_49_steps_are_noops():
    env = _env()
    wm = env.wm
    env.reset()
    for i in range(CHUNK_LENGTH - 1):
        obs, r, term, trunc, info = env.step(_act())
        assert float(r.reshape(-1)[0]) == 0.0
        assert not bool(term.reshape(-1)[0]) and not bool(trunc.reshape(-1)[0])
    assert wm.n == 0                       # ★ 前 49 步一次 WM 都没调


def test_wm_fires_exactly_once_on_fiftieth_step():
    env = _env()
    wm = env.wm
    env.reset()
    for _ in range(CHUNK_LENGTH):
        env.step(_act())
    assert wm.n == 1


def test_truncated_only_after_two_chunks():
    env = _env()
    env.reset()
    for _ in range(CHUNK_LENGTH):
        _, _, _, trunc, _ = env.step(_act())
    assert not bool(trunc.reshape(-1)[0])          # 第 1 段末不截断
    for _ in range(CHUNK_LENGTH):
        _, _, term, trunc, _ = env.step(_act())
    assert bool(trunc.reshape(-1)[0])              # 第 2 段末截断
    assert not bool(term.reshape(-1)[0])           # terminated 恒 False


def test_pbrs_reward_is_gamma_phi_next_minus_phi_prev():
    """_StubBase 的 ψ[0] 依次是 1,2,3... → Φ 依次 1,2,3。"""
    env = _env(gamma=0.9)
    env.reset()                                    # 基座第 1 次调用 → Φ_prev = 1
    r = None
    for _ in range(CHUNK_LENGTH):
        _, r, _, _, _ = env.step(_act())
    # 第 2 次基座调用 → Φ_next = 2;reward = 0.9*2 - 1 = 0.8
    assert float(r.reshape(-1)[0]) == pytest.approx(0.8, abs=1e-6)


def test_phi_is_not_zeroed_at_truncation():
    """★ 与仓库既有 potential_shaping 约定相反:truncated 时 Φ 不置零。

    若置零,第二段 reward 会变成 -Φ_prev(负数);不置零则是 gamma*Φ_next - Φ_prev。
    """
    env = _env(gamma=0.9)
    env.reset()
    for _ in range(CHUNK_LENGTH):
        env.step(_act())                           # 第 1 段:Φ 1→2
    r = None
    for _ in range(CHUNK_LENGTH):
        _, r, _, trunc, _ = env.step(_act())       # 第 2 段:Φ 2→3,且 truncated
    assert bool(trunc.reshape(-1)[0])
    assert float(r.reshape(-1)[0]) == pytest.approx(0.9 * 3 - 2, abs=1e-6)
    assert float(r.reshape(-1)[0]) > 0             # 置零的话会是 -2


def test_two_segment_return_telescopes():
    """r_0 + gamma*r_1 == gamma^2*Phi_2 - Phi_0。"""
    g = 0.9
    env = _env(gamma=g)
    env.reset()                                    # Phi_0 = 1
    r0 = r1 = None
    for _ in range(CHUNK_LENGTH):
        _, r0, _, _, _ = env.step(_act())          # Phi_1 = 2
    for _ in range(CHUNK_LENGTH):
        _, r1, _, _, _ = env.step(_act())          # Phi_2 = 3
    total = float(r0.reshape(-1)[0]) + g * float(r1.reshape(-1)[0])
    assert total == pytest.approx(g * g * 3 - 1, abs=1e-6)


def test_serve_called_once_per_chunk():
    base = _StubBase()
    env = _env(base=base)
    env.reset()                                    # 1 次(起点)
    for _ in range(CHUNK_LENGTH):
        env.step(_act())
    assert base.n == 2                             # 起点 1 次 + 本段末 1 次


def test_num_envs_is_one_and_action_space_dim():
    env = _env()
    assert env.num_envs == 1
    assert env.action_space.shape[-1] == ACTION_DIM


def test_window_token_advances_after_each_chunk():
    env = _env()
    obs, _ = env.reset()
    t0 = obs["_wm_window_token"]
    for _ in range(CHUNK_LENGTH):
        obs, _, _, _, _ = env.step(_act())
    assert obs["_wm_window_token"] != t0


class _IdentityScaler:
    def scale(self, action):
        return action

    def unscale(self, action):
        return action


class _IdentityStandardizer:
    def standardize(self, state):
        return state


def test_chunk_wrapper_exposes_one_transition_per_wm_call():
    """Actor 一次输出 50x16 residual，wrapper 内部攒满后才向 trainer 返回。"""
    wm = _StubWM()
    base = _StubBase()
    inner = _env(wm=wm, base=base, gamma=0.995)
    wrapped = ChunkResidualEnvWrapper(
        inner, base, _IdentityScaler(), _IdentityStandardizer(),
        chunk_length=CHUNK_LENGTH, base_action_mode="replan",
        reward_shaping_mode="none", gamma=0.995,
    )

    obs, _ = wrapped.reset()
    assert obs["observation.base_action"].shape == (1, CHUNK_LENGTH * ACTION_DIM)

    next_obs, reward, terminated, truncated, info = wrapped.step(
        torch.zeros(1, CHUNK_LENGTH * ACTION_DIM))

    assert wm.n == 1
    assert next_obs["_wm_window_token"] != obs["_wm_window_token"]
    assert info["scaled_action"].shape == (1, CHUNK_LENGTH * ACTION_DIM)
    assert reward.shape == (1,)
    assert not terminated.item() and not truncated.item()
