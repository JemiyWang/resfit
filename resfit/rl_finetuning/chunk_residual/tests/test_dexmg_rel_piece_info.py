"""③a' object-aware:dexmg 把 rel_piece 经 info 透出(不进 observation.state)。

关键架构约束(handoff §3):rel_piece 是特权 Φ 信息,observation.state 必须保持 18 维
(actor/critic + 部署不变),rel_piece 像 stage_id 一样经 info 透给 V。
"""
import numpy as np
import pytest

from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper
from resfit.rl_finetuning.chunk_residual.object_state import compute_eef_rel_piece_from_env


class _FakeSim:
    def __init__(self, body_xpos):
        class _Data:
            def get_body_xpos(_self, name):
                return np.asarray(body_xpos[name], dtype=np.float64)
        self.data = _Data()


class _FakeRobosuiteEnv:
    """最小 TwoArm 仿真 stub:eef(sim 实时) + piece(sim) + step/reset 出 1 个 lowdim key。"""
    def __init__(self):
        self._eef0_xpos = np.array([1., 0., 0.])
        self._eef1_xpos = np.array([0., 2., 0.])
        self.sim = _FakeSim({"piece_1_root": [0., 0., 0.], "piece_2_root": [0., 0., 1.]})

    def _obs(self):
        return {"robot0_eef_pos": np.zeros(3, dtype=np.float32)}

    def reset(self):
        return self._obs()

    def step(self, action):
        return self._obs(), 0.0, False, {}


def _make_wrapper(state_mode):
    w = RobosuiteGymWrapper.__new__(RobosuiteGymWrapper)
    w.env = _FakeRobosuiteEnv()
    w.state_mode = state_mode
    w.env_name = "FakeTwoArm"
    w._stage_det = None                       # 无检测器 → 跳过 stage_id 分支
    w.episode_steps = 0
    w.expected_image_keys = []
    w._get_expected_low_dim_keys = lambda name: ["robot0_eef_pos"]   # 3 维本体
    return w


def test_observation_state_stays_lowdim_in_eef_piece():
    """eef_piece 模式 observation.state 不被拼大(仍是本体 3 维),特权不泄漏给策略。"""
    w = _make_wrapper("eef_piece")
    obs, info = w.reset()
    assert obs["observation.state"].shape == (3,)      # 本体维度,NOT 3+12
    obs2, *_ , info2 = w.step(np.zeros(7))
    assert obs2["observation.state"].shape == (3,)


def test_reset_emits_rel_piece_in_info_eef_piece():
    w = _make_wrapper("eef_piece")
    _, info = w.reset()
    assert "rel_piece" in info
    rel = np.asarray(info["rel_piece"])
    assert rel.shape == (12,)
    expected = compute_eef_rel_piece_from_env(w.env)
    np.testing.assert_allclose(rel, expected)


def test_step_emits_rel_piece_in_info_eef_piece():
    w = _make_wrapper("eef_piece")
    w.reset()
    _, _, _, _, info = w.step(np.zeros(7))
    assert "rel_piece" in info
    rel = np.asarray(info["rel_piece"])
    assert rel.shape == (12,)
    np.testing.assert_allclose(rel, compute_eef_rel_piece_from_env(w.env))


def test_eef_mode_no_rel_piece_in_info():
    """默认 eef 模式:不算也不透出 rel_piece(零回归、零额外 sim 开销)。"""
    w = _make_wrapper("eef")
    _, info_r = w.reset()
    _, _, _, _, info_s = w.step(np.zeros(7))
    assert "rel_piece" not in info_r
    assert "rel_piece" not in info_s
