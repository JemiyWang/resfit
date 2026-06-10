"""stage 检测器排序契约 + chunk wrapper 的 stage 闩锁/归零行为。

检测器排序用 fake env(纯逻辑);wrapper 行为用会发 info["stage_id"] 的 fake vec env。
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper
from resfit.rl_finetuning.chunk_residual.stage_detectors import (
    NUM_STAGES,
    get_stage_detector,
    threading_stage,
    threepiece_stage,
)

D, L = 4, 5


# ---------- 检测器排序契约 ----------
class _FakeGripper:
    pass


class _FakeRobot:
    def __init__(self):
        self.gripper = [_FakeGripper()]


class _FakePiece:                 # piece_1
    contact_geoms = ["p1g0", "p1g1"]


class _FakePiece2:                # piece_2
    contact_geoms = ["p2g0", "p2g1"]


def _is_piece2(object_geoms):
    return "p2g0" in object_geoms


class _FakeTPEnv:
    """list-gripper 变体(非 dict);grasp=piece1, grasp2=piece2。"""
    def __init__(self, second=False, first=False, grasp=False, grasp2=False):
        self._second, self._first = second, first
        self._grasp, self._grasp2 = grasp, grasp2
        self.robots = [_FakeRobot()]
        self.piece_1 = _FakePiece()
        self.piece_2 = _FakePiece2()

    def _check_second_piece_is_assembled(self):
        return self._second

    def _check_first_piece_is_assembled(self):
        return self._first

    def _check_grasp(self, gripper, object_geoms):
        return self._grasp2 if _is_piece2(object_geoms) else self._grasp


def test_threepiece_stage_success_is_4():
    assert threepiece_stage(_FakeTPEnv(second=True)) == 4


def test_threepiece_stage_first_assembled_is_2():
    assert threepiece_stage(_FakeTPEnv(first=True)) == 2


def test_threepiece_stage_grasped_is_1():
    # 抓起 piece1 但尚未装好 → 应为阶段 1
    assert threepiece_stage(_FakeTPEnv(grasp=True)) == 1


def test_threepiece_stage_start_is_0():
    assert threepiece_stage(_FakeTPEnv()) == 0


def test_threepiece_stage_priority_success_over_lower():
    # 同时满足时,高阶段优先
    assert threepiece_stage(_FakeTPEnv(second=True, first=True, grasp=True)) == 4


def test_registry_unknown_task_returns_none():
    assert get_stage_detector("NoSuchTask") is None
    assert get_stage_detector("TwoArmThreePieceAssembly") is threepiece_stage


# ---------- 真环境建模:robot.gripper 是 dict(迭代得 str 键),_check_grasp 只认对象 ----------
class _GripperObj:
    pass


class _DictGripperRobot:
    def __init__(self):
        self.gripper = {"right": _GripperObj(), "left": _GripperObj()}  # 迭代得键(str)


class _RealisticTPEnv:
    """真环境:gripper=dict(迭代得 str 键);_check_grasp 收到 str 键恒 False。"""
    def __init__(self, second=False, first=False, grasp=False, grasp2=False):
        self._second, self._first = second, first
        self._grasp, self._grasp2 = grasp, grasp2
        self.robots = [_DictGripperRobot(), _DictGripperRobot()]
        self.piece_1 = _FakePiece()
        self.piece_2 = _FakePiece2()

    def _check_second_piece_is_assembled(self):
        return self._second

    def _check_first_piece_is_assembled(self):
        return self._first

    def _check_grasp(self, gripper, object_geoms):
        if isinstance(gripper, str):
            return False
        return self._grasp2 if _is_piece2(object_geoms) else self._grasp


def test_threepiece_stage_grasped_with_dict_gripper_is_1():
    # 真环境 gripper=dict;抓住 piece1 未装好 → 应为 1
    assert threepiece_stage(_RealisticTPEnv(grasp=True)) == 1


def test_threepiece_stage_assembled_but_still_grasped_is_1():
    # piece1 已靠近 base(first 谓词 True)但仍被握住 → 还没释放,应判 1,不应跳 2
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=True)) == 1


def test_threepiece_stage_assembled_and_released_is_2():
    # 靠近 base 且已释放 → 2
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=False)) == 2


def test_threepiece_stage_piece2_grasped_is_3():
    # piece1 已装好 + 正握 piece2 → 3
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=False, grasp2=True)) == 3


def test_threepiece_stage_piece2_grasped_priority_over_piece1_still_grasped():
    # piece1 已装好,piece2 已抓起,且 piece1 仍被(误)握 → 仍判 3(piece2 抓起优先)
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=True, grasp2=True)) == 3


def test_num_stages_threepiece_is_5():
    assert NUM_STAGES["TwoArmThreePieceAssembly"] == 5


# ---------- wrapper 的 stage 闩锁/归零 ----------
class _FakeBase:
    class _Cfg:
        image_features = {"observation.images.cam": None}
    config = _Cfg()

    def reset(self, env_ids=None):
        pass

    def get_action_chunk(self, raw_obs, chunk_length):
        b = raw_obs["observation.state"].shape[0]
        return torch.full((b, chunk_length, D), 0.5)


class _IdentityScaler:
    def scale(self, a):
        return torch.clamp(a, -1, 1)

    def unscale(self, a):
        return a


class _IdentityStd:
    def standardize(self, s):
        return s


class _StageVecEnv:
    """每步按 seq 发 info['stage_id'];term_at 步触发 terminated。"""
    def __init__(self, seq, term_at=None):
        self.action_space = gym.spaces.Box(-1, 1, (D,), dtype=np.float32)
        self.seq, self.term_at, self.i = seq, term_at, 0

    def _obs(self):
        return {"observation.state": torch.zeros(1, 3),
                "observation.images.cam": torch.zeros(1, 3, 4, 4)}

    def reset(self, **kw):
        self.i = 0
        return self._obs(), {}

    def step(self, action):
        s = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        term = torch.tensor([self.term_at is not None and self.i >= self.term_at])
        return (self._obs(), torch.tensor([0.0]), term,
                torch.tensor([False]), {"stage_id": np.array([s])})


def _mk_wrapper(env, bonus=0.0):
    return ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                   chunk_length=L, stage_reward_bonus=bonus,
                                   reward_shaping_mode="staged")


def test_wrapper_latches_max_stage_within_chunk():
    # stage 升到 2 后回落到 1,0 → 闩锁停在 2(供 reward);stage_id 解耦=瞬时(随回落)
    env = _StageVecEnv([0, 1, 2, 1, 0])
    w = _mk_wrapper(env)
    w.reset()
    obs, _, _, _, info = w.step(torch.zeros(1, L * D))
    assert info["max_stage_in_chunk"] == 2                # 闩锁:chunk 内最高
    assert w._stage == 2                                  # 闩锁保持,不随回落(reward 用)
    assert int(obs["observation.stage_id"][0, 0]) == 0    # stage_id 解耦=瞬时(chunk 末已回落到 0)


def test_wrapper_resets_stage_on_done():
    # 第 3 步 terminate,期间到过 stage 2 → max_in_chunk=2,但归零后 obs.stage_id=0
    env = _StageVecEnv([1, 2, 2, 0, 0], term_at=3)
    w = _mk_wrapper(env)
    w.reset()
    obs, _, term, _, info = w.step(torch.zeros(1, L * D))
    assert term.item() is True
    assert info["max_stage_in_chunk"] == 2
    assert int(obs["observation.stage_id"][0, 0]) == 0   # done 后新 episode 从 0 起


def test_wrapper_stage_starts_at_zero_on_reset():
    env = _StageVecEnv([0, 0, 0, 0, 0])
    w = _mk_wrapper(env)
    obs, _ = w.reset()
    assert int(obs["observation.stage_id"][0, 0]) == 0


def test_wrapper_no_stage_id_key_stays_zero():
    # info 无 stage_id(无检测器任务)→ 不崩,stage 恒 0
    class _NoStageEnv(_StageVecEnv):
        def step(self, action):
            self.i += 1
            return (self._obs(), torch.tensor([0.0]), torch.tensor([False]),
                    torch.tensor([False]), {})
    w = _mk_wrapper(_NoStageEnv([0]))
    w.reset()
    obs, _, _, _, info = w.step(torch.zeros(1, L * D))
    assert int(obs["observation.stage_id"][0, 0]) == 0
    assert info["max_stage_in_chunk"] == 0


def test_wrapper_staged_reward_adds_bonus():
    # stage 0→2(峰值 2),bonus=1.0 → 奖励含 +2.0(env 基础奖励为 0)
    env = _StageVecEnv([0, 1, 2, 1, 0])
    w = _mk_wrapper(env, bonus=1.0)
    w.reset()
    _, reward, _, _, info = w.step(torch.zeros(1, L * D))
    assert info["max_stage_in_chunk"] == 2
    assert float(reward[0]) == 2.0


def test_wrapper_no_bonus_is_regression():
    # bonus=0.0 → 与现状逐位相同(奖励仍为 env 的 0.0)
    env = _StageVecEnv([0, 1, 2, 1, 0])
    w = _mk_wrapper(env, bonus=0.0)
    w.reset()
    _, reward, _, _, _ = w.step(torch.zeros(1, L * D))
    assert float(reward[0]) == 0.0


def test_wrapper_staged_reward_done_midchunk_uses_peak():
    # 第 3 步 terminate,期间到过 stage 2 → 用归零前峰值算 Δ=2 → +2.0
    env = _StageVecEnv([1, 2, 2, 0, 0], term_at=3)
    w = _mk_wrapper(env, bonus=1.0)
    w.reset()
    _, reward, term, _, info = w.step(torch.zeros(1, L * D))
    assert term.item() is True
    assert info["max_stage_in_chunk"] == 2
    assert float(reward[0]) == 2.0


def test_wrapper_staged_reward_midepisode_start_stage():
    # 已在 stage 2 进入本 chunk,本 chunk 升到 3 → Δ=1,bonus=1.0 → +1.0
    env = _StageVecEnv([2, 2, 3, 2, 2])
    w = _mk_wrapper(env, bonus=1.0)
    w.reset()
    w._stage = 2                       # 模拟中途进入(上个 chunk 已到 stage 2)
    _, reward, _, _, info = w.step(torch.zeros(1, L * D))
    assert info["max_stage_in_chunk"] == 3
    assert float(reward[0]) == 1.0


# ---------- potential(PBS)模式接线 ----------
def _mk_pbs_wrapper(env, bonus=1.0, gamma=0.9):
    return ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                   chunk_length=L, stage_reward_bonus=bonus,
                                   reward_shaping_mode="potential", gamma=gamma)


def test_wrapper_potential_shaping_not_done():
    # start=0 进入,峰值 2,未 done → F = bonus·(γ·2 − 0) = 1·(0.9·2) = 1.8
    env = _StageVecEnv([0, 1, 2, 2, 2])
    w = _mk_pbs_wrapper(env)
    w.reset()
    _, reward, _, _, info = w.step(torch.zeros(1, L * D))
    assert info["max_stage_in_chunk"] == 2
    assert float(reward[0]) == pytest.approx(1.8)


def test_wrapper_potential_terminal_zeroes_phi():
    # 已在 stage 2 进入,本 chunk term → Φ(s')=0 → F = −bonus·2 = −2.0
    env = _StageVecEnv([2, 2, 2, 0, 0], term_at=3)
    w = _mk_pbs_wrapper(env)
    w.reset()
    w._stage = 2
    _, reward, term, _, _ = w.step(torch.zeros(1, L * D))
    assert term.item() is True
    assert float(reward[0]) == pytest.approx(-2.0)


# ---------- threading 3 段检测器契约 ----------
class _FakeNeedle:
    contact_geoms = ["ndl0", "ndl1"]

class _FakeTripod:
    contact_geoms = ["trp0", "trp1"]

def _is_tripod(object_geoms):
    return "trp0" in object_geoms


class _FakeThreadEnv:
    """list-gripper fake;grasp_needle / grasp_tripod 独立控制。"""
    def __init__(self, success=False, grasp_needle=False, grasp_tripod=False):
        self._success = success
        self._gn, self._gt = grasp_needle, grasp_tripod
        self.robots = [_FakeRobot()]          # 复用本文件已有的 _FakeRobot(gripper=[_FakeGripper()])
        self.needle = _FakeNeedle()
        self.tripod = _FakeTripod()

    def _check_success(self):
        return self._success

    def _check_grasp(self, gripper, object_geoms):
        return self._gt if _is_tripod(object_geoms) else self._gn


def test_threading_stage_success_is_2():
    assert threading_stage(_FakeThreadEnv(success=True)) == 2

def test_threading_stage_both_grasped_is_1():
    assert threading_stage(_FakeThreadEnv(grasp_needle=True, grasp_tripod=True)) == 1

def test_threading_stage_only_needle_is_0():
    # 仅抓针、未抓脚架 → 仍 0(stage 1 要求两物都抓)
    assert threading_stage(_FakeThreadEnv(grasp_needle=True)) == 0

def test_threading_stage_only_tripod_is_0():
    assert threading_stage(_FakeThreadEnv(grasp_tripod=True)) == 0

def test_threading_stage_start_is_0():
    assert threading_stage(_FakeThreadEnv()) == 0

def test_threading_stage_priority_success_over_grasp():
    assert threading_stage(_FakeThreadEnv(success=True, grasp_needle=True, grasp_tripod=True)) == 2

def test_threading_registered():
    assert NUM_STAGES["TwoArmThreading"] == 3
    assert get_stage_detector("TwoArmThreading") is threading_stage
