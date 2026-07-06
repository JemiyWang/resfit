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
    lifttray_stage,
    threading_stage,
    threepiece_stage,
    pouring_stage,
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


def test_threepiece_stage_success_is_3():
    assert threepiece_stage(_FakeTPEnv(second=True)) == 3


def test_threepiece_stage_first_assembled_is_2():
    assert threepiece_stage(_FakeTPEnv(first=True)) == 2


def test_threepiece_stage_both_grasped_is_1():
    # 新口径:piece1 和 piece2 同时被抓起 → 阶段 1
    assert threepiece_stage(_FakeTPEnv(grasp=True, grasp2=True)) == 1


def test_threepiece_stage_only_piece1_grasped_is_0():
    # 仅抓 piece1、未抓 piece2 → 仍 0(stage 1 要求两件都抓)
    assert threepiece_stage(_FakeTPEnv(grasp=True)) == 0


def test_threepiece_stage_only_piece2_grasped_is_0():
    assert threepiece_stage(_FakeTPEnv(grasp2=True)) == 0


def test_threepiece_stage_start_is_0():
    assert threepiece_stage(_FakeTPEnv()) == 0


def test_threepiece_stage_priority_success_over_lower():
    # 同时满足时,高阶段优先 → 成功段(3)
    assert threepiece_stage(_FakeTPEnv(second=True, first=True, grasp=True)) == 3


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


def test_threepiece_stage_both_grasped_dict_gripper_is_1():
    # 真环境 gripper=dict;piece1+piece2 都抓住、未装好 → 1
    assert threepiece_stage(_RealisticTPEnv(grasp=True, grasp2=True)) == 1


def test_threepiece_stage_only_piece1_dict_gripper_is_0():
    # 真环境仅抓 piece1 → 0(stage 1 要求两件都抓)
    assert threepiece_stage(_RealisticTPEnv(grasp=True)) == 0


def test_threepiece_stage_assembled_still_grasped_is_2():
    # 新口径:piece1 已装好(first 谓词 True)即判 2,不再要求先释放
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=True)) == 2


def test_threepiece_stage_assembled_and_released_is_2():
    # piece1 装好(无论是否仍握)→ 2
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=False)) == 2


def test_threepiece_stage_piece2_grasped_after_assembled_is_2():
    # 新口径:piece1 装好后抓 piece2 仍归 2(装配里程碑主导,抓 piece2 不再单列)
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=False, grasp2=True)) == 2


def test_threepiece_stage_success_over_all_is_3():
    # piece2 装好(成功)优先于任何低阶段
    assert threepiece_stage(
        _RealisticTPEnv(second=True, first=True, grasp=True, grasp2=True)) == 3


def test_num_stages_threepiece_is_4():
    assert NUM_STAGES["TwoArmThreePieceAssembly"] == 4


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


# ---------- lifttray 4 段检测器契约 ----------
class _FakeBox:
    def __init__(self, name):
        self.name = name


class _FakeLiftTrayEnv:
    """success + obj0/obj1 是否与 pot_base 接触 独立控制。
    check_contact(geoms_1, geoms_2) 仅 geoms_1=='pot_base' 时按 obj.name 返回其 on_tray。"""
    def __init__(self, success=False, obj0_on=False, obj1_on=False):
        self._success = success
        self._on = {"obj0": obj0_on, "obj1": obj1_on}
        self.obj0 = _FakeBox("obj0")
        self.obj1 = _FakeBox("obj1")

    def _check_success(self):
        return self._success

    def check_contact(self, geoms_1, geoms_2=None):
        if geoms_1 != "pot_base":
            return False
        return self._on.get(getattr(geoms_2, "name", None), False)


def test_lifttray_stage_start_is_0():
    assert lifttray_stage(_FakeLiftTrayEnv()) == 0


def test_lifttray_stage_one_block_on_tray_is_1():
    assert lifttray_stage(_FakeLiftTrayEnv(obj0_on=True)) == 1
    assert lifttray_stage(_FakeLiftTrayEnv(obj1_on=True)) == 1   # 顺序无关,哪块都算 1


def test_lifttray_stage_both_on_tray_is_2():
    assert lifttray_stage(_FakeLiftTrayEnv(obj0_on=True, obj1_on=True)) == 2


def test_lifttray_stage_success_is_3():
    assert lifttray_stage(_FakeLiftTrayEnv(success=True)) == 3


def test_lifttray_stage_priority_success_over_lower():
    assert lifttray_stage(_FakeLiftTrayEnv(success=True, obj0_on=True, obj1_on=True)) == 3


def test_lifttray_stage_sequential_pick_separates_two_stages():
    # 核心意图:obj1 先上盘→1;obj0 后上盘(两块都在)→2 → 两次搬运落在不同段(持久里程碑)
    assert lifttray_stage(_FakeLiftTrayEnv(obj1_on=True)) == 1
    assert lifttray_stage(_FakeLiftTrayEnv(obj0_on=True, obj1_on=True)) == 2


def test_lifttray_registered():
    assert NUM_STAGES["TwoArmLiftTray"] == 4
    assert get_stage_detector("TwoArmLiftTray") is lifttray_stage


# ---------- pouring 5 段检测器契约 ----------
class _FakeCup:
    contact_geoms = ["cup0", "cup1"]

class _FakeBowl:
    contact_geoms = ["bowl0", "bowl1"]

class _FakeBall:
    contact_geoms = ["ball0"]

def _is_bowl(object_geoms):
    return "bowl0" in object_geoms


class _FakePouringEnv:
    """GR1 dict-gripper fake;grasp_cup/grasp_bowl 独立控制;check_contact 返回 ball_in_bowl。"""
    def __init__(self, success=False, ball_in_bowl=False, grasp_cup=False, grasp_bowl=False):
        self._success = success
        self._bib = ball_in_bowl
        self._gc, self._gb = grasp_cup, grasp_bowl
        self.robots = [_DictGripperRobot()]     # GR1 单机器人,gripper=dict(迭代得 str 键)
        self.cup = _FakeCup()
        self.bowl = _FakeBowl()
        self.ball = _FakeBall()

    def _check_success(self):
        return self._success

    def check_contact(self, a, b):               # 检测器调 check_contact(env.bowl, env.ball)
        return self._bib

    def _check_grasp(self, gripper, object_geoms):
        if isinstance(gripper, str):
            return False
        return self._gb if _is_bowl(object_geoms) else self._gc


def test_pouring_stage_start_is_0():
    assert pouring_stage(_FakePouringEnv()) == 0

def test_pouring_stage_cup_grasped_is_1():
    assert pouring_stage(_FakePouringEnv(grasp_cup=True)) == 1

def test_pouring_stage_ball_in_bowl_is_2():
    assert pouring_stage(_FakePouringEnv(ball_in_bowl=True)) == 2

def test_pouring_stage_ball_in_bowl_and_bowl_grasped_is_3():
    assert pouring_stage(_FakePouringEnv(ball_in_bowl=True, grasp_bowl=True)) == 3

def test_pouring_stage_success_is_4():
    assert pouring_stage(_FakePouringEnv(success=True)) == 4

def test_pouring_stage_priority_success_over_lower():
    # 同时满足 → 高阶段优先 = 成功(4)
    assert pouring_stage(_FakePouringEnv(
        success=True, ball_in_bowl=True, grasp_cup=True, grasp_bowl=True)) == 4

def test_pouring_stage_bowl_grasped_before_pour_is_0():
    # 倒球前就抓碗(球未入碗、cup 未抓)→ 仍 0(stage 3 要求球已入碗;不误触发)
    assert pouring_stage(_FakePouringEnv(grasp_bowl=True)) == 0

def test_pouring_stage_ball_in_bowl_not_success_is_3_when_carrying():
    # 球入碗 + 抓碗但未 success → 3(搬运中)
    assert pouring_stage(_FakePouringEnv(ball_in_bowl=True, grasp_bowl=True, success=False)) == 3

def test_pouring_stage_monotonic_separation():
    # 三相位依次:抓cup(1)→倒入+松cup(2)→抓碗搬运(3),证明 5 段能切开
    seq = [
        _FakePouringEnv(grasp_cup=True),                              # 1
        _FakePouringEnv(ball_in_bowl=True, grasp_cup=False),          # 2
        _FakePouringEnv(ball_in_bowl=True, grasp_bowl=True),          # 3
    ]
    assert [pouring_stage(e) for e in seq] == [1, 2, 3]

def test_pouring_registered():
    assert NUM_STAGES["TwoArmPouring"] == 5
    assert get_stage_detector("TwoArmPouring") is pouring_stage
