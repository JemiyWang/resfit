"""offline_stage_replay 的 hdf5 proprio 拼装走 env-aware(env_hint)。

_demo_base_actions 喂 base ACT 的 observation.state 必须按 task 拼对维度
(pouring=36/lifttray=38),否则 KeyError 或喂错维度给 ACT。这里 mock base_policy
捕获它收到的 observation.state 维度,隔离验证拼装(不需真 ACT/env)。
"""
import h5py
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.offline_stage_replay import _demo_base_actions


class _RecBasePolicy:
    """记录每帧 select_action 收到的 observation.state 维度;返回固定动作。"""
    def __init__(self):
        self.state_dims = []

    def reset(self):
        pass

    def select_action(self, raw_obs):
        self.state_dims.append(int(raw_obs["observation.state"].shape[-1]))
        return torch.zeros(1, 7)


class _IdentityScaler:
    def scale(self, x):
        return torch.as_tensor(x, dtype=torch.float32)


def _make_pouring_grp(f, T):
    g = f.create_group("data/demo_0")
    g.create_dataset("obs/agentview_image", data=np.zeros((T, 84, 84, 3), np.uint8))
    fields = {"robot0_right_eef_pos": 3, "robot0_right_eef_quat": 4, "robot0_right_gripper_qpos": 11,
              "robot0_left_eef_pos": 3, "robot0_left_eef_quat": 4, "robot0_left_gripper_qpos": 11}
    for k, d in fields.items():
        g.create_dataset(f"obs/{k}", data=np.zeros((T, d), np.float32))
    return g


def test_demo_base_actions_pouring_state_dim(tmp_path):
    T = 4
    p = str(tmp_path / "pour.hdf5")
    with h5py.File(p, "w") as f:
        _make_pouring_grp(f, T)
    bp = _RecBasePolicy()
    with h5py.File(p, "r") as f:
        grp = f["data/demo_0"]
        out = _demo_base_actions(bp, grp, ["observation.images.agentview"],
                                 _IdentityScaler(), "cpu", env_hint="TwoArmPouring")
    assert out.shape == (T, 7)
    assert bp.state_dims == [36] * T        # 每帧喂 ACT 的 state 都是 36 维(humanoid)


# ---------------------------------------------------------------------------
# no-stage hdf5 路:build_offline_buffer 对 TwoArmPouring 不调 replay_instant_stages
# (detector=None → 历史 bug:TypeError:'NoneType' object is not callable)
# ---------------------------------------------------------------------------

import json
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from resfit.rl_finetuning.chunk_residual import offline_stage_replay as _osr
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import expected_low_dim_keys


def _make_pouring_hdf5_full(path, T=4):
    """最小完整 pouring HDF5:含 env_args、states、actions、humanoid obs + agentview 图。"""
    env_args = json.dumps({
        "env_name": "TwoArmPouring",
        "env_kwargs": {},
    })
    fields = {"robot0_right_eef_pos": 3, "robot0_right_eef_quat": 4, "robot0_right_gripper_qpos": 11,
              "robot0_left_eef_pos": 3, "robot0_left_eef_quat": 4, "robot0_left_gripper_qpos": 11}
    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env_args"] = env_args
        g = data_grp.create_group("demo_0")
        g.create_dataset("states", data=np.zeros((T, 5), dtype=np.float32))
        g.create_dataset("actions", data=np.zeros((T, 7), dtype=np.float32))
        for k, d in fields.items():
            g.create_dataset(f"obs/{k}", data=np.zeros((T, d), dtype=np.float32))
        g.create_dataset("obs/agentview_image", data=np.zeros((T, 4, 4, 3), dtype=np.uint8))
        g.attrs["model_file"] = "dummy_model"


class _IdScalerFull:
    def scale(self, a): return torch.as_tensor(a, dtype=torch.float32)
    def unscale(self, a): return a


class _IdStdFull:
    def standardize(self, s): return torch.as_tensor(s, dtype=torch.float32)


class _FakeRbFull:
    def __init__(self): self.items = []
    def add(self, td): self.items.append(td)


def test_build_offline_buffer_no_stage_hdf5_pouring(tmp_path, monkeypatch):
    """hdf5 路 + env_hint=TwoArmPouring(no-stage):不起 env / 不调 replay_instant_stages,
    stage_id 和 max_stage 全 0,不 raise TypeError:'NoneType' object is not callable。
    这是 bug 修复前会 crash 的路径。
    """
    T = 4
    hdf5 = str(tmp_path / "pouring.hdf5")
    _make_pouring_hdf5_full(hdf5, T)

    # 记录 make_replay_env 和 replay_instant_stages 的调用次数
    calls = {"make_env": 0, "replay_stages": 0}

    def _fake_make_env(_path):
        calls["make_env"] += 1
        # 若修复前走到这里并调用 detector(None)(env),会崩;修复后不该到这里
        class _FakeEnv:
            def close(self): pass
        return _FakeEnv(), "TwoArmPouring"

    def _fake_replay_stages(env, states, *, model_file, detector, ep_meta=None):
        calls["replay_stages"] += 1
        # 修复前 detector=None → TypeError:'NoneType' object is not callable
        return detector(env)

    monkeypatch.setattr(_osr, "make_replay_env", _fake_make_env)
    monkeypatch.setattr(_osr, "replay_instant_stages", _fake_replay_stages)

    rb = _FakeRbFull()
    # no-stage 路:stage_cache=None → 走 else 分支;env_hint="TwoArmPouring" → detector=None
    added = _osr.build_offline_buffer(
        rb, hdf5,
        action_scaler=_IdScalerFull(),
        state_standardizer=_IdStdFull(),
        image_keys=["observation.images.agentview"],
        bonus=1.0, mode="none", gamma=0.99,
        stage_cache=None, potential=None, subgoal=None,
        base_mode="gt", env_hint="TwoArmPouring",
    )

    assert added == T - 1, f"预期 {T-1} 条 transition,实得 {added}"
    assert calls["replay_stages"] == 0, "no-stage 任务不应调 replay_instant_stages"
    assert calls["make_env"] == 0, "no-stage 任务不应起 env"

    # stage_id 和 max_stage 全 0
    for td in rb.items:
        assert float(td["obs"]["observation.stage_id"].abs().max()) == 0.0, \
            "no-stage 的 stage_id 应为 0"
        assert float(td["max_stage"].abs()) == 0.0, \
            "no-stage 的 max_stage 应为 0"
