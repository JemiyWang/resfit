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
