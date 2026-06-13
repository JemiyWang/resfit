import numpy as np
from resfit.rl_finetuning.chunk_residual.libero_env import LiberoGymWrapper


class _StubLiberoEnv:
    """假 LIBERO OffScreenRenderEnv:4-tuple + 原始 obs 键。"""
    def __init__(self): self.t = 0
    def _obs(self):
        return {"agentview_image": np.zeros((128, 128, 3), np.uint8),
                "robot0_eye_in_hand_image": np.zeros((128, 128, 3), np.uint8),
                "robot0_eef_pos": np.ones(3), "robot0_eef_quat": np.array([0., 0, 0, 1.]),
                "robot0_gripper_qpos": np.array([0.04, -0.04])}
    def reset(self): self.t = 0; return self._obs()
    def set_init_state(self, s): return self._obs()
    def step(self, a): self.t += 1; return self._obs(), 0.0, False, {}
    def close(self): pass


def test_wrapper_reset_returns_obs_info_with_contract_keys():
    w = LiberoGymWrapper(_StubLiberoEnv(), init_states=np.zeros((1, 10)), num_steps_wait=2)
    obs, info = w.reset()
    assert obs["observation.state"].shape == (8,)   # per-env 1-D,与 dexmg 对齐(stack 后 (N,8))
    assert "observation.images.agentview" in obs
    assert "observation.images.robot0_eye_in_hand" in obs
    assert obs["observation.images.agentview"].shape == (3, 224, 224)   # CHW
    assert obs["observation.images.agentview"].dtype == np.float32
    assert 0.0 <= float(obs["observation.images.agentview"].max()) <= 1.0
    assert w.action_space.shape == (7,)


def test_wrapper_step_returns_5tuple():
    w = LiberoGymWrapper(_StubLiberoEnv(), init_states=np.zeros((1, 10)), num_steps_wait=0)
    w.reset()
    obs, r, term, trunc, info = w.step(np.zeros(7, np.float32))
    assert isinstance(term, (bool, np.bool_)) and isinstance(trunc, (bool, np.bool_))
    assert "observation.state" in obs and "success" in info


def test_wrapper_reset_runs_dummy_steps():
    env = _StubLiberoEnv()
    LiberoGymWrapper(env, init_states=np.zeros((1, 10)), num_steps_wait=5).reset()
    assert env.t == 5
