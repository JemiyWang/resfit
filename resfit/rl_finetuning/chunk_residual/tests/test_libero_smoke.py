"""LIBERO-residual live smoke(opt-in)。默认 skip——真起 LIBERO env 需 libero-residual conda env
(robosuite1.4.1)+ MUJOCO_GL=egl + GPU,不在 residual 单测套里跑。

手动跑(在 libero-residual env):
  LIBERO_RESIDUAL_SMOKE=1 MUJOCO_GL=egl \
  conda run -p /mnt/mnt/data/envs/libero-residual python -m pytest \
    resfit/rl_finetuning/chunk_residual/tests/test_libero_smoke.py -v

端到端残差 RL live smoke(另需先起 pi0_libero serve,见 MEMORY project_pi0_libero_run_recipe):
  conda run -p /mnt/mnt/data/envs/libero-residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    --env_family libero --libero_suite libero_spatial --libero_task_id 0 \
    --libero_stats_json /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json \
    --base_policy_type pi05 --base_action_mode queue --chunk_length 1 --pi0_action_dim 7 \
    --pi0_host 0.0.0.0 --pi0_port 8000 --reward_shaping none --offline_fraction 0 --total_steps 50
"""
import os
import numpy as np
import pytest

RUN = os.environ.get("LIBERO_RESIDUAL_SMOKE") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="opt-in:需 libero-residual env + MUJOCO_GL=egl + GPU")


def test_libero_env_constructs_and_steps():
    """真起单个 LIBERO env,reset+step 几步,验证契约 obs/action 维度。"""
    from resfit.rl_finetuning.chunk_residual.libero_env import make_libero_env
    env, prompt = make_libero_env("libero_spatial", 0)
    obs, info = env.reset()
    assert obs["observation.state"].shape == (8,)
    assert obs["observation.images.agentview"].shape == (3, 224, 224)
    assert env.action_space.shape == (7,)
    assert isinstance(prompt, str) and len(prompt) > 0
    for _ in range(3):
        obs, r, term, trunc, info = env.step(np.zeros(7, np.float32))
        assert "success" in info
    env.close()
