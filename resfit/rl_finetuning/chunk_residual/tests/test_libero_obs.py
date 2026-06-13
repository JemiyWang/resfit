import json
import numpy as np
import pytest
from resfit.rl_finetuning.chunk_residual.libero_obs import (
    quat2axisangle, resize_with_pad, flip_resize_image,
    assemble_libero_state, adapt_4tuple, build_libero_serve_obs, load_libero_norm_stats,
)


def test_quat2axisangle_identity_quat_is_zero():
    assert np.allclose(quat2axisangle(np.array([0., 0, 0, 1.])), np.zeros(3))


def test_quat2axisangle_clamps_w_over_one():
    assert np.allclose(quat2axisangle(np.array([0., 0, 0, 1.5])), np.zeros(3))


def test_resize_with_pad_shape_and_dtype():
    out = resize_with_pad(np.zeros((128, 256, 3), np.uint8), 224, 224)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_flip_resize_image_shape_dtype():
    img = np.zeros((100, 120, 3), np.uint8); img[0, 0] = 255
    out = flip_resize_image(img)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_assemble_libero_state_is_8d_axisangle():
    obs = {"robot0_eef_pos": np.ones(3), "robot0_eef_quat": np.array([0., 0, 0, 1.]),
           "robot0_gripper_qpos": np.array([0.04, -0.04])}
    s = assemble_libero_state(obs)
    assert s.shape == (8,) and s.dtype == np.float32
    assert np.allclose(s[:3], 1.0) and np.allclose(s[3:6], 0.0)


def test_adapt_4tuple_success_terminates():
    o, r, term, trunc, info = adapt_4tuple({"x": 1}, 1.0, True, {})
    assert term is True and trunc is False and info["success"] is True


def test_adapt_4tuple_timeout_truncates():
    o, r, term, trunc, info = adapt_4tuple({"x": 1}, 0.0, True, {})
    assert term is False and trunc is True


def test_build_libero_serve_obs_flat_schema():
    raw = {"observation.images.agentview": np.zeros((3, 4, 4), np.float32),
           "observation.images.robot0_eye_in_hand": np.zeros((3, 4, 4), np.float32),
           "observation.state": np.arange(8, dtype=np.float32)[None, :]}
    out = build_libero_serve_obs(raw, base_key="observation.images.agentview",
                                 wrist_key="observation.images.robot0_eye_in_hand",
                                 state_key="observation.state", prompt="do it", env_index=0)
    assert set(out) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    assert out["observation/image"].shape == (224, 224, 3) and out["observation/image"].dtype == np.uint8
    assert out["observation/state"].shape == (8,) and out["prompt"] == "do it"


def test_load_libero_norm_stats_keynames(tmp_path):
    # LeRobot meta/stats.json:键名 actions/state(兜底也认 action/observation.state)
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({
        "actions": {"mean": [0.0] * 7, "std": [1.0] * 7},
        "state": {"mean": [0.0] * 8, "std": [2.0] * 8},
    }))
    st = load_libero_norm_stats(str(p))
    assert st["action_mean"].shape == (7,) and st["action_std"].shape == (7,)
    assert st["state_mean"].shape == (8,) and np.allclose(st["state_std"], 2.0)
