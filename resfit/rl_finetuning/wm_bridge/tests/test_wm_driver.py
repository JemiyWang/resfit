import numpy as np
import torch

from resfit.rl_finetuning.wm_bridge.wm_driver import (
    CAMERA_KEYS, CHUNK_LENGTH, N_PREVIOUS, WM_TOKEN_SLOTS, WM_TOKEN_STEPS,
    ActionNormalizer, frames_to_obs_images, pack_act_tokens,
    split_predicted_frames,
)


def _norm():
    return ActionNormalizer(min_val=-np.ones(16, np.float32),
                            max_val=np.ones(16, np.float32))


def test_normalize_roundtrip():
    n = ActionNormalizer(min_val=np.full(16, -2.0, np.float32),
                         max_val=np.full(16, 6.0, np.float32))
    a = np.linspace(-2.0, 6.0, 16, dtype=np.float32)
    np.testing.assert_allclose(n.denormalize(n.normalize(a)), a, atol=1e-5)


def test_normalize_maps_endpoints_to_pm1():
    n = ActionNormalizer(min_val=np.full(16, -2.0, np.float32),
                         max_val=np.full(16, 6.0, np.float32))
    np.testing.assert_allclose(n.normalize(np.full(16, -2.0, np.float32)),
                               -np.ones(16), atol=1e-6)
    np.testing.assert_allclose(n.normalize(np.full(16, 6.0, np.float32)),
                               np.ones(16), atol=1e-6)


def test_pack_act_tokens_shape_and_dtype():
    acts = np.zeros((CHUNK_LENGTH, 16), dtype=np.float32)
    tok = pack_act_tokens(acts, _norm())
    assert tok.shape == (1, WM_TOKEN_STEPS, WM_TOKEN_SLOTS)
    assert tok.dtype == torch.bfloat16


def test_pack_act_tokens_subsamples_every_other_action():
    acts = np.zeros((CHUNK_LENGTH, 16), dtype=np.float32)
    acts[:, 0] = np.arange(CHUNK_LENGTH)          # 0..49
    tok = pack_act_tokens(acts, _norm()).float().numpy()
    # interval=2 → 取 index 0,2,4,...,48;归一化恒等(min=-1,max=1)
    np.testing.assert_allclose(tok[0, :, 0], np.arange(0, CHUNK_LENGTH, 2), atol=1e-2)


def test_pack_act_tokens_zero_pads_slots_16_to_30():
    acts = np.ones((CHUNK_LENGTH, 16), dtype=np.float32)
    tok = pack_act_tokens(acts, _norm()).float().numpy()
    assert np.all(tok[0, :, 16:] == 0.0)


def test_split_predicted_frames_drops_history():
    video = torch.zeros(3, 3, N_PREVIOUS + WM_TOKEN_STEPS, 192, 256)
    video[:, :, :N_PREVIOUS] = 1.0                # 历史段标记
    pred = split_predicted_frames(video)
    assert pred.shape == (3, 3, WM_TOKEN_STEPS, 192, 256)
    assert torch.all(pred == 0.0)                 # 历史段已被切掉


def test_frames_to_obs_images_keys_shape_and_range():
    # WM 输出值域 [-1,1];-1 应映射到 0.0,+1 应映射到 1.0
    frames = torch.full((3, 3, WM_TOKEN_STEPS, 192, 256), -1.0)
    obs = frames_to_obs_images(frames, t_index=0)
    assert set(obs.keys()) == set(CAMERA_KEYS)
    for k in CAMERA_KEYS:
        assert obs[k].shape == (1, 3, 84, 84)
        assert torch.all(obs[k] >= 0.0) and torch.all(obs[k] <= 1.0)
        assert torch.allclose(obs[k], torch.zeros_like(obs[k]), atol=1e-5)
