"""WM 张量契约:动作打包 / 帧解包 / 归一化。

绕开的上游坑(均不改上游):
- 坑1 rl_release.yaml 指向的 infer.yaml 缺 min_val/max_val → 本模块自带归一化统计量
- 坑2 wm_utils.py:193 用 CPU FloatTensor 减可能在 CUDA 上的 actions → 本模块自己打包
- 坑4 dynamics_model.py:227 把 device 当尺寸传 → 永远显式传 act_tokens,不走兜底
- 坑6 custom_pipeline.py:997 退出时 .train() → 调用方每次 infer 后显式 .eval()
- 坑7 动作维度在 14/16/30/32 之间漂移 → 显式钉死 16 并断言
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

ACTION_DIM = 16
CHUNK_LENGTH = 50
WM_TOKEN_STEPS = 25
WM_TOKEN_SLOTS = 30
WM_ACTION_INTERVAL = 2          # 50 动作 @30Hz → 25 token;WM 出 25 帧 @15Hz,同覆盖 1.667s
N_PREVIOUS = 4                  # 历史帧数
FRAME_H, FRAME_W = 192, 256
AGENT_IMG = 84                  # min_vit PatchEmbed2.num_patch=81 写死,只能 84

# 三视角。顺序即 WM video 张量第 0 维的顺序,不可改。
CAMERA_KEYS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)


class ActionNormalizer:
    """min-max 到 [-1,1],逐维。统计量由本模块持有,不依赖上游 config。"""

    def __init__(self, min_val, max_val):
        self.min_val = np.asarray(min_val, dtype=np.float32).reshape(-1)
        self.max_val = np.asarray(max_val, dtype=np.float32).reshape(-1)
        assert self.min_val.shape == (ACTION_DIM,), \
            f"min_val 须 ({ACTION_DIM},),got {self.min_val.shape}"
        assert self.max_val.shape == (ACTION_DIM,), \
            f"max_val 须 ({ACTION_DIM},),got {self.max_val.shape}"
        span = self.max_val - self.min_val
        assert np.all(span > 0), "max_val 须逐维严格大于 min_val"
        self._span = span

    def normalize(self, a):
        arr = np.asarray(a, dtype=np.float32)
        return (2.0 * (arr - self.min_val) / self._span - 1.0).astype(np.float32)

    def denormalize(self, a):
        arr = np.asarray(a, dtype=np.float32)
        return ((arr + 1.0) * 0.5 * self._span + self.min_val).astype(np.float32)


def pack_act_tokens(actions_physical, normalizer: ActionNormalizer) -> torch.Tensor:
    """(50,16) 物理空间绝对关节动作 → (1,25,30) bf16 act_tokens。"""
    arr = np.asarray(actions_physical, dtype=np.float32)
    assert arr.shape == (CHUNK_LENGTH, ACTION_DIM), \
        f"actions 须 ({CHUNK_LENGTH},{ACTION_DIM}),got {arr.shape}"
    sampled = arr[::WM_ACTION_INTERVAL]
    assert sampled.shape[0] == WM_TOKEN_STEPS, \
        f"抽样后须 {WM_TOKEN_STEPS} 步,got {sampled.shape[0]}"
    normed = normalizer.normalize(sampled)
    tokens = np.zeros((WM_TOKEN_STEPS, WM_TOKEN_SLOTS), dtype=np.float32)
    tokens[:, :ACTION_DIM] = normed
    return torch.from_numpy(tokens).unsqueeze(0).to(torch.bfloat16)


def split_predicted_frames(video: torch.Tensor) -> torch.Tensor:
    """(V,C,29,H,W) → (V,C,25,H,W),切掉前 4 帧历史。"""
    assert video.ndim == 5, f"video 须 5 维 (V,C,T,H,W),got {tuple(video.shape)}"
    total = N_PREVIOUS + WM_TOKEN_STEPS
    assert video.shape[2] == total, f"时间维须 {total},got {video.shape[2]}"
    return video[:, :, N_PREVIOUS:]


def frames_to_obs_images(frames: torch.Tensor, t_index: int) -> dict:
    """(V,C,T,H,W) 值域[-1,1] 的第 t_index 帧 → {camera_key: [1,3,84,84] float[0,1]}。"""
    assert frames.ndim == 5, f"frames 须 5 维,got {tuple(frames.shape)}"
    assert frames.shape[0] == len(CAMERA_KEYS), \
        f"视角数须 {len(CAMERA_KEYS)},got {frames.shape[0]}"
    out = {}
    for v, key in enumerate(CAMERA_KEYS):
        img = frames[v, :, t_index].float()            # (3,H,W) in [-1,1]
        img = (img + 1.0) * 0.5                        # → [0,1]
        img = img.clamp(0.0, 1.0).unsqueeze(0)         # (1,3,H,W)
        img = F.interpolate(img, size=(AGENT_IMG, AGENT_IMG),
                            mode="bilinear", align_corners=False)
        out[key] = img.contiguous()
    return out


def build_obs_window(frames: torch.Tensor, t_indices) -> torch.Tensor:
    """从预测帧取 4 帧作为下一段的 obs 窗口 → (V,C,4,H,W)。"""
    idx = list(t_indices)
    assert len(idx) == N_PREVIOUS, f"须 {N_PREVIOUS} 个索引,got {len(idx)}"
    return frames[:, :, idx].contiguous()
