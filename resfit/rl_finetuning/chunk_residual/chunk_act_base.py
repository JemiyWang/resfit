"""不修改 ACT 的前提下,一次性从 ACTPolicy 拿整段 chunk(原始动作尺度)。

复用 ACTPolicy 的 normalize_inputs / 图像重组 / model / unnormalize_outputs。
"""
from __future__ import annotations

import torch


@torch.no_grad()
def get_action_chunk(base_policy, raw_obs: dict, chunk_length: int) -> torch.Tensor:
    """返回 [B, chunk_length, action_dim] 的原始尺度动作 chunk。"""
    base_policy.eval()
    batch = base_policy.normalize_inputs(raw_obs)          # modeling_act.py:164
    batch = dict(batch)
    batch["observation.images"] = [batch[k] for k in base_policy.config.image_features]  # :167
    chunk = base_policy.model(batch)[0]                    # [B, chunk_size, D] 归一化空间 :209
    chunk = base_policy.unnormalize_outputs({"action": chunk})["action"]  # 原始尺度 :210
    return chunk[:, :chunk_length]
