# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""stage-conditioned residual 用的轻量工具。

stage_id 在 obs 里是 float([B,1] 或 [B])，取值 0..num_stages-1（瞬时阶段，可回退）。
这里把它转成 one-hot 并可拼到 prop 向量上。纯函数，无副作用，便于单测。
"""
import torch
import torch.nn.functional as F


def stage_onehot(stage_id: torch.Tensor, num_stages: int) -> torch.Tensor:
    """stage_id [B,1] 或 [B] float -> one-hot [B, num_stages] float32。

    越界值 clamp 到 [0, num_stages-1]（瞬时 stage 理论上不越界，clamp 仅作防御）。
    """
    idx = stage_id.reshape(stage_id.shape[0]).long().clamp_(0, num_stages - 1)
    oh = F.one_hot(idx, num_classes=num_stages)
    return oh.to(dtype=torch.float32, device=stage_id.device)


def append_stage(prop: torch.Tensor, stage_id: torch.Tensor, num_stages: int) -> torch.Tensor:
    """把 stage one-hot 拼到 prop 末尾：[B, P] -> [B, P + num_stages]。

    one-hot 对齐到 prop 的 device（不是 stage_id 的）：env wrapper 的 stage_id
    可能在 CPU（torch.full 默认），而 prop 在 GPU，否则 cat 跨设备报错。
    """
    return torch.cat([prop, stage_onehot(stage_id, num_stages).to(prop.device)], dim=-1)
