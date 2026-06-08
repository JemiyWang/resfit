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


def stage_budget_factor(stage_id: torch.Tensor, budget: torch.Tensor, num_stages: int) -> torch.Tensor:
    """stage_id [B,1] 或 [B] float -> 每样本残差幅度乘子 [B,1] float32。

    budget: 1-D 张量，长度 num_stages，budget[k] = stage k 的乘子。
    越界 stage_id clamp 到 [0, num_stages-1]。返回乘子在 stage_id 的 device 上。
    """
    idx = stage_id.reshape(stage_id.shape[0]).long().clamp_(0, num_stages - 1)
    fac = budget.to(idx.device)[idx]
    return fac.reshape(-1, 1).to(torch.float32)


def parse_stage_budget(arg: "str | None", num_stages: int) -> "list[float] | None":
    """解析 --stage_budget 字符串 -> list[float]；None 表关。长度必须 == num_stages。"""
    if arg is None:
        return None
    vals = [float(x) for x in arg.split(",")]
    if len(vals) != num_stages:
        raise ValueError(f"stage_budget 长度 {len(vals)} != num_stages {num_stages}")
    return vals


def append_subgoal(prop: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
    """把潜子目标 z(维度任意)拼到 prop 末尾:[B, P] -> [B, P + rep_dim]。

    z 对齐到 prop 的 device(env 给的 z 可能在 CPU,prop 在 GPU)。与 append_stage 同模板,
    但 z 已是连续向量,无需 one-hot。
    """
    return torch.cat([prop, subgoal.to(prop.device)], dim=-1)
