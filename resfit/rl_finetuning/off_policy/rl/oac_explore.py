# OAC(Optimistic Actor-Critic, Ciosek et al. NeurIPS 2019)探索偏移的纯函数。
# 只做"把探索分布均值沿 Q 乐观上界梯度偏移"的数学,不依赖具体 agent,便于单测。
# 参考: /mnt/mnt/data/wjm/residual/oac-explore/optimistic_exploration.py
from __future__ import annotations

import math
from typing import Callable

import torch


def q_upper_bound(q_per_head: torch.Tensor, beta_ub: float) -> torch.Tensor:
    """Q 乐观上界 μ_Q + β_UB·σ_Q,用 critic ensemble 的均值/标准差。

    q_per_head: [K, B, 1] 或 [K, B];K=critic 头数。
    返回: [B]。K=2 时 σ 退化为 |Q1-Q2|/2(unbiased=False)。
    """
    q = q_per_head.squeeze(-1) if q_per_head.dim() == 3 else q_per_head   # [K, B]
    mu_q = q.mean(0)                                                      # [B]
    sigma_q = q.std(0, unbiased=False)                                    # [B]
    return mu_q + beta_ub * sigma_q


def optimistic_mean_shift(
    mu_T: torch.Tensor,
    std_T: torch.Tensor,
    q_ub_fn: Callable[[torch.Tensor], torch.Tensor],
    delta: float,
) -> torch.Tensor:
    """把均值 mu_T 沿 q_ub_fn 的梯度方向偏移,KL 预算 δ;协方差不变。

    mu_T, std_T: [B, A]。q_ub_fn(action[B,A]) -> Q_UB[B](已含 β_UB·σ)。
    返回偏移后的均值 mu_E[B,A](已 detach)。delta<=0 时原样返回 mu_T。
    数学(对角 Σ=std²):mu_E = mu_T + √(2δ)·Σ·g / sqrt(gᵀΣg),g=∇_a Q_UB|_{mu_T}。
    """
    if delta <= 0.0:
        return mu_T.detach()

    mu_leaf = mu_T.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        q_ub = q_ub_fn(mu_leaf)                       # [B]
        grad = torch.autograd.grad(q_ub.sum(), mu_leaf)[0]   # [B, A]

    sigma = std_T ** 2                                # [B, A]
    denom = torch.sqrt((grad ** 2 * sigma).sum(-1, keepdim=True)) + 1e-6   # [B, 1]
    mu_c = math.sqrt(2.0 * delta) * (sigma * grad) / denom
    return (mu_leaf + mu_c).detach()
