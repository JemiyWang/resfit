"""离线 HIQL action-free value(模块 ③a)的纯逻辑。

设计见 docs/superpowers/specs/2026-06-07-hiql-value-design.md。
学法:goal-reaching 内部 reward(末步=1 否则 0)+ expectile TD,EMA target。
"""
import copy

import numpy as np
import torch
import torch.nn as nn


def expectile_loss(diff, expectile):
    """expectile regression 损失 L_tau(u) = |tau - 1[u<0]| * u^2,u=diff=y-V。

    tau>0.5 时对低估(diff>0,V<y)惩罚更重 -> 学上侧 expectile(乐观 value)。
    tau=0.5 退化为 0.5*MSE。返回标量。
    """
    weight = torch.where(diff < 0, 1.0 - expectile, expectile)
    return (weight * diff.pow(2)).mean()


def discounted_target(reward, next_v, done, gamma):
    """action-free TD target y = r + gamma*(1-done)*V(s')。

    done=1(终止)时 y=r,不 bootstrap(与 critic 的 Q-target 一致)。
    入参均为 [B] 或 [B,1] 张量;done 为 float(0/1)。
    """
    return reward + gamma * (1.0 - done) * next_v


class ValueMLP(nn.Module):
    """lowdim state -> 标量 V(s) 的小 MLP。state_dim/hidden 存为属性,便于 save/load 重建。"""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s):
        return self.net(s)
