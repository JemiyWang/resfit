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
