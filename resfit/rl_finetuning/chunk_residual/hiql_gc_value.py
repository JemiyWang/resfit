"""goal-conditioned HIQL value(分层路 Phase 1)的纯逻辑。

与单任务 V-as-Φ 的 hiql_value.py 正交:本模块学 V(s, φ([g,s])),带 10 维归一化瓶颈
表征 φ + 双 critic 集成 + EMA target,供高层 AWR(Phase 2)与低层子目标条件(Phase 3)复用。
设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from resfit.rl_finetuning.chunk_residual.hiql_value import expectile_loss


def _mlp(in_dim, hidden, out_dim, n_hidden=2):
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class RelativeGoalEncoder(nn.Module):
    """φ([g,s]):concat(targets=g, bases=s) -> MLP -> rep_dim,再归一化到半径 sqrt(rep_dim)。"""

    def __init__(self, state_dim, rep_dim=10, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.net = _mlp(2 * state_dim, hidden, rep_dim)

    def forward(self, g, s):
        rep = self.net(torch.cat([g, s], dim=-1))
        rep = rep / (rep.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return rep
