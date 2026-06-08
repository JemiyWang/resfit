"""HIQL 高层策略 π^h(z|s,g)(分层路 Phase 2)。

AWR 从 Phase 1 冻结的 goal-conditioned value 抽取:输出 k 步后子目标潜表征 z=φ(s_t,s_{t+k})
上的高斯。设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import torch
import torch.nn as nn

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import _mlp, sample_gc_goals


class HighActor(nn.Module):
    """π^h(z | s, g):concat(s,g) -> _mlp -> mean(rep_dim);log_std 为 state-independent 参数。"""

    def __init__(self, state_dim, rep_dim=10, hidden=256, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.hidden = hidden
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mean = _mlp(2 * state_dim, hidden, rep_dim)
        self.log_std = nn.Parameter(torch.zeros(rep_dim))

    def forward(self, s, g):
        mean = self.mean(torch.cat([s, g], dim=-1))
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
        return torch.distributions.Normal(mean, std)


def awr_weight(adv, beta, clip=100.0):
    """AWR 权重 exp(beta·adv),上界 clip(防爆)。adv 为张量,返回同形状张量。"""
    return torch.exp(beta * adv).clamp(max=clip)
