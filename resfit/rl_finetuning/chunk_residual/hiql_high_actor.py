"""HIQL 高层策略 π^h(z|s,g)(分层路 Phase 2)。

AWR 从 Phase 1 冻结的 goal-conditioned value 抽取:输出 k 步后子目标潜表征 z=φ(s_t,s_{t+k})
上的高斯。设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import copy

import numpy as np
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


def train_high_actor(data, vf, *, way_steps=25, beta=1.0, lr=3e-4,
                     batch_size=256, steps=50_000, hidden=256, seed=0):
    """AWR 抽高层 π^h。vf:冻结 GoalConditionedVF。复用 Phase 1 的 data(build_gc_data)。

    优势 Ã^h = min V(s_{t+k},g) − min V(s_t,g);回归目标 z=vf.phi(s_t, s_{t+k})。
    k 步航点 way=min(s_idx+k, demo末);goal 混采 sample_gc_goals。返回训练后的 HighActor。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vf = copy.deepcopy(vf).eval()
    for p in vf.parameters():
        p.requires_grad_(False)
    states = data["states"]
    D = states.shape[1]
    rep_dim = vf.rep_dim
    ha = HighActor(D, rep_dim, hidden)
    opt = torch.optim.Adam(ha.parameters(), lr=lr)
    s_idx, traj_id = data["s_idx"], data["traj_id"]
    last_arr = np.array([data["last_idx_of"][int(d)] for d in traj_id], dtype=np.int64)
    n = len(s_idx)
    bs = min(batch_size, n)
    for _ in range(steps):
        b = rng.integers(0, n, size=bs)
        si, tj = s_idx[b], traj_id[b]
        wi = np.minimum(si + way_steps, last_arr[b])
        gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"],
                             rng, n_total=len(states))
        s, sw, g = states[si], states[wi], states[gi]
        with torch.no_grad():
            vs1, vs2 = vf(s, g)
            vw1, vw2 = vf(sw, g)
            adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
            w = awr_weight(adv, beta)
            z_tgt = vf.phi(s, sw)
        dist = ha(s, g)
        logp = dist.log_prob(z_tgt).sum(-1)
        loss = -(w * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ha
