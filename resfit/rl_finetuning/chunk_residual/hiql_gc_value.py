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


class GoalConditionedVF(nn.Module):
    """V(s, φ([g,s])),双 critic 集成。

    约定 phi(s, g) = goal_encoder(targets=g, bases=s):第一参恒为"基准状态",第二参为
    "目标/子目标状态"。forward(s, g) -> (v1, v2)。
    """

    def __init__(self, state_dim, rep_dim=10, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.hidden = hidden
        self.goal_encoder = RelativeGoalEncoder(state_dim, rep_dim, hidden)
        self.v1 = _mlp(state_dim + rep_dim, hidden, 1)
        self.v2 = _mlp(state_dim + rep_dim, hidden, 1)

    def phi(self, s, g):
        return self.goal_encoder(g, s)

    def forward(self, s, g):
        x = torch.cat([s, self.phi(s, g)], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)


def stage_entries_from_instant(instant_stages):
    """逐帧瞬时 stage(int 数组) -> 各更高 stage 首次到达的下标(升序 int64 数组)。

    用运行最大值 latch 消抖;入口=latch 比前一帧大的位置。全 0 返回空数组。
    """
    instant = np.asarray(instant_stages).astype(np.int64)
    if len(instant) == 0:
        return np.empty(0, dtype=np.int64)
    latch = np.maximum.accumulate(instant)
    inc = np.flatnonzero(np.diff(latch, prepend=latch[0] - (latch[0] > 0)) > 0)
    return inc.astype(np.int64)


def build_gc_data(seqs, stage_entries):
    """list[(T,D)] 标准化 state + list[within-demo stage 入口下标] -> 扁平训练数据 dict。

    返回:states(N,D tensor)、s_idx/sn_idx(每 transition 的 s/s' 全局下标)、done(tensor)、
    traj_id、last_idx_of(demo->末态全局下标)、stage_entries_of(demo->入口全局下标 array)。
    T<2 的 demo 跳过。
    """
    states, s_idx, sn_idx, done, traj_id = [], [], [], [], []
    last_idx_of, stage_entries_of = {}, {}
    offset = 0
    for d, seq in enumerate(seqs):
        seq = np.asarray(seq, dtype=np.float32)
        T = len(seq)
        if T < 2:
            continue
        states.append(seq)
        g_idx = np.arange(offset, offset + T)
        last_idx_of[d] = int(g_idx[-1])
        ent = np.asarray(stage_entries[d], dtype=np.int64)
        ent = ent[ent < T]
        stage_entries_of[d] = g_idx[ent] if len(ent) else g_idx[[-1]]
        for t in range(T - 1):
            s_idx.append(offset + t)
            sn_idx.append(offset + t + 1)
            done.append(1.0 if t == T - 2 else 0.0)
            traj_id.append(d)
        offset += T
    return {
        "states": torch.tensor(np.concatenate(states, axis=0), dtype=torch.float32),
        "s_idx": np.asarray(s_idx, dtype=np.int64),
        "sn_idx": np.asarray(sn_idx, dtype=np.int64),
        "done": torch.tensor(done, dtype=torch.float32),
        "traj_id": np.asarray(traj_id, dtype=np.int64),
        "last_idx_of": last_idx_of,
        "stage_entries_of": stage_entries_of,
    }


def sample_gc_goals(idx, traj_id, last_idx_of, stage_entries_of, rng,
                    *, n_total, p_curr=0.2, p_traj=0.5, p_rand=0.3):
    """HIQL 混采 + stage 入口锚:current(p_curr)/future(p_traj)/random(p_rand)。

    future = 同 demo 中 >= 当前下标的 stage 入口态里均匀取一个,无则取末态。
    idx/traj_id 为 (B,) np 数组。返回 (B,) goal 全局下标。
    """
    B = len(idx)
    goal = rng.integers(0, n_total, size=B)
    fut = np.empty(B, dtype=np.int64)
    for j in range(B):
        ent = stage_entries_of[int(traj_id[j])]
        cand = ent[ent >= idx[j]]
        fut[j] = int(rng.choice(cand)) if len(cand) else last_idx_of[int(traj_id[j])]
    denom = max(1.0 - p_curr, 1e-8)
    goal = np.where(rng.random(B) < p_traj / denom, fut, goal)
    goal = np.where(rng.random(B) < p_curr, idx, goal)
    return goal.astype(np.int64)
