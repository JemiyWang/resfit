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
        self.mean_net = _mlp(2 * state_dim, hidden, rep_dim)
        self.log_std = nn.Parameter(torch.zeros(rep_dim))

    def forward(self, s, g):
        mean = self.mean_net(torch.cat([s, g], dim=-1))
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
        return torch.distributions.Normal(mean, std)


def awr_weight(adv, beta, clip=100.0):
    """AWR 权重 exp(beta·adv),上界 clip(防爆)。adv 为张量,返回同形状张量。"""
    return torch.exp(beta * adv).clamp(max=clip)


def sample_high_goal_target(s_idx, traj_id, last_idx_of, rng, *, way_steps,
                            n_total, high_p_randomgoal=0.0):
    """HIQL GCSDataset 高段(逐字照 gc_dataset.py:122-135),返回 (goal_idx, target_idx)。

    traj goal: 线性插值 round(min(si+1,final)·d + final·(1−d)) ∈ [si+1, final],永不命中 current;
    traj target = min(si+way, traj_goal)。random goal(prob high_p_randomgoal)的 target=min(si+way, final)。
    前置:每个 si < last_idx_of[traj_id](build_gc_data 保证);否则 si==final 时 traj_goal 退化为 current。
    """
    si = np.asarray(s_idx, dtype=np.int64)
    B = len(si)
    final = np.array([last_idx_of[int(t)] for t in traj_id], dtype=np.int64)
    dist = rng.random(B)
    traj_goal = np.round(np.minimum(si + 1, final) * dist + final * (1 - dist)).astype(np.int64)
    traj_target = np.minimum(si + way_steps, traj_goal)
    rand_goal = rng.integers(0, n_total, size=B)
    rand_target = np.minimum(si + way_steps, final)
    pick = rng.random(B) < high_p_randomgoal
    goal = np.where(pick, rand_goal, traj_goal)
    target = np.where(pick, rand_target, traj_target)
    return goal.astype(np.int64), target.astype(np.int64)


def train_high_actor(data, vf, *, way_steps=25, beta=1.0, lr=3e-4,
                     batch_size=256, steps=50_000, hidden=256, seed=0,
                     target_mode="fixed_waypoint", high_p_randomgoal=0.0,
                     adv_agg="min"):
    """AWR 抽高层 π^h。vf:冻结 GoalConditionedVF。复用 Phase 1 的 data(build_gc_data)。

    优势 Ã^h 由 adv_agg 决定(见下);回归目标 z=vf.phi(s_t, s_{t+k})。返回训练后的 HighActor。
    adv_agg:
      - 'min'(默认,底层旧行为):Ã^h = min(vw1,vw2) − min(vs1,vs2)。
      - 'mean'(对齐 HIQL):Ã^h = 0.5(vw1+vw2) − 0.5(vs1+vs2)。
    target_mode:
      - 'fixed_waypoint'(默认):wi=min(si+way,demo末),goal 混采 sample_gc_goals(与现状逐位等价)。
      - 'clamp_to_goal':HIQL GCSDataset 高段——goal 线性插值轨迹态,wi=min(si+way, goal)。
    """
    if target_mode not in ("fixed_waypoint", "clamp_to_goal"):
        raise ValueError(f"unknown target_mode: {target_mode!r}")
    if adv_agg not in ("min", "mean"):
        raise ValueError(f"unknown adv_agg: {adv_agg!r}")
    if high_p_randomgoal != 0.0 and target_mode != "clamp_to_goal":
        raise ValueError(
            f"high_p_randomgoal={high_p_randomgoal} 仅在 target_mode='clamp_to_goal' 下生效;"
            f"当前 target_mode={target_mode!r} 会静默忽略它")
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
        if target_mode == "fixed_waypoint":
            wi = np.minimum(si + way_steps, last_arr[b])
            gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"],
                                 rng, n_total=len(states))
        else:
            gi, wi = sample_high_goal_target(si, tj, data["last_idx_of"], rng,
                                             way_steps=way_steps, n_total=len(states),
                                             high_p_randomgoal=high_p_randomgoal)
        s, sw, g = states[si], states[wi], states[gi]
        with torch.no_grad():
            vs1, vs2 = vf(s, g)
            vw1, vw2 = vf(sw, g)
            if adv_agg == "min":
                adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
            else:  # "mean"
                adv = 0.5 * (vw1 + vw2) - 0.5 * (vs1 + vs2)
            w = awr_weight(adv, beta)
            z_tgt = vf.phi(s, sw)
        dist = ha(s, g)
        logp = dist.log_prob(z_tgt).sum(-1)
        loss = -(w * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ha


def save_high_actor(path, model, *, gc_value_ckpt, way_steps, beta,
                    target_mode="fixed_waypoint", high_p_randomgoal=0.0,
                    adv_agg="min"):
    """存 high_actor.pt:权重 + 维度 + gc_value_ckpt/way_steps/beta + target_mode/high_p_randomgoal。"""
    torch.save({
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "rep_dim": model.rep_dim,
        "hidden": model.hidden,
        "log_std_min": model.log_std_min,
        "log_std_max": model.log_std_max,
        "gc_value_ckpt": gc_value_ckpt,
        "way_steps": way_steps,
        "beta": beta,
        "target_mode": target_mode,
        "high_p_randomgoal": high_p_randomgoal,
        "adv_agg": adv_agg,
    }, path)


def load_high_actor(path, map_location="cpu"):
    """读 high_actor.pt,重建 HighActor(eval),返回 (model, info)。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = HighActor(ckpt["state_dim"], ckpt["rep_dim"], ckpt["hidden"],
                      log_std_min=ckpt.get("log_std_min", -5.0),
                      log_std_max=ckpt.get("log_std_max", 2.0))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    info = {k: ckpt[k] for k in ("gc_value_ckpt", "way_steps", "beta")}
    info["target_mode"] = ckpt.get("target_mode", "fixed_waypoint")
    info["high_p_randomgoal"] = ckpt.get("high_p_randomgoal", 0.0)
    info["adv_agg"] = ckpt.get("adv_agg", "min")
    return model, info
