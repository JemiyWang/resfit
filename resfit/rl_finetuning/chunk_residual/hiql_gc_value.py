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


def expectile_loss_weighted(adv, diff, expectile):
    """两参 expectile(逐字照 HIQL hiql.py:20-22):门控看 adv 符号,平方的是 diff。

    weight = tau      if adv >= 0   # adv=q-V_target,正=该转移是赚的 -> 把 V 往上拉
    weight = 1 - tau  if adv <  0
    与单参 expectile_loss 的区别:门控量(adv)与被平方量(diff)解耦。adv==diff 时两者等价。
    返回标量(.mean())。
    """
    weight = torch.where(adv >= 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def _mlp(in_dim, hidden, out_dim, n_hidden=2, use_layer_norm=False):
    act = nn.GELU if use_layer_norm else nn.ReLU
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), act()]
        if use_layer_norm:
            layers += [nn.LayerNorm(hidden)]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class RelativeGoalEncoder(nn.Module):
    """φ([g,s]):concat([targets, bases]) -> MLP -> rep_dim,再归一化到半径 sqrt(rep_dim)。"""

    def __init__(self, state_dim, rep_dim=10, hidden=256, use_layer_norm=False):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.net = _mlp(2 * state_dim, hidden, rep_dim, use_layer_norm=use_layer_norm)

    def forward(self, targets, bases):
        rep = self.net(torch.cat([targets, bases], dim=-1))
        rep = rep / (rep.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return rep


class GoalConditionedVF(nn.Module):
    """V(s, φ([g,s])),双 critic 集成。

    约定 phi(s, g) = goal_encoder(targets=g, bases=s):第一参恒为"基准状态",第二参为
    "目标/子目标状态"。forward(s, g) -> (v1, v2)。
    """

    def __init__(self, state_dim, rep_dim=10, hidden=256, use_layer_norm=False):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.hidden = hidden
        self.use_layer_norm = use_layer_norm
        self.goal_encoder = RelativeGoalEncoder(state_dim, rep_dim, hidden, use_layer_norm=use_layer_norm)
        self.v1 = _mlp(state_dim + rep_dim, hidden, 1, use_layer_norm=use_layer_norm)
        self.v2 = _mlp(state_dim + rep_dim, hidden, 1, use_layer_norm=use_layer_norm)

    def phi(self, s, g):
        return self.goal_encoder(g, s)

    def forward(self, s, g):
        x = torch.cat([s, self.phi(s, g)], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)


def stage_entries_from_instant(instant_stages):
    """逐帧瞬时 stage(int 数组) -> 各更高 stage 首次到达的下标(升序 int64 数组)。

    用运行最大值 latch 消抖;入口=latch 比前一帧大的位置。全 0 返回空数组。
    若 demo 首帧已处于 stage k>0,index 0 记为 stage k 入口;比 k 低的 stage 在本 demo 不存在,不产生条目。
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
                    *, n_total, p_curr=0.2, p_traj=0.5, p_rand=0.3,
                    future_mode="stage_entry", discount=0.99):
    """HIQL 混采:current(p_curr)/future(p_traj)/random(p_rand)。

    future_mode 决定"未来目标"怎么从同 demo 里取:
      - 'stage_entry'(默认,本项目原口径):>= 当前下标的 stage 入口态里均匀取一个,无则末态。
        语义清晰(里程碑),但只覆盖入口帧,中间态作为目标几乎没被训到 -> V 在那留洞。
      - 'geometric'(HIQL geom_sample=1 口径):同 demo 内按几何分布(参数 1-discount)取
        min(idx + Geom, 末态),覆盖全部中间态、不留洞,更贴论文。
    idx/traj_id 为 (B,) np 数组。返回 (B,) goal 全局下标。
    P(random) = 1 − p_curr − p_traj;三者必须和为 1(断言保证)。
    """
    assert abs(p_curr + p_traj + p_rand - 1.0) < 1e-6, \
        f"p_curr+p_traj+p_rand must sum to 1, got {p_curr}+{p_traj}+{p_rand}"
    idx = np.asarray(idx, dtype=np.int64)
    B = len(idx)
    goal = rng.integers(0, n_total, size=B)
    final = np.array([last_idx_of[int(t)] for t in traj_id], dtype=np.int64)
    if future_mode == "geometric":
        offset = np.ceil(np.log(1.0 - rng.random(B)) / np.log(discount)).astype(np.int64)
        fut = np.minimum(idx + offset, final)
    elif future_mode == "stage_entry":
        fut = np.empty(B, dtype=np.int64)
        for j in range(B):
            ent = stage_entries_of[int(traj_id[j])]
            cand = ent[ent >= idx[j]]
            fut[j] = int(rng.choice(cand)) if len(cand) else final[j]
    else:
        raise ValueError(f"unknown future_mode: {future_mode!r}")
    denom = max(1.0 - p_curr, 1e-8)
    goal = np.where(rng.random(B) < p_traj / denom, fut, goal)
    goal = np.where(rng.random(B) < p_curr, idx, goal)
    return goal.astype(np.int64)


def train_gc_value(data, *, gamma=0.99, expectile=0.7, ema=0.005, lr=3e-4,
                   batch_size=256, steps=50_000, rep_dim=10, hidden=256, seed=0,
                   future_mode="stage_entry", use_layer_norm=False,
                   value_loss_mode="shared_min"):
    """在扁平 GC 数据上训 action-free expectile goal-conditioned value。

    reward r(s,g)=0 if s==g else -1;到达 goal 或 demo 末步都截断 bootstrap。
    EMA target。返回 (model, v_stats)。future_mode 透传给 sample_gc_goals。
    value_loss_mode:
      - 'shared_min'(默认,现状):两 critic 都回归 y=r+γ·mask·min(nv1,nv2),残差自门控 expectile。
      - 'hiql'(对齐参考):per-critic 目标 q_i=r+γ·mask·nv_i(不取 min)、adv=q−V_target 门控的
        两参 expectile、当前态 V 走 target 网。复刻 HIQL compute_value_loss 的 expectile+双 critic
        两处(mask 仍含 (1-done),done-mask 不在本轮范围)。
    """
    if value_loss_mode not in ("shared_min", "hiql"):
        raise ValueError(f"unknown value_loss_mode: {value_loss_mode!r}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    states = data["states"]
    D = states.shape[1]
    model = GoalConditionedVF(D, rep_dim, hidden, use_layer_norm=use_layer_norm)
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(data["s_idx"])
    bs = min(batch_size, n)
    s_idx, sn_idx, traj_id = data["s_idx"], data["sn_idx"], data["traj_id"]
    done_all = data["done"]
    for _ in range(steps):
        b = rng.integers(0, n, size=bs)
        si, sni, tj = s_idx[b], sn_idx[b], traj_id[b]
        gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"],
                             rng, n_total=len(states),
                             future_mode=future_mode, discount=gamma)
        s = states[si]
        s_next = states[sni]
        g = states[gi]
        success = torch.tensor(si == gi, dtype=torch.float32)
        reward = success - 1.0
        mask = (1.0 - success) * (1.0 - done_all[b])
        if value_loss_mode == "shared_min":
            with torch.no_grad():
                nv1, nv2 = target(s_next, g)
                nv = torch.minimum(nv1, nv2)
                y = reward + gamma * mask * nv
            v1, v2 = model(s, g)
            loss = expectile_loss(y - v1, expectile) + expectile_loss(y - v2, expectile)
        else:  # "hiql"
            with torch.no_grad():
                nv1, nv2 = target(s_next, g)
                nv = torch.minimum(nv1, nv2)
                q = reward + gamma * mask * nv
                v1t, v2t = target(s, g)            # 当前态 V 走 target 网
                v_t = 0.5 * (v1t + v2t)
                adv = q - v_t
                q1 = reward + gamma * mask * nv1   # per-critic 目标,不取 min
                q2 = reward + gamma * mask * nv2
            v1, v2 = model(s, g)
            loss = (expectile_loss_weighted(adv, q1 - v1, expectile)
                    + expectile_loss_weighted(adv, q2 - v2, expectile))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), model.parameters()):
                tp.mul_(1.0 - ema).add_(ema * mp)
    with torch.no_grad():
        gl = np.array([data["last_idx_of"][int(d)] for d in traj_id], dtype=np.int64)
        vv1, vv2 = model(states[s_idx], states[gl])
        vv = torch.minimum(vv1, vv2)
        v_stats = {"min": float(vv.min()), "max": float(vv.max()), "mean": float(vv.mean())}
    return model, v_stats


def save_gc_value(path, model, *, v_stats, mean, std, dataset_id,
                  state_mode="eef_piece", rel_piece_stats=None,
                  value_loss_mode="shared_min"):
    """存 gc_value.pt:权重 + 维度 + v_stats + state mean/std + dataset_id + state_mode
    + value_loss_mode(provenance,旧档无此键时 load_gc_value 回退 'shared_min')
    (+ eef_piece 的 rel_piece mean/std,供 Phase 3 online 同源标准化)。"""
    payload = {
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "rep_dim": model.rep_dim,
        "hidden": model.hidden,
        "use_layer_norm": model.use_layer_norm,
        "v_stats": v_stats,
        "mean": mean,
        "std": std,
        "dataset_id": dataset_id,
        "state_mode": state_mode,
        "value_loss_mode": value_loss_mode,
    }
    if rel_piece_stats is not None:
        payload["rel_piece_mean"], payload["rel_piece_std"] = rel_piece_stats
    torch.save(payload, path)


def load_gc_value(path, map_location="cpu"):
    """读 gc_value.pt,重建 GoalConditionedVF(eval),返回 (model, info)。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = GoalConditionedVF(ckpt["state_dim"], ckpt["rep_dim"], ckpt["hidden"],
                              use_layer_norm=ckpt.get("use_layer_norm", False))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    info = {k: ckpt[k] for k in ("v_stats", "mean", "std", "dataset_id")}
    info["state_mode"] = ckpt.get("state_mode", "eef_piece")
    info["value_loss_mode"] = ckpt.get("value_loss_mode", "shared_min")
    info["rel_piece_mean"] = ckpt.get("rel_piece_mean")
    info["rel_piece_std"] = ckpt.get("rel_piece_std")
    return model, info


@torch.no_grad()
def critic_divergence(model, seqs):
    """双 critic 在真态上的分化度诊断:corr(v1, v2) 与 mean|v1−v2|。

    g 取各 demo 末态(与 verify 状态侧同口径)。corr 越低、|Δ| 越大 = 两 critic 越分化 =
    min(v1,v2) ensemble 越有意义。shared_min 下两 critic 易趋同(corr→1);hiql per-critic 目标应更分化。
    返回 {'corr': float, 'mean_abs_diff': float}。
    """
    v1s, v2s = [], []
    for seq in seqs:
        s = torch.as_tensor(np.ascontiguousarray(seq), dtype=torch.float32)
        g = torch.as_tensor(np.ascontiguousarray(np.broadcast_to(seq[-1], seq.shape)),
                            dtype=torch.float32)
        a, b = model(s, g)
        v1s.append(a.numpy())
        v2s.append(b.numpy())
    v1, v2 = np.concatenate(v1s), np.concatenate(v2s)
    if len(v1) < 2:
        return {"corr": float("nan"), "mean_abs_diff": float(np.abs(v1 - v2).mean())}
    return {"corr": float(np.corrcoef(v1, v2)[0, 1]),
            "mean_abs_diff": float(np.abs(v1 - v2).mean())}
