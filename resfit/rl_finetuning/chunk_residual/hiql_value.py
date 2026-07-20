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


def build_transitions(state_seqs):
    """list of [T_i, D] 数组(每条 demo 的标准化 state 序列)-> (s, s_next, done) float32 张量。

    每条长 T 的 demo 产出 T-1 个 transition;done 在该 demo 末 transition=1(到达 goal=轨迹终点)。
    T<2 的 demo 跳过。done 形状 [N,1]。
    """
    s_list, sn_list, done_list = [], [], []
    for seq in state_seqs:
        seq = np.asarray(seq, dtype=np.float32)
        T = seq.shape[0]
        if T < 2:
            continue
        s_list.append(seq[:-1])
        sn_list.append(seq[1:])
        d = np.zeros(T - 1, dtype=np.float32)
        d[-1] = 1.0
        done_list.append(d)
    s = torch.from_numpy(np.concatenate(s_list, axis=0))
    sn = torch.from_numpy(np.concatenate(sn_list, axis=0))
    done = torch.from_numpy(np.concatenate(done_list, axis=0)).unsqueeze(1)
    return s, sn, done


def build_transitions_with_rewards(state_seqs, success_flags):
    """带成功/失败标签的 transitions -> (s, s_next, done, reward)。

    与 build_transitions 相比:done 的语义与逐位取值完全相同(轨迹末 transition=1),
    额外产出 reward——成功轨迹末端 +1、失败轨迹末端 −1、中间步 0(RISE value 口径)。

    ★ 跳过 T<2 的逻辑与 build_transitions 逐位一致,且 success_flags 在同一次遍历里消费:
      若分两次构造(seqs 一次、flags 一次),被跳过的短 demo 会让标签整体错位一格且不报错。
    """
    assert len(success_flags) == len(state_seqs), \
        f"success_flags 长度 {len(success_flags)} != state_seqs {len(state_seqs)}"
    s_list, sn_list, done_list, rew_list = [], [], [], []
    for seq, ok in zip(state_seqs, success_flags):
        seq = np.asarray(seq, dtype=np.float32)
        T = seq.shape[0]
        if T < 2:
            continue
        s_list.append(seq[:-1])
        sn_list.append(seq[1:])
        d = np.zeros(T - 1, dtype=np.float32)
        d[-1] = 1.0
        done_list.append(d)
        r = np.zeros(T - 1, dtype=np.float32)
        r[-1] = 1.0 if ok else -1.0
        rew_list.append(r)
    s = torch.from_numpy(np.concatenate(s_list, axis=0))
    sn = torch.from_numpy(np.concatenate(sn_list, axis=0))
    done = torch.from_numpy(np.concatenate(done_list, axis=0)).unsqueeze(1)
    reward = torch.from_numpy(np.concatenate(rew_list, axis=0)).unsqueeze(1)
    return s, sn, done, reward


def save_value(path, model, *, v_stats, mean, std, dataset_id,
               state_mode="eef", rel_piece_stats=None,
               act_feat_signature=None, act_weight_sha=None):
    """存 value.pt:权重 + 维度 + V 统计 + state mean/std + dataset_id + state_mode(+ rel_piece stats)。

    state_mode=eef(18)|eef_piece(30);eef_piece 时 rel_piece_stats=(mean(12,),std(12,)),
    ③b online 推理标准化 rel_piece 用(③a' object-aware)。
    """
    payload = {
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "hidden": model.hidden,
        "v_stats": v_stats,
        "mean": mean,
        "std": std,
        "dataset_id": dataset_id,
        "state_mode": state_mode,
    }
    if rel_piece_stats is not None:
        payload["rel_piece_mean"], payload["rel_piece_std"] = rel_piece_stats
    if act_feat_signature is not None:
        payload["act_feat_signature"] = dict(act_feat_signature)
    if act_weight_sha is not None:
        payload["act_weight_sha"] = str(act_weight_sha)
    torch.save(payload, path)


def load_value(path, map_location="cpu"):
    """读 value.pt,重建 ValueMLP(eval 模式),返回 (model, info_dict)。

    info_dict 含 v_stats / mean / std / dataset_id。
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = ValueMLP(ckpt["state_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    info = {k: ckpt[k] for k in ("v_stats", "mean", "std", "dataset_id")}
    info["state_dim"] = ckpt["state_dim"]
    info["state_mode"] = ckpt.get("state_mode", "eef")
    info["rel_piece_mean"] = ckpt.get("rel_piece_mean")
    info["rel_piece_std"] = ckpt.get("rel_piece_std")
    info["act_feat_signature"] = ckpt.get("act_feat_signature")
    info["act_weight_sha"] = ckpt.get("act_weight_sha")
    return model, info


def train_value(s, s_next, done, *, reward=None, gamma=0.99, expectile=0.7, ema=0.005,
                lr=3e-4, batch_size=256, steps=50000, hidden=256, seed=0):
    """在 (s, s_next, done) 上训 action-free IQL expectile value。

    reward=None(默认):goal-reaching 内部 reward = done(末步=1 否则 0),与既有实现逐位等价。
    reward 显式传入(success_signed 路):成功轨迹末 +1、失败轨迹末 −1、中间 0
    (来自 build_transitions_with_rewards)。EMA target net 稳定 bootstrap。
    返回 (model, v_stats),v_stats = 训练后全数据上 V 的 {min,max,mean}。
    """
    torch.manual_seed(seed)
    n, d = s.shape
    model = ValueMLP(d, hidden)
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    reward = done if reward is None else reward  # 默认 r_t = 1 if done else 0 == done
    bs = min(batch_size, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,))
        with torch.no_grad():
            y = discounted_target(reward[idx], target(s_next[idx]), done[idx], gamma)
        v = model(s[idx])
        loss = expectile_loss(y - v, expectile)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), model.parameters()):
                tp.mul_(1.0 - ema).add_(ema * mp)
    with torch.no_grad():
        allv = model(s)
        v_stats = {"min": float(allv.min()), "max": float(allv.max()), "mean": float(allv.mean())}
    return model, v_stats
