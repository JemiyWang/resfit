"""把 ③a 的冻结 value 当 PBS 势函数 Φ(模块 ③b)。

设计见 docs/superpowers/specs/2026-06-07-hiql-potential-design.md。
potential_shaping 是通用 PBS 公式(Φ 可为 int stage 或 V(state)*scale);
HiqlPotential 加载冻结 value 并把标准化 state 映射到 Φ。
"""
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_value import load_value


def potential_shaping(phi_start, phi_next, *, bonus, gamma, done):
    """通用 PBS 整形:F = bonus*(gamma*phi_next - phi_start),done 时 phi_next=0。

    phi_start/phi_next 可为 float 或单元素张量(online b=1);返回 float。
    """
    pn = 0.0 if done else float(phi_next)
    return bonus * (gamma * pn - float(phi_start))


def gc_subgoal_shaping(potential, s_start, s_end, z_start, *, bonus, gamma, done):
    """A2 单步 shaping:F = bonus·(γ·Φ(s_end,z_start) − Φ(s_start,z_start))。

    phi_start 与 phi_next 用**同一个 z_start**(保证单步 PBS);done 时 phi_next=0
    (由 potential_shaping 处理)。返回 float。
    """
    phi_a = potential.phi(s_start, z_start)
    phi_b = potential.phi(s_end, z_start)
    return potential_shaping(phi_a, phi_b, bonus=bonus, gamma=gamma, done=done)


class HiqlPotential:
    """加载 ③a 的冻结 value,把标准化 state 映射到 PBS 势函数值 phi = V(state)*scale。

    scale = auto_scale * phi_scale,auto_scale=(num_stages-1)/(v_max-v_min) 让 V 动态范围
    匹配现状整数 stage Φ 的 [0,num_stages-1]。只缩放不平移(PBS 下平移会引入存活项)。
    """

    def __init__(self, model, scale, device="cpu",
                 state_mode="eef", rel_piece_mean=None, rel_piece_std=None,
                 feature_mean=None, feature_std=None,
                 act_feat_signature=None, act_weight_sha=None):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scale = float(scale)
        self.device = device
        # ③a' object-aware:eef_piece 时 V 是 30 维(18 本体 + 12 标准化 rel_piece)。
        # phi 入参仍是 18 维 std state + raw rel_piece,在此用训练同款 stats 标准化 rel 再拼,
        # 保证 online/offline 喂 V 的 30 维表示与 train_hiql_value 逐位同源。
        self.state_mode = state_mode
        self.feature_mean = None
        self.feature_std = None
        self.act_feat_signature = act_feat_signature
        self.act_weight_sha = act_weight_sha
        if state_mode == "act_feat":
            if feature_mean is None or feature_std is None:
                raise ValueError("state_mode='act_feat' requires feature mean/std in value checkpoint")
            self.feature_mean = torch.as_tensor(np.asarray(feature_mean), dtype=torch.float32, device=device)
            self.feature_std = torch.as_tensor(np.asarray(feature_std), dtype=torch.float32, device=device)
            if int(self.feature_mean.numel()) != int(self.model.state_dim):
                raise ValueError(
                    f"act_feat mean dim {self.feature_mean.numel()} != value state_dim {self.model.state_dim}"
                )
            if int(self.feature_std.numel()) != int(self.model.state_dim):
                raise ValueError(
                    f"act_feat std dim {self.feature_std.numel()} != value state_dim {self.model.state_dim}"
                )
        self.rel_mean = None
        self.rel_std = None
        if state_mode == "eef_piece":
            if rel_piece_mean is None or rel_piece_std is None:
                raise ValueError("state_mode='eef_piece' 需 rel_piece_mean/std(value.pt 应已存)")
            self.rel_mean = torch.as_tensor(np.asarray(rel_piece_mean),
                                            dtype=torch.float32, device=device)
            self.rel_std = torch.as_tensor(np.asarray(rel_piece_std),
                                           dtype=torch.float32, device=device)

    @classmethod
    def from_ckpt(cls, path, *, num_stages, phi_scale=1.0, device="cpu"):
        model, info = load_value(path, map_location=device)
        vmin, vmax = info["v_stats"]["min"], info["v_stats"]["max"]
        phi_range = max(int(num_stages) - 1, 1)
        auto_scale = phi_range / max(vmax - vmin, 1e-6)
        return cls(model, scale=auto_scale * phi_scale, device=device,
                   state_mode=info["state_mode"],
                   rel_piece_mean=info["rel_piece_mean"],
                   rel_piece_std=info["rel_piece_std"],
                   feature_mean=info["mean"],
                   feature_std=info["std"],
                   act_feat_signature=info.get("act_feat_signature"),
                   act_weight_sha=info.get("act_weight_sha"))

    def _value_input(self, state_std, rel_piece_raw):
        """构造喂 V 的输入:eef 模式直接 18 维;eef_piece 模式拼 30 维(标准化 rel)。"""
        x = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)
        if self.state_mode != "eef_piece":
            return x
        if rel_piece_raw is None:
            raise ValueError("state_mode='eef_piece' 的 phi 需传 rel_piece_raw(12,)")
        rel = torch.as_tensor(np.asarray(rel_piece_raw), dtype=torch.float32, device=self.device)
        rel_n = (rel - self.rel_mean) / self.rel_std
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if rel_n.ndim == 1:
            rel_n = rel_n.unsqueeze(0)
        return torch.cat([x, rel_n], dim=-1)

    def standardize_features(self, raw_feat):
        if self.state_mode != "act_feat":
            raise ValueError("standardize_features is only valid for state_mode='act_feat'")
        x = torch.as_tensor(raw_feat, dtype=torch.float32, device=self.device)
        return (x - self.feature_mean) / self.feature_std

    @torch.no_grad()
    def phi(self, state_std, rel_piece_raw=None):
        """[B,18] 已标准化 state(+ eef_piece 模式 raw rel_piece[B,12]/(12,))-> [B] 势函数值 = V*scale。

        eef 模式:rel_piece_raw 被忽略(逐位等价旧单参行为)。
        eef_piece 模式:用存储的 rel mean/std 标准化 rel,拼成 30 维喂 V(与训练同源)。
        """
        return self.model(self._value_input(state_std, rel_piece_raw)).squeeze(-1) * self.scale


class GcSubgoalPotential:
    """A2:把冻结 goal-conditioned value 当 PBS 势函数 Φ(s,z)=mean(V(s,z))·scale。

    与 HiqlPotential(单状态 V(s))正交:这里 phi 第二参是子目标 rep z(不是 rel_piece),
    用 value_from_rep 直接吃 z(z 须在半径 sqrt(rep_dim) 球面,renorm_subgoal=True 保证)。
    """

    def __init__(self, gc_value, scale, device="cpu"):
        self.model = gc_value.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scale = float(scale)
        self.device = device
        self.is_subgoal = True

    @classmethod
    def from_ckpt(cls, gc_value_ckpt, *, num_stages, phi_scale=1.0, device="cpu"):
        from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
        model, info = load_gc_value(gc_value_ckpt, map_location=device)
        vmin, vmax = info["v_stats"]["min"], info["v_stats"]["max"]
        phi_range = max(int(num_stages) - 1, 1)
        auto_scale = phi_range / max(vmax - vmin, 1e-6)
        return cls(model, scale=auto_scale * phi_scale, device=device)

    @torch.no_grad()
    def phi(self, s, z):
        """s:[B,state_dim] 已标准化 state;z:[B,rep_dim] 子目标 rep -> [B] 势函数值。"""
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        v1, v2 = self.model.value_from_rep(s, z)
        return 0.5 * (v1 + v2) * self.scale
