"""HIQL 分层路运行时 helper(Phase 3):载冻结 gc_value+high_actor,在线/离线算潜子目标 z。

约束:给 actor/critic 的 observation.state 恒 18 维;30 维 eef_piece(18+12 标准化 rel)只用于
算 z;z(10 维)是唯一进策略的 object-aware 信号。online/offline 用同一套 rel mean/std 标准化。
设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import load_high_actor


class HiqlSubgoal:
    def __init__(self, gc_value, high_actor, goal30, rel_mean, rel_std, device="cpu"):
        self.vf = gc_value.to(device).eval()
        self.ha = high_actor.to(device).eval()
        for m in (self.vf, self.ha):
            for p in m.parameters():
                p.requires_grad_(False)
        self.rep_dim = gc_value.rep_dim
        self.device = device
        self.goal30 = torch.as_tensor(np.asarray(goal30), dtype=torch.float32, device=device).reshape(-1)
        self.rel_mean = torch.as_tensor(np.asarray(rel_mean), dtype=torch.float32, device=device)
        self.rel_std = torch.as_tensor(np.asarray(rel_std), dtype=torch.float32, device=device)

    @classmethod
    def from_ckpts(cls, gc_value_ckpt, high_actor_ckpt, *, goal30, device="cpu"):
        gc, info = load_gc_value(gc_value_ckpt, map_location=device)
        ha, _ = load_high_actor(high_actor_ckpt, map_location=device)
        assert info["state_mode"] == "eef_piece", "分层路 gc_value 须 eef_piece(object-aware)"
        return cls(gc, ha, goal30, info["rel_piece_mean"], info["rel_piece_std"], device=device)

    def build_state30(self, state_std, rel_raw):
        """18 维已标准化 state(tensor [B,18]) + raw rel_piece([B,12] np/tensor) -> [B,30] tensor。"""
        x = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        rel = torch.as_tensor(np.asarray(rel_raw), dtype=torch.float32, device=self.device)
        if rel.ndim == 1:
            rel = rel.unsqueeze(0)
        rel_n = (rel - self.rel_mean) / self.rel_std
        return torch.cat([x, rel_n], dim=-1)

    @torch.no_grad()
    def subgoal_online(self, state_std, rel_raw):
        """在线:z = π^h(s30, goal30)(取分布均值,确定性)。返回 [B, rep_dim]。"""
        s30 = self.build_state30(state_std, rel_raw)
        g30 = self.goal30.unsqueeze(0).expand(s30.shape[0], -1)
        return self.ha(s30, g30).mean

    @torch.no_grad()
    def subgoal_waypoint(self, s30_base, s30_target):
        """离线:z = φ(base=s_t, target=s_{t+k})(真航点)。返回 [B, rep_dim]。"""
        b = torch.as_tensor(s30_base, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(s30_target, dtype=torch.float32, device=self.device)
        return self.vf.phi(b, t)
