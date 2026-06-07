"""把 ③a 的冻结 value 当 PBS 势函数 Φ(模块 ③b)。

设计见 docs/superpowers/specs/2026-06-07-hiql-potential-design.md。
potential_shaping 是通用 PBS 公式(Φ 可为 int stage 或 V(state)*scale);
HiqlPotential 加载冻结 value 并把标准化 state 映射到 Φ。
"""
import torch

from resfit.rl_finetuning.chunk_residual.hiql_value import load_value


def potential_shaping(phi_start, phi_next, *, bonus, gamma, done):
    """通用 PBS 整形:F = bonus*(gamma*phi_next - phi_start),done 时 phi_next=0。

    phi_start/phi_next 可为 float 或单元素张量(online b=1);返回 float。
    """
    pn = 0.0 if done else float(phi_next)
    return bonus * (gamma * pn - float(phi_start))


class HiqlPotential:
    """加载 ③a 的冻结 value,把标准化 state 映射到 PBS 势函数值 phi = V(state)*scale。

    scale = auto_scale * phi_scale,auto_scale=(num_stages-1)/(v_max-v_min) 让 V 动态范围
    匹配现状整数 stage Φ 的 [0,num_stages-1]。只缩放不平移(PBS 下平移会引入存活项)。
    """

    def __init__(self, model, scale, device="cpu"):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scale = float(scale)
        self.device = device

    @classmethod
    def from_ckpt(cls, path, *, num_stages, phi_scale=1.0, device="cpu"):
        model, info = load_value(path, map_location=device)
        vmin, vmax = info["v_stats"]["min"], info["v_stats"]["max"]
        auto_scale = (num_stages - 1) / max(vmax - vmin, 1e-6)
        return cls(model, scale=auto_scale * phi_scale, device=device)

    @torch.no_grad()
    def phi(self, state_std):
        """state_std: [B,state_dim] 已标准化 -> [B] 势函数值 = V(state)*scale。"""
        return self.model(state_std.to(self.device)).squeeze(-1) * self.scale
