"""PyTorch 版 VLA-anchored residual-flow actor(从 kai0/rlt JAX 版移植)。

接口对齐 ResFiT Actor:forward(obs, std) -> utils.TruncatedNormal,返回的均值是"残差"
(decoded_delta);QAgent 在 loss/执行时再加 base_action。
冻结 AE 作为子模块(requires_grad=False),梯度只流到速度网络(经 decode(z_corr) 回传)。
"""
from __future__ import annotations

import torch
from torch import nn

from resfit.rl_finetuning.off_policy.common_utils import utils


def smooth_clip(x: torch.Tensor, limit: float) -> torch.Tensor:
    limit = float(limit)
    if limit <= 0.0:
        return x
    return limit * torch.tanh(x / limit)


class ResidualFlowActor(nn.Module):
    def __init__(self, repr_dim: int, patch_repr_dim: int, prop_dim: int, action_dim: int,
                 chunk_length: int, action_dim_per_step: int, frozen_ae,
                 feature_dim: int = 128, hidden_dim: int = 512, num_layers: int = 3,
                 latent_delta_scale: float = 0.05, action_delta_clip: float = 0.2,
                 velocity_init_scale: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        # 接口对齐 ResFiT Actor 的构造签名;当前实现无 spatial_emb 分支,故不使用
        self.patch_repr_dim = patch_repr_dim
        self.flat_action_dim = action_dim
        self.chunk_length = chunk_length
        self.action_dim_per_step = action_dim_per_step
        self.latent_dim = frozen_ae.latent_dim
        self.latent_delta_scale = latent_delta_scale
        self.action_delta_clip = action_delta_clip

        # 冻结 AE
        self.frozen_ae = frozen_ae
        for p in self.frozen_ae.parameters():
            p.requires_grad = False

        # 图像特征压缩(仿 Actor 非 spatial_emb 分支:Linear repr_dim->feature_dim)
        comp = [nn.Linear(repr_dim, feature_dim)]
        if use_layer_norm:
            comp.append(nn.LayerNorm(feature_dim))
        comp.append(nn.ReLU())
        self.compress = nn.Sequential(*comp)

        # s_p 维 = feature_dim + prop_dim;velocity 输入 = z_ref + z_rl + LN(s_p) + a_ref
        self.s_p_dim = feature_dim + prop_dim
        self.s_p_norm = nn.LayerNorm(self.s_p_dim) if use_layer_norm else nn.Identity()
        in_dim = self.latent_dim + self.latent_dim + self.s_p_dim + self.flat_action_dim

        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        self.velocity_mlp = nn.Sequential(*layers)
        self.velocity_out = nn.Linear(hidden_dim, self.latent_dim)
        # zero-init 末层 → 起步残差=0,action=base
        if velocity_init_scale == 0.0:
            nn.init.zeros_(self.velocity_out.weight)
        else:
            nn.init.normal_(self.velocity_out.weight, std=velocity_init_scale)
        nn.init.zeros_(self.velocity_out.bias)

    def trainable_parameters(self):
        """供注入时建优化器:排除冻结 AE。"""
        return [p for n, p in self.named_parameters() if not n.startswith("frozen_ae.")]

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_ae.eval()   # AE 始终 eval(仅 LayerNorm,行为不变,但保持语义清晰)
        return self

    def forward(self, obs: dict, std: float):
        a_ref = obs["observation.base_action"]                       # [B, FLAT]
        feat = self.compress(obs["feat"].flatten(1, -1))             # [B, feature_dim]
        s_p = torch.cat([feat, obs["observation.state"]], dim=-1)    # [B, s_p_dim]
        s_p = self.s_p_norm(s_p)

        z_ref = self.frozen_ae.encode(a_ref).detach()                # stopgrad
        a_ref_hat = self.frozen_ae.decode(z_ref).detach()            # stopgrad
        # z_rl:flow 的源 latent;确定性 TD3 下恒置 0,预留给未来 RL 修正(M2.3+)
        z_rl = torch.zeros_like(z_ref)

        h = torch.cat([z_ref, z_rl, s_p, a_ref], dim=-1)
        h = self.velocity_mlp(h)
        velocity = self.velocity_out(h)
        latent_delta = torch.tanh(velocity) * self.latent_delta_scale
        z_corr = z_ref + latent_delta
        a_corr_hat = self.frozen_ae.decode(z_corr)                   # 梯度经此回传到 velocity
        decoded_delta = smooth_clip(a_corr_hat - a_ref_hat, self.action_delta_clip)
        return utils.TruncatedNormal(decoded_delta, std)
