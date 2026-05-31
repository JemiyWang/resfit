"""PyTorch 版动作 chunk 自编码器(从 kai0/rlt 的 JAX 版移植)。

时序卷积 AE,作用在已归一化的展平 chunk [B, chunk_length*action_dim] 上。
LayerNorm 始终作用在 channel 维(channels-last 视角最后一维)。
"""
from __future__ import annotations

import torch
from torch import nn


class _Conv1dBlock(nn.Module):
    """channels-last 输入 [B,T,C] 的 Conv1d + LayerNorm(C) + ReLU。"""

    def __init__(self, channels: int, kernel: int, use_layer_norm: bool):
        super().__init__()
        pad = kernel // 2  # SAME(kernel 为奇数,如 3 → pad=1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad)
        self.norm = nn.LayerNorm(channels) if use_layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,C]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B,T,C]
        if self.norm is not None:
            y = self.norm(y)
        return torch.relu(y)


class ActionAutoencoder(nn.Module):
    def __init__(self, action_dim: int, chunk_length: int, latent_dim: int = 64,
                 hidden_dim: int = 256, conv_layers: int = 2, conv_kernel: int = 3,
                 use_layer_norm: bool = True):
        super().__init__()
        if conv_kernel % 2 == 0:
            raise ValueError(f"conv_kernel must be odd for SAME padding, got {conv_kernel}")
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.flat_action_dim = chunk_length * action_dim

        self.encoder_input_dense = nn.Linear(action_dim, hidden_dim)
        self.encoder_convs = nn.ModuleList(
            [_Conv1dBlock(hidden_dim, conv_kernel, use_layer_norm) for _ in range(conv_layers)]
        )
        self.encoder_latent_dense = nn.Linear(hidden_dim, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim) if use_layer_norm else None

        self.decoder_input_dense = nn.Linear(latent_dim, chunk_length * hidden_dim)
        self.decoder_convs = nn.ModuleList(
            [_Conv1dBlock(hidden_dim, conv_kernel, use_layer_norm) for _ in range(conv_layers)]
        )
        self.decoder_output_dense = nn.Linear(hidden_dim, action_dim)

    def _check(self, x: torch.Tensor):
        if x.ndim != 2 or x.shape[-1] != self.flat_action_dim:
            raise ValueError(f"expected [B,{self.flat_action_dim}], got {tuple(x.shape)}")

    def encode(self, action_flat: torch.Tensor) -> torch.Tensor:
        self._check(action_flat)
        b = action_flat.shape[0]
        x = action_flat.reshape(b, self.chunk_length, self.action_dim)
        x = torch.relu(self.encoder_input_dense(x))   # [B,T,H]
        for blk in self.encoder_convs:
            x = blk(x)
        x = x.mean(dim=1)                              # [B,H] 沿时间平均
        z = self.encoder_latent_dense(x)              # [B,LAT]
        if self.latent_norm is not None:
            z = self.latent_norm(z)
        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"expected [B,{self.latent_dim}], got {tuple(latent.shape)}")
        b = latent.shape[0]
        x = self.decoder_input_dense(latent)
        x = x.reshape(b, self.chunk_length, self.hidden_dim)
        x = torch.relu(x)
        for blk in self.decoder_convs:
            x = blk(x)
        seq = self.decoder_output_dense(x)            # [B,T,D]
        return seq.reshape(b, self.flat_action_dim)

    def forward(self, action_flat: torch.Tensor, return_latent: bool = False):
        z = self.encode(action_flat)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon


def action_autoencoder_loss(model: ActionAutoencoder, action_flat: torch.Tensor,
                            velocity_loss_weight: float = 0.1):
    """L1 重构 + 速度损失(相邻步差分 L1)。返回 (loss, metrics)。

    注意:M2 里 AE 是冻结的——若仅用于监控,请在 torch.no_grad() 下调用,
    避免经冻结 encoder/decoder 构建不必要的计算图。
    """
    recon = model(action_flat)
    recon_l1 = (recon - action_flat).abs().mean()

    b = action_flat.shape[0]
    tgt = action_flat.reshape(b, model.chunk_length, model.action_dim)
    rec = recon.reshape(b, model.chunk_length, model.action_dim)
    if model.chunk_length > 1:
        tgt_v = tgt[:, 1:, :] - tgt[:, :-1, :]
        rec_v = rec[:, 1:, :] - rec[:, :-1, :]
        velocity_l1 = (rec_v - tgt_v).abs().mean()
    else:
        velocity_l1 = torch.zeros((), dtype=recon.dtype, device=recon.device)

    loss = recon_l1 + velocity_loss_weight * velocity_l1
    return loss, {"loss": loss.detach(), "recon_l1": recon_l1.detach(),
                  "velocity_l1": velocity_l1.detach()}
