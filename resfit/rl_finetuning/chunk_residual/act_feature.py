"""冻结 ACT encoder 特征器(act_feat);不改 ACT,复用其组件。

纯函数(pool/concat/signature)只依赖 torch/numpy,单测可跑;ActFeatureExtractor 需真 ACT。
"""
from __future__ import annotations

import torch


def pool_encoder_out(encoder_out: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    """[seq, B, dim] -> [B, dim]。pooling='mean' 对 token 维求均值。"""
    if pooling != "mean":
        raise ValueError(f"unsupported pooling: {pooling}")
    return encoder_out.mean(dim=0)


def concat_proprio(emb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """[B, D_emb] ⊕ [B, D_proprio] -> [B, D_emb+D_proprio] float32(proprio 在尾部)。"""
    return torch.cat([emb.float(), proprio.float()], dim=-1)


def act_feat_signature(act_ckpt_id, image_keys, proprio_key, pooling) -> dict:
    """同源校验签名(纯数据)。"""
    return {
        "act_ckpt_id": str(act_ckpt_id),
        "image_keys": list(image_keys),
        "proprio_key": str(proprio_key),
        "pooling": str(pooling),
    }
