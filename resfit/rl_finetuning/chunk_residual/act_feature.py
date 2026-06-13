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


class ActFeatureExtractor:
    """冻结 ACT,用 model.encoder 的 forward hook 抓 encoder_out → 池化 ⊕ 原始 proprio。

    注意:__init__ 会原地修改传入的 act_policy:调用 .eval() 并冻结所有参数
    (requires_grad=False)。调用方不应在此之后再对该 policy 调用 .train()。
    """

    def __init__(self, act_policy, image_keys, proprio_key="observation.state",
                 pooling="mean", proprio_dim=18):
        self.act = act_policy
        self.act.eval()
        for p in self.act.parameters():
            p.requires_grad_(False)
        self.image_keys = list(image_keys)
        if hasattr(act_policy, "config") and getattr(act_policy.config, "image_features", None) is not None:
            canonical = list(act_policy.config.image_features.keys())
            assert list(image_keys) == canonical, (
                f"image_keys 顺序须与 act.config.image_features 一致: {list(image_keys)!r} vs {canonical!r}")
        self.proprio_key = proprio_key
        self.pooling = pooling
        self._proprio_dim = proprio_dim

    @property
    def feature_dim(self) -> int:
        return int(self.act.config.dim_model) + self._proprio_dim

    @torch.no_grad()
    def embed_batch(self, raw_obs: dict) -> torch.Tensor:
        self.act.eval()
        _p = next(self.act.parameters(), None)
        dev = _p.device if _p is not None else torch.device("cpu")
        raw_obs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw_obs.items()}
        batch = dict(self.act.normalize_inputs(raw_obs))
        batch["observation.images"] = [batch[k] for k in self.image_keys]
        captured = {}
        h = self.act.model.encoder.register_forward_hook(
            lambda m, i, o: captured.__setitem__("enc", o))
        try:
            self.act.model(batch)
        finally:
            h.remove()
        emb = pool_encoder_out(captured["enc"], self.pooling)   # [B, D_emb]
        proprio = torch.as_tensor(raw_obs[self.proprio_key], dtype=torch.float32).to(emb.device)
        return concat_proprio(emb, proprio)

    def signature(self, act_ckpt_id) -> dict:
        return act_feat_signature(act_ckpt_id, self.image_keys, self.proprio_key, self.pooling)
