"""冻结 ACT encoder 特征器(act_feat);不改 ACT,复用其组件。

纯函数(pool/concat/signature)只依赖 torch/numpy,单测可跑;ActFeatureExtractor 需真 ACT。
"""
from __future__ import annotations

import hashlib
import warnings

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

    def weight_fingerprint(self) -> str:
        """对冻结的 self.act 算权重指纹（同源校验用）。"""
        return act_weight_fingerprint(self.act)


def act_weight_fingerprint(policy) -> str:
    """对 policy.state_dict() 的确定性 sha256 指纹（hexdigest）。

    确定性：按 key 排序；浮点张量先 detach().cpu().float().contiguous() 去掉
    设备/内存布局/当前精度差异；整型/bool 缓冲按原 dtype 取字节。key/dtype/shape/bytes
    全部进哈希。保证：同一份内存权重 → 同一 hash；改任一权重 → hash 变。
    不保证 fp16 存档 vs fp32 存档相等（那本就是两份不同的值）。
    """
    h = hashlib.sha256()
    sd = policy.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k]
        if not torch.is_tensor(t):
            continue
        t = t.detach().cpu()
        if t.is_floating_point():
            t = t.float()
        t = t.contiguous()
        h.update(k.encode("utf-8") + b"\x00")
        h.update(str(t.dtype).encode("utf-8") + b"\x00")
        h.update(repr(tuple(t.shape)).encode("utf-8") + b"\x00")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assert_act_base_samesource(*, gv_sha, cache_sha, base_sha, allow_mismatch=False):
    """act_feat 在线 base ACT 同源校验（纯逻辑，便于单测）。

    - cache_sha 与 gv_sha 两边都有且不等 → 离线就异源（cache≠gc_value），raise。
    - gv_sha is None（旧 cache/gc_value 无指纹）→ 无法校验，warn 并跳过。
    - base_sha == gv_sha → 同源，静默返回。
    - base_sha != gv_sha → 默认 raise；allow_mismatch=True 则降级 warn。
    """
    if cache_sha and gv_sha and cache_sha != gv_sha:
        raise ValueError(
            f"[act_feat] cache 与 gc_value 权重指纹不符: {cache_sha} vs {gv_sha}（离线就异源）")
    if gv_sha is None:
        warnings.warn(
            "[act_feat] 产物无权重指纹（旧 cache/gc_value），无法校验在线 base 同源；"
            "务必先过一致性 smoke", stacklevel=2)
        return
    if base_sha == gv_sha:
        return
    if allow_mismatch:
        warnings.warn(
            f"[act_feat] 在线 base 权重指纹 != 离线（{base_sha} vs {gv_sha}），"
            "--allow_act_base_mismatch 已放行", stacklevel=2)
        return
    raise ValueError(
        f"[act_feat] 在线 base_policy 权重 != 离线 build cache 的 ACT"
        f"（指纹 {base_sha} vs {gv_sha}）；确认同源，或加 --allow_act_base_mismatch 放行")
