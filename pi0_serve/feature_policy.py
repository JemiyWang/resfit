"""零侵入 serve 特征器:在 Policy.infer 返回里透出 pi05 prefix 池化特征。

pool_prefix 纯函数(只依赖 jax)可单测;真前向(后续 task)惰性 import pi0,只在真 serve 进程触达。
本文件用 openpi 的 .venv 运行(因 jax/openpi 都在那)。
设计见 resfit/docs/superpowers/specs/2026-06-14-hiql-value-pi0-feat-via-serve-design.md。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def pool_prefix(prefix_out, mask, pooling="last"):
    """prefix_out [B,L,D] + mask [B,L](bool/0-1) → [B,D]。

    pooling: 'last'(默认) 或 'mean'
    last = 末**有效** token(按 mask.sum-1,非 [:, -1]);mean = 有效 token 均值(排除 padding)。
    """
    if pooling == "last":
        # prefix 始终含有效 image token,mask.sum>=1;all-padding 不应发生
        last_idx = mask.astype(jnp.int32).sum(axis=1) - 1
        return prefix_out[jnp.arange(prefix_out.shape[0]), last_idx]
    if pooling == "mean":
        m = mask.astype(prefix_out.dtype)[..., None]
        return (prefix_out * m).sum(axis=1) / jnp.clip(m.sum(axis=1), 1.0, None)
    raise ValueError(f"unknown pooling {pooling!r}")


def make_prefix_feat_fn():
    """真前向(惰性 import pi0):obs → 复现 Policy.infer 预处理 → embed_prefix → prefix llm 前向 → pool。

    真 API 底稿:pi0.sample_actions:317-320 + pi0_value._forward_backbone:266-273。返回 [D] np.float32。
    单测不触达此(走注入 stub);真权重在 opt-in smoke / 起真 serve 时验证。
    """
    import jax
    import jax.numpy as _jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    def _fn(inner, obs, pooling):
        inputs = jax.tree.map(lambda x: x, obs)                       # copy(同 Policy.infer)
        inputs = inner._input_transform(inputs)                        # 同款预处理(同源命门)
        inputs = jax.tree.map(lambda x: _jnp.asarray(x)[None, ...], inputs)  # 加 batch
        observation = _model.Observation.from_dict(inputs)
        model = inner._model
        tok, mask, ar = model.embed_prefix(observation)               # image+prompt,不含 state
        (prefix_out, _), _ = model.PaliGemma.llm(
            [tok, None], mask=make_attn_mask(mask, ar), positions=_jnp.cumsum(mask, 1) - 1)
        pooled = pool_prefix(prefix_out, mask, pooling)               # [1, D]
        return np.asarray(pooled[0], dtype=np.float32)

    return _fn


class FeaturePolicy:
    """包住原 Policy,在 infer 返回里加 prefix_feat。零侵入:不改原 Policy/协议。

    prefix_feat_fn(inner, obs, pooling) -> [D];默认 = make_prefix_feat_fn()(真前向)。
    """

    def __init__(self, inner, pooling="last", prefix_feat_fn=None):
        self.inner = inner
        self.pooling = pooling
        self._prefix_feat_fn = prefix_feat_fn or make_prefix_feat_fn()

    @property
    def metadata(self):
        return self.inner.metadata

    def infer(self, obs, **kw):
        outputs = self.inner.infer(obs, **kw)                         # 原 actions/state 不动
        feat = self._prefix_feat_fn(self.inner, obs, self.pooling)
        outputs["prefix_feat"] = np.asarray(feat, dtype=np.float32)
        return outputs


def wrap_with_feature(inner, pooling="last", prefix_feat_fn=None):
    """把一个 Policy 包成透特征的 FeaturePolicy。"""
    return FeaturePolicy(inner, pooling=pooling, prefix_feat_fn=prefix_feat_fn)
