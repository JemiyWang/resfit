"""零侵入 serve 特征器:在 Policy.infer 返回里透出 pi05 prefix 池化特征。

pool_prefix 纯函数(只依赖 jax)可单测;真前向(后续 task)惰性 import pi0,只在真 serve 进程触达。
本文件用 openpi 的 .venv 运行(因 jax/openpi 都在那)。
设计见 resfit/docs/superpowers/specs/2026-06-14-hiql-value-pi0-feat-via-serve-design.md。
"""
from __future__ import annotations

import jax.numpy as jnp


def pool_prefix(prefix_out, mask, pooling="last"):
    """prefix_out [B,L,D] + mask [B,L](bool/0-1) → [B,D]。

    last = 末**有效** token(按 mask.sum-1,非 [:, -1]);mean = 有效 token 均值(排除 padding)。
    """
    if pooling == "last":
        last_idx = mask.astype(jnp.int32).sum(axis=1) - 1
        return prefix_out[jnp.arange(prefix_out.shape[0]), last_idx]
    if pooling == "mean":
        m = mask.astype(prefix_out.dtype)[..., None]
        return (prefix_out * m).sum(axis=1) / jnp.clip(m.sum(axis=1), 1.0, None)
    raise ValueError(f"unknown pooling {pooling!r}")
