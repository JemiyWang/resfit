"""Smoke eval: drive a pi0/pi05 base policy (via websocket) on the three-piece dexmg env.

First version validates the cross-process link only (connect + step loop + sanity asserts),
not success rate. Grows into the Task 10 base-policy gate later.
"""
from __future__ import annotations

import numpy as np


def check_action(arr, action_dim, abs_limit=5.0):
    """Assert a base-policy action chunk-step is sane; raise ValueError otherwise."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"action must be 2-D [B, dim], got shape {a.shape}")
    if a.shape[-1] != action_dim:
        raise ValueError(f"action last dim must be {action_dim}, got {a.shape[-1]}")
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError(f"action must be floating dtype, got {a.dtype}")
    if not np.isfinite(a).all():
        raise ValueError("action contains NaN/Inf")
    peak = float(np.abs(a).max())
    if peak > abs_limit:
        raise ValueError(f"action abs max {peak:.3f} exceeds limit {abs_limit}")
