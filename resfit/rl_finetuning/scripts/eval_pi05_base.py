"""Smoke eval: drive a pi0/pi05 base policy (via websocket) on the three-piece dexmg env.

First version validates the cross-process link only (connect + step loop + sanity asserts),
not success rate. Grows into the Task 10 base-policy gate later.
"""
from __future__ import annotations

import time

import numpy as np


def check_action(arr, action_dim: int, abs_limit: float = 5.0) -> None:
    """Assert a base-policy action chunk-step is sane; raise ValueError otherwise."""
    if hasattr(arr, "detach"):  # handles both CPU and CUDA torch tensors
        arr = arr.detach().cpu().numpy()
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"action must be 2-D [B, dim], got shape {a.shape}")
    if a.shape[1] != action_dim:
        raise ValueError(f"action last dim must be {action_dim}, got {a.shape[1]}")
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError(f"action must be floating dtype, got {a.dtype}")
    if not np.isfinite(a).all():
        raise ValueError("action contains NaN/Inf")
    peak = float(np.abs(a).max())
    if peak > abs_limit:
        raise ValueError(f"action abs max {peak:.3f} exceeds limit {abs_limit}")


def _is_done(terminated, truncated):
    def _any(x):
        return bool(x.any()) if hasattr(x, "any") else bool(x)
    return _any(terminated) or _any(truncated)


def run_smoke(env, base_policy, n_episodes, max_steps, action_dim):
    """Run pure-base-policy rollouts; collect diagnostics. Raises on insane actions."""
    report = {
        "episodes": [],
        "infer_times": [],
        "action_min": float("inf"),
        "action_max": float("-inf"),
    }
    for _ in range(n_episodes):
        obs, _info = env.reset()
        base_policy.reset()
        steps = 0
        for _t in range(max_steps):
            t0 = time.perf_counter()
            action = base_policy.select_action(obs)
            report["infer_times"].append(time.perf_counter() - t0)
            check_action(action, action_dim)
            a = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
            report["action_min"] = min(report["action_min"], float(a.min()))
            report["action_max"] = max(report["action_max"], float(a.max()))
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if _is_done(terminated, truncated):
                break
        report["episodes"].append(steps)
    return report
