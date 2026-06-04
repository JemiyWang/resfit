"""Smoke eval: drive a pi0/pi05 base policy (via websocket) on the three-piece dexmg env.

First version validates the cross-process link only (connect + step loop + sanity asserts),
not success rate. Grows into the Task 10 base-policy gate later.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


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


def run_smoke(env, base_policy, n_episodes: int, max_steps: int, action_dim: int) -> dict:
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


ACTION_DIM = 14
TASK_NAME = "TwoArmThreePieceAssembly"
PROMPT = "assemble the three pieces"
IMAGE_KEY_MAP = {
    "observation.images.agentview": "base",
    "observation.images.robot0_eye_in_hand": "left_wrist",
    "observation.images.robot1_eye_in_hand": "right_wrist",
}


def format_report(report) -> str:
    n = len(report["episodes"])
    times = report["infer_times"]
    avg_ms = (sum(times) / len(times) * 1000.0) if times else 0.0
    return (
        f"smoke report: {n} episodes, lengths={report['episodes']}, "
        f"total_steps={sum(report['episodes'])}, "
        f"avg_infer={avg_ms:.1f}ms, "
        f"action_range=[{report['action_min']:.3f}, {report['action_max']:.3f}]"
    )


def build_smoke_env(device, camera_size=84):
    """Bare three-piece vec env (no residual wrapper)."""
    from resfit.dexmg.environments.dexmg import create_vectorized_env
    return create_vectorized_env(
        env_name=TASK_NAME, num_envs=1, device=device, camera_size=camera_size
    )


def build_pi0_base_policy(host, port, device):
    """Connect to the pi0/pi05 websocket server and wrap as a step-level base policy."""
    from resfit.rl_finetuning.config.residual_td3 import BasePolicyConfig
    from resfit.lerobot.policies.pi05 import load_pi05_base_policy

    cfg = BasePolicyConfig(
        type="pi05",
        host=host,
        port=port,
        action_dim=ACTION_DIM,
        execute_horizon=30,
        prompt=PROMPT,
        image_key_map=dict(IMAGE_KEY_MAP),
    )
    return load_pi05_base_policy(cfg, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description="pi0/pi05 base-policy cross-process smoke eval (three-piece)")
    parser.add_argument("--host", default="127.0.0.1", help="pi0 websocket server host")
    parser.add_argument("--port", type=int, default=8000, help="pi0 websocket server port")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device} host={args.host} port={args.port}")

    try:
        base_policy = build_pi0_base_policy(args.host, args.port, device)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"failed to connect base policy at {args.host}:{args.port} -- "
            f"is the pi0 server running? (see plan Task 9b). cause: {exc}"
        ) from exc

    env = build_smoke_env(device)
    report = run_smoke(env, base_policy, args.n_episodes, args.max_steps, ACTION_DIM)
    print(format_report(report))
    print("[smoke] PASS: cross-process link ran without crashing.")


if __name__ == "__main__":
    main()
