from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import torch


def clone_tensor_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone tensor values so z perturbations never mutate caller-owned batches."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}


@contextmanager
def _preserve_training_mode(module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def _require_subgoal_obs(agent, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    if not getattr(agent, "subgoal_conditioned", False):
        raise ValueError("agent.subgoal_conditioned is false; no subgoal path to diagnose")
    if "observation.subgoal" not in obs:
        raise KeyError("observation.subgoal")
    return obs["observation.subgoal"]


def _obs_with_feat(agent, obs: dict[str, torch.Tensor], *, detach_feat: bool) -> dict[str, torch.Tensor]:
    out = clone_tensor_obs(obs)
    if "feat" not in out:
        out["feat"] = agent._encode(out, augment=False)
    if detach_feat:
        out["feat"] = out["feat"].detach()
    return out


def _mean_q(q: torch.Tensor) -> torch.Tensor:
    # Project critic returns [num_q, batch, 1]. Keep this tolerant for tests.
    if q.ndim >= 3:
        return q.mean(dim=0)
    return q


def _actor_mean(agent, obs: dict[str, torch.Tensor], stddev: float) -> torch.Tensor:
    return agent.actor.forward(obs, stddev).mean


def _variant_obs(
    obs: dict[str, torch.Tensor],
    *,
    name: str,
    noise_scale: float,
) -> dict[str, torch.Tensor]:
    out = clone_tensor_obs(obs)
    z = out["observation.subgoal"]
    if name == "shuffle":
        out["observation.subgoal"] = z[torch.randperm(z.shape[0], device=z.device)]
    elif name == "zero":
        out["observation.subgoal"] = torch.zeros_like(z)
    elif name == "noise":
        out["observation.subgoal"] = z + noise_scale * torch.randn_like(z)
    else:
        raise ValueError(f"unknown subgoal sensitivity variant: {name}")
    return out


@torch.no_grad()
def measure_subgoal_sensitivity(
    agent,
    obs: dict[str, torch.Tensor],
    action: torch.Tensor,
    *,
    stddev: float = 0.0,
    variants: Iterable[str] = ("shuffle", "zero", "noise"),
    noise_scale: float = 0.1,
) -> dict[str, float]:
    """Measure Q/action changes when only observation.subgoal is perturbed."""
    _require_subgoal_obs(agent, obs)
    with _preserve_training_mode(agent):
        base_obs = _obs_with_feat(agent, obs, detach_feat=True)
        q0 = _mean_q(agent.critic(base_obs["feat"], agent._critic_prop(base_obs), action))
        a0 = _actor_mean(agent, base_obs, stddev)

        q_scale = q0.float().std(unbiased=False).clamp_min(1e-6)
        a_scale = a0.norm(dim=-1).mean().clamp_min(1e-6)

        metrics: dict[str, float] = {}
        for name in variants:
            alt_obs = _variant_obs(base_obs, name=name, noise_scale=noise_scale)
            q1 = _mean_q(agent.critic(alt_obs["feat"], agent._critic_prop(alt_obs), action))
            a1 = _actor_mean(agent, alt_obs, stddev)

            dq = (q1 - q0).abs().mean()
            da = (a1 - a0).norm(dim=-1).mean()
            metrics[f"{name}/dq_abs_mean"] = float(dq.item())
            metrics[f"{name}/dq_rel_to_qstd"] = float((dq / q_scale).item())
            metrics[f"{name}/da_l2_mean"] = float(da.item())
            metrics[f"{name}/da_rel"] = float((da / a_scale).item())
        return metrics


def measure_subgoal_gradient_sensitivity(
    agent,
    obs: dict[str, torch.Tensor],
    action: torch.Tensor,
) -> dict[str, float]:
    """Measure local Q-gradient magnitude with respect to z and state."""
    _require_subgoal_obs(agent, obs)
    with _preserve_training_mode(agent):
        grad_obs = _obs_with_feat(agent, obs, detach_feat=True)
        z = grad_obs["observation.subgoal"].detach().clone().requires_grad_(True)
        state = grad_obs["observation.state"].detach().clone().requires_grad_(True)
        grad_obs["observation.subgoal"] = z
        grad_obs["observation.state"] = state

        q = _mean_q(agent.critic(grad_obs["feat"], agent._critic_prop(grad_obs), action)).mean()
        grad_z, grad_state = torch.autograd.grad(q, (z, state), retain_graph=False)

        z_rms = grad_z.square().mean().sqrt()
        state_rms = grad_state.square().mean().sqrt()
        return {
            "grad_z_rms": float(z_rms.item()),
            "grad_state_rms": float(state_rms.item()),
            "grad_z_over_state": float((z_rms / state_rms.clamp_min(1e-8)).item()),
        }
