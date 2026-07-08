"""Critic-only warmup for chunk_residual RL (aligns offpolicy _run_critic_warmup).

Runs `steps` critic-only gradient updates (actor frozen) on batches produced by
`sample_batch_fn`, WITHOUT stepping the environment. Intended to run once, right
after the online buffer is filled and before actor training begins.
"""


def run_critic_warmup(agent, steps, sample_batch_fn, *, log_every=1000):
    """Do `steps` critic-only updates and return the last metrics dict (or None).

    agent.update is called as agent.update(batch, 0.0, False, bc_batch=None,
    ref_agent=None): stddev=0.0 target smoothing, update_actor=False (critic only),
    no BC. `sample_batch_fn()` must return a fresh training batch per call.
    """
    metrics = None
    for i in range(steps):
        batch = sample_batch_fn()
        metrics = agent.update(batch, 0.0, False, bc_batch=None, ref_agent=None)
        if log_every and (i + 1) % log_every == 0:
            cl = metrics.get("train/critic_loss") if isinstance(metrics, dict) else None
            tail = f" critic_loss={cl:.4f}" if isinstance(cl, (int, float)) else ""
            print(f"[critic-warmup] {i + 1}/{steps}{tail}")
    return metrics
