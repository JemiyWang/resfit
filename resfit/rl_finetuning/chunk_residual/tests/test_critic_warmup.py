from resfit.rl_finetuning.chunk_residual.critic_warmup import run_critic_warmup


class _MockAgent:
    def __init__(self):
        self.calls = []

    def update(self, batch, stddev, update_actor, bc_batch=None, ref_agent=None):
        self.calls.append(dict(stddev=stddev, update_actor=update_actor,
                               bc_batch=bc_batch, ref_agent=ref_agent))
        return {"critic_loss": 0.5}


def test_runs_critic_only_n_times():
    agent = _MockAgent()
    sampled = {"n": 0}

    def sample_fn():
        sampled["n"] += 1
        return object()

    out = run_critic_warmup(agent, 5, sample_fn, log_every=0)
    assert len(agent.calls) == 5
    assert sampled["n"] == 5
    assert all(c["update_actor"] is False for c in agent.calls)
    assert all(c["stddev"] == 0.0 for c in agent.calls)
    assert all(c["bc_batch"] is None and c["ref_agent"] is None for c in agent.calls)
    assert out == {"critic_loss": 0.5}


def test_zero_steps_is_noop():
    agent = _MockAgent()
    out = run_critic_warmup(agent, 0, lambda: object(), log_every=0)
    assert agent.calls == []
    assert out is None
