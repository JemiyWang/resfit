import torch
from torch import nn


class _FakeDist:
    def __init__(self, mean):
        self.mean = mean


class _FakeActor(nn.Module):
    def forward(self, obs, std):
        feat = obs["feat"].flatten(1)
        prop = obs["observation.state"]
        z = obs["observation.subgoal"]
        base = obs["observation.base_action"]
        mean = torch.cat([feat[:, :1] + z[:, :1], prop[:, :1] - z[:, 1:2], base[:, :1]], dim=-1)
        return _FakeDist(mean)


class _FakeCritic(nn.Module):
    def forward(self, feat, prop, action):
        z = prop[:, -2:]
        q = action[:, :1] + 2.0 * z[:, :1] - z[:, 1:2]
        return torch.stack([q, q + 1.0], dim=0)


class _FakeAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.subgoal_conditioned = True
        self.critic = _FakeCritic()
        self.actor = _FakeActor()

    def _encode(self, obs, augment):
        del augment
        return obs["image"].float()

    def _critic_prop(self, obs):
        return torch.cat([obs["observation.state"], obs["observation.subgoal"]], dim=-1)


def _obs():
    return {
        "image": torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2),
        "observation.state": torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        "observation.base_action": torch.tensor([[0.1], [0.2], [0.3]]),
        "observation.subgoal": torch.tensor([[1.0, 0.0], [2.0, 1.0], [4.0, 3.0]]),
    }


def test_subgoal_sensitivity_reports_q_and_actor_changes():
    from resfit.rl_finetuning.chunk_residual.subgoal_sensitivity import (
        measure_subgoal_sensitivity,
    )

    agent = _FakeAgent()
    action = torch.tensor([[0.0], [1.0], [2.0]])

    metrics = measure_subgoal_sensitivity(
        agent,
        _obs(),
        action,
        stddev=0.0,
        variants=("zero",),
    )

    assert metrics["zero/dq_abs_mean"] > 0
    assert metrics["zero/dq_rel_to_qstd"] > 0
    assert metrics["zero/da_l2_mean"] > 0
    assert metrics["zero/da_rel"] > 0


def test_subgoal_gradient_sensitivity_reports_nonzero_z_gradient():
    from resfit.rl_finetuning.chunk_residual.subgoal_sensitivity import (
        measure_subgoal_gradient_sensitivity,
    )

    agent = _FakeAgent()
    action = torch.tensor([[0.0], [1.0], [2.0]])

    metrics = measure_subgoal_gradient_sensitivity(agent, _obs(), action)

    assert metrics["grad_z_rms"] > 0
    assert metrics["grad_state_rms"] == 0
    assert metrics["grad_z_over_state"] > 0


def test_diagnose_subgoal_sensitivity_parser_accepts_required_paths():
    from resfit.rl_finetuning.chunk_residual.diagnose_subgoal_sensitivity import build_parser

    args = build_parser().parse_args([
        "--run_dir", "outputs_chunk/run",
        "--offline_buffer_cache", "outputs_chunk/run_offcache",
        "--batch_size", "32",
        "--device", "cpu",
        "--variants", "shuffle", "zero",
    ])

    assert args.run_dir == "outputs_chunk/run"
    assert args.offline_buffer_cache == "outputs_chunk/run_offcache"
    assert args.batch_size == 32
    assert args.device == "cpu"
    assert args.variants == ["shuffle", "zero"]
