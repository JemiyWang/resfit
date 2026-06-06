import torch

from resfit.rl_finetuning.off_policy.rl.q_agent import bc_target


def test_bc_target_residual_subtracts_base():
    action = torch.tensor([[1.0, 2.0, 3.0]])
    base = torch.tensor([[0.5, 1.0, 1.0]])
    out = bc_target(action, base, residual_actor=True)
    assert torch.allclose(out, action - base)


def test_bc_target_demo_zero_when_action_equals_base():
    # offline GT demo: base==action -> 残差 target=0
    a = torch.randn(4, 6)
    out = bc_target(a, a.clone(), residual_actor=True)
    assert torch.allclose(out, torch.zeros_like(a))


def test_bc_target_non_residual_returns_action_ignores_base():
    # 非残差 actor: target=action,且不解引用 base(传 None 也不报错)
    action = torch.tensor([[1.0, 2.0]])
    out = bc_target(action, None, residual_actor=False)
    assert torch.allclose(out, action)


import math
import pytest
from tensordict import TensorDict
from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

_C, _H, _W = 3, 84, 84
_CAM = "observation.images.agentview"
_SD, _FLAT, _NS, _BS = 5, 12, 5, 4


def _residual_agent(bc_coef):
    cfg = QAgentConfig()
    cfg.device = "cpu"
    cfg.critic.loss.type = "mse"
    cfg.bc_loss_coef = bc_coef
    cfg.bc_loss_dynamic = 0
    agent = QAgent(obs_shape=(_C, _H, _W), prop_shape=(_SD,), action_dim=_FLAT,
                   rl_cameras=[_CAM], cfg=cfg, residual_actor=True, num_stages=_NS)
    agent.train(True)
    agent.actor_target.train(True)
    return agent


def _obs():
    return {
        _CAM: torch.rand(_BS, _C, _H, _W),
        "observation.state": torch.randn(_BS, _SD),
        "observation.base_action": torch.tanh(torch.randn(_BS, _FLAT)),
        "observation.stage_id": torch.randint(0, _NS, (_BS, 1)).float(),
    }


def _rl_batch():
    return TensorDict({
        "obs": TensorDict(_obs(), batch_size=[_BS]),
        "action": torch.tanh(torch.randn(_BS, _FLAT)),
        ("next", "reward"): torch.zeros(_BS),
        "gamma": torch.full((_BS,), 0.99),
        "nonterminal": torch.ones(_BS),
        ("next", "obs"): TensorDict(_obs(), batch_size=[_BS]),
    }, batch_size=[_BS])


def _bc_batch():
    return TensorDict({
        "obs": TensorDict(_obs(), batch_size=[_BS]),
        "action": torch.tanh(torch.randn(_BS, _FLAT)),
    }, batch_size=[_BS])


@pytest.mark.manual
def test_residual_bc_update_runs_and_reports_bc_loss():
    """残差 actor 带 bc_batch 跑 update -> update_actor_rft -> _compute_actor_bc_loss(assert 已解)。
    metrics 含有限的 rft/bc_loss + actor_loss_total。依赖 VitEncoder,故 manual;CPU。"""
    torch.manual_seed(0)
    agent = _residual_agent(bc_coef=0.1)
    metrics = agent.update(_rl_batch(), stddev=0.05, update_actor=True,
                           bc_batch=_bc_batch(), ref_agent=agent)
    assert math.isfinite(metrics["rft/bc_loss"])
    assert math.isfinite(metrics["train/actor_loss_total"])


@pytest.mark.manual
def test_default_off_no_bc_loss_metric():
    """bc_batch=None(默认关)走 update_actor(非 rft),无 rft/bc_loss。"""
    torch.manual_seed(0)
    agent = _residual_agent(bc_coef=0.0)
    metrics = agent.update(_rl_batch(), stddev=0.05, update_actor=True,
                           bc_batch=None, ref_agent=None)
    assert "rft/bc_loss" not in metrics
    assert math.isfinite(metrics["train/actor_loss_total"])


def test_cli_demo_bc_coef_default_zero():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    assert build_parser().parse_args([]).demo_bc_coef == 0.0


def test_cli_demo_bc_coef_parses():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    assert build_parser().parse_args(["--demo_bc_coef", "0.1"]).demo_bc_coef == 0.1
