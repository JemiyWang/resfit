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
