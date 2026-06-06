import torch
from resfit.rl_finetuning.chunk_residual.hiql_value import expectile_loss


def test_expectile_half_equals_half_mse():
    diff = torch.tensor([2.0, -3.0, 1.0])
    loss = expectile_loss(diff, 0.5)
    assert torch.allclose(loss, 0.5 * diff.pow(2).mean())


def test_expectile_asymmetric_weights():
    # diff>0 (低估, V<y) 用权重 tau; diff<0 用权重 1-tau
    pos = expectile_loss(torch.tensor([2.0]), 0.7)
    neg = expectile_loss(torch.tensor([-2.0]), 0.7)
    assert torch.allclose(pos, torch.tensor(0.7 * 4.0))
    assert torch.allclose(neg, torch.tensor(0.3 * 4.0))
    assert pos > neg


def test_expectile_returns_scalar():
    loss = expectile_loss(torch.randn(8), 0.7)
    assert loss.shape == torch.Size([])


from resfit.rl_finetuning.chunk_residual.hiql_value import discounted_target


def test_target_done_no_bootstrap():
    y = discounted_target(reward=torch.tensor([1.0]), next_v=torch.tensor([5.0]),
                          done=torch.tensor([1.0]), gamma=0.99)
    assert torch.allclose(y, torch.tensor([1.0]))


def test_target_not_done_bootstraps():
    y = discounted_target(reward=torch.tensor([0.0]), next_v=torch.tensor([5.0]),
                          done=torch.tensor([0.0]), gamma=0.99)
    assert torch.allclose(y, torch.tensor([0.99 * 5.0]))


def test_target_batch():
    y = discounted_target(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 9.0]),
                          torch.tensor([0.0, 1.0]), 0.9)
    assert torch.allclose(y, torch.tensor([1.8, 1.0]))
