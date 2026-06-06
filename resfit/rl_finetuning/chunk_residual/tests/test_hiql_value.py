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


from resfit.rl_finetuning.chunk_residual.hiql_value import ValueMLP


def test_valuemlp_output_shape():
    m = ValueMLP(state_dim=18, hidden=32)
    out = m(torch.randn(4, 18))
    assert out.shape == (4, 1)


def test_valuemlp_records_dims():
    m = ValueMLP(state_dim=18, hidden=64)
    assert m.state_dim == 18 and m.hidden == 64


def test_valuemlp_trainable():
    m = ValueMLP(18, 32)
    m(torch.randn(4, 18)).sum().backward()
    assert all(p.grad is not None for p in m.parameters())


import numpy as np
from resfit.rl_finetuning.chunk_residual.hiql_value import build_transitions


def test_build_transitions_counts_and_done():
    seqs = [np.zeros((3, 2)), np.zeros((2, 2))]  # 3帧->2 trans, 2帧->1 trans
    s, sn, done = build_transitions(seqs)
    assert s.shape == (3, 2) and sn.shape == (3, 2) and done.shape == (3, 1)
    # demo1 done=[0,1], demo2 done=[1]
    assert torch.allclose(done.squeeze(1), torch.tensor([0.0, 1.0, 1.0]))


def test_build_transitions_alignment():
    seq = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    s, sn, done = build_transitions([seq])
    assert torch.allclose(s, torch.tensor([[0., 0.], [1., 1.]]))
    assert torch.allclose(sn, torch.tensor([[1., 1.], [2., 2.]]))


def test_build_transitions_skips_short():
    s, sn, done = build_transitions([np.zeros((1, 2)), np.zeros((3, 2))])
    assert s.shape == (2, 2)  # 仅 3 帧的 demo 贡献(1 帧的跳过)


from resfit.rl_finetuning.chunk_residual.hiql_value import save_value, load_value


def test_save_load_roundtrip(tmp_path):
    m = ValueMLP(6, 16)
    vstats = {"min": 0.0, "max": 1.0, "mean": 0.5}
    mean, std = torch.zeros(6), torch.ones(6)
    p = str(tmp_path / "value.pt")
    save_value(p, m, v_stats=vstats, mean=mean, std=std, dataset_id="dummy/ds")
    m2, info = load_value(p)
    x = torch.randn(3, 6)
    assert torch.allclose(m(x), m2(x))          # 权重一致
    assert m2.state_dim == 6 and m2.hidden == 16  # 维度重建正确
    assert info["v_stats"] == vstats
    assert info["dataset_id"] == "dummy/ds"
    assert torch.allclose(info["mean"], mean) and torch.allclose(info["std"], std)
