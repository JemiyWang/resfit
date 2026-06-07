import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping
from resfit.rl_finetuning.chunk_residual.hiql_potential import HiqlPotential
from resfit.rl_finetuning.chunk_residual.hiql_value import ValueMLP, save_value


def test_potential_shaping_not_done():
    # F = bonus*(gamma*phi_next - phi_start)
    f = potential_shaping(1.0, 2.0, bonus=1.0, gamma=0.99, done=False)
    assert abs(f - (0.99 * 2.0 - 1.0)) < 1e-6


def test_potential_shaping_done_zeroes_next():
    # done -> phi_next=0 -> F = -bonus*phi_start
    f = potential_shaping(3.0, 9.9, bonus=2.0, gamma=0.99, done=True)
    assert abs(f - (2.0 * (0.0 - 3.0))) < 1e-6


def test_potential_shaping_accepts_tensor_scalar():
    # online 传 [1] tensor 标量,应与 float 同结果
    f = potential_shaping(torch.tensor([1.0]), torch.tensor([2.0]),
                          bonus=1.0, gamma=0.99, done=False)
    assert abs(float(f) - (0.99 * 2.0 - 1.0)) < 1e-6


def _make_fake_ckpt(tmp_path, vmin, vmax):
    m = ValueMLP(state_dim=3, hidden=8)
    p = str(tmp_path / "value.pt")
    save_value(p, m, v_stats={"min": vmin, "max": vmax, "mean": 0.5 * (vmin + vmax)},
               mean=torch.zeros(3), std=torch.ones(3), dataset_id="dummy")
    return p, m


def test_hiqlpotential_auto_scale(tmp_path):
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    # num_stages=5 -> 动态范围目标 [0,4];auto_scale=(5-1)/(2-0)=2.0;phi_scale=1
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    assert abs(pot.scale - 2.0) < 1e-6


def test_hiqlpotential_phi_equals_v_times_scale(tmp_path):
    p, model = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    x = torch.randn(4, 3)
    with torch.no_grad():
        expected = model(x).squeeze(-1) * 2.0
    out = pot.phi(x)
    assert out.shape == (4,)
    assert torch.allclose(out, expected, atol=1e-5)


def test_hiqlpotential_phi_scale_multiplies(tmp_path):
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=0.5, device="cpu")
    assert abs(pot.scale - 1.0) < 1e-6   # 2.0 * 0.5
