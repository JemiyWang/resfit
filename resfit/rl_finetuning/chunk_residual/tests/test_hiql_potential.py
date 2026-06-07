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


import numpy as np
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import transition_rewards


class _StubPot:
    """phi(state_seq[T,D]) = 每行第一元素,模拟 V(state)。"""
    def phi(self, state_seq):
        return state_seq[:, 0]


def test_transition_rewards_hiql_uses_v():
    instant = np.array([0, 1, 2])                 # 3 帧 -> 2 transition,T=3
    state_seq = torch.tensor([[10.], [20.], [30.]])  # phi = [10,20,30]
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.99,
                           success=True, potential=_StubPot(), state_seq=state_seq)
    exp0 = potential_shaping(10.0, 20.0, bonus=1.0, gamma=0.99, done=False)        # t=0 not done
    exp1 = 1.0 + potential_shaping(20.0, 30.0, bonus=1.0, gamma=0.99, done=True)   # t=1 done(base+1)
    assert abs(r[0] - exp0) < 1e-5
    assert abs(r[1] - exp1) < 1e-5


def test_transition_rewards_none_matches_stage_baseline():
    # potential=None(默认)走现状 latch 路径,逐位等价
    instant = np.array([0, 1, 2])
    from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import shaping_reward
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.99, success=True)
    exp0 = shaping_reward(0, 1, mode="potential", bonus=1.0, gamma=0.99, done=False)
    exp1 = 1.0 + shaping_reward(1, 2, mode="potential", bonus=1.0, gamma=0.99, done=True)
    assert abs(r[0] - exp0) < 1e-5 and abs(r[1] - exp1) < 1e-5


from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def test_parser_potential_source_defaults():
    args = build_parser().parse_args(["--task", "TwoArmThreePieceAssembly"])
    assert args.potential_source == "stage"
    assert args.hiql_value_ckpt is None
    assert args.phi_scale == 1.0


def test_parser_potential_source_hiql():
    args = build_parser().parse_args(
        ["--task", "TwoArmThreePieceAssembly", "--potential_source", "hiql",
         "--hiql_value_ckpt", "v.pt", "--phi_scale", "0.5"])
    assert args.potential_source == "hiql"
    assert args.hiql_value_ckpt == "v.pt"
    assert args.phi_scale == 0.5
