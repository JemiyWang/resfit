import numpy as np
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


# --- ③a' object-aware:eef_piece 模式 phi 拼 30 维(state18_std | (rel-mean)/std)---

def _make_objaware_ckpt(tmp_path, vmin=0.0, vmax=2.0):
    """30 维 value ckpt(state_mode=eef_piece)+ rel_piece mean/std,模拟 ③a' 产物。"""
    m = ValueMLP(state_dim=30, hidden=8)
    rel_mean = np.arange(12, dtype=np.float32)          # 非零非一,验证确实用上
    rel_std = np.linspace(0.5, 3.0, 12).astype(np.float32)
    p = str(tmp_path / "value_obj.pt")
    save_value(p, m, v_stats={"min": vmin, "max": vmax, "mean": 0.5 * (vmin + vmax)},
               mean=torch.zeros(18), std=torch.ones(18), dataset_id="dummy",
               state_mode="eef_piece", rel_piece_stats=(rel_mean, rel_std))
    return p, m, rel_mean, rel_std


def test_hiqlpotential_reads_state_mode_and_rel_stats(tmp_path):
    p, _, rel_mean, rel_std = _make_objaware_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    assert pot.state_mode == "eef_piece"
    assert pot.model.state_dim == 30
    assert np.allclose(pot.rel_mean.cpu().numpy(), rel_mean)
    assert np.allclose(pot.rel_std.cpu().numpy(), rel_std)


def test_hiqlpotential_phi_eef_piece_builds_30dim_batched(tmp_path):
    """offline 口径:state[T,18] + rel[T,12] -> V(concat[state18, (rel-mean)/std])*scale。"""
    p, model, rel_mean, rel_std = _make_objaware_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    state18 = torch.randn(4, 18)
    rel_raw = torch.randn(4, 12)
    rel_n = (rel_raw - torch.as_tensor(rel_mean)) / torch.as_tensor(rel_std)
    x30 = torch.cat([state18, rel_n], dim=-1)
    with torch.no_grad():
        expected = model(x30).squeeze(-1) * pot.scale
    out = pot.phi(state18, rel_raw)
    assert out.shape == (4,)
    assert torch.allclose(out, expected, atol=1e-5)


def test_hiqlpotential_phi_eef_piece_online_single(tmp_path):
    """online 口径:state[1,18] + rel (12,) numpy -> [1] 输出,与 batched 同公式。"""
    p, model, rel_mean, rel_std = _make_objaware_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    state18 = torch.randn(1, 18)
    rel_raw = np.random.RandomState(0).randn(12).astype(np.float32)
    rel_n = (torch.as_tensor(rel_raw) - torch.as_tensor(rel_mean)) / torch.as_tensor(rel_std)
    x30 = torch.cat([state18, rel_n.unsqueeze(0)], dim=-1)
    with torch.no_grad():
        expected = model(x30).squeeze(-1) * pot.scale
    out = pot.phi(state18, rel_raw)
    assert out.shape == (1,)
    assert torch.allclose(out, expected, atol=1e-5)


def test_hiqlpotential_phi_eef_piece_requires_rel(tmp_path):
    p, _, _, _ = _make_objaware_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    with pytest.raises((ValueError, TypeError)):
        pot.phi(torch.randn(1, 18))      # eef_piece 模式缺 rel_piece 必须报错


def test_hiqlpotential_phi_eef_mode_ignores_rel(tmp_path):
    """eef 模式(旧默认):phi 无视 rel,与单参旧行为逐位一致。"""
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)   # state_dim=3, state_mode 默认 eef
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    assert pot.state_mode == "eef"
    x = torch.randn(4, 3)
    assert torch.allclose(pot.phi(x), pot.phi(x, rel_piece_raw=np.zeros(12)), atol=1e-6)


from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import transition_rewards


class _StubPot:
    """phi(state_seq[T,D]) = 每行第一元素,模拟 V(state)。"""
    def phi(self, state_seq, rel_piece_seq=None):
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


class _RelStubPot:
    """phi = state第一列 + rel第一列;rel 非空时必须被透传进来才对得上。"""
    def phi(self, state_seq, rel_piece_seq=None):
        base = np.asarray(state_seq)[:, 0]
        if rel_piece_seq is not None:
            base = base + np.asarray(rel_piece_seq)[:, 0]
        return base


def test_transition_rewards_passes_rel_piece_seq():
    # ③a' object-aware:offline 用 18 维 std state + raw rel_piece,phi 内部拼 30 维。
    instant = np.array([0, 1, 2])
    state_seq = torch.tensor([[10.], [20.], [30.]])
    rel_seq = np.array([[1., 9.], [2., 9.], [3., 9.]])   # 第一列加进 phi → phi=[11,22,33]
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.99,
                           success=True, potential=_RelStubPot(),
                           state_seq=state_seq, rel_piece_seq=rel_seq)
    exp0 = potential_shaping(11.0, 22.0, bonus=1.0, gamma=0.99, done=False)
    exp1 = 1.0 + potential_shaping(22.0, 33.0, bonus=1.0, gamma=0.99, done=True)
    assert abs(r[0] - exp0) < 1e-5 and abs(r[1] - exp1) < 1e-5


def test_transition_fields_threads_rel_piece_seq():
    # transition_fields 是 build_offline_buffer 的入口,必须把 rel_piece_seq 透到 phi。
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import transition_fields
    instant = np.array([0, 1, 2])
    state_seq = torch.tensor([[10.], [20.], [30.]])
    rel_seq = np.array([[1., 9.], [2., 9.], [3., 9.]])
    fld = transition_fields(instant, bonus=1.0, mode="potential", gamma=0.99,
                            success=True, potential=_RelStubPot(),
                            state_seq=state_seq, rel_piece_seq=rel_seq)
    exp0 = potential_shaping(11.0, 22.0, bonus=1.0, gamma=0.99, done=False)
    assert abs(fld["reward"][0] - exp0) < 1e-5


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
