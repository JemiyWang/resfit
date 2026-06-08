# tests/test_hiql_gc_value.py
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import RelativeGoalEncoder


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    """多核机上 torch 对微小张量开满线程会颠簸;本模块测试限 1 线程,跑完恢复(避免泄漏到其它测试文件)。"""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def test_relative_goal_encoder_shape_and_norm():
    enc = RelativeGoalEncoder(state_dim=30, rep_dim=10, hidden=64)
    s = torch.randn(8, 30)
    g = torch.randn(8, 30)
    z = enc(g, s)                      # forward(targets=g, bases=s)
    assert z.shape == (8, 10)
    # 归一化到球面半径 sqrt(rep_dim)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.full((8,), float(np.sqrt(10))), atol=1e-4)


def test_goal_conditioned_vf_forward_and_phi():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    vf = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64)
    s = torch.randn(5, 30)
    g = torch.randn(5, 30)
    v1, v2 = vf(s, g)
    assert v1.shape == (5,) and v2.shape == (5,)
    z = vf.phi(s, g)
    assert z.shape == (5, 10)
    assert torch.allclose(z, vf.goal_encoder(g, s), atol=1e-6)
    (v1.sum() + v2.sum()).backward()
    assert any(p.grad is not None for p in vf.parameters())


def test_stage_entries_from_instant():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import stage_entries_from_instant
    instant = np.array([0, 0, 1, 1, 2, 1, 2, 3], dtype=np.int8)
    entries = stage_entries_from_instant(instant)
    assert entries.tolist() == [2, 4, 7]
    assert stage_entries_from_instant(np.zeros(5, dtype=np.int8)).tolist() == []


def test_build_gc_data_and_sample_goals():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, sample_gc_goals
    seqs = [np.arange(8).reshape(4, 2).astype(np.float32),
            (np.arange(6).reshape(3, 2) + 100).astype(np.float32)]
    stage_entries = [np.array([2]), np.array([1])]
    data = build_gc_data(seqs, stage_entries)
    assert len(data["s_idx"]) == 5
    assert data["states"].shape == (7, 2)
    assert data["done"].tolist() == [0, 0, 1, 0, 1]
    assert data["last_idx_of"][0] == 3
    assert data["stage_entries_of"][0].tolist() == [2]
    assert data["stage_entries_of"][1].tolist() == [4 + 1]

    rng = np.random.default_rng(0)
    s_i = data["s_idx"]
    g = sample_gc_goals(s_i, data["traj_id"], data["last_idx_of"],
                        data["stage_entries_of"], rng, n_total=7,
                        p_curr=1.0, p_traj=0.0, p_rand=0.0)
    assert g.tolist() == list(s_i)

    g2 = sample_gc_goals(s_i, data["traj_id"], data["last_idx_of"],
                         data["stage_entries_of"], rng, n_total=7,
                         p_curr=0.0, p_traj=1.0, p_rand=0.0)
    for i, gi in zip(s_i, g2):
        d = data["traj_id"][list(s_i).index(i)]
        allowed = set(data["stage_entries_of"][d][data["stage_entries_of"][d] >= i].tolist())
        allowed.add(data["last_idx_of"][d])
        assert gi in allowed


def test_build_gc_data_skips_short_demos():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data
    # demo 1 是 T=1(应被跳过),demo 0/2 正常
    seqs = [np.zeros((3, 2), np.float32), np.zeros((1, 2), np.float32), np.ones((2, 2), np.float32)]
    data = build_gc_data(seqs, [np.array([], np.int64), np.array([], np.int64), np.array([], np.int64)])
    # 只有 demo 0(2 transition)+ demo 2(1 transition)= 3 transitions;T=1 的 demo 1 被跳过
    assert len(data["s_idx"]) == 3
    # 跳过的 demo(id=1)不出现在 traj_id 里,也不在 last_idx_of
    assert 1 not in set(data["traj_id"].tolist())
    assert 1 not in data["last_idx_of"]


def test_sample_gc_goals_pure_random():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, sample_gc_goals
    seq = np.arange(10).reshape(5, 2).astype(np.float32)
    data = build_gc_data([seq], [np.array([], np.int64)])
    rng = np.random.default_rng(0)
    g = sample_gc_goals(data["s_idx"], data["traj_id"], data["last_idx_of"],
                        data["stage_entries_of"], rng, n_total=5,
                        p_curr=0.0, p_traj=0.0, p_rand=1.0)
    # 纯随机:每个 goal 都是合法全局下标 [0, n_total)
    assert g.shape == data["s_idx"].shape
    assert ((g >= 0) & (g < 5)).all()


def test_train_gc_value_learns_progress():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    model, v_stats = train_gc_value(
        data, gamma=0.99, expectile=0.7, ema=0.01, lr=1e-3,
        batch_size=9, steps=2000, rep_dim=8, hidden=64, seed=0)
    assert v_stats["max"] > v_stats["min"]
    states = data["states"]
    g = states[9].repeat(10, 1)
    with torch.no_grad():
        v1, v2 = model(states, g)
        v = torch.minimum(v1, v2)
    assert v[-3:].mean() > v[:3].mean()


def test_gc_value_save_load_roundtrip(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, save_gc_value, load_gc_value)
    model = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64)
    p = str(tmp_path / "gc_value.pt")
    save_gc_value(p, model, v_stats={"min": -3.0, "max": 0.0, "mean": -1.0},
                  mean=torch.zeros(30), std=torch.ones(30),
                  dataset_id="ds", state_mode="eef_piece",
                  rel_piece_stats=(np.zeros(12), np.ones(12)))
    m2, info = load_gc_value(p)
    s, g = torch.randn(4, 30), torch.randn(4, 30)
    v1a, _ = model(s, g)
    v1b, _ = m2(s, g)
    assert torch.allclose(v1a, v1b, atol=1e-6)
    assert info["state_mode"] == "eef_piece"
    assert info["v_stats"]["min"] == -3.0
    assert info["rel_piece_mean"] is not None
