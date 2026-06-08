# tests/test_hiql_gc_value.py
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import RelativeGoalEncoder


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
