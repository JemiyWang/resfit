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


def test_sample_gc_goals_geometric_future_covers_intermediate():
    """future_mode='geometric'(HIQL geom_sample=1 口径):未来目标按几何分布落在 [idx, last]
    内任意帧,覆盖到非 stage 入口的中间态——补 stage_entry 采样'目标空间留洞'的关键。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import sample_gc_goals
    # 单 demo:全局下标 0..99;stage 入口只有末态(模拟无中间里程碑 -> stage_entry 会把未来全堆到 99)
    last_idx_of = {0: 99}
    stage_entries_of = {0: np.array([99], dtype=np.int64)}
    idx = np.zeros(2000, dtype=np.int64)         # 全部从起始态 0 出发
    traj = np.zeros(2000, dtype=np.int64)
    rng = np.random.default_rng(0)
    g = sample_gc_goals(idx, traj, last_idx_of, stage_entries_of, rng,
                        n_total=100, p_curr=0.0, p_traj=1.0, p_rand=0.0,
                        future_mode="geometric", discount=0.9)
    # 全落在本 demo 内、>= 起始
    assert g.min() >= 0 and g.max() <= 99
    # 关键:铺开到大量中间帧(不像 stage_entry 那样全堆末态 99)
    assert len(np.unique(g)) > 20
    assert np.mean(g == 99) < 0.5
    # 几何均值 offset ≈ 1/(1-0.9)=10(从 idx=0 出发 -> goal≈offset)
    assert 5.0 < g.mean() < 20.0


def test_train_gc_value_passes_future_mode_through():
    """train_gc_value 要把 future_mode 透传到 sample_gc_goals:非法 mode 应在采样时抛 ValueError
    (若没透传,会是 TypeError/不报错)——以此证明开关真的接通,而非被忽略。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    with pytest.raises(ValueError):
        train_gc_value(data, steps=1, batch_size=4, rep_dim=4, hidden=16,
                       future_mode="bogus_mode", seed=0)


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


def test_mlp_layer_norm_modules_present():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import _mlp
    m_plain = _mlp(8, 16, 4, use_layer_norm=False)
    m_ln = _mlp(8, 16, 4, use_layer_norm=True)
    has = lambda m, t: any(isinstance(x, t) for x in m.modules())
    assert not has(m_plain, torch.nn.LayerNorm) and has(m_plain, torch.nn.ReLU)
    assert has(m_ln, torch.nn.LayerNorm) and has(m_ln, torch.nn.GELU)
    x = torch.randn(3, 8)
    assert m_ln(x).shape == (3, 4)            # 前向 shape 不变

def test_gc_value_layer_norm_forward_and_phi_sphere():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    vf = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, use_layer_norm=True)
    assert vf.use_layer_norm is True
    s, g = torch.randn(5, 30), torch.randn(5, 30)
    v1, v2 = vf(s, g)
    assert v1.shape == (5,) and v2.shape == (5,)
    z = vf.phi(s, g)
    assert torch.allclose(z.norm(dim=-1), torch.full((5,), float(np.sqrt(10))), atol=1e-4)

def test_gc_value_layer_norm_save_load_roundtrip(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, save_gc_value, load_gc_value)
    model = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, use_layer_norm=True)
    p = str(tmp_path / "gc_ln.pt")
    save_gc_value(p, model, v_stats={"min": -3.0, "max": 0.0, "mean": -1.0},
                  mean=torch.zeros(30), std=torch.ones(30),
                  dataset_id="ds", state_mode="eef_piece",
                  rel_piece_stats=(np.zeros(12), np.ones(12)))
    m2, _ = load_gc_value(p)                  # 须按 use_layer_norm=True 重建,否则 load_state_dict 失配报错
    assert m2.use_layer_norm is True
    s, g = torch.randn(4, 30), torch.randn(4, 30)
    v1a, _ = model(s, g); v1b, _ = m2(s, g)
    assert torch.allclose(v1a, v1b, atol=1e-6)
