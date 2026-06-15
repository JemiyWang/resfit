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


def test_expectile_loss_weighted_gates_on_adv():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import expectile_loss_weighted
    from resfit.rl_finetuning.chunk_residual.hiql_value import expectile_loss
    tau = 0.7
    # 门控用 adv:正/负/0(0 归入 >=0 -> tau),平方的是 diff
    adv = torch.tensor([1.0, -1.0, 0.0])
    diff = torch.tensor([2.0, 2.0, 2.0])
    out = expectile_loss_weighted(adv, diff, tau)
    expected = (0.7 * 4 + 0.3 * 4 + 0.7 * 4) / 3       # weight=[.7,.3,.7], diff²=4, 取 mean
    assert abs(float(out) - expected) < 1e-6
    # adv 与 diff 异号时,与"用 diff 自门控"的单参版不同(证明门控真的看 adv)
    adv2, diff2 = torch.tensor([-1.0]), torch.tensor([2.0])
    assert abs(float(expectile_loss_weighted(adv2, diff2, tau)) - 1.2) < 1e-6   # 0.3*4
    assert abs(float(expectile_loss(diff2, tau)) - 2.8) < 1e-6                  # 0.7*4
    # adv==diff 时退化为标准 expectile,两者一致
    z = torch.tensor([1.5, -0.5, 2.0])
    assert torch.allclose(expectile_loss_weighted(z, z, tau), expectile_loss(z, tau), atol=1e-7)


def test_train_gc_value_loss_mode_default_equiv_and_hiql_differs():
    # 注意:此处验证的是底层函数 train_gc_value 的**签名默认**(shared_min),
    # 与 Task1 改动的 **CLI argparse 默认**(hiql)是两个层面、互不影响。
    # CLI 入口(train_hiql_gc_value.py)现在默认 hiql,但底层函数签名未变仍默认 shared_min,
    # 因此这里断言"不传 value_loss_mode == 显式传 shared_min"依然正确。
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(12).reshape(12, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(gamma=0.99, expectile=0.7, ema=0.01, lr=1e-3,
              batch_size=8, steps=300, rep_dim=8, hidden=32, seed=0)
    # 默认 == 显式 shared_min,逐张量 atol=0(证明默认分支没被动过)
    m_def, _ = train_gc_value(data, **kw)
    m_sm, _ = train_gc_value(data, value_loss_mode="shared_min", **kw)
    for k, a in m_def.state_dict().items():
        assert torch.equal(a, m_sm.state_dict()[k]), f"default vs shared_min 不等: {k}"
    # hiql 模式确实走了不同分支 -> 权重与 shared_min 不同
    m_hi, _ = train_gc_value(data, value_loss_mode="hiql", **kw)
    diff = max(float((m_hi.state_dict()[k] - m_sm.state_dict()[k]).abs().max())
               for k in m_sm.state_dict())
    assert diff > 1e-6, "hiql 与 shared_min 应训出不同权重"


def test_train_gc_value_hiql_learns_progress():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    model, v_stats = train_gc_value(
        data, gamma=0.99, expectile=0.7, ema=0.01, lr=1e-3,
        batch_size=9, steps=2000, rep_dim=8, hidden=64, seed=0, value_loss_mode="hiql")
    assert v_stats["max"] > v_stats["min"]
    states = data["states"]
    g = states[9].repeat(10, 1)
    with torch.no_grad():
        v1, v2 = model(states, g)
        v = torch.minimum(v1, v2)
    assert v[-3:].mean() > v[:3].mean()


def test_train_gc_value_rejects_bad_loss_mode():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    with pytest.raises(ValueError):
        train_gc_value(data, steps=1, batch_size=4, rep_dim=4, hidden=16,
                       value_loss_mode="bogus", seed=0)


def test_gc_value_save_load_value_loss_mode(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, save_gc_value, load_gc_value)
    import torch as _t
    model = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64)
    p = str(tmp_path / "gc_vlm.pt")
    save_gc_value(p, model, v_stats={"min": -3.0, "max": 0.0, "mean": -1.0},
                  mean=_t.zeros(30), std=_t.ones(30), dataset_id="ds",
                  state_mode="eef_piece", rel_piece_stats=(np.zeros(12), np.ones(12)),
                  value_loss_mode="hiql")
    _m, info = load_gc_value(p)
    assert info["value_loss_mode"] == "hiql"
    # 旧档(无该键)回退 shared_min
    ckpt = _t.load(p, weights_only=False)
    del ckpt["value_loss_mode"]
    _t.save(ckpt, p)
    _m2, info2 = load_gc_value(p)
    assert info2["value_loss_mode"] == "shared_min"


def test_critic_divergence_keys_and_range():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, critic_divergence)
    torch.manual_seed(0)
    vf = GoalConditionedVF(state_dim=4, rep_dim=4, hidden=16)
    seqs = [np.random.RandomState(1).randn(6, 4).astype(np.float32),
            np.random.RandomState(2).randn(5, 4).astype(np.float32)]
    d = critic_divergence(vf, seqs)
    assert set(d) == {"corr", "mean_abs_diff"}
    assert -1.0 <= d["corr"] <= 1.0
    assert np.isfinite(d["corr"])
    assert d["mean_abs_diff"] >= 0.0


def test_value_mask_mode_default_equivalence():
    """底层默认(不传)与显式 'done_aware' 逐位等价(默认路径未被破坏)。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(steps=300, batch_size=16, rep_dim=8, hidden=32, lr=1e-3, ema=0.01, seed=0)
    m0, _ = train_gc_value(data, **kw)
    m1, _ = train_gc_value(data, value_mask_mode="done_aware", **kw)
    for a, b in zip(m0.state_dict().values(), m1.state_dict().values()):
        assert torch.equal(a, b)


def test_value_mask_mode_hiql_differs_and_validates():
    """'hiql' 模式去掉 (1-done) → 训出与 'done_aware' 不同;非法值报错。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(steps=300, batch_size=16, rep_dim=8, hidden=32, lr=1e-3, ema=0.01, seed=0)
    m_done, _ = train_gc_value(data, value_mask_mode="done_aware", **kw)
    m_hiql, _ = train_gc_value(data, value_mask_mode="hiql", **kw)
    diffs = [not torch.equal(a, b) for a, b in
             zip(m_done.state_dict().values(), m_hiql.state_dict().values())]
    assert any(diffs)
    with pytest.raises(ValueError):
        train_gc_value(data, value_mask_mode="bogus", **kw)


def test_gc_value_parser_defaults_aligned_to_hiql():
    """对齐 HIQL:不带 flag 时 CLI 默认 = geometric / LN 开 / hiql 口径;旧选项仍可显式回退。"""
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser
    req = ["--hdf5", "x.hdf5", "--dataset", "some/ds", "--stage_cache", "s.npz"]
    a = build_parser().parse_args(req)
    assert a.goal_future_mode == "geometric"
    assert a.use_layer_norm == 1
    assert a.value_loss_mode == "hiql"
    assert a.value_mask_mode == "hiql"
    assert a.value_rep_mode == "goal_only"
    # 旧口径仍可显式回退
    b = build_parser().parse_args(req + ["--goal_future_mode", "stage_entry",
                                         "--use_layer_norm", "0",
                                         "--value_loss_mode", "shared_min",
                                         "--value_mask_mode", "done_aware",
                                         "--value_rep_mode", "concat"])
    assert b.goal_future_mode == "stage_entry"
    assert b.use_layer_norm == 0
    assert b.value_loss_mode == "shared_min"
    assert b.value_mask_mode == "done_aware"
    assert b.value_rep_mode == "concat"


def test_rep_mode_default_concat_and_dims():
    """默认(不传)= concat:goal 编码器第一层吃 2*state_dim;goal_only 吃 state_dim。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import RelativeGoalEncoder
    enc_default = RelativeGoalEncoder(state_dim=30, rep_dim=10, hidden=64)
    assert enc_default.net[0].in_features == 60          # concat[g,s]
    enc_go = RelativeGoalEncoder(state_dim=30, rep_dim=10, hidden=64, rep_mode="goal_only")
    assert enc_go.net[0].in_features == 30               # goal-only


def test_goal_only_phi_ignores_state():
    """goal_only 下 phi(s,g) 只依赖 g(扰动 s 不变);concat 下扰动 s 会变。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    g = torch.randn(4, 30)
    s1 = torch.randn(4, 30)
    s2 = torch.randn(4, 30)
    vf_go = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="goal_only")
    assert torch.allclose(vf_go.phi(s1, g), vf_go.phi(s2, g), atol=1e-6)
    vf_cat = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="concat")
    assert not torch.allclose(vf_cat.phi(s1, g), vf_cat.phi(s2, g), atol=1e-4)
    with pytest.raises(ValueError):
        GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="bogus")


def test_value_rep_mode_default_equivalence():
    """train_gc_value 底层默认(不传)与显式 'concat' 逐位等价。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(steps=300, batch_size=16, rep_dim=8, hidden=32, lr=1e-3, ema=0.01, seed=0)
    m0, _ = train_gc_value(data, **kw)
    m1, _ = train_gc_value(data, value_rep_mode="concat", **kw)
    for a, b in zip(m0.state_dict().values(), m1.state_dict().values()):
        assert torch.equal(a, b)


def test_gc_value_rep_mode_save_load_roundtrip(tmp_path):
    """save/load 往返保 value_rep_mode 并按之重建维度;旧档(无键)回退 concat。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, save_gc_value, load_gc_value)
    vf = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="goal_only")
    p = tmp_path / "v.pt"
    save_gc_value(str(p), vf, v_stats={"min": 0.0, "max": 0.0, "mean": 0.0},
                  mean=torch.zeros(30), std=torch.ones(30), dataset_id="x")
    m, info = load_gc_value(str(p))
    assert info["value_rep_mode"] == "goal_only"
    assert m.goal_encoder.net[0].in_features == 30
    # 旧档:删掉键 → 回退 concat(2*state_dim)
    ckpt = torch.load(str(p), weights_only=False)
    del ckpt["value_rep_mode"]
    cat = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="concat")
    ckpt["state_dict"] = cat.state_dict()
    torch.save(ckpt, str(p))
    m2, info2 = load_gc_value(str(p))
    assert info2["value_rep_mode"] == "concat"
    assert m2.goal_encoder.net[0].in_features == 60


def test_train_gc_value_device_cpu_finite():
    """device='cpu' 时 v_stats 有限(smoke test,与默认行为等价)。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(steps=20, batch_size=8, rep_dim=4, hidden=16, seed=0)
    _model, v_stats = train_gc_value(data, device="cpu", **kw)
    assert np.isfinite(v_stats["min"])
    assert np.isfinite(v_stats["max"])
    assert np.isfinite(v_stats["mean"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_gc_value_device_cuda_finite():
    """device='cuda' 时 v_stats 也有限(不要求 cpu/cuda 数值逐位相等)。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(steps=20, batch_size=8, rep_dim=4, hidden=16, seed=0)
    model_cuda, v_stats = train_gc_value(data, device="cuda", **kw)
    assert np.isfinite(v_stats["min"])
    assert np.isfinite(v_stats["max"])
    assert np.isfinite(v_stats["mean"])
    # 确认 model 参数在 GPU 上
    assert next(model_cuda.parameters()).is_cuda
