import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    """多核机上 torch 对微小张量开满线程会颠簸;本模块测试限 1 线程,跑完恢复(防泄漏到其它测试文件)。"""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def test_high_actor_forward_dist():
    ha = HighActor(state_dim=30, rep_dim=10, hidden=64)
    s = torch.randn(6, 30)
    g = torch.randn(6, 30)
    dist = ha(s, g)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (6, 10)
    z = dist.rsample()
    assert z.shape == (6, 10)
    lp = dist.log_prob(z).sum(-1)
    assert lp.shape == (6,)
    # std 与输入无关(log_std 是 nn.Parameter,不是 MLP 输出)
    assert torch.allclose(dist.scale[0], dist.scale[1])


def test_awr_weight():
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import awr_weight
    adv = torch.tensor([-1.0, 0.0, 1.0, 100.0])
    w = awr_weight(adv, beta=1.0, clip=100.0)
    assert torch.allclose(w[:3], torch.exp(torch.tensor([-1.0, 0.0, 1.0])), atol=1e-5)
    assert float(w[3]) == 100.0
    w2 = awr_weight(adv, beta=0.5, clip=100.0)
    assert torch.allclose(w2[2], torch.exp(torch.tensor(0.5)), atol=1e-5)


def test_train_high_actor_predicts_forward_waypoint():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor

    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    vf, _ = train_gc_value(data, steps=2000, batch_size=16, rep_dim=8, hidden=64,
                           lr=1e-3, ema=0.01, seed=0)
    ha = train_high_actor(data, vf, way_steps=5, beta=1.0, steps=2000,
                          batch_size=16, hidden=64, lr=1e-3, seed=0)

    states = data["states"]
    st = states[2:3]
    g = states[-1:].clone()
    with torch.no_grad():
        z_pred = ha(st, g).mean
        z_fwd = vf.phi(st, states[7:8])
        z_stay = vf.phi(st, st)
    assert (z_pred - z_fwd).norm() < (z_pred - z_stay).norm()


def test_high_actor_save_load_roundtrip(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
        HighActor, save_high_actor, load_high_actor)
    ha = HighActor(state_dim=30, rep_dim=10, hidden=64)
    p = str(tmp_path / "high_actor.pt")
    save_high_actor(p, ha, gc_value_ckpt="gc_value.pt", way_steps=25, beta=1.0)
    ha2, info = load_high_actor(p)
    s, g = torch.randn(3, 30), torch.randn(3, 30)
    assert torch.allclose(ha(s, g).mean, ha2(s, g).mean, atol=1e-6)
    assert info["way_steps"] == 25 and info["gc_value_ckpt"] == "gc_value.pt"
    assert info["beta"] == 1.0


def test_sample_high_goal_target_traj_branch():
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import sample_high_goal_target
    last_idx_of = {0: 99}
    si = np.zeros(3000, dtype=np.int64)          # 全从 t=0 出发
    tj = np.zeros(3000, dtype=np.int64)
    rng = np.random.default_rng(0)
    goal, target = sample_high_goal_target(si, tj, last_idx_of, rng,
                                           way_steps=25, n_total=100, high_p_randomgoal=0.0)
    # traj goal ∈ [si+1, final],永不命中 current(=si=0)
    assert goal.min() >= 1 and goal.max() <= 99
    # target = min(si+way, goal) = min(25, goal)
    assert np.all(target == np.minimum(25, goal))
    assert np.all(target <= goal)

def test_sample_high_goal_target_random_branch():
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import sample_high_goal_target
    last_idx_of = {0: 99}
    si = np.full(4000, 10, dtype=np.int64)
    tj = np.zeros(4000, dtype=np.int64)
    rng = np.random.default_rng(1)
    goal, target = sample_high_goal_target(si, tj, last_idx_of, rng,
                                           way_steps=25, n_total=100, high_p_randomgoal=1.0)
    # 全 random:goal ∈ [0,100),target = min(10+25, 99) = 35(恒定)
    assert goal.min() >= 0 and goal.max() < 100
    assert np.all(target == 35)


def test_train_high_actor_rejects_bad_target_mode():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    vf, _ = train_gc_value(data, steps=1, batch_size=8, rep_dim=4, hidden=16, seed=0)
    with pytest.raises(ValueError):
        train_high_actor(data, vf, steps=1, batch_size=8, hidden=16, target_mode="bogus")

def test_train_high_actor_clamp_collapses_to_near_goal():
    """clamp_to_goal:goal 近(dist<way)时子目标应收敛到 goal 而非固定 +way 航点。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor
    seq = np.arange(40).reshape(40, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    vf, _ = train_gc_value(data, steps=3000, batch_size=32, rep_dim=8, hidden=64,
                           lr=1e-3, ema=0.01, seed=0)
    ha = train_high_actor(data, vf, way_steps=10, beta=1.0, steps=4000, batch_size=32,
                          hidden=64, lr=1e-3, seed=0, target_mode="clamp_to_goal")
    states = data["states"]
    st = states[5:6]
    g_near = states[8:9]          # dist=3 < way=10 -> 子目标应≈goal(states[8])
    with torch.no_grad():
        z_pred = ha(st, g_near).mean
        z_goal = vf.phi(st, states[8:9])     # 落 goal
        z_way = vf.phi(st, states[15:16])    # 固定 +10 航点
    assert (z_pred - z_goal).norm() < (z_pred - z_way).norm()
