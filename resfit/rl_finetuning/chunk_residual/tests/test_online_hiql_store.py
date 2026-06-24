import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.online_hiql_store import (
    OnlineHiqlStore, sample_value_batch, sample_high_actor_batch)


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def test_store_accumulates_and_builds_data():
    store = OnlineHiqlStore(max_transitions=1000)
    assert len(store) == 0 and not store.ready(1)
    store.add_episode(np.arange(8, dtype=np.float32).reshape(4, 2))   # 3 transitions
    store.add_episode((np.arange(6, dtype=np.float32) + 100).reshape(3, 2))  # 2 transitions
    assert len(store) == 5
    assert store.ready(5) and not store.ready(6)
    data = store.data()
    assert data["states"].shape == (7, 2)
    assert len(data["s_idx"]) == 5
    assert data["done"].tolist() == [0, 0, 1, 0, 1]


def test_store_skips_short_episode_and_caches_until_dirty():
    store = OnlineHiqlStore(max_transitions=1000)
    store.add_episode(np.zeros((1, 3), np.float32))   # T=1 -> 跳过
    assert len(store) == 0
    store.add_episode(np.ones((3, 3), np.float32))    # 2 transitions
    d1 = store.data()
    d2 = store.data()
    assert d1 is d2                                   # 未变 -> 缓存复用(同一对象)
    store.add_episode(np.ones((2, 3), np.float32))
    d3 = store.data()
    assert d3 is not d1                               # 脏后重建


def test_store_fifo_drops_oldest_over_cap():
    store = OnlineHiqlStore(max_transitions=4)
    store.add_episode(np.zeros((3, 1), np.float32))   # 2 trans
    store.add_episode(np.ones((3, 1), np.float32))    # 2 trans -> total 4
    store.add_episode(np.full((3, 1), 2.0, np.float32))  # +2 -> 超 4,丢最旧
    assert len(store) <= 4
    # 最旧(全 0)那条应被丢弃
    assert not np.any(store.data()["states"].numpy() == 0.0)


def test_sample_value_batch_shapes_and_success():
    store = OnlineHiqlStore()
    store.add_episode(np.arange(20, dtype=np.float32).reshape(10, 2))
    data = store.data()
    rng = np.random.default_rng(0)
    s, s_next, g, success, done = sample_value_batch(
        data, 32, rng, future_mode="geometric", gamma=0.99)
    assert s.shape == (32, 2) and s_next.shape == (32, 2) and g.shape == (32, 2)
    assert success.shape == (32,) and done.shape == (32,)
    assert set(np.unique(success.numpy()).tolist()) <= {0.0, 1.0}


def test_sample_high_actor_batch_waypoint_in_range():
    store = OnlineHiqlStore()
    store.add_episode(np.arange(40, dtype=np.float32).reshape(20, 2))
    data = store.data()
    rng = np.random.default_rng(0)
    s, sw, g = sample_high_actor_batch(
        data, 16, rng, way_steps=5, future_mode="geometric", gamma=0.99)
    assert s.shape == (16, 2) and sw.shape == (16, 2) and g.shape == (16, 2)
