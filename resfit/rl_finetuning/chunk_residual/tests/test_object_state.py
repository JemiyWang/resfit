import numpy as np
import pytest

from resfit.rl_finetuning.chunk_residual.object_state import (
    eef_rel_piece, rel_piece_stats, compute_eef_rel_piece, PIECE_ROOT_BODIES,
    read_eef_positions, compute_eef_rel_piece_from_env)


def test_piece_root_bodies_constant():
    assert PIECE_ROOT_BODIES == ("piece_1_root", "piece_2_root")


def test_eef_rel_piece_order_and_values():
    eef0 = [1.0, 0.0, 0.0]
    eef1 = [0.0, 2.0, 0.0]
    p1 = [0.0, 0.0, 0.0]
    p2 = [0.0, 0.0, 1.0]
    out = eef_rel_piece([eef0, eef1], [p1, p2])
    assert out.shape == (12,)
    # 顺序 [eef0-p1, eef0-p2, eef1-p1, eef1-p2]
    np.testing.assert_allclose(out[0:3], [1, 0, 0])
    np.testing.assert_allclose(out[3:6], [1, 0, -1])
    np.testing.assert_allclose(out[6:9], [0, 2, 0])
    np.testing.assert_allclose(out[9:12], [0, 2, -1])


def test_eef_rel_piece_dtype_float32():
    out = eef_rel_piece([[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]])
    assert out.dtype == np.float32
    assert out.shape == (12,)


def test_rel_piece_stats_mean_std_and_floor():
    a = np.array([[1., 2., 3.], [3., 6., 3.]], dtype=np.float32)
    m, s = rel_piece_stats(a)
    np.testing.assert_allclose(m, [2., 4., 3.])
    assert s[2] == pytest.approx(1e-6)   # std=0 被抬到下限
    assert (s > 0).all()
    assert m.dtype == np.float32 and s.dtype == np.float32


class _FakeSim:
    """mock mujoco sim:body_name -> xpos。"""
    def __init__(self, body_xpos):
        class _Data:
            def get_body_xpos(_self, name):
                return np.asarray(body_xpos[name], dtype=np.float64)
        self.data = _Data()


def test_compute_eef_rel_piece_reads_sim():
    sim = _FakeSim({"piece_1_root": [0., 0., 0.], "piece_2_root": [0., 0., 1.]})
    out = compute_eef_rel_piece(sim, [[1., 0., 0.], [0., 2., 0.]])
    assert out.shape == (12,)
    np.testing.assert_allclose(out[0:3], [1, 0, 0])    # eef0 - p1
    np.testing.assert_allclose(out[3:6], [1, 0, -1])   # eef0 - p2


class _FakeEnv:
    """mock TwoArm env: _eefN_xpos(sim 实时 eef) + sim.data.get_body_xpos(piece)。"""
    def __init__(self, eef0, eef1, piece_xpos):
        self._eef0_xpos = np.asarray(eef0, dtype=np.float64)
        self._eef1_xpos = np.asarray(eef1, dtype=np.float64)
        self.sim = _FakeSim(piece_xpos)


def test_read_eef_positions():
    env = _FakeEnv([1., 0., 0.], [0., 2., 0.], {})
    eefs = read_eef_positions(env)
    assert len(eefs) == 2
    np.testing.assert_allclose(eefs[0], [1, 0, 0])
    np.testing.assert_allclose(eefs[1], [0, 2, 0])
    assert eefs[0].dtype == np.float32


def test_compute_eef_rel_piece_from_env():
    env = _FakeEnv([1., 0., 0.], [0., 2., 0.],
                   {"piece_1_root": [0., 0., 0.], "piece_2_root": [0., 0., 1.]})
    out = compute_eef_rel_piece_from_env(env)
    assert out.shape == (12,)
    np.testing.assert_allclose(out[0:3], [1, 0, 0])    # eef0 - p1
    np.testing.assert_allclose(out[3:6], [1, 0, -1])   # eef0 - p2
    np.testing.assert_allclose(out[6:9], [0, 2, 0])    # eef1 - p1
    np.testing.assert_allclose(out[9:12], [0, 2, -1])  # eef1 - p2


# ---- object-aware 物体名从 env 解析(threading/three_piece/fallback)----
from resfit.rl_finetuning.chunk_residual.object_state import get_object_root_bodies


class _Obj:
    def __init__(self, root): self.root_body = root

class _ThreadingEnv:
    def __init__(self):
        self.needle = _Obj("needle_obj_root")
        self.tripod = _Obj("tripod_obj_root")

class _ThreePieceEnv:
    def __init__(self):
        self.piece_1 = _Obj("piece_1_root")
        self.piece_2 = _Obj("piece_2_root")

class _UnknownEnv:
    pass


def test_get_object_root_bodies_threading():
    assert get_object_root_bodies(_ThreadingEnv()) == ("needle_obj_root", "tripod_obj_root")

def test_get_object_root_bodies_threepiece():
    assert get_object_root_bodies(_ThreePieceEnv()) == ("piece_1_root", "piece_2_root")

def test_get_object_root_bodies_fallback_to_constant():
    assert get_object_root_bodies(_UnknownEnv()) == PIECE_ROOT_BODIES


class _ThreadingFullEnv:
    """mock threading env:_eefN_xpos(sim 实时)+ sim.data.get_body_xpos + needle/tripod 属性。"""
    def __init__(self):
        import numpy as np
        self.needle = _Obj("needle_obj_root")
        self.tripod = _Obj("tripod_obj_root")
        self._eef0_xpos = np.array([1., 0., 0.])
        self._eef1_xpos = np.array([0., 2., 0.])
        body = {"needle_obj_root": [0., 0., 0.], "tripod_obj_root": [0., 0., 1.]}
        class _Data:
            def get_body_xpos(_s, name): return np.asarray(body[name], dtype=np.float64)
        class _Sim:
            data = _Data()
        self.sim = _Sim()


def test_compute_from_env_resolves_threading_bodies_no_arg():
    # 不传 bodies → 应自动用 needle/tripod root,而非三件套默认
    out = compute_eef_rel_piece_from_env(_ThreadingFullEnv())
    assert out.shape == (12,)
    np.testing.assert_allclose(out[0:3], [1, 0, 0])    # eef0 - needle
    np.testing.assert_allclose(out[3:6], [1, 0, -1])   # eef0 - tripod
    np.testing.assert_allclose(out[6:9], [0, 2, 0])    # eef1 - needle
    np.testing.assert_allclose(out[9:12], [0, 2, -1])  # eef1 - tripod
