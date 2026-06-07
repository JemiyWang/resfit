import numpy as np
import pytest

from resfit.rl_finetuning.chunk_residual.object_state import (
    eef_rel_piece, rel_piece_stats, compute_eef_rel_piece, PIECE_ROOT_BODIES)


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
