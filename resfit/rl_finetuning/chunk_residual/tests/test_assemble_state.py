import numpy as np
import pytest

from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    assemble_state, assemble_state18, STATE18_KEYS, STATE_DIM_BY_MODE)


def _fake_obs(T=5):
    rs = np.random.RandomState(0)
    return {k: rs.randn(T, d).astype(np.float32) for k, d in STATE18_KEYS}


def test_eef_mode_equals_assemble_state18():
    obs = _fake_obs()
    np.testing.assert_array_equal(assemble_state(obs, "eef"), assemble_state18(obs))


def test_eef_piece_shape_and_concat():
    obs = _fake_obs(T=5)
    rp = np.arange(5 * 12, dtype=np.float32).reshape(5, 12)
    out = assemble_state(obs, "eef_piece", rel_piece=rp)
    assert out.shape == (5, 30)
    np.testing.assert_array_equal(out[:, :18], assemble_state18(obs))
    np.testing.assert_array_equal(out[:, 18:], rp)


def test_eef_piece_requires_rel_piece():
    with pytest.raises(ValueError):
        assemble_state(_fake_obs(), "eef_piece")


def test_eef_piece_wrong_rel_shape_raises():
    with pytest.raises(ValueError):
        assemble_state(_fake_obs(T=5), "eef_piece", rel_piece=np.zeros((5, 9)))


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        assemble_state(_fake_obs(), "bogus")


def test_state_dim_by_mode_table():
    assert STATE_DIM_BY_MODE == {"eef": 18, "eef_piece": 30}
