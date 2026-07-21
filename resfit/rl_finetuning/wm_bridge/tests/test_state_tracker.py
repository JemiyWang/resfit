import numpy as np
import pytest

from resfit.rl_finetuning.wm_bridge.state_tracker import ProprioTracker


def test_initial_proprio_is_returned_as_copy():
    init = np.arange(16, dtype=np.float32)
    t = ProprioTracker(init)
    got = t.proprio
    got[0] = 999.0                      # 改返回值不得污染内部状态
    assert t.proprio[0] == 0.0


def test_advance_takes_last_action_of_chunk():
    t = ProprioTracker(np.zeros(16, dtype=np.float32))
    chunk = np.arange(50 * 16, dtype=np.float32).reshape(50, 16)
    out = t.advance(chunk)
    np.testing.assert_allclose(out, chunk[-1])
    np.testing.assert_allclose(t.proprio, chunk[-1])


def test_advance_rejects_wrong_action_dim():
    t = ProprioTracker(np.zeros(16, dtype=np.float32))
    with pytest.raises(AssertionError):
        t.advance(np.zeros((50, 14), dtype=np.float32))


def test_init_rejects_wrong_dim():
    with pytest.raises(AssertionError):
        ProprioTracker(np.zeros(14, dtype=np.float32))
