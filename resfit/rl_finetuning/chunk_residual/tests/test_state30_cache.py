import numpy as np

from resfit.rl_finetuning.chunk_residual.state30_cache import (
    save_state30_cache, load_state30_cache)


def test_state30_cache_roundtrip(tmp_path):
    seqs = [np.arange(10 * 30).reshape(10, 30).astype(np.float32),
            np.ones((4, 30), np.float32)]
    p = str(tmp_path / "c.npz")
    save_state30_cache(p, seqs)
    out = load_state30_cache(p)
    assert len(out) == 2
    assert out[0].shape == (10, 30)
    assert out[1].shape == (4, 30)
    assert np.allclose(out[0], seqs[0])
    assert np.allclose(out[1], seqs[1])
