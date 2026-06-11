import numpy as np

from resfit.rl_finetuning.chunk_residual.state30_cache import (
    save_state30_cache, load_state30_cache)
from resfit.rl_finetuning.chunk_residual.state30_cache import load_state30_cache_v2


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


def test_state30_cache_v2_roundtrip(tmp_path):
    seqs = [np.ones((5, 30), np.float32), np.zeros((3, 30), np.float32)]
    mean = np.arange(12).astype(np.float32)
    std = (np.arange(12) + 1).astype(np.float32)
    p = str(tmp_path / "c.npz")
    save_state30_cache(p, seqs, rel_stats=(mean, std), dataset_id="ds/x")
    s, rel, ds = load_state30_cache_v2(p)
    assert len(s) == 2 and np.allclose(s[0], seqs[0])
    assert rel is not None and np.allclose(rel[0], mean) and np.allclose(rel[1], std)
    assert ds == "ds/x"


def test_state30_cache_v1_loaded_as_no_stats(tmp_path):
    p = str(tmp_path / "old.npz")
    save_state30_cache(p, [np.ones((2, 30), np.float32)])
    s, rel, ds = load_state30_cache_v2(p)
    assert len(s) == 1 and rel is None and ds is None


def test_old_loader_still_reads_v2_seqs(tmp_path):
    p = str(tmp_path / "c.npz")
    save_state30_cache(p, [np.ones((2, 30), np.float32)],
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)),
                       dataset_id="d")
    assert len(load_state30_cache(p)) == 1
