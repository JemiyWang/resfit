import numpy as np

from resfit.rl_finetuning.chunk_residual.state30_cache import (
    save_state30_cache,
    load_state30_cache,
    load_state30_cache_v2,
    state30_cache_reuse,
)


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


def test_state30_cache_v2_rel_stats_without_dataset_id(tmp_path):
    p = str(tmp_path / "c.npz")
    save_state30_cache(p, [np.ones((2, 30), np.float32)],
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)))
    s, rel, ds = load_state30_cache_v2(p)
    assert len(s) == 1
    assert rel is not None and np.allclose(rel[1], 1.0)
    assert ds is None


def _write_v2(p, n=3, dataset_id="ds"):
    seqs = [np.ones((2, 30), np.float32) for _ in range(n)]
    save_state30_cache(p, seqs,
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)),
                       dataset_id=dataset_id)


def test_reuse_hit_full(tmp_path):
    p = str(tmp_path / "c.npz"); _write_v2(p, n=3, dataset_id="ds")
    out = state30_cache_reuse(p, dataset_id="ds", num_demos=None)
    assert out is not None
    seqs, rel = out
    assert len(seqs) == 3 and np.allclose(rel[1], 1.0)


def test_reuse_skip_partial_num_demos(tmp_path):
    p = str(tmp_path / "c.npz"); _write_v2(p)
    assert state30_cache_reuse(p, dataset_id="ds", num_demos=20) is None


def test_reuse_skip_missing_file(tmp_path):
    assert state30_cache_reuse(str(tmp_path / "nope.npz"), dataset_id="ds", num_demos=None) is None


def test_reuse_skip_old_format(tmp_path):
    p = str(tmp_path / "old.npz")
    save_state30_cache(p, [np.ones((2, 30), np.float32)])   # v1,无 stats
    assert state30_cache_reuse(p, dataset_id="ds", num_demos=None) is None


def test_reuse_skip_dataset_mismatch(tmp_path):
    p = str(tmp_path / "c.npz"); _write_v2(p, dataset_id="ds_a")
    assert state30_cache_reuse(p, dataset_id="ds_b", num_demos=None) is None


def test_reuse_skip_v2_without_stored_dataset_id(tmp_path):
    # v2 缓存有 rel_stats 但没存 dataset_id(ds 读回 None)→ 对真 dataset_id 应拒绝
    p = str(tmp_path / "c.npz")
    save_state30_cache(p, [np.ones((2, 30), np.float32)],
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)))
    assert state30_cache_reuse(p, dataset_id="ankile/x", num_demos=None) is None
