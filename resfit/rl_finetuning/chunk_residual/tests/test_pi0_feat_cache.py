import numpy as np
from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import (
    save_pi0_feat_cache, pi0_feat_cache_reuse, load_pi0_feat_cache,
)

SIG = {"serve_ckpt_id": "pi05_base", "serve_metadata": {"a": 1}, "image_keys": ["agentview_image"],
       "proprio_key": "state18", "pooling": "last", "prompt": "do it", "dataset_id": "ds", "num_demos": None}


def _seqs():
    return [np.arange(6, dtype=np.float32).reshape(3, 2), np.ones((2, 2), np.float32)]


def test_save_reuse_roundtrip(tmp_path):
    p = str(tmp_path / "f.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    got = pi0_feat_cache_reuse(p, signature=SIG, num_demos=None)
    assert got is not None
    seqs, (mean, std) = got
    assert len(seqs) == 2 and np.allclose(seqs[0], _seqs()[0])
    assert np.allclose(mean, 0) and np.allclose(std, 1)


def test_reuse_signature_mismatch_returns_none(tmp_path):
    p = str(tmp_path / "f.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    assert pi0_feat_cache_reuse(p, signature=dict(SIG, pooling="mean"), num_demos=None) is None


def test_reuse_partial_num_demos_returns_none(tmp_path):
    p = str(tmp_path / "f.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    assert pi0_feat_cache_reuse(p, signature=SIG, num_demos=1) is None


def test_reuse_missing_file_returns_none(tmp_path):
    assert pi0_feat_cache_reuse(str(tmp_path / "nope.npz"), signature=SIG, num_demos=None) is None


def test_load_returns_seqs_stats_sig(tmp_path):
    p = str(tmp_path / "f.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    seqs, stats, sig = load_pi0_feat_cache(p)
    assert len(seqs) == 2 and sig["serve_ckpt_id"] == "pi05_base"
