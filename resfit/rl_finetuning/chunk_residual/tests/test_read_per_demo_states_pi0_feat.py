import numpy as np
import pytest
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import save_pi0_feat_cache

SIG = {"serve_ckpt_id": "pi05_base", "serve_metadata": {}, "image_keys": ["agentview_image"],
       "proprio_key": "state18", "pooling": "last", "prompt": "x", "dataset_id": "ds", "num_demos": None}


def test_pi0_feat_cache_hit_returns_seqs_and_stats(tmp_path):
    cache = str(tmp_path / "c.npz")
    seqs = [np.zeros((3, 5), np.float32), np.zeros((2, 5), np.float32)]
    save_pi0_feat_cache(cache, seqs, (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=SIG)
    out_seqs, standardizer, feat_stats = read_per_demo_states(
        "ignored.hdf5", "ds", "pi0_feat", pi0_feat_cache=cache,
        pi0_feat_signature=dict(SIG, num_demos=None))
    assert len(out_seqs) == 2 and out_seqs[0].shape[1] == 5
    assert standardizer is None
    mean, std = feat_stats
    assert mean.shape == (5,)


def test_pi0_feat_cache_missing_raises(tmp_path):
    with pytest.raises(RuntimeError):
        read_per_demo_states("ignored.hdf5", "ds", "pi0_feat",
                             pi0_feat_cache=str(tmp_path / "nope.npz"),
                             pi0_feat_signature=dict(SIG, num_demos=None))
