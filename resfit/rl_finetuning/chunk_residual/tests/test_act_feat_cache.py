import numpy as np

from resfit.rl_finetuning.chunk_residual.act_feat_cache import (
    save_act_feat_cache, act_feat_cache_reuse,
)

SIG = {"act_ckpt_id": "ckptA", "image_keys": ["observation.images.agentview"],
       "proprio_key": "observation.state", "pooling": "mean",
       "dataset_id": "ds", "num_demos": None}


def test_roundtrip(tmp_path):
    p = str(tmp_path / "c.npz")
    seqs = [np.ones((3, 5), np.float32), np.zeros((2, 5), np.float32)]
    stats = (np.arange(5, dtype=np.float32), np.ones(5, np.float32))
    save_act_feat_cache(p, seqs, stats, signature=SIG)
    got = act_feat_cache_reuse(p, signature=SIG, num_demos=None)
    assert got is not None
    gseqs, gstats = got
    assert len(gseqs) == 2 and gseqs[0].shape == (3, 5)
    assert np.allclose(gstats[0], stats[0]) and np.allclose(gstats[1], stats[1])


def test_signature_mismatch_returns_none(tmp_path):
    p = str(tmp_path / "c.npz")
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=SIG)
    bad = dict(SIG, act_ckpt_id="ckptB")
    assert act_feat_cache_reuse(p, signature=bad, num_demos=None) is None


def test_partial_num_demos_not_reused(tmp_path):
    p = str(tmp_path / "c.npz")
    sig_partial = dict(SIG, num_demos=1)
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=sig_partial)
    # 请求全量但缓存是部分量 → 不命中
    assert act_feat_cache_reuse(p, signature=SIG, num_demos=None) is None


def test_fp16_option_roundtrip(tmp_path):
    p = str(tmp_path / "c.npz")
    seqs = [np.full((2, 5), 0.5, np.float32)]
    save_act_feat_cache(p, seqs, (np.zeros(5, np.float32), np.ones(5, np.float32)),
                        signature=SIG, fp16=True)
    gseqs, _ = act_feat_cache_reuse(p, signature=SIG, num_demos=None)
    assert gseqs[0].dtype == np.float32 and np.allclose(gseqs[0], 0.5, atol=1e-3)
