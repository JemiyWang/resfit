import numpy as np

from resfit.rl_finetuning.chunk_residual.act_feat_cache import (
    save_act_feat_cache, act_feat_cache_reuse,
)
from resfit.rl_finetuning.chunk_residual.act_feat_cache import load_act_feat_cache

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


def test_num_demos_param_overrides_signature(tmp_path):
    # 缓存按 num_demos=None 存;若调用方 signature 里写了 num_demos=None 但 param 传 5,应不命中
    p = str(tmp_path / "c.npz")
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=SIG)
    assert act_feat_cache_reuse(p, signature=SIG, num_demos=5) is None
    assert act_feat_cache_reuse(p, signature=SIG, num_demos=None) is not None


def test_load_returns_sha_when_present(tmp_path):
    p = str(tmp_path / "c.npz")
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)),
                        signature=SIG, act_weight_sha="deadbeef")
    seqs, stats, sig, sha = load_act_feat_cache(p)
    assert sha == "deadbeef"
    assert sig == SIG and len(seqs) == 1


def test_load_returns_none_sha_when_absent(tmp_path):
    # 旧缓存（不带 sha）→ 第 4 项 None，且 reuse 仍照常命中（向后兼容）
    p = str(tmp_path / "c.npz")
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=SIG)
    seqs, stats, sig, sha = load_act_feat_cache(p)
    assert sha is None
    assert act_feat_cache_reuse(p, signature=SIG, num_demos=None) is not None
