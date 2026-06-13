import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states


class _StubExtractor:
    """每帧返回 [4+18];记录调用次数(验证命中缓存跳过)。"""
    calls = 0

    def embed_batch(self, raw):
        _StubExtractor.calls += 1
        b = raw["observation.state"].shape[0]
        emb = torch.zeros(b, 4)
        return torch.cat([emb, torch.as_tensor(raw["observation.state"], dtype=torch.float32)], -1)


def _raw_seq():
    return [{"observation.images.agentview": torch.zeros(3, 3, 4, 4),
             "observation.state": torch.ones(3, 18)}]


def test_act_feat_branch_builds_then_caches(tmp_path):
    cache = str(tmp_path / "act.npz")
    _StubExtractor.calls = 0
    seqs, std, stats = read_per_demo_states(
        "ignored.hdf5", "ds", state_mode="act_feat", num_demos=None,
        act_feat_cache=cache, act_extractor=_StubExtractor(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        _raw_obs_seqs=_raw_seq())
    assert std is None
    assert seqs[0].shape[1] == 4 + 18
    assert stats is not None and np.asarray(stats[0]).shape[0] == 22
    after_build = _StubExtractor.calls
    assert after_build == 1
    # 再读一次:命中缓存,extractor 不应再被调用
    seqs2, _, _ = read_per_demo_states(
        "ignored.hdf5", "ds", state_mode="act_feat", num_demos=None,
        act_feat_cache=cache, act_extractor=_StubExtractor(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        _raw_obs_seqs=_raw_seq())
    assert _StubExtractor.calls == after_build    # 命中缓存,未再调用
    assert np.allclose(seqs[0], seqs2[0])
