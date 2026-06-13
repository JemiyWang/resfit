import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    read_per_demo_states, _build_raw_obs_seqs,
)
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import STATE18_KEYS


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


def test_build_raw_obs_seqs_image_preprocessing(tmp_path):
    import h5py
    T = 3
    p = str(tmp_path / "mini.hdf5")
    with h5py.File(p, "w") as f:
        g = f.create_group("data/demo_0")
        # HWC uint8 image like real dexmg hdf5
        g.create_dataset("obs/agentview_image",
                         data=(np.arange(T * 84 * 84 * 3).reshape(T, 84, 84, 3) % 256).astype(np.uint8))
        for k, dim in STATE18_KEYS:
            g.create_dataset(f"obs/{k}", data=np.zeros((T, dim), np.float32))
    seqs = _build_raw_obs_seqs(p, ["observation.images.agentview"], "observation.state", num_demos=None)
    img = seqs[0]["observation.images.agentview"]
    assert img.shape == (T, 3, 84, 84)          # CHW
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0   # scaled to [0,1]
    assert seqs[0]["observation.state"].shape == (T, 18)
