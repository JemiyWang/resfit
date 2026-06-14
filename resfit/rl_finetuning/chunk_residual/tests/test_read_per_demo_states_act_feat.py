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


def test_act_feat_build_standardizes_proprio():
    import torch
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states

    captured = {}

    class _RecExtractor:
        """记录被喂进的 proprio,便于直接断言它已被 dataset-标准化(绕开 act_feat 末标准化的干扰)。"""
        def embed_batch(self, raw):
            captured["proprio"] = torch.as_tensor(raw["observation.state"], dtype=torch.float32).clone()
            b = raw["observation.state"].shape[0]
            return torch.cat([torch.zeros(b, 2),
                              torch.as_tensor(raw["observation.state"], dtype=torch.float32)], -1)

    class _StubStd:
        def standardize(self, x):            # (x - 1)/2,便于断言确实被应用
            return (torch.as_tensor(x, dtype=torch.float32) - 1.0) / 2.0

    raw = [{"observation.images.agentview": torch.zeros(3, 3, 4, 4),
            "observation.state": torch.full((3, 18), 5.0)}]   # raw proprio = 5
    read_per_demo_states(
        "ignored.hdf5", "ds", state_mode="act_feat", num_demos=None,
        act_feat_cache=None, act_extractor=_RecExtractor(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        state_standardizer=_StubStd(), _raw_obs_seqs=raw)
    # extractor 收到的 proprio 应已被标准化:(5-1)/2 = 2.0(证明 build 在喂 extractor 前应用了 standardizer)
    assert torch.allclose(captured["proprio"], torch.full((3, 18), 2.0))


def test_act_feat_lerobot_source(tmp_path):
    import torch
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states

    class _StubDS:
        def __init__(self):
            self.episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([3])}
        def __getitem__(self, i):
            return {"observation.images.agentview": torch.zeros(3, 4, 4),
                    "observation.state": torch.full((18,), 5.0), "action": torch.zeros(7)}

    class _RecExtractor:
        captured = {}
        def embed_batch(self, ro):
            _RecExtractor.captured["proprio"] = ro["observation.state"].clone()
            b = ro["observation.state"].shape[0]
            return torch.cat([torch.zeros(b, 4), torch.as_tensor(ro["observation.state"], dtype=torch.float32)], -1)

    class _StubStd:
        def standardize(self, x):
            return (torch.as_tensor(x, dtype=torch.float32) - 1.0) / 2.0

    seqs, std, stats = read_per_demo_states(
        "ignored.hdf5", "repo/id", state_mode="act_feat", num_demos=None,
        act_feat_cache=None, act_extractor=_RecExtractor(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        state_standardizer=_StubStd(), data_source="lerobot", _lerobot_ds=_StubDS())
    assert std is None and seqs[0].shape == (3, 22)
    # proprio 经 dataset-标准化:(5-1)/2 = 2.0
    assert torch.allclose(_RecExtractor.captured["proprio"], torch.full((3, 18), 2.0))
