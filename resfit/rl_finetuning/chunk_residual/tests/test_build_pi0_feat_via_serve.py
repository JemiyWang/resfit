import h5py
import numpy as np
from resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve import (
    assemble_pi0_feat_seqs, pi0_feat_signature, build_main,
)


def test_assemble_concats_proprio_and_standardizes():
    raw = [np.ones((2, 3), np.float32), np.full((2, 3), 3.0, np.float32)]   # prefix_feat(T,3)
    prop = [np.zeros((2, 2), np.float32), np.full((2, 2), 4.0, np.float32)] # proprio(T,2)
    seqs, mean, std = assemble_pi0_feat_seqs(raw, prop)
    assert seqs[0].shape == (2, 5)                          # 3 + 2
    allcat = np.concatenate(seqs, axis=0)
    assert np.allclose(allcat.mean(0), 0, atol=1e-5)        # 全量标准化
    assert mean.shape == (5,) and std.shape == (5,)


def test_signature_changes_with_pooling():
    s1 = pi0_feat_signature("ds", None, image_keys=["a"], proprio_key="p", pooling="last",
                            prompt="x", serve_ckpt_id="ck", serve_metadata={})
    s2 = pi0_feat_signature("ds", None, image_keys=["a"], proprio_key="p", pooling="mean",
                            prompt="x", serve_ckpt_id="ck", serve_metadata={})
    assert s1 != s2


class _StubClient:
    def __init__(self):
        self.calls = 0
        self.metadata = {"server": "stub"}

    def get_server_metadata(self):
        return self.metadata

    def infer(self, obs):
        self.calls += 1
        return {"actions": np.zeros((1, 7), np.float32), "prefix_feat": np.ones(3, np.float32)}


def _make_hdf5(path, demos=2, T=3):
    with h5py.File(path, "w") as f:
        for i in range(demos):
            g = f.create_group(f"data/demo_{i}")
            g.create_dataset("obs/agentview_image", data=np.zeros((T, 4, 4, 3), np.uint8))
            g.create_dataset("obs/state18", data=np.full((T, 2), float(i + 1), np.float32))


def test_build_main_writes_cache_and_uses_client(tmp_path):
    hdf5 = str(tmp_path / "t.hdf5"); _make_hdf5(hdf5)
    cache = str(tmp_path / "c.npz")
    client = _StubClient()
    build_main(client, hdf5=hdf5, dataset_id="ds", image_keys=["agentview_image"],
               proprio_key="state18", prompt="x", pooling="last", serve_ckpt_id="pi05_base",
               out_cache=cache, num_demos=None)
    assert client.calls == 6                                # 2 demos * 3 frames
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    seqs, (mean, std), sig = load_pi0_feat_cache(cache)
    assert len(seqs) == 2 and seqs[0].shape[1] == 3 + 2     # prefix3 + proprio2
    assert sig["serve_ckpt_id"] == "pi05_base"
