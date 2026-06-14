"""③a' object-aware:build_offline_buffer 在 eef_piece value 下 replay rel_piece 并传给 Φ。

observation.state 仍存 18 维(actor/critic 不变);rel_piece 只用于重算 potential reward。
用极小 HDF5 + monkeypatch sim-replay(不起真 robosuite)做快速单测。
"""
import numpy as np
import h5py
import torch

from resfit.rl_finetuning.chunk_residual import offline_stage_replay as osr
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    STATE18_KEYS, save_stage_cache)


def _make_tiny_hdf5(path, T=4):
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("states", data=np.zeros((T, 5), dtype=np.float32))
        g.create_dataset("actions", data=np.zeros((T, 7), dtype=np.float32))
        for k, d in STATE18_KEYS:
            g.create_dataset(f"obs/{k}", data=np.zeros((T, d), dtype=np.float32))
        g.create_dataset("obs/agentview_image", data=np.zeros((T, 4, 4, 3), dtype=np.uint8))
        g.attrs["model_file"] = "dummy_model"
    return T


class _IdScaler:
    def scale(self, a): return torch.as_tensor(a, dtype=torch.float32)
    def unscale(self, a): return a


class _IdStd:
    def standardize(self, s): return torch.as_tensor(s, dtype=torch.float32)


class _RelPot:
    """eef_piece value:phi = state第一列 + rel第一列(rel 必须被 build 传进来)。"""
    state_mode = "eef_piece"
    def phi(self, state_seq, rel_piece_seq=None):
        base = np.asarray(state_seq)[:, 0].astype(np.float32)
        assert rel_piece_seq is not None, "eef_piece 时 build 必须传 rel_piece_seq"
        return base + np.asarray(rel_piece_seq)[:, 0]


class _FakeRb:
    def __init__(self): self.items = []
    def add(self, td): self.items.append(td)


def test_build_offline_buffer_replays_rel_for_eef_piece(tmp_path, monkeypatch):
    hdf5 = str(tmp_path / "tiny.hdf5")
    T = _make_tiny_hdf5(hdf5)
    # stage 缓存命中 → 不走 stage replay;rel 必须独立 replay
    stage_cache = str(tmp_path / "stages.npz")
    save_stage_cache(stage_cache, {"demo_0": np.array([0, 1, 2, 3], dtype=np.int8)})

    calls = {"rel": 0}
    rel_arr = np.tile(np.arange(12, dtype=np.float32), (T, 1))   # (T,12),第一列=0
    rel_arr[:, 0] = np.arange(T, dtype=np.float32) * 100.0       # 第一列 = 0,100,200,300

    def _fake_make_env(_path):
        class _E:
            def close(self): pass
        return _E(), "FakeEnv"

    def _fake_replay_rel(env, states, *, model_file, ep_meta=None):
        calls["rel"] += 1
        return rel_arr

    monkeypatch.setattr(osr, "make_replay_env", _fake_make_env)
    monkeypatch.setattr(osr, "replay_eef_rel_piece", _fake_replay_rel)

    rb = _FakeRb()
    added = osr.build_offline_buffer(
        rb, hdf5, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
        image_keys=["observation.images.agentview"], bonus=1.0, mode="potential",
        gamma=0.99, stage_cache=stage_cache, potential=_RelPot())

    assert added == T - 1
    assert calls["rel"] == 1                       # eef_piece 必须 replay 一次 rel
    # 验证 reward 用了 rel:t=0 phi_start=state0[0]+rel0[0]=0+0=0, phi_next=state1[0]+rel1[0]=0+100=100
    from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping
    exp0 = potential_shaping(0.0, 100.0, bonus=1.0, gamma=0.99, done=False)
    got0 = float(rb.items[0]["next"]["reward"])
    assert abs(got0 - exp0) < 1e-4


def test_build_offline_buffer_no_rel_replay_for_eef(tmp_path, monkeypatch):
    """eef value(state_mode 缺省):不 replay rel,零额外开销、零回归。"""
    hdf5 = str(tmp_path / "tiny.hdf5")
    _make_tiny_hdf5(hdf5)
    stage_cache = str(tmp_path / "stages.npz")
    save_stage_cache(stage_cache, {"demo_0": np.array([0, 1, 2, 3], dtype=np.int8)})

    called = {"rel": 0}
    monkeypatch.setattr(osr, "replay_eef_rel_piece",
                        lambda *a, **k: called.__setitem__("rel", called["rel"] + 1))

    rb = _FakeRb()
    osr.build_offline_buffer(
        rb, hdf5, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
        image_keys=["observation.images.agentview"], bonus=1.0, mode="staged",
        gamma=0.99, stage_cache=stage_cache, potential=None)
    assert called["rel"] == 0


# --- offline buffer 缓存签名:必须区分 Φ 身份(否则 30 维 object-aware 错误复用 18 维 V 的缓存)---

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    build_parser, _offline_buffer_signature)


class _SigFakePot:
    def __init__(self, scale, state_mode):
        self.scale = scale
        self.state_mode = state_mode


def _sig_args(extra):
    base = ["--task", "TwoArmThreePieceAssembly", "--offline_dataset_path", "/tmp/x.hdf5"]
    return build_parser().parse_args(base + extra)


def test_signature_distinguishes_value_ckpt_and_state_mode():
    img = ["observation.images.agentview"]
    a18 = _sig_args(["--potential_source", "hiql", "--reward_shaping", "potential",
                     "--hiql_value_ckpt", "v18.pt"])
    a30 = _sig_args(["--potential_source", "hiql", "--reward_shaping", "potential",
                     "--hiql_value_ckpt", "v30.pt"])
    s18 = _offline_buffer_signature(a18, img, 100, "potential",
                                    potential=_SigFakePot(2.0, "eef"))
    s30 = _offline_buffer_signature(a30, img, 100, "potential",
                                    potential=_SigFakePot(3.0, "eef_piece"))
    assert s18 != s30           # 不同 value/state_mode/scale → 不同签名 → 不会错误复用缓存


def test_signature_stage_source_backward_compatible():
    """stage 源签名不含 hiql 键 → 不动现有 stage 缓存(向后兼容)。"""
    img = ["observation.images.agentview"]
    s = _offline_buffer_signature(_sig_args([]), img, 100, "staged")
    assert "hiql_value_ckpt" not in s
    assert "potential_source" not in s


def test_build_offline_buffer_lerobot_no_stage():
    import numpy as np
    import torch
    from tensordict import TensorDict
    from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import build_offline_buffer

    class _StubDS:
        def __init__(self):
            self.episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([4])}
        def __getitem__(self, i):
            return {"observation.images.agentview": torch.zeros(3, 4, 4),
                    "observation.state": torch.zeros(18), "action": torch.zeros(7)}

    class _StubScaler:
        def scale(self, x): return torch.as_tensor(x, dtype=torch.float32)
    class _StubStd:
        def standardize(self, x): return torch.as_tensor(x, dtype=torch.float32)
    class _StubSubgoal:
        state_mode = "act_feat"
        def subgoal_waypoint(self, b, t): return torch.zeros(b.shape[0], 10)

    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=100, device="cpu"))
    n = build_offline_buffer(
        rb, "ignored.hdf5", action_scaler=_StubScaler(), state_standardizer=_StubStd(),
        image_keys=["observation.images.agentview"], bonus=1.0, mode="none", gamma=0.99,
        subgoal=_StubSubgoal(), way_steps=2,
        act_feat_seqs=[np.zeros((4, 530), np.float32)],
        data_source="lerobot", lerobot_repo_id="repo/id", lerobot_root="/x", _lerobot_ds=_StubDS(),
        base_mode="gt")
    assert n == 3                                  # T=4 → 3 transitions
    td = rb.sample(1)
    assert "observation.subgoal" in td["obs"].keys()
    assert float(td["obs"]["observation.stage_id"].abs().max()) == 0.0   # no-stage
