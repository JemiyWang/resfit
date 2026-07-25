"""Task 15:pi0_feat 建缓存脚本的 teleavatar/block 路。

block 实机数据是 LeRobot 格式、三相机(top_head/hand_left/hand_right)、16 维 state,
既有的 dexmg_hdf5 / libero 两路都读不了。teleavatar 路照 piper_deploy.py 的权威 obs
schema(嵌套 {"state","images":{...},"prompt"})拼 obs,客户端侧 resize_with_pad 到 224
(与 libero 路同款,→ 与部署同源)。
"""
import numpy as np
import pytest

from resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve import (
    TELEAVATAR_CAM_MAP, build_main_teleavatar, build_parser,
    build_teleavatar_serve_obs,
)


# ---- build_teleavatar_serve_obs:嵌套 schema + 224 resize + cam 映射 ----

def _chw(v):
    return np.full((3, 40, 60), v, np.float32)   # 非方形,验 resize_with_pad


def test_obs_is_nested_schema_matching_piper_deploy():
    obs = build_teleavatar_serve_obs(
        {"top_head": _chw(0.1), "hand_left": _chw(0.2), "hand_right": _chw(0.3)},
        state=np.arange(16, dtype=np.float32), prompt="build block")
    assert set(obs.keys()) == {"state", "images", "prompt"}
    assert set(obs["images"].keys()) == {"top_head", "hand_left", "hand_right"}
    assert obs["prompt"] == "build block"


def test_images_resized_to_224_chw():
    obs = build_teleavatar_serve_obs(
        {"top_head": _chw(0.1), "hand_left": _chw(0.2), "hand_right": _chw(0.3)},
        state=np.zeros(16, np.float32), prompt="p")
    for cam in ("top_head", "hand_left", "hand_right"):
        assert obs["images"][cam].shape == (3, 224, 224)


def test_state_is_passed_through_as_float32_vector():
    obs = build_teleavatar_serve_obs(
        {"top_head": _chw(0.1), "hand_left": _chw(0.2), "hand_right": _chw(0.3)},
        state=np.arange(16, dtype=np.float64), prompt="p")
    assert obs["state"].shape == (16,)
    assert obs["state"].dtype == np.float32
    np.testing.assert_allclose(obs["state"], np.arange(16))


def test_cam_map_covers_the_three_teleavatar_cameras():
    assert set(TELEAVATAR_CAM_MAP.keys()) == {
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right"}
    assert set(TELEAVATAR_CAM_MAP.values()) == {"top_head", "hand_left", "hand_right"}


# ---- CLI ----

def test_data_source_accepts_teleavatar():
    args = build_parser().parse_args([
        "--host", "h", "--port", "9000", "--data_source", "teleavatar",
        "--repo_id", "block_success", "--lerobot_root", "/x",
        "--prompt", "build block", "--serve_ckpt_id", "kai0-block",
        "--out_cache", "/tmp/c.npz"])
    assert args.data_source == "teleavatar"
    assert args.repo_id == "block_success"


# ---- 端到端(stub client + monkeypatch LeRobot 读取)----

class _StubClient:
    def infer(self, obs):
        # 校验拿到的是嵌套 schema(否则 serve 会 KeyError)
        assert "images" in obs and "state" in obs and "prompt" in obs
        assert set(obs["images"]) == {"top_head", "hand_left", "hand_right"}
        return {"prefix_feat": np.arange(8, dtype=np.float32)}

    def get_server_metadata(self):
        return {}


class _RecordingClient(_StubClient):
    def __init__(self):
        self.observations = []

    def infer(self, obs):
        self.observations.append(obs)
        return super().infer(obs)


def _patch_lerobot(monkeypatch, n_eps, T):
    import numpy as np
    from resfit.rl_finetuning.chunk_residual import build_pi0_feat_cache_via_serve as B

    def fake_list(root):
        return list(range(n_eps))

    def fake_read(root, ep, cameras, proprio_key="observation.state", **kw):
        return {
            "images": {k: np.zeros((T, 3, 40, 60), np.uint8) for k in cameras},
            "state": np.arange(T * 16, dtype=np.float32).reshape(T, 16),
        }

    monkeypatch.setattr(B, "list_teleavatar_episodes", fake_list, raising=False)
    monkeypatch.setattr(B, "read_teleavatar_episode_batched", fake_read, raising=False)


def test_build_main_teleavatar_end_to_end(tmp_path, monkeypatch):
    """3 集 × 5 帧 → 逐帧 serve → 缓存;签名带 serve_ckpt_id/三相机键。"""
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    _patch_lerobot(monkeypatch, n_eps=3, T=5)
    out = str(tmp_path / "block_success.npz")

    build_main_teleavatar(
        _StubClient(), lerobot_root="/x", repo_id="block_success",
        prompt="build block", pooling="mean", serve_ckpt_id="kai0-block",
        out_cache=out, num_demos=None)

    seqs, stats, sig = load_pi0_feat_cache(out)
    assert len(seqs) == 3
    # dim = D_emb(8) + proprio(16) = 24
    assert seqs[0].shape[1] == 24
    assert sig["serve_ckpt_id"] == "kai0-block"
    assert sig["prompt"] == "build block"
    assert set(sig["image_keys"]) == {"top_head", "hand_left", "hand_right"}


def test_paper_policy_state_is_14_but_cached_proprio_stays_16(
    tmp_path,
    monkeypatch,
):
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import (
        load_pi0_feat_cache,
    )

    _patch_lerobot(monkeypatch, n_eps=1, T=4)
    client = _RecordingClient()
    out = str(tmp_path / "paper.npz")
    build_main_teleavatar(
        client,
        lerobot_root="/x",
        repo_id="paper_success",
        prompt="put the paper roll on the holder",
        pooling="mean",
        serve_ckpt_id="pi05_paper_awbc_19999",
        out_cache=out,
        policy_state_dim=14,
    )
    assert all(obs["state"].shape == (14,) for obs in client.observations)
    seqs, _, sig = load_pi0_feat_cache(out)
    assert seqs[0].shape[1] == 8 + 16
    assert sig["policy_state_dim"] == 14


def test_build_main_teleavatar_respects_num_demos(tmp_path, monkeypatch):
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    _patch_lerobot(monkeypatch, n_eps=10, T=4)
    out = str(tmp_path / "c.npz")
    build_main_teleavatar(
        _StubClient(), lerobot_root="/x", repo_id="block_fail",
        prompt="build block", pooling="mean", serve_ckpt_id="kai0-block",
        out_cache=out, num_demos=2)
    seqs, _, _ = load_pi0_feat_cache(out)
    assert len(seqs) == 2


def test_shards_partition_episodes_without_overlap(tmp_path, monkeypatch):
    """4 分片跨步切 10 集 → 各分片不重叠,并集=全部。"""
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    _patch_lerobot(monkeypatch, n_eps=10, T=4)
    counts = []
    for i in range(4):
        out = str(tmp_path / f"shard{i}.npz")
        build_main_teleavatar(
            _StubClient(), lerobot_root="/x", repo_id="block_success",
            prompt="build block", pooling="mean", serve_ckpt_id="kai0-block",
            out_cache=out, num_shards=4, shard_index=i)
        seqs, _, _ = load_pi0_feat_cache(out)
        counts.append(len(seqs))
    # eps[0::4]=0,4,8→3; [1::4]=1,5,9→3; [2::4]=2,6→2; [3::4]=3,7→2
    assert counts == [3, 3, 2, 2]
    assert sum(counts) == 10          # 并集=全部 10 集


def test_num_demos_applied_after_shard_not_before(tmp_path, monkeypatch):
    """★ 先分片再截 num_demos:每分片都非空。反序会让高 index 分片拿空集(SMOKE bug)。"""
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    _patch_lerobot(monkeypatch, n_eps=20, T=4)
    for i in range(4):
        out = str(tmp_path / f"s{i}.npz")
        build_main_teleavatar(
            _StubClient(), lerobot_root="/x", repo_id="block_success",
            prompt="p", pooling="mean", serve_ckpt_id="k",
            out_cache=out, num_shards=4, shard_index=i, num_demos=2)
        seqs, _, _ = load_pi0_feat_cache(out)
        assert len(seqs) == 2, f"shard {i} 应有 2 集(先分片再截),got {len(seqs)}"


def test_shard_index_out_of_range_raises(tmp_path, monkeypatch):
    _patch_lerobot(monkeypatch, n_eps=8, T=4)
    with pytest.raises(AssertionError):
        build_main_teleavatar(
            _StubClient(), lerobot_root="/x", repo_id="block_success",
            prompt="p", pooling="mean", serve_ckpt_id="k",
            out_cache=str(tmp_path / "x.npz"), num_shards=4, shard_index=4)


def test_cli_exposes_shard_args():
    args = build_parser().parse_args([
        "--host", "h", "--port", "9000", "--data_source", "teleavatar",
        "--repo_id", "block_success", "--lerobot_root", "/x",
        "--serve_ckpt_id", "k", "--out_cache", "/o",
        "--num_shards", "4", "--shard_index", "2"])
    assert args.num_shards == 4 and args.shard_index == 2


def test_cli_exposes_policy_state_dim():
    args = build_parser().parse_args([
        "--host", "h", "--port", "9000", "--data_source", "teleavatar",
        "--repo_id", "paper_success", "--lerobot_root", "/x",
        "--serve_ckpt_id", "paper", "--out_cache", "/o",
        "--policy_state_dim", "14",
    ])
    assert args.policy_state_dim == 14
