from pathlib import Path

import numpy as np

from resfit.rl_finetuning.wm_bridge.block_offline_cache import (
    EndpointRecord,
    EndpointStore,
    dataset_manifest,
    endpoint_fingerprint,
    replay_fingerprint,
    resolve_replay_generation,
)
from resfit.rl_finetuning.wm_bridge.block_offline_chunk import EpisodeRef
from resfit.rl_finetuning.wm_bridge.wm_driver import CAMERA_KEYS


def _episode_refs(root, lengths):
    result = []
    for episode_id, num_frames in enumerate(lengths):
        parquet = root / f"episode_{episode_id:06d}.parquet"
        parquet.write_bytes(f"parquet-{episode_id}-{num_frames}".encode())
        videos = {}
        for key in CAMERA_KEYS:
            path = root / f"{episode_id}-{key}.mp4"
            path.write_bytes(f"video-{episode_id}-{key}".encode())
            videos[key] = str(path)
        result.append(EpisodeRef(
            root=str(root), episode_id=episode_id,
            parquet_path=str(parquet), video_paths=videos,
            num_frames=num_frames))
    return tuple(result)


def _record(indices):
    n = len(indices)
    return EndpointRecord(
        frame_indices=np.asarray(indices, dtype=np.int64),
        base_actions=np.zeros((n, 50, 16), dtype=np.float32),
        prefix_features=np.zeros((n, 2), dtype=np.float32),
        proprio=np.zeros((n, 16), dtype=np.float32),
    )


def _mark_complete(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "buffer_meta.json").write_text("{}", encoding="utf-8")
    (directory / "bridge_meta.json").write_text("{}", encoding="utf-8")


def test_endpoint_fingerprint_ignores_selected_demo_count(tmp_path):
    episodes = _episode_refs(tmp_path, lengths=[51, 101])
    manifest = dataset_manifest(episodes)
    fp_a = endpoint_fingerprint(
        manifest=manifest, pi0_serve_ckpt_id="kai0-a",
        prompt="build block", camera_keys=CAMERA_KEYS)
    fp_b = endpoint_fingerprint(
        manifest=manifest, pi0_serve_ckpt_id="kai0-a",
        prompt="build block", camera_keys=CAMERA_KEYS)
    assert fp_a == fp_b


def test_replay_fingerprint_changes_for_value_gamma_and_num_demos():
    base = dict(
        endpoint_fp="end-a", value_sha256="value-a", gamma=0.995,
        num_demos=2, stats_sha256="stats-a", action_scale=0.2,
        min_range_per_dim=0.1, image_size=84, image_keys=CAMERA_KEYS,
        n_step=1)
    assert replay_fingerprint(**base) != replay_fingerprint(
        **{**base, "value_sha256": "value-b"})
    assert replay_fingerprint(**base) != replay_fingerprint(
        **{**base, "num_demos": 295})


def test_endpoint_store_round_trip_and_rejects_partial_file(tmp_path):
    store = EndpointStore(str(tmp_path), "fp")
    record = _record(indices=[0, 50])
    store.save_episode(7, record)
    loaded = store.load_episode(7)
    np.testing.assert_array_equal(loaded.frame_indices, [0, 50])
    (store.directory / "episode_000008.npz.tmp").write_bytes(b"partial")
    assert store.load_episode(8) is None


def test_force_rebuild_selects_new_generation(tmp_path):
    first = resolve_replay_generation(str(tmp_path), "replay-fp", False)
    _mark_complete(first)
    assert resolve_replay_generation(str(tmp_path), "replay-fp", False) == first
    rebuilt = resolve_replay_generation(str(tmp_path), "replay-fp", True)
    assert rebuilt != first
