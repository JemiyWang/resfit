import inspect
from pathlib import Path

import numpy as np
import pytest

from resfit.rl_finetuning.wm_bridge import block_offline_cache as cache_module
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


def test_endpoint_fingerprint_has_no_selected_demo_count_parameter():
    assert "num_demos" not in inspect.signature(
        endpoint_fingerprint).parameters


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("manifest", {"root": "/other", "episodes": []}),
        ("pi0_serve_ckpt_id", "kai0-b"),
        ("prompt", "other prompt"),
        ("camera_keys", tuple(reversed(CAMERA_KEYS))),
        ("chunk_length", 25),
        ("stride", 25),
    ],
)
def test_endpoint_fingerprint_changes_for_every_identity_field(field, changed):
    base = dict(
        manifest={"root": "/dataset", "episodes": []},
        pi0_serve_ckpt_id="kai0-a",
        prompt="build block",
        camera_keys=CAMERA_KEYS,
        chunk_length=50,
        stride=50,
    )
    assert endpoint_fingerprint(**base) != endpoint_fingerprint(
        **{**base, field: changed})


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("endpoint_fp", "end-b"),
        ("value_sha256", "value-b"),
        ("gamma", 0.99),
        ("num_demos", 295),
        ("stats_sha256", "stats-b"),
        ("action_scale", 0.1),
        ("min_range_per_dim", 0.2),
        ("image_size", 96),
        ("image_keys", ("other.camera",)),
        ("n_step", 2),
    ],
)
def test_replay_fingerprint_changes_for_every_identity_field(field, changed):
    base = dict(
        endpoint_fp="end-a", value_sha256="value-a", gamma=0.995,
        num_demos=2, stats_sha256="stats-a", action_scale=0.2,
        min_range_per_dim=0.1, image_size=84, image_keys=CAMERA_KEYS,
        n_step=1)
    assert replay_fingerprint(**base) != replay_fingerprint(
        **{**base, field: changed})


def test_dataset_manifest_rejects_empty_episode_sequence():
    with pytest.raises(ValueError, match="at least one episode"):
        dataset_manifest(())


def test_endpoint_store_round_trip_and_rejects_partial_file(tmp_path):
    store = EndpointStore(str(tmp_path), "fp")
    record = _record(indices=[0, 50])
    store.save_episode(7, record)
    loaded = store.load_episode(7)
    np.testing.assert_array_equal(loaded.frame_indices, [0, 50])
    (store.directory / "episode_000008.npz.tmp").write_bytes(b"partial")
    assert store.load_episode(8) is None


@pytest.mark.parametrize(
    "indices",
    [
        np.asarray([np.nan]),
        np.asarray([0.5]),
    ],
)
def test_endpoint_store_rejects_non_integral_or_nonfinite_indices(
    tmp_path,
    indices,
):
    store = EndpointStore(str(tmp_path), "fp")
    record = _record(indices=[0])
    invalid = EndpointRecord(
        frame_indices=indices,
        base_actions=record.base_actions,
        prefix_features=record.prefix_features,
        proprio=record.proprio,
    )
    with pytest.raises(ValueError, match="invalid endpoint cache record"):
        store.save_episode(7, invalid)


def test_endpoint_store_uses_unique_temporary_paths(tmp_path, monkeypatch):
    store = EndpointStore(str(tmp_path), "fp")
    sources = []
    original_replace = cache_module.os.replace

    def record_replace(source, destination):
        sources.append(Path(source))
        original_replace(source, destination)

    monkeypatch.setattr(cache_module.os, "replace", record_replace)
    store.save_episode(7, _record(indices=[0]))
    store.save_episode(7, _record(indices=[0]))

    assert len(set(sources)) == 2
    assert all(path.parent == store.directory for path in sources)


def test_force_rebuild_selects_new_generation(tmp_path):
    first = resolve_replay_generation(str(tmp_path), "replay-fp", False)
    _mark_complete(first)
    assert resolve_replay_generation(str(tmp_path), "replay-fp", False) == first
    rebuilt = resolve_replay_generation(str(tmp_path), "replay-fp", True)
    assert rebuilt != first
