# Block Offline Replay Sparse Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated per-frame MP4 open/seek/decode operations with an episode-local sparse frame cache while preserving existing endpoint caches and replay semantics.

**Architecture:** A reusable TorchCodec helper batch-decodes the sorted endpoint indices once per camera. `BlockEpisodeReader` normalizes those sparse frames once and serves later endpoint and replay lookups from memory. A real-data gate proves exact pixel equality and at least 3x speedup before the interrupted seed-0 run is restarted with its existing endpoint cache.

**Tech Stack:** Python 3.10, PyTorch 2.7, TorchCodec, OpenCV, pandas, NumPy, TensorDict, pytest, tmux.

## Global Constraints

- Preserve exactly 6,989 offline transitions for the full `block_success` dataset.
- Preserve chunk boundaries, terminal flags, actions, features, rewards, tensor schema, image resolution, and normalization.
- Reuse valid endpoint `.npz` files; do not delete or rewrite them.
- Require exact `uint8` pixel equality between the accepted optimized decoder and the legacy OpenCV decoder.
- Stop before production restart if pixel equality fails; do not mix old endpoint features with different replay pixels.
- Decode each required frame at most once per episode.
- Keep additional resident memory below 512 MiB.
- Require at least 3x end-to-end reader speedup on three representative real episodes.
- Keep WM port 9000, pi0 port 8001, and advantage port 8002 services running while restarting only seed 0 on GPU 7.
- Do not reuse the interrupted process's incomplete replay generation.

## File Structure

- Modify `resfit/rl_finetuning/chunk_residual/teleavatar_batch_source.py`
  - Owns reusable sparse TorchCodec video decoding and input validation.
- Modify `resfit/rl_finetuning/chunk_residual/tests/test_teleavatar_batch_source.py`
  - Owns isolated sparse decoder unit tests with a fake decoder.
- Modify `resfit/rl_finetuning/wm_bridge/block_offline_chunk.py`
  - Owns episode-local normalized frame caching and offline build integration.
- Modify `resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py`
  - Owns cache reuse, transition equivalence, and profiling-log unit tests.
- Create `resfit/rl_finetuning/wm_bridge/verify_block_sparse_decode.py`
  - Owns explicit real-data pixel, runtime, and memory verification.
- Create `resfit/rl_finetuning/wm_bridge/tests/test_verify_block_sparse_decode.py`
  - Owns verification-report and failure-gate unit tests.

---

### Task 1: Snapshot and Gracefully Stop the Seed-0 Builder

**Files:**
- Read: `outputs_imagination/block_shore_mixed50_seed0.log`
- Preserve: `cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae/`

**Interfaces:**
- Consumes: tmux session `block_mixed_s0`, trainer PID resolved from the live process table.
- Produces: a stopped seed-0 builder, a recorded endpoint-cache file count, and unchanged WM/pi0/adv service PIDs.

- [ ] **Step 1: Record the exact live targets**

Run:

```bash
tmux list-panes -t block_mixed_s0 -F '#{session_name} #{pane_id} #{pane_pid} #{pane_current_command}'
ps -eo pid,ppid,etime,state,%cpu,rss,args
rg -c '^offline_build_progress' outputs_imagination/block_shore_mixed50_seed0.log
ls cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae
```

Expected:

- exactly one `launch_block_imagination.sh 7 0` wrapper and one
  `launch_imagination` trainer are associated with `block_mixed_s0`;
- service PIDs for ports 9000, 8001, and 8002 remain independent;
- endpoint files are named `episode_XXXXXX.npz`.

- [ ] **Step 2: Save endpoint-cache integrity evidence**

Run:

```bash
sha256sum cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae/episode_000000.npz
sha256sum cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae/episode_000001.npz
sha256sum cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae/episode_000002.npz
```

Expected: three SHA-256 lines are captured for comparison after restart.

- [ ] **Step 3: Interrupt only the seed-0 pane**

Run:

```bash
tmux send-keys -t block_mixed_s0 C-c
```

Expected: the wrapper, trainer, and its `tee` child exit; the tmux pane remains
available.

- [ ] **Step 4: Verify service and cache preservation**

Run:

```bash
ps -p 1698375,1698376,1698377 -o pid,etime,state,%cpu,rss,args
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
ls cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae
```

Expected:

- the seed-0 trainer no longer appears on GPU 7;
- WM, pi0, and advantage services remain alive;
- the endpoint file count is unchanged.

---

### Task 2: Add Sparse TorchCodec Frame Decoding

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/teleavatar_batch_source.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_teleavatar_batch_source.py`

**Interfaces:**
- Consumes: `teleavatar_video_path(root, ep, cam, chunks_size)`.
- Produces: `read_teleavatar_episode_frames(root, ep, cameras, frame_indices, chunks_size=1000) -> dict[str, torch.Tensor]`, where every value has shape `(K, 3, H, W)` and dtype `torch.uint8`.

- [ ] **Step 1: Write failing sparse-decoder tests**

Append these imports and helpers to
`test_teleavatar_batch_source.py`:

```python
from types import SimpleNamespace

import torch

import resfit.rl_finetuning.chunk_residual.teleavatar_batch_source as source


class _FakeDecoder:
    calls = []

    def __init__(self, path):
        self.path = path
        self.metadata = SimpleNamespace(num_frames=121)

    def get_frames_at(self, indices):
        type(self).calls.append((self.path, tuple(indices)))
        data = torch.stack([
            torch.full((3, 2, 2), index, dtype=torch.uint8)
            for index in indices
        ])
        return SimpleNamespace(data=data)


def test_sparse_read_batches_once_per_camera(monkeypatch):
    _FakeDecoder.calls = []
    monkeypatch.setattr(source, "_make_video_decoder", _FakeDecoder)
    cameras = ["cam_a", "cam_b", "cam_c"]

    images = source.read_teleavatar_episode_frames(
        "/dataset", 7, cameras, [0, 50, 69, 100, 120])

    assert set(images) == set(cameras)
    assert len(_FakeDecoder.calls) == 3
    assert all(call[1] == (0, 50, 69, 100, 120)
               for call in _FakeDecoder.calls)
    assert all(tuple(value.shape) == (5, 3, 2, 2)
               for value in images.values())
    assert all(value.dtype == torch.uint8 for value in images.values())


@pytest.mark.parametrize(
    "indices, message",
    [
        ([], "non-empty"),
        ([0, 0], "sorted unique"),
        ([50, 0], "sorted unique"),
        ([-1, 0], "non-negative"),
        ([0, 121], "out of range"),
    ],
)
def test_sparse_read_rejects_invalid_indices(monkeypatch, indices, message):
    monkeypatch.setattr(source, "_make_video_decoder", _FakeDecoder)
    with pytest.raises((ValueError, IndexError), match=message):
        source.read_teleavatar_episode_frames(
            "/dataset", 0, ["cam"], indices)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_teleavatar_batch_source.py -q
```

Expected: FAIL because `read_teleavatar_episode_frames` and
`_make_video_decoder` do not exist.

- [ ] **Step 3: Implement the minimal sparse decoder**

Add to `teleavatar_batch_source.py`:

```python
def _make_video_decoder(path):
    from torchcodec.decoders import VideoDecoder
    return VideoDecoder(path)


def _validated_sparse_indices(frame_indices):
    indices = list(frame_indices)
    if not indices:
        raise ValueError("frame_indices must be non-empty")
    if any(isinstance(index, bool) or not isinstance(index, (int, np.integer))
           for index in indices):
        raise ValueError("frame_indices must contain integers")
    indices = [int(index) for index in indices]
    if any(index < 0 for index in indices):
        raise ValueError("frame_indices must be non-negative")
    if indices != sorted(set(indices)):
        raise ValueError("frame_indices must be sorted unique")
    return indices


def read_teleavatar_episode_frames(
    root,
    ep,
    cameras,
    frame_indices,
    chunks_size: int = CHUNKS_SIZE,
):
    """Batch-decode selected CHW RGB uint8 frames once per camera."""
    indices = _validated_sparse_indices(frame_indices)
    images = {}
    for camera in cameras:
        path = teleavatar_video_path(
            root, ep, camera, chunks_size=chunks_size)
        decoder = _make_video_decoder(path)
        num_frames = int(decoder.metadata.num_frames)
        if indices[-1] >= num_frames:
            raise IndexError(
                f"frame index {indices[-1]} out of range for "
                f"episode={ep} camera={camera} num_frames={num_frames}")
        data = decoder.get_frames_at(indices).data
        if (
            data.dtype != torch.uint8
            or data.ndim != 4
            or data.shape[0] != len(indices)
            or data.shape[1] != 3
        ):
            raise ValueError(
                f"invalid decoded frames for episode={ep} camera={camera}: "
                f"shape={tuple(data.shape)} dtype={data.dtype}")
        images[camera] = data
    return images
```

Add `import torch` at module level because output validation depends on the
public tensor dtype.

- [ ] **Step 4: Run focused and existing decoder tests**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_teleavatar_batch_source.py -q
```

Expected: all tests PASS, including existing full-episode decoding tests.

- [ ] **Step 5: Commit the sparse decoder**

```bash
git add resfit/rl_finetuning/chunk_residual/teleavatar_batch_source.py resfit/rl_finetuning/chunk_residual/tests/test_teleavatar_batch_source.py
git commit -m "perf: batch decode sparse block frames"
```

---

### Task 3: Cache Sparse Frames Per Offline Episode

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/block_offline_chunk.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py`

**Interfaces:**
- Consumes: `read_teleavatar_episode_frames(...) -> dict[str, torch.Tensor]`.
- Produces:
  - `BlockEpisodeReader.prepare_native_frames(frame_indices) -> None`;
  - `BlockEpisodeReader.native_frames(frame_index) -> np.ndarray` with shape
    `(3, 3, H, W)` and dtype `float32`;
  - one `offline_episode_profile` log line per completed episode.

- [ ] **Step 1: Write failing reader-cache tests**

Import `BlockEpisodeReader` in `test_block_offline_chunk.py`, then add:

```python
def _write_block_reader_parquet(path, num_frames):
    pd.DataFrame({
        "action": [np.full(16, index, np.float32)
                   for index in range(num_frames)],
        "observation.state": [np.full(18, index, np.float32)
                              for index in range(num_frames)],
    }).to_parquet(path)


def test_block_reader_prepares_sparse_frames_once(tmp_path):
    episode = synthetic_episode(tmp_path, num_frames=51)
    _write_block_reader_parquet(episode.parquet_path, 51)
    calls = []

    def fake_loader(root, episode_id, cameras, frame_indices):
        calls.append((root, episode_id, tuple(cameras), tuple(frame_indices)))
        return {
            camera: torch.stack([
                torch.full((3, 2, 2), index, dtype=torch.uint8)
                for index in frame_indices
            ])
            for camera in cameras
        }

    reader = BlockEpisodeReader(episode, frame_loader=fake_loader)
    reader.prepare_native_frames([0, 50])

    first = reader.native_frames(0)
    second = reader.native_frames(0)
    assert len(calls) == 1
    assert first is second
    assert first.shape == (len(CAMERA_KEYS), 3, 2, 2)
    assert first.dtype == np.float32
    assert np.all(first == -1.0)
    assert np.allclose(reader.native_frames(50), 50 / 127.5 - 1.0)


def test_block_reader_rejects_unprepared_frame(tmp_path):
    episode = synthetic_episode(tmp_path, num_frames=51)
    _write_block_reader_parquet(episode.parquet_path, 51)
    reader = BlockEpisodeReader(
        episode, frame_loader=lambda *args: {})
    with pytest.raises(RuntimeError, match="not prepared"):
        reader.native_frames(0)
```

- [ ] **Step 2: Write a failing build-integration test**

Extend `FakeReader`:

```python
class FakeReader:
    instances = []

    def __init__(self, episode=None, num_frames=None):
        type(self).instances.append(self)
        self.prepared = []
        self.decoder_backend = "fake"
        # keep the existing actions and states initialization

    def prepare_native_frames(self, frame_indices):
        self.prepared.append(tuple(frame_indices))
```

Add:

```python
def test_build_prepares_union_once_and_profiles_episode(tmp_path, capsys):
    FakeReader.instances = []
    build_block_offline_buffer(
        ListReplay(),
        str(tmp_path),
        action_scaler=IdentityActionScaler(),
        state_standardizer=IdentityStateStandardizer(),
        image_keys=list(CAMERA_KEYS),
        gamma=0.5,
        num_demos=None,
        base_policy=FakeBase(),
        scorer=SumFeatureScorer(),
        endpoint_store=MemoryEndpointStore(),
        episodes=(synthetic_episode(tmp_path, num_frames=120),),
        reader_factory=FakeReader,
    )

    assert FakeReader.instances[0].prepared == [(0, 50, 69, 100, 119)]
    output = capsys.readouterr().out
    assert "offline_episode_profile episode=0" in output
    assert "decoder_backend=fake" in output
    assert "transitions=3" in output
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py -q
```

Expected: FAIL because the production reader has no sparse preparation method
and the build loop neither prepares nor profiles an episode.

- [ ] **Step 4: Implement the episode-local cache**

Change `BlockEpisodeReader` to:

```python
class BlockEpisodeReader:
    decoder_backend = "torchcodec_sparse"

    def __init__(self, episode: EpisodeRef, *, frame_loader=None):
        self.episode = episode
        frame = pd.read_parquet(
            episode.parquet_path,
            columns=["action", "observation.state"],
        )
        self.actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        self.states = np.stack(
            frame["observation.state"].to_numpy()).astype(np.float32)[:, :16]
        if frame_loader is None:
            from resfit.rl_finetuning.chunk_residual.teleavatar_batch_source import (
                read_teleavatar_episode_frames,
            )
            frame_loader = read_teleavatar_episode_frames
        self._frame_loader = frame_loader
        self._native_frame_cache = None

    def prepare_native_frames(self, frame_indices) -> None:
        indices = tuple(frame_indices)
        decoded = self._frame_loader(
            self.episode.root,
            self.episode.episode_id,
            list(CAMERA_KEYS),
            list(indices),
        )
        arrays = []
        for camera in CAMERA_KEYS:
            value = decoded[camera]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            arrays.append(np.asarray(value, dtype=np.uint8))
        stacked = np.stack(arrays, axis=1)
        if stacked.shape[0] != len(indices):
            raise ValueError(
                f"decoded frame count {stacked.shape[0]} != {len(indices)}")
        normalized = stacked.astype(np.float32) / 127.5 - 1.0
        self._native_frame_cache = {
            frame_index: normalized[position]
            for position, frame_index in enumerate(indices)
        }

    def native_frames(self, frame_index: int) -> np.ndarray:
        if (
            self._native_frame_cache is None
            or frame_index not in self._native_frame_cache
        ):
            raise RuntimeError(
                f"native frame {frame_index} was not prepared for "
                f"episode {self.episode.episode_id}")
        return self._native_frame_cache[frame_index]
```

Remove `_read_rgb_frame()` from production use, but retain it as
`read_rgb_frame_opencv()` for the explicit equivalence verifier in Task 4.

- [ ] **Step 5: Prepare once and add per-episode timing**

In `build_block_offline_buffer()`, import `time` and structure the episode loop
as follows:

```python
slices = plan_episode_chunks(episode.num_frames)
required_indices = sorted({
    frame_index
    for item in slices
    for frame_index in (item.start, item.end)
})
reader = reader_factory(episode=episode)

decode_started = time.perf_counter()
reader.prepare_native_frames(required_indices)
decode_seconds = time.perf_counter() - decode_started

endpoint_started = time.perf_counter()
# existing endpoint cache load/validation/build block
endpoint_seconds = time.perf_counter() - endpoint_started

replay_started = time.perf_counter()
# existing `for item in slices:` transition construction block
replay_seconds = time.perf_counter() - replay_started

print(
    f"offline_episode_profile episode={episode.episode_id} "
    f"decoder_backend={getattr(reader, 'decoder_backend', 'unknown')} "
    f"sparse_frames={len(required_indices)} "
    f"decode_seconds={decode_seconds:.6f} "
    f"endpoint_seconds={endpoint_seconds:.6f} "
    f"replay_seconds={replay_seconds:.6f} "
    f"endpoint_cache={'hit' if endpoint_record_was_cached else 'miss'} "
    f"transitions={len(slices)}"
)
```

Track `endpoint_record_was_cached` immediately after the first successful
cache load and before any missing record is rebuilt.

- [ ] **Step 6: Run focused and bridge test suites**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py -q
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests -q
```

Expected: both commands PASS. Existing transition schema, terminal, cache-hit,
and reward tests remain green.

- [ ] **Step 7: Commit reader integration**

```bash
git add resfit/rl_finetuning/wm_bridge/block_offline_chunk.py resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py
git commit -m "perf: reuse sparse frames in block replay"
```

---

### Task 4: Add the Real-Data Correctness and Performance Gate

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/verify_block_sparse_decode.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_verify_block_sparse_decode.py`

**Interfaces:**
- Consumes:
  - `catalog_episodes()`;
  - `plan_episode_chunks()`;
  - `read_rgb_frame_opencv()`;
  - `read_teleavatar_episode_frames()`.
- Produces:
  - `verify_episode(episode) -> dict`;
  - CLI exit status 0 only when every sampled pixel is exact and aggregate
    speedup is at least 3.0.

- [ ] **Step 1: Write failing report and gate tests**

Create `test_verify_block_sparse_decode.py`:

```python
import pytest

from resfit.rl_finetuning.wm_bridge.verify_block_sparse_decode import (
    check_report,
)


def test_check_report_accepts_exact_fast_bounded_result():
    report = {
        "pixel_equal": True,
        "legacy_seconds": 12.0,
        "sparse_seconds": 2.0,
        "peak_extra_mib": 120.0,
    }
    assert check_report(report)["speedup"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"pixel_equal": False}, "pixel equality"),
        ({"sparse_seconds": 5.0}, "3.0x"),
        ({"peak_extra_mib": 513.0}, "512 MiB"),
    ],
)
def test_check_report_rejects_failed_gate(change, message):
    report = {
        "pixel_equal": True,
        "legacy_seconds": 12.0,
        "sparse_seconds": 2.0,
        "peak_extra_mib": 120.0,
    }
    report.update(change)
    with pytest.raises(RuntimeError, match=message):
        check_report(report)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_verify_block_sparse_decode.py -q
```

Expected: FAIL because the verification module does not exist.

- [ ] **Step 3: Implement the verification gate**

Create `verify_block_sparse_decode.py` with:

```python
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from resfit.rl_finetuning.chunk_residual.teleavatar_batch_source import (
    read_teleavatar_episode_frames,
)
from resfit.rl_finetuning.wm_bridge.block_offline_chunk import (
    CAMERA_KEYS,
    catalog_episodes,
    plan_episode_chunks,
    read_rgb_frame_opencv,
)


def required_indices(episode):
    return sorted({
        index
        for item in plan_episode_chunks(episode.num_frames)
        for index in (item.start, item.end)
    })


def verify_episode(episode):
    indices = required_indices(episode)
    before_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    legacy_started = time.perf_counter()
    legacy = {
        camera: np.stack([
            read_rgb_frame_opencv(
                episode.video_paths[camera], frame_index)
            for frame_index in indices
        ])
        for camera in CAMERA_KEYS
    }
    legacy_seconds = time.perf_counter() - legacy_started

    sparse_started = time.perf_counter()
    sparse = read_teleavatar_episode_frames(
        episode.root,
        episode.episode_id,
        list(CAMERA_KEYS),
        indices,
    )
    sparse_seconds = time.perf_counter() - sparse_started
    after_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    pixel_equal = all(np.array_equal(
        legacy[camera],
        sparse[camera].detach().cpu().numpy(),
    ) for camera in CAMERA_KEYS)
    return {
        "episode": episode.episode_id,
        "frames": len(indices),
        "pixel_equal": pixel_equal,
        "legacy_seconds": legacy_seconds,
        "sparse_seconds": sparse_seconds,
        "peak_extra_mib": max(0.0, after_mib - before_mib),
    }


def check_report(report):
    if not report["pixel_equal"]:
        raise RuntimeError("pixel equality gate failed")
    speedup = report["legacy_seconds"] / report["sparse_seconds"]
    if speedup < 3.0:
        raise RuntimeError(f"speedup {speedup:.3f} is below 3.0x")
    if report["peak_extra_mib"] >= 512.0:
        raise RuntimeError("peak additional memory reached 512 MiB")
    return {**report, "speedup": speedup}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episodes", default="0,8,28")
    args = parser.parse_args(argv)

    selected_ids = [int(value) for value in args.episodes.split(",")]
    episodes = {
        episode.episode_id: episode
        for episode in catalog_episodes(args.dataset_root)
    }
    results = [
        verify_episode(episodes[episode_id])
        for episode_id in selected_ids
    ]
    aggregate = {
        "pixel_equal": all(item["pixel_equal"] for item in results),
        "legacy_seconds": sum(item["legacy_seconds"] for item in results),
        "sparse_seconds": sum(item["sparse_seconds"] for item in results),
        "peak_extra_mib": max(item["peak_extra_mib"] for item in results),
    }
    checked = check_report(aggregate)
    print(json.dumps(
        {"episodes": results, "aggregate": checked},
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run verifier unit tests**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_verify_block_sparse_decode.py -q
```

Expected: PASS.

- [ ] **Step 5: Run exact-pixel and performance verification on real data**

Run:

```bash
bash -c 'source resfit/lerobot/shell/torchcodec_env.sh; /mnt/mnt/data/envs/residual/bin/python -m resfit.rl_finetuning.wm_bridge.verify_block_sparse_decode --dataset-root /mnt/mnt/data/domains_rise/block/block_success --episodes 0,8,28'
```

Expected:

- process exits 0;
- aggregate `pixel_equal` is `true`;
- aggregate `speedup` is at least `3.0`;
- aggregate `peak_extra_mib` is below `512.0`.

If this command exits nonzero, stop the implementation before production
restart. Preserve the endpoint cache and return to the approved design's
OpenCV sequential fallback decision.

- [ ] **Step 6: Run the complete affected test surface**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_teleavatar_batch_source.py resfit/rl_finetuning/wm_bridge/tests -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit the verification gate**

```bash
git add resfit/rl_finetuning/wm_bridge/verify_block_sparse_decode.py resfit/rl_finetuning/wm_bridge/tests/test_verify_block_sparse_decode.py
git commit -m "test: gate sparse block video decoding"
```

---

### Task 5: Restart Seed 0 and Verify Cache Reuse

**Files:**
- Run: `launch_block_imagination.sh`
- Inspect: `outputs_imagination/block_shore_mixed50_seed0.log`
- Inspect: `outputs_imagination/block_shore_mixed50_seed0/bridge_run_config.json`

**Interfaces:**
- Consumes: verified implementation, unchanged endpoint cache, live services on ports 9000/8001/8002.
- Produces: restarted GPU-7 seed-0 experiment with faster preprocessing and complete replay-build statistics.

- [ ] **Step 1: Reconfirm services and GPU 7 availability**

Run:

```bash
ps -p 1698375,1698376,1698377 -o pid,etime,state,%cpu,rss,args
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
ss -ltnp
```

Expected:

- service PIDs remain alive;
- ports 9000, 8001, and 8002 are listening;
- no seed-0 trainer occupies GPU 7.

- [ ] **Step 2: Restart in the existing tmux session**

Run:

```bash
tmux send-keys -t block_mixed_s0 'cd /mnt/mnt/data/resfit && bash launch_block_imagination.sh 7 0' Enter
```

Expected: one new wrapper, trainer, and `tee` process start; no duplicate seed-0
trainer exists.

- [ ] **Step 3: Verify partial endpoint-cache reuse**

Run after the startup banner and first cached episode complete:

```bash
sed -n '1,80p' outputs_imagination/block_shore_mixed50_seed0.log
rg -n 'offline_source|endpoint_cache|offline_episode_profile|offline_endpoint_progress|Traceback|Exception|CUDA out of memory|nan' outputs_imagination/block_shore_mixed50_seed0.log
```

Expected:

- startup reports `endpoint_cache=partial` and `replay_cache=miss`;
- early cached episodes log `endpoint_cache=hit`;
- those episodes contain no `offline_endpoint_progress` lines;
- no traceback, OOM, or NaN appears.

- [ ] **Step 4: Verify cached files were not rewritten**

Run:

```bash
sha256sum cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae/episode_000000.npz
sha256sum cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae/episode_000001.npz
sha256sum cache/block_mixed_replay/endpoints/7204cfb95a84594ec8a2a6b6be5ae47295da5c8d423616397003a5a769054eae/episode_000002.npz
```

Expected: all three hashes match Task 1 exactly.

- [ ] **Step 5: Measure ten-minute production throughput**

Record `rg -c '^offline_build_progress'` at the start and after ten minutes,
while checking profile lines and process health:

```bash
rg -c '^offline_build_progress' outputs_imagination/block_shore_mixed50_seed0.log
rg -n '^offline_episode_profile' outputs_imagination/block_shore_mixed50_seed0.log
ps -eo pid,ppid,etime,state,%cpu,rss,args
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Expected:

- at least 1,000 transitions are built in ten minutes;
- resident-memory growth remains below 512 MiB above the post-start baseline;
- profile lines show one sparse decode per episode;
- the trainer and all three services remain alive.

- [ ] **Step 6: Verify preprocessing completion**

Monitor the log until:

```text
offline_build_stats episodes=295 ... transitions=6989 ...
```

Expected:

- `transitions=6989`;
- endpoint hits plus misses equal 295 minus skipped short episodes;
- `bridge_run_config.json` and `bridge_meta.json` contain complete build stats;
- RL updates begin and GPU 7 utilization rises above preprocessing-idle levels.

- [ ] **Step 7: Run completion verification**

Run:

```bash
git status --short
git log -5 --oneline
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_teleavatar_batch_source.py resfit/rl_finetuning/wm_bridge/tests -q
```

Expected:

- only pre-existing unrelated worktree changes remain;
- all affected tests pass;
- the latest implementation commits are present;
- the production run is alive beyond replay preprocessing.
