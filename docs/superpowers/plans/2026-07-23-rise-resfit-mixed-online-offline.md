# RISE × ResFiT Mixed Online/Offline Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bridge-owned block-success offline replay source so every production TD3 update uses exactly 128 imagined-online and 128 real-offline 50-step transitions.

**Architecture:** Keep the original RISE repository, the generic ResFiT trainer, and the generic ResFiT offline builders unchanged. Add pure chunk/index and two-level cache modules under `wm_bridge`, inject bridge-specific `count_offline_transitions` and `build_offline_buffer` functions, then let the existing trainer construct, cache, sample, and concatenate its native online/offline replay buffers.

**Tech Stack:** Python 3.10, PyTorch, TorchRL/TensorDict, NumPy, pandas/Parquet, OpenCV, pytest, Bash, existing kai0 websocket client and `Kai0HiqlScorer`.

## Global Constraints

- Offline replay source is exactly `/mnt/mnt/data/domains_rise/block/block_success`; `block_fail` remains imagination-reset-only.
- `offline_fraction=0.5`, `batch_size=256`, `chunk_length=50`, `n_step=1`, `gamma=imagination_gamma=0.995`.
- One offline transition is `(frame[t], actions[t:t+50], frame[t+50])`.
- Regular starts use stride 50; add one terminal-aligned start `T-51` when needed; skip episodes shorter than 51 frames.
- Store expert scaled action in replay; store frozen kai0 scaled chunks in current/next `observation.base_action`.
- Reward is exactly `gamma * Phi(s_next) - Phi(s)` for regular and terminal chunks; no success bonus.
- Keep kai0, HIQL, and the world model frozen.
- Production mixed replay requires `base_policy_type=pi05`, `base_action_mode=replan`, `actor=raw`, no dummy scorer, no relabeling, no stage/subgoal conditioning, and no online HIQL finetuning.
- Use `offline_base_mode=gt` only as the documented trainer-compatibility marker; bridge metadata is authoritative and must say `actual_base_mode=kai0_chunk`.
- Do not modify `/mnt/mnt/data/data2/RL/RISE`, `chunk_residual/train_chunk_residual.py`, or `chunk_residual/offline_stage_replay.py`.
- Preserve unrelated dirty-worktree and staged changes. Every commit lists exact paths.
- Follow TDD: add a focused failing test, observe the expected failure, implement the minimum behavior, rerun the focused test, then run the affected bridge suite.

---

## File Structure

- Create `resfit/rl_finetuning/wm_bridge/block_offline_chunk.py`
  - episode catalog and chunk-index rules;
  - Parquet/action/state and indexed-video access;
  - unique endpoint collection through kai0;
  - conversion to online-compatible TensorDict transitions;
  - bridge-compatible count/build entry points.
- Create `resfit/rl_finetuning/wm_bridge/block_offline_cache.py`
  - dataset manifests and stable fingerprints;
  - per-episode atomic endpoint cache;
  - replay generation-directory resolution;
  - bridge metadata and cache-hit validation.
- Create `resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py`
  - index, source, endpoint reuse, PBRS, terminal, TensorDict schema tests.
- Create `resfit/rl_finetuning/wm_bridge/tests/test_block_offline_cache.py`
  - fingerprint, generation, atomic-write, corruption, and invalidation tests.
- Modify `resfit/rl_finetuning/wm_bridge/builder.py`
  - parse offline bridge arguments;
  - construct scorer/cache runtime once;
  - expose fake offline count/build closures sharing the existing base/scorer.
- Modify `resfit/rl_finetuning/wm_bridge/launch_imagination.py`
  - register two additional injectable symbols;
  - translate bridge offline arguments into trainer-compatible passthrough;
  - write `bridge_run_config.json`.
- Modify `resfit/rl_finetuning/wm_bridge/contract.py`
  - validate the generic trainer hook points by source;
  - enforce mixed-replay arguments only when offline chunk mode is enabled.
- Modify `resfit/rl_finetuning/wm_bridge/tests/test_contract.py`
  - mixed-mode pass/fail matrix and pure-online regression.
- Modify `resfit/rl_finetuning/wm_bridge/tests/test_launcher.py`
  - five-symbol injection, parent-package registration, and offline translation.
- Modify `launch_block_imagination.sh`
  - approved production parameters, cache root, and distinct output/W&B names.
- Create `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`
  - static production-launch flag and exact 128/128 batch assertions.

---

### Task 1: Episode Catalog and Exact 50-step Chunk Indexing

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/block_offline_chunk.py`
- Create: `resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py`

**Interfaces:**
- Produces: `ChunkSlice(start: int, end: int, terminal: bool)`.
- Produces: `EpisodeRef(root: str, episode_id: int, parquet_path: str, video_paths: dict[str, str], num_frames: int)`.
- Produces: `plan_episode_chunks(num_frames: int, chunk_length: int = 50) -> tuple[ChunkSlice, ...]`.
- Produces: `catalog_episodes(dataset_root: str, num_demos: int | None = None) -> tuple[EpisodeRef, ...]`.
- Produces: `count_block_chunk_transitions(dataset_root: str, num_demos: int | None = None) -> int`.

- [ ] **Step 1: Write the failing chunk-boundary tests**

Add:

```python
import pytest

from resfit.rl_finetuning.wm_bridge.block_offline_chunk import (
    plan_episode_chunks,
)


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    [
        (50, ()),
        (51, ((0, 50, True),)),
        (101, ((0, 50, False), (50, 100, True))),
        (120, ((0, 50, False), (50, 100, False), (69, 119, True))),
    ],
)
def test_plan_episode_chunks(num_frames, expected):
    got = tuple((x.start, x.end, x.terminal)
                for x in plan_episode_chunks(num_frames))
    assert got == expected


def test_chunk_plan_has_no_duplicate_aligned_terminal():
    got = plan_episode_chunks(151)
    assert [x.start for x in got] == [0, 50, 100]
    assert sum(x.terminal for x in got) == 1
```

- [ ] **Step 2: Run the focused tests and observe the missing-module failure**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py -v
```

Expected: FAIL during collection with
`ModuleNotFoundError: ...wm_bridge.block_offline_chunk`.

- [ ] **Step 3: Implement immutable chunk and episode records plus the index rule**

Add:

```python
from __future__ import annotations

from dataclasses import dataclass
import glob
import os

import pandas as pd

from resfit.rl_finetuning.wm_bridge.wm_driver import CAMERA_KEYS, CHUNK_LENGTH

_EPISODES_PER_CHUNK = 1000


@dataclass(frozen=True)
class ChunkSlice:
    start: int
    end: int
    terminal: bool


@dataclass(frozen=True)
class EpisodeRef:
    root: str
    episode_id: int
    parquet_path: str
    video_paths: dict[str, str]
    num_frames: int


def plan_episode_chunks(num_frames: int,
                        chunk_length: int = CHUNK_LENGTH) -> tuple[ChunkSlice, ...]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if num_frames < chunk_length + 1:
        return ()
    terminal_start = num_frames - chunk_length - 1
    starts = list(range(0, terminal_start + 1, chunk_length))
    if starts[-1] != terminal_start:
        starts.append(terminal_start)
    return tuple(
        ChunkSlice(start=t, end=t + chunk_length,
                   terminal=(t == terminal_start))
        for t in starts
    )


def _episode_paths(root: str, episode_id: int):
    chunk_id = episode_id // _EPISODES_PER_CHUNK
    parquet = os.path.join(
        root, f"data/chunk-{chunk_id:03d}/episode_{episode_id:06d}.parquet")
    videos = {
        key: os.path.join(
            root,
            f"videos/chunk-{chunk_id:03d}/{key}/episode_{episode_id:06d}.mp4")
        for key in CAMERA_KEYS
    }
    return parquet, videos


def catalog_episodes(dataset_root: str,
                     num_demos: int | None = None) -> tuple[EpisodeRef, ...]:
    paths = sorted(glob.glob(os.path.join(
        dataset_root, "data", "chunk-*", "episode_*.parquet")))
    if num_demos is not None:
        paths = paths[:num_demos]
    result = []
    for parquet_path in paths:
        episode_id = int(
            os.path.basename(parquet_path).removeprefix("episode_").removesuffix(".parquet"))
        expected_parquet, videos = _episode_paths(dataset_root, episode_id)
        if os.path.abspath(parquet_path) != os.path.abspath(expected_parquet):
            raise ValueError(f"unexpected parquet layout: {parquet_path}")
        num_frames = len(pd.read_parquet(parquet_path, columns=["frame_index"]))
        result.append(EpisodeRef(
            dataset_root, episode_id, parquet_path, videos, num_frames))
    if not result:
        raise ValueError(f"no episodes found under {dataset_root}")
    return tuple(result)


def count_block_chunk_transitions(dataset_root: str,
                                  num_demos: int | None = None) -> int:
    return sum(len(plan_episode_chunks(ep.num_frames))
               for ep in catalog_episodes(dataset_root, num_demos))
```

- [ ] **Step 4: Add catalog tests using a temporary LeRobot directory**

Create two minimal parquet files with `frame_index` and assert:

```python
def test_catalog_and_count_use_sorted_episode_ids(tmp_path):
    _write_episode_parquet(tmp_path, episode_id=2, num_frames=120)
    _write_episode_parquet(tmp_path, episode_id=0, num_frames=51)
    episodes = catalog_episodes(str(tmp_path))
    assert [x.episode_id for x in episodes] == [0, 2]
    assert count_block_chunk_transitions(str(tmp_path)) == 4


def test_num_demos_limits_sorted_prefix(tmp_path):
    _write_episode_parquet(tmp_path, episode_id=0, num_frames=51)
    _write_episode_parquet(tmp_path, episode_id=1, num_frames=101)
    assert count_block_chunk_transitions(str(tmp_path), num_demos=1) == 1
```

The helper writes:

```python
def _write_episode_parquet(root, episode_id, num_frames):
    chunk = episode_id // 1000
    path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_id:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame_index": range(num_frames)}).to_parquet(path)
```

- [ ] **Step 5: Run the task tests**

Run the focused test file. Expected: all tests PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  resfit/rl_finetuning/wm_bridge/block_offline_chunk.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py
git commit -m "feat: add block offline chunk catalog"
```

---

### Task 2: Two-level Fingerprints and Atomic Endpoint Cache

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/block_offline_cache.py`
- Create: `resfit/rl_finetuning/wm_bridge/tests/test_block_offline_cache.py`

**Interfaces:**
- Consumes: `EpisodeRef` from Task 1.
- Produces: `EndpointRecord(frame_indices, base_actions, prefix_features, proprio)`.
- Produces: `dataset_manifest(episodes) -> dict`.
- Produces: `endpoint_fingerprint(...) -> str` independent of `num_demos`.
- Produces: `replay_fingerprint(...) -> str` including selected demo count and value/scaler identity.
- Produces: `EndpointStore(root: str, fingerprint: str, force_rebuild: bool = False)`.
- Produces: `resolve_replay_generation(cache_root, replay_fp, force_rebuild) -> str`.

- [ ] **Step 1: Write failing fingerprint-level tests**

Add these concrete test helpers:

```python
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
```

```python
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
```

- [ ] **Step 2: Run and verify the missing-module failure**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_cache.py -v
```

Expected: collection FAIL because `block_offline_cache` does not exist.

- [ ] **Step 3: Implement canonical hashing and manifest construction**

Use canonical JSON and file SHA helpers:

```python
ENDPOINT_SCHEMA = 1
REPLAY_SCHEMA = 1


def _stable_sha(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: str, hash_content: bool) -> dict:
    stat = os.stat(path)
    identity = {
        "path": os.path.realpath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        identity["sha256"] = file_sha256(path)
    return identity


def dataset_manifest(episodes: Sequence[EpisodeRef]) -> dict:
    return {
        "root": os.path.realpath(episodes[0].root),
        "episodes": [
            {
                "id": ep.episode_id,
                "frames": ep.num_frames,
                "parquet": _file_identity(ep.parquet_path, hash_content=True),
                "videos": {
                    key: _file_identity(path, hash_content=False)
                    for key, path in sorted(ep.video_paths.items())
                },
            }
            for ep in episodes
        ],
    }
```

`_file_identity` returns real path, size, `mtime_ns`, and a SHA-256 only for
metadata/parquet files. Missing videos raise `FileNotFoundError`.

- [ ] **Step 4: Implement endpoint/replay fingerprint functions**

```python
def endpoint_fingerprint(*, manifest, pi0_serve_ckpt_id, prompt,
                         camera_keys, chunk_length=50, stride=50):
    return _stable_sha({
        "schema": ENDPOINT_SCHEMA,
        "manifest": manifest,
        "chunk_length": chunk_length,
        "stride": stride,
        "terminal_rule": "append_T_minus_L_minus_1",
        "camera_keys": list(camera_keys),
        "prompt": prompt,
        "action_dim": 16,
        "pi0_serve_ckpt_id": pi0_serve_ckpt_id,
    })


def replay_fingerprint(*, endpoint_fp, value_sha256, gamma, num_demos,
                       stats_sha256, action_scale, min_range_per_dim,
                       image_size, image_keys, n_step):
    return _stable_sha({
        "schema": REPLAY_SCHEMA,
        "endpoint_fp": endpoint_fp,
        "value_sha256": value_sha256,
        "gamma": float(gamma),
        "num_demos": num_demos,
        "stats_sha256": stats_sha256,
        "action_scale": float(action_scale),
        "min_range_per_dim": float(min_range_per_dim),
        "image_size": int(image_size),
        "image_keys": sorted(image_keys),
        "n_step": int(n_step),
    })
```

- [ ] **Step 5: Write failing atomic-cache and generation tests**

```python
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
```

- [ ] **Step 6: Implement atomic per-episode NPZ and replay generations**

Implement the record and store directly:

```python
@dataclass(frozen=True)
class EndpointRecord:
    frame_indices: np.ndarray
    base_actions: np.ndarray
    prefix_features: np.ndarray
    proprio: np.ndarray


def _validate_record(record):
    n = len(record.frame_indices)
    valid = (
        record.frame_indices.ndim == 1
        and record.base_actions.shape == (n, 50, 16)
        and record.prefix_features.ndim == 2
        and record.prefix_features.shape[0] == n
        and record.proprio.shape == (n, 16)
        and all(np.all(np.isfinite(value)) for value in (
            record.base_actions, record.prefix_features, record.proprio))
    )
    if not valid:
        raise ValueError("invalid endpoint cache record")


class EndpointStore:
    def __init__(self, root, fingerprint, force_rebuild=False):
        self.directory = Path(root) / "endpoints" / fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        self.force_rebuild = force_rebuild

    def _path(self, episode_id):
        return self.directory / f"episode_{episode_id:06d}.npz"

    def load_episode(self, episode_id):
        path = self._path(episode_id)
        if self.force_rebuild or not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            record = EndpointRecord(
                data["frame_indices"], data["base_actions"],
                data["prefix_features"], data["proprio"])
        _validate_record(record)
        return record

    def save_episode(self, episode_id, record):
        _validate_record(record)
        final_path = self._path(episode_id)
        temp_path = final_path.with_suffix(".npz.tmp")
        with temp_path.open("wb") as stream:
            np.savez_compressed(
                stream, frame_indices=record.frame_indices,
                base_actions=record.base_actions,
                prefix_features=record.prefix_features,
                proprio=record.proprio)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, final_path)


def resolve_replay_generation(cache_root, replay_fp, force_rebuild):
    parent = Path(cache_root) / "replay" / replay_fp
    parent.mkdir(parents=True, exist_ok=True)
    complete = sorted(
        path for path in parent.glob("gen-*")
        if (path / "buffer_meta.json").is_file()
        and (path / "bridge_meta.json").is_file())
    if complete and not force_rebuild:
        return str(complete[-1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    created = parent / f"gen-{stamp}-{uuid.uuid4().hex[:8]}"
    created.mkdir()
    return str(created)
```

This produces `replay/<replay-fp>/gen-<UTC timestamp>-<8 hex>/`. Normal
resolution reuses the newest complete generation; force rebuild creates a new
one without deleting older caches.

- [ ] **Step 7: Run cache tests and commit**

Run the cache test file, then:

```bash
git add \
  resfit/rl_finetuning/wm_bridge/block_offline_cache.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_cache.py
git commit -m "feat: add block offline cache fingerprints"
```

---

### Task 3: Endpoint Collection, PBRS, and TensorDict Replay Construction

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/block_offline_chunk.py`
- Modify: `resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py`

**Interfaces:**
- Consumes: `EndpointStore`, `EndpointRecord` from Task 2.
- Consumes: existing `Kai0ImaginationBase.query(raw_obs)`.
- Consumes: existing `Kai0HiqlScorer.phi(prefix_feature, raw_proprio)`.
- Produces: `collect_episode_endpoints(...) -> EndpointRecord`.
- Produces: `build_block_offline_buffer(offline_rb, dataset_root, *, action_scaler, state_standardizer, image_keys, gamma, num_demos, base_policy, scorer, endpoint_store, episodes=None, reader_factory=BlockEpisodeReader, **_ignored) -> BuildStats`.
- The public build function accepts extra generic trainer keyword arguments through `**_ignored` but never interprets generic stage/GT semantics.

- [ ] **Step 1: Write failing endpoint-reuse and shape tests**

Use these concrete fakes for endpoint and replay tests:

```python
def synthetic_episode(root, num_frames=51, episode_id=0):
    return EpisodeRef(
        root=str(root), episode_id=episode_id,
        parquet_path=str(root / f"episode_{episode_id:06d}.parquet"),
        video_paths={key: str(root / f"{episode_id}-{key}.mp4")
                     for key in CAMERA_KEYS},
        num_frames=num_frames)


class FakeReader:
    def __init__(self, episode=None, num_frames=None):
        n = num_frames if num_frames is not None else episode.num_frames
        self.actions = np.stack([
            np.full(16, i, dtype=np.float32) for i in range(n)])
        self.states = np.stack([
            np.full(16, i, dtype=np.float32) for i in range(n)])

    def native_frames(self, frame_index):
        return np.zeros((len(CAMERA_KEYS), 3, 192, 256), dtype=np.float32)


class FakeBase:
    def __init__(self):
        self.tokens = []

    def query(self, obs):
        token = obs["_wm_window_token"]
        self.tokens.append(token)
        value = float(token.rsplit(":", 1)[-1])
        return (
            np.full((50, 16), value, np.float32),
            np.array([value, value + 1], np.float32),
        )


class MemoryEndpointStore:
    def __init__(self):
        self.records = {}

    def load_episode(self, episode_id):
        return self.records.get(episode_id)

    def save_episode(self, episode_id, record):
        self.records[episode_id] = record


class IdentityActionScaler:
    def scale(self, value):
        return value.float()


class IdentityStateStandardizer:
    def standardize(self, value):
        return value.float()


class SumFeatureScorer:
    def phi(self, feature, proprio):
        return float(np.asarray(feature, dtype=np.float32).sum())

    @staticmethod
    def phi_for_frame(frame_index):
        return 2.0 * frame_index + 1.0


class ListReplay:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)
```

```python
def test_collect_episode_endpoints_queries_each_unique_frame_once(tmp_path):
    slices = plan_episode_chunks(120)
    reader = FakeReader(num_frames=120)
    base = FakeBase()
    record = collect_episode_endpoints(
        reader, episode_id=3, slices=slices, base_policy=base)
    assert record.frame_indices.tolist() == [0, 50, 69, 100, 119]
    assert len(base.tokens) == 5
    assert record.base_actions.shape == (5, 50, 16)
    assert record.proprio.shape == (5, 16)
```

- [ ] **Step 2: Run the focused test and verify `collect_episode_endpoints` is missing**

Expected: import failure naming the missing function.

- [ ] **Step 3: Implement indexed episode reading and endpoint collection**

Add `BlockEpisodeReader`:

```python
class BlockEpisodeReader:
    def __init__(self, episode: EpisodeRef):
        self.episode = episode
        frame = pd.read_parquet(
            episode.parquet_path,
            columns=["action", "observation.state"])
        self.actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        self.states = np.stack(
            frame["observation.state"].to_numpy()).astype(np.float32)[:, :16]

    def native_frames(self, frame_index: int) -> np.ndarray:
        views = [
            _read_rgb_frame(self.episode.video_paths[key], frame_index)
            for key in CAMERA_KEYS
        ]
        return np.stack(views).astype(np.float32) / 127.5 - 1.0
```

Implement indexed RGB reading explicitly:

```python
def _read_rgb_frame(path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"failed to read {path} at frame {frame_index}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb, (2, 0, 1)).copy()
```

Collect sorted unique endpoints from every slice start/end:

```python
def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} expected finite shape {shape}, got {array.shape}")
    return array


def _finite_vector(value, name):
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def collect_episode_endpoints(reader, *, episode_id, slices, base_policy):
    indices = sorted({i for item in slices for i in (item.start, item.end)})
    bases, features, states = [], [], []
    for frame_index in indices:
        state = reader.states[frame_index]
        raw_obs = {
            "_wm_native_frames": reader.native_frames(frame_index),
            "_wm_window_token": f"offline:{episode_id}:{frame_index}",
            "observation.state": torch.from_numpy(state).unsqueeze(0),
        }
        base, feature = base_policy.query(raw_obs)
        bases.append(_finite_array(base, (50, 16), "base_action"))
        features.append(_finite_vector(feature, "prefix_feature"))
        states.append(_finite_array(state, (16,), "proprio"))
    return EndpointRecord(
        frame_indices=np.asarray(indices, dtype=np.int64),
        base_actions=np.stack(bases),
        prefix_features=np.stack(features),
        proprio=np.stack(states),
    )
```

- [ ] **Step 4: Write failing transition-schema and exact-PBRS tests**

Construct an in-memory replay stub whose `add(td)` appends entries. Use simple
identity scaler/standardizer fakes and assert:

```python
def test_build_transition_matches_online_schema_and_exact_pbrs(tmp_path):
    rb = ListReplay()
    stats = build_block_offline_buffer(
        rb, str(tmp_path),
        action_scaler=IdentityActionScaler(),
        state_standardizer=IdentityStateStandardizer(),
        image_keys=list(CAMERA_KEYS),
        gamma=0.5, num_demos=None,
        base_policy=FakeBase(), scorer=SumFeatureScorer(),
        endpoint_store=MemoryEndpointStore(),
        episodes=(synthetic_episode(tmp_path, num_frames=51),),
        reader_factory=FakeReader,
    )
    item = rb.items[0][0]
    assert tuple(item["action"].shape) == (800,)
    assert tuple(item["obs"]["observation.base_action"].shape) == (800,)
    assert tuple(item["next"]["obs"]["observation.base_action"].shape) == (800,)
    assert item["next"]["done"].item() is True
    assert item["next"]["reward"].item() == pytest.approx(
        0.5 * SumFeatureScorer.phi_for_frame(50)
        - SumFeatureScorer.phi_for_frame(0))
    assert stats.transitions == 1
```

Add these exact schema assertions to the same test:

```python
for key in CAMERA_KEYS:
    assert item["obs"][key].dtype == torch.uint8
    assert tuple(item["obs"][key].shape) == (3, 84, 84)
    assert item["next"]["obs"][key].dtype == torch.uint8
assert tuple(item["obs"]["observation.state"].shape) == (16,)
assert tuple(item["next"]["obs"]["observation.state"].shape) == (16,)
assert item["obs"]["observation.stage_id"].item() == 0.0
assert item["next"]["obs"]["observation.stage_id"].item() == 0.0
assert item["max_stage"].item() == 0.0
assert item["_priority"].item() == 10.0
```

- [ ] **Step 5: Implement image conversion, scaling, scorer lookup, and replay add**

Reuse `frames_to_obs_images`:

```python
def _agent_images(native_frames):
    window = torch.from_numpy(native_frames[:, :, None])
    images = frames_to_obs_images(window, t_index=0)
    return {key: (value[0].mul(255).round().to(torch.uint8))
            for key, value in images.items()}
```

For each chunk:

```python
def _endpoint_at(record, frame_index):
    positions = np.flatnonzero(record.frame_indices == frame_index)
    if len(positions) != 1:
        raise ValueError(f"endpoint {frame_index} missing or duplicated")
    pos = int(positions[0])
    return (record.base_actions[pos], record.prefix_features[pos],
            record.proprio[pos])


def _replay_obs(native_frames, raw_state, scaled_base, state_standardizer):
    obs = _agent_images(native_frames)
    obs["observation.state"] = state_standardizer.standardize(
        torch.from_numpy(raw_state)).cpu()
    obs["observation.base_action"] = scaled_base
    obs["observation.stage_id"] = torch.tensor(0.0)
    return obs


base, feature, raw_state = _endpoint_at(endpoint_record, item.start)
next_base, next_feature, next_raw_state = _endpoint_at(
    endpoint_record, item.end)
expert = torch.from_numpy(reader.actions[item.start:item.end])
expert_scaled = action_scaler.scale(expert).reshape(-1).cpu()
base_scaled = action_scaler.scale(
    torch.from_numpy(base)).reshape(-1).cpu()
next_base_scaled = action_scaler.scale(
    torch.from_numpy(next_base)).reshape(-1).cpu()
current_obs = _replay_obs(
    reader.native_frames(item.start), raw_state, base_scaled,
    state_standardizer)
next_obs = _replay_obs(
    reader.native_frames(item.end), next_raw_state, next_base_scaled,
    state_standardizer)
reward = gamma * scorer.phi(next_feature, next_raw_state) \
         - scorer.phi(feature, raw_state)
```

Build a single-item TensorDict matching `add_chunk_transition`:

```python
transition = TensorDict({
    "obs": TensorDict(current_obs, batch_size=[]),
    "next": TensorDict({
        "obs": TensorDict(next_obs, batch_size=[]),
        "done": torch.tensor(item.terminal, dtype=torch.bool),
        "reward": torch.tensor(reward, dtype=torch.float32),
    }, batch_size=[]),
    "action": expert_scaled,
    "max_stage": torch.tensor(0.0),
    "_priority": torch.tensor(10.0),
}, batch_size=[]).unsqueeze(0)
offline_rb.add(transition)
```

Before adding, reject any non-finite reward, state, action, base action, or
feature. Cache each completed episode endpoint record atomically. Return:

```python
@dataclass(frozen=True)
class BuildStats:
    episodes: int
    skipped_short_episodes: int
    transitions: int
    endpoint_hits: int
    endpoint_misses: int
    reward_mean: float
    reward_std: float
    potential_delta_mean: float
    potential_delta_std: float
    residual_norm_mean: float
    expert_saturation_fraction: float
    base_saturation_fraction: float
```

For each transition append `phi_next - phi_current`, reward, residual norm,
and the boolean masks `abs(expert_scaled) >= 0.999` and
`abs(base_scaled) >= 0.999`. Compute the scalar fields with `np.mean` and
`np.std`; use `0.0` for every aggregate when no transition was added.

- [ ] **Step 6: Add tests for cache resume, terminal overlap, and no bonus**

Add the reusable runner and concrete tests:

```python
def _run_fake_build(tmp_path, num_frames, store, base):
    rb = ListReplay()
    build_block_offline_buffer(
        rb, str(tmp_path),
        action_scaler=IdentityActionScaler(),
        state_standardizer=IdentityStateStandardizer(),
        image_keys=list(CAMERA_KEYS), gamma=0.5, num_demos=None,
        base_policy=base, scorer=SumFeatureScorer(),
        endpoint_store=store,
        episodes=(synthetic_episode(tmp_path, num_frames=num_frames),),
        reader_factory=FakeReader)
    return rb


def test_second_build_reuses_endpoint_cache_without_base_queries(tmp_path):
    store = MemoryEndpointStore()
    _run_fake_build(tmp_path, 120, store, FakeBase())
    second_base = FakeBase()
    _run_fake_build(tmp_path, 120, store, second_base)
    assert second_base.tokens == []


def test_unaligned_terminal_order_done_and_residual_target(tmp_path):
    rb = _run_fake_build(
        tmp_path, 120, MemoryEndpointStore(), FakeBase())
    starts = [int(batch[0]["action"][0].item()) for batch in rb.items]
    dones = [bool(batch[0]["next"]["done"].item()) for batch in rb.items]
    assert starts == [0, 50, 69]
    assert dones == [False, False, True]
    terminal = rb.items[-1][0]
    residual = (terminal["action"]
                - terminal["obs"]["observation.base_action"])
    expected = torch.arange(50, dtype=torch.float32).repeat_interleave(16)
    torch.testing.assert_close(residual, expected)
```

The exact-PBRS test from Step 4 proves terminal chunks receive no additional
success bonus; this step proves terminal alignment and cache reuse.

- [ ] **Step 7: Run Task 1–3 tests and commit**

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_cache.py -v
git add \
  resfit/rl_finetuning/wm_bridge/block_offline_chunk.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_chunk.py
git commit -m "feat: build kai0 block offline replay"
```

---

### Task 4: Bridge Arguments, Fingerprinted Cache Resolution, and Contracts

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/builder.py:20-117`
- Modify: `resfit/rl_finetuning/wm_bridge/contract.py:1-145`
- Modify: `resfit/rl_finetuning/wm_bridge/tests/test_contract.py`
- Create: `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`

**Interfaces:**
- Produces bridge args:
  `offline_chunk_dataset: str | None`,
  `offline_chunk_cache_root: str | None`,
  `offline_rebuild: bool`.
- Produces: `prepare_offline_runtime(bridge_args, passthrough) -> (OfflineRuntime | None, list[str])`.
- `OfflineRuntime` contains episode catalog, endpoint/replay fingerprints,
  endpoint store, selected replay generation, and `bridge_meta`.
- Produces: `contract.check_mixed_replay_args(trainer_args, offline_chunk_dataset)`.
- Produces: `contract.check_mixed_scorer(scorer, enabled: bool)`.
- Changes: `build_imagination_factories(bridge_args, offline_runtime=None)`.

- [ ] **Step 1: Write failing parser and contract tests**

Add tests proving pure online still passes and mixed mode rejects each bad
value:

```python
def _mixed_args(**overrides):
    values = {
        "offline_fraction": 0.5,
        "batch_size": 256,
        "base_policy_type": "pi05",
        "base_action_mode": "replan",
        "chunk_length": 50,
        "n_step": 1,
        "actor": "raw",
        "relabel": False,
        "stage_conditioned": False,
        "subgoal_conditioned": False,
        "online_finetune_value": False,
        "online_finetune_high_actor": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_pure_online_does_not_require_mixed_flags():
    contract.check_mixed_replay_args(
        _mixed_args(offline_fraction=0.0),
        offline_chunk_dataset=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("offline_fraction", 0.4),
        ("batch_size", 255),
        ("base_policy_type", "act"),
        ("actor", "flow"),
        ("relabel", True),
        ("online_finetune_value", True),
    ],
)
def test_mixed_contract_rejects_invalid_configuration(field, value):
    args = _mixed_args(**{field: value})
    with pytest.raises(ContractError, match=field):
        contract.check_mixed_replay_args(
            args, offline_chunk_dataset="/data/block_success")


def test_mixed_mode_rejects_dummy_scorer_even_with_debug_flag():
    with pytest.raises(ContractError, match="DummyScorer"):
        contract.check_mixed_scorer(DummyScorer(), enabled=True)


def test_pure_online_keeps_existing_dummy_scorer_policy():
    contract.check_mixed_scorer(DummyScorer(), enabled=False)
```

Parser test:

```python
def test_parse_bridge_offline_args_are_removed_from_passthrough():
    bridge, rest = parse_bridge_args([
        "--value_ckpt", "/tmp/value.pt",
        "--offline_chunk_dataset", "/data/block_success",
        "--offline_chunk_cache_root", "/cache/block",
        "--offline_rebuild",
        "--batch_size", "256",
    ])
    assert bridge.offline_chunk_dataset == "/data/block_success"
    assert bridge.offline_rebuild is True
    assert "--offline_chunk_dataset" not in rest
    assert rest == ["--batch_size", "256"]
```

- [ ] **Step 2: Run focused contract/parser tests and observe failures**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py -v
```

Expected: FAIL for missing args/functions.

- [ ] **Step 3: Add bridge arguments and exact mixed configuration parser**

Extend `parse_bridge_args`:

```python
p.add_argument("--offline_chunk_dataset", default=None)
p.add_argument("--offline_chunk_cache_root", default=None)
p.add_argument("--offline_rebuild", action="store_true")
```

Add a small parser in `contract.py` for trainer values that affect mixed
semantics:

```python
def parse_mixed_passthrough(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--offline_fraction", type=float, default=0.0)
    p.add_argument("--offline_num_demos", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--base_policy_type", default="act")
    p.add_argument("--actor", default="raw")
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--relabel", action="store_true")
    p.add_argument("--stage_conditioned", action="store_true")
    p.add_argument("--online_finetune_value", action="store_true")
    p.add_argument("--online_finetune_high_actor", action="store_true")
    p.add_argument("--output_dir", default="outputs_chunk")
    args, _ = p.parse_known_args(argv)
    return args
```

`check_mixed_replay_args` returns immediately when
`offline_chunk_dataset is None`; otherwise enforce every global constraint,
and require
`os.path.basename(os.path.realpath(dataset)) == "block_success"`.
`check_mixed_scorer` rejects `DummyScorer` whenever `enabled=True`, even when
the general imagination debug flag would otherwise allow it.

- [ ] **Step 4: Implement runtime fingerprint/cache preparation**

In `builder.py`, add:

```python
@dataclass(frozen=True)
class OfflineRuntime:
    dataset_root: str
    all_episodes: tuple[EpisodeRef, ...]
    num_demos: int | None
    endpoint_store: EndpointStore
    replay_cache_dir: str
    bridge_meta: dict

    def selected_episodes(self, override: int | None = None):
        limit = self.num_demos if override is None else override
        return self.all_episodes if limit is None else self.all_episodes[:limit]
```

`prepare_offline_runtime`:

1. catalogs the complete dataset for the endpoint manifest;
2. applies `offline_num_demos` only to the selected build episodes;
3. hashes `value_ckpt` and `<dataset>/meta/stats.json`;
4. computes endpoint and replay fingerprints;
5. chooses a replay generation, creating a new one when
   `offline_rebuild=True`;
6. appends or replaces these trainer flags exactly once:

```text
--offline_dataset_path <block_success>
--offline_buffer_cache <fingerprinted generation>
--offline_base_mode gt
```

Implement strict option replacement:

```python
def _replace_option(argv, name, value):
    output, index = [], 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ContractError(f"{name} requires exactly one value")
            index += 2
            continue
        if token.startswith(name + "="):
            index += 1
            continue
        output.append(token)
        index += 1
    output.extend([name, str(value)])
    return output
```

Call it once for each authoritative trainer option. This removes existing
well-formed values and rejects a bare malformed option instead of guessing.

In `launch_imagination.main`, call the functions in this order:

```python
bridge_args, passthrough = parse_bridge_args(argv)
contract.check_passthrough_runtime_args(
    passthrough, imagination_gamma=bridge_args.imagination_gamma)
offline_runtime, passthrough = prepare_offline_runtime(
    bridge_args, passthrough)
factories = build_imagination_factories(
    bridge_args, offline_runtime=offline_runtime)
```

Inside `build_imagination_factories`, create the scorer as today, then call:

```python
contract.check_mixed_scorer(
    scorer, enabled=(offline_runtime is not None))
```

- [ ] **Step 5: Write and implement bridge metadata output**

Test that metadata contains:

```python
assert meta["trainer_compat_mode"] == "gt"
assert meta["actual_base_mode"] == "kai0_chunk"
assert meta["offline_reward"] == "gamma_phi_next_minus_phi"
assert meta["online_batch_size"] == 128
assert meta["offline_batch_size"] == 128
assert meta["pi0_serve_ckpt_id"] == "pi05_block_awbc_49999"
```

Implement and call this before invoking the trainer:

```python
def write_bridge_run_config(output_dir, metadata):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "bridge_run_config.json"
    temp_path = directory / ".bridge_run_config.json.tmp"
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, final_path)


def write_bridge_cache_meta(replay_cache_dir, metadata):
    directory = Path(replay_cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "bridge_meta.json"
    temp_path = directory / ".bridge_meta.json.tmp"
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, final_path)
```

Parse `--output_dir` with the same small passthrough parser used for mixed
arguments, defaulting to the trainer default `outputs_chunk`. Call both
`write_bridge_run_config(parsed.output_dir, metadata)` and
`write_bridge_cache_meta(offline_runtime.replay_cache_dir, metadata)` before
installing fakes.

- [ ] **Step 6: Run focused tests and commit**

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py -v
git add \
  resfit/rl_finetuning/wm_bridge/builder.py \
  resfit/rl_finetuning/wm_bridge/contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py
git commit -m "feat: configure block mixed replay bridge"
```

---

### Task 5: Inject Offline Count/Build Functions Without Modifying the Trainer

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/builder.py:48-117`
- Modify: `resfit/rl_finetuning/wm_bridge/launch_imagination.py:20-68`
- Modify: `resfit/rl_finetuning/wm_bridge/tests/test_launcher.py`
- Modify: `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`

**Interfaces:**
- Consumes: `OfflineRuntime` from Task 4.
- Produces fake
  `count_offline_transitions(dataset_path, num_demos=None) -> int`.
- Produces fake `build_offline_buffer(offline_rb, dataset_path, **kwargs)`.
- Produces `make_offline_factories(offline_runtime, state, scorer) -> dict[str, Callable]`.
- The fake builder shares the exact `state["base"]` and `scorer` instances
  already used by `ImaginationVecEnv`.

- [ ] **Step 1: Extend the failing launcher injection test to five symbols**

Change the target list and factory call:

```python
leaf_names = (
    "resfit.dexmg.environments.dexmg",
    "resfit.rl_finetuning.utils.evaluate_dexmg",
    "resfit.lerobot.policies.pi05",
    "resfit.rl_finetuning.chunk_residual.offline_stage_replay",
)
install_fakes({
    "create_vectorized_env": lambda **kw: "ENV",
    "run_dexmg_evaluation": lambda **kw: {},
    "load_pi05_base_policy": lambda *a, **k: "BASE",
    "count_offline_transitions": lambda *a, **k: 7,
    "build_offline_buffer": lambda *a, **k: "BUILT",
})
offline = sys.modules[
    "resfit.rl_finetuning.chunk_residual.offline_stage_replay"]
assert offline.count_offline_transitions("ignored") == 7
assert offline.build_offline_buffer(None, "ignored") == "BUILT"
```

- [ ] **Step 2: Run the launcher test and observe `_TARGETS` lookup failure**

Expected: FAIL with `KeyError` for the first offline symbol.

- [ ] **Step 3: Register both offline symbols on the same fake module**

Extend `_TARGETS`:

```python
"count_offline_transitions":
    "resfit.rl_finetuning.chunk_residual.offline_stage_replay",
"build_offline_buffer":
    "resfit.rl_finetuning.chunk_residual.offline_stage_replay",
```

Keep `install_fakes` unchanged: its existing reuse of a module marked
`_wm_bridge_fake` correctly installs the second symbol on the same leaf
module.

- [ ] **Step 4: Add failing fake count/build closure tests**

Use a concrete runtime and monkeypatch the heavy builder:

```python
from resfit.rl_finetuning.wm_bridge import builder as builder_module


class RuntimeStub:
    dataset_root = "/data/block_success"
    endpoint_store = object()

    def selected_episodes(self, num_demos=None):
        episodes = (
            types.SimpleNamespace(num_frames=51),
            types.SimpleNamespace(num_frames=101),
        )
        return episodes if num_demos is None else episodes[:num_demos]


def test_offline_factories_count_and_forward_shared_objects(monkeypatch):
    runtime = RuntimeStub()
    shared_base, shared_scorer = object(), object()
    state = {"base": shared_base}
    captured = {}

    def fake_build(rb, path, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(transitions=3)

    monkeypatch.setattr(builder_module,
                        "build_block_offline_buffer", fake_build)
    factories = make_offline_factories(runtime, state, shared_scorer)
    assert factories["count_offline_transitions"](
        runtime.dataset_root, num_demos=2) == 3
    with pytest.raises(ContractError, match="dataset_path"):
        factories["count_offline_transitions"](
            "/another/root", num_demos=2)

    factories["build_offline_buffer"](
        object(), runtime.dataset_root, num_demos=2,
        base_policy="generic", base_mode="gt", gamma=0.995)
    assert captured["base_policy"] is shared_base
    assert captured["scorer"] is shared_scorer
    assert captured["endpoint_store"] is runtime.endpoint_store
    assert len(captured["episodes"]) == 2
```

- [ ] **Step 5: Implement the closures in `build_imagination_factories`**

When `offline_runtime is not None`, add:

```python
def _check_offline_source(dataset_path, runtime):
    if os.path.realpath(dataset_path) != os.path.realpath(runtime.dataset_root):
        raise ContractError(
            f"dataset_path {dataset_path!r} does not match "
            f"{runtime.dataset_root!r}")


def format_offline_build_stats(stats, runtime):
    return (
        f"offline_source=block_success "
        f"offline_transitions={stats.transitions} "
        f"trainer_compat_mode=gt actual_offline_base=kai0_chunk "
        f"offline_reward=gamma_phi_next_minus_phi "
        f"replay_cache={runtime.replay_cache_dir}"
    )


def fake_count_offline_transitions(dataset_path, num_demos=None):
    _check_offline_source(dataset_path, offline_runtime)
    episodes = offline_runtime.selected_episodes(num_demos)
    return sum(len(plan_episode_chunks(ep.num_frames)) for ep in episodes)


def fake_build_offline_buffer(offline_rb, dataset_path, **kwargs):
    _check_offline_source(dataset_path, offline_runtime)
    if state["base"] is None:
        raise RuntimeError("kai0 base must be created before offline replay")
    stats = build_block_offline_buffer(
        offline_rb, dataset_path,
        base_policy=state["base"],
        scorer=scorer,
        endpoint_store=offline_runtime.endpoint_store,
        episodes=offline_runtime.selected_episodes(kwargs.get("num_demos")),
        **kwargs,
    )
    print(format_offline_build_stats(stats, offline_runtime))
```

Avoid passing duplicate `base_policy`, `base_mode`, or generic reward
arguments: pop those keys first, assert `base_mode == "gt"` is only the
compatibility marker, then supply the shared base explicitly.

- [ ] **Step 6: Validate hook points without importing the heavy generic module**

Add `contract.check_offline_hook_points()` that inspects
`train_chunk_residual.main` source and requires both exact lazy imports:

```text
from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
    build_offline_buffer, count_offline_transitions)
```

and the exact calls to both functions. Call this check before installing
fakes. This prevents silent trainer drift without importing
robosuite/dexmimicgen.

- [ ] **Step 7: Run launcher/contract tests and commit**

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_launcher.py \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py -v
git add \
  resfit/rl_finetuning/wm_bridge/builder.py \
  resfit/rl_finetuning/wm_bridge/launch_imagination.py \
  resfit/rl_finetuning/wm_bridge/contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_launcher.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py
git commit -m "feat: inject block offline replay builder"
```

---

### Task 6: Production Launch Parameters and End-to-end Verification Gates

**Files:**
- Modify: `launch_block_imagination.sh`
- Modify: `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`

**Interfaces:**
- Production launcher consumes the existing WM, kai0, and advantage serves.
- Production output directory:
  `/mnt/mnt/data/resfit/outputs_imagination/block_shore_mixed50_seed${SEED}`.
- Production endpoint/replay cache root:
  `/mnt/mnt/data/resfit/cache/block_mixed_replay`.

- [ ] **Step 1: Write a failing static launch-contract test**

Read the shell script and use `shlex`-compatible token matching:

```python
def test_production_launch_enables_exact_mixed_configuration():
    text = Path("launch_block_imagination.sh").read_text()
    required = {
        "--offline_chunk_dataset": "${BLK}/block_success",
        "--offline_chunk_cache_root":
            "/mnt/mnt/data/resfit/cache/block_mixed_replay",
        "--offline_fraction": "0.5",
        "--batch_size": "256",
        "--actor": "raw",
        "--action_scale": "0.2",
        "--min_range_per_dim": "0.1",
        "--demo_bc_coef": "0.1",
        "--bc_coef_final": "0.01",
        "--critic_warmup_steps": "10000",
        "--learning_starts": "10000",
    }
    for flag, value in required.items():
        assert f"{flag} {value}" in text
    assert "--no_stage_balanced" in text
    assert "--offline_base_mode base_policy" not in text
    assert "--offline_base_mode gt" not in text


def test_production_batch_is_exactly_128_plus_128():
    batch_size, fraction = 256, 0.5
    assert int(batch_size * (1 - fraction)) == 128
    assert int(batch_size * fraction) == 128
```

- [ ] **Step 2: Run the static test and verify missing flags fail**

Run the new test file. Expected: FAIL listing the first absent offline flag.

- [ ] **Step 3: Update the production launch script**

Add:

```bash
--offline_chunk_dataset ${BLK}/block_success \
--offline_chunk_cache_root /mnt/mnt/data/resfit/cache/block_mixed_replay \
--offline_fraction 0.5 --batch_size 256 \
--actor raw --action_scale 0.2 --min_range_per_dim 0.1 \
--demo_bc_coef 0.1 --bc_coef_final 0.01 \
--critic_warmup_steps 10000 --learning_starts 10000 \
--no_stage_balanced \
```

Change:

```bash
OUT=/mnt/mnt/data/resfit/outputs_imagination/block_shore_mixed50_seed${SEED}
--wandb_name block_shore_mixed50_seed${SEED}
```

Keep `block_success` and `block_fail` in `--init_state_dataset`. Do not pass
generic `--offline_dataset_path`, `--offline_buffer_cache`, or
`--offline_base_mode`; the bridge injects all three authoritative values.

- [ ] **Step 4: Run syntax, focused tests, and the full bridge suite**

```bash
bash -n launch_block_imagination.sh
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py -v
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests -v
```

Expected: Bash exits 0; focused tests PASS; full bridge suite PASS.

- [ ] **Step 5: Run pure-online regression without live serves**

Exercise argument preparation with no `--offline_chunk_dataset` and assert:

- passthrough does not gain any offline argument;
- no endpoint/replay directories are created;
- factory dictionary contains only the original three symbols;
- existing imagination contract tests remain byte-for-byte behaviorally
  equivalent.

Expected: PASS.

- [ ] **Step 6: Run opt-in two-episode live cache smoke**

With kai0 serve active on port 8001:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/mnt/mnt/data/resfit \
HF_LEROBOT_HOME=/mnt/mnt/data/domains_rise/block HF_HUB_OFFLINE=1 \
/mnt/mnt/data/envs/residual/bin/python -m \
  resfit.rl_finetuning.wm_bridge.launch_imagination \
  --value_ckpt /mnt/mnt/data/resfit/outputs_chunk/block_value_pi0feat.pt \
  --wm_host 127.0.0.1 --wm_port 9000 \
  --init_state_dataset /mnt/mnt/data/domains_rise/block/block_success \
  --init_state_dataset /mnt/mnt/data/domains_rise/block/block_fail \
  --pi0_serve_ckpt_id pi05_block_awbc_49999 \
  --offline_chunk_dataset /mnt/mnt/data/domains_rise/block/block_success \
  --offline_chunk_cache_root /mnt/mnt/data/resfit/cache/block_mixed_replay_smoke \
  --offline_num_demos 2 --offline_fraction 0.5 --batch_size 256 \
  --base_policy_type pi05 --base_action_mode replan --chunk_length 50 \
  --actor raw --action_scale 0.2 --min_range_per_dim 0.1 \
  --n_step 1 --gamma 0.995 --imagination_gamma 0.995 \
  --reward_shaping none --potential_source stage \
  --pi0_host 127.0.0.1 --pi0_port 8001 \
  --pi0_prompt "build block" --pi0_action_dim 16 \
  --dataset block_success --critic_warmup_steps 2 \
  --learning_starts 50 --utd 1 --smoke \
  --wandb_mode disabled \
  --output_dir /mnt/mnt/data/resfit/outputs_imagination/block_mixed_smoke
```

Expected startup evidence:

```text
offline_source=block_success
trainer_compat_mode=gt
actual_offline_base=kai0_chunk
online_batch_size=128 offline_batch_size=128
offline_reward=gamma_phi_next_minus_phi
```

Expected completion evidence: replay contains finite transitions, two critic
warmup steps run, and at least one actor update is finite. Rerun the identical
command and expect endpoint and replay cache hits with zero offline endpoint
kai0 queries.

- [ ] **Step 7: Inspect production dry-run metadata**

Run argument preparation with the production script tokens without starting
serves. Assert `bridge_run_config.json` would record:

```json
{
  "online_batch_size": 128,
  "offline_batch_size": 128,
  "trainer_compat_mode": "gt",
  "actual_base_mode": "kai0_chunk",
  "offline_reward": "gamma_phi_next_minus_phi"
}
```

- [ ] **Step 8: Commit Task 6**

```bash
git add \
  launch_block_imagination.sh \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py
git commit -m "feat: launch block training with mixed replay"
```

---

## Final Verification

- [ ] Run whitespace and shell checks:

```bash
git diff --check
bash -n launch_block_imagination.sh
```

- [ ] Run the complete bridge test suite:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests -v
```

- [ ] Run affected chunk-residual regression tests:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/tests/test_mixed_batch.py \
  resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup.py -v
```

If an exact listed regression file does not exist, locate the corresponding
test with:

```bash
rg --files resfit/rl_finetuning | rg \
  'test_.*(chunk|mixed|warmup).*\.py$'
```

and run every matching existing test; do not silently omit the category.

- [ ] Confirm protected upstream files are unchanged relative to the
implementation base commit:

```bash
git diff --exit-code <implementation-base> -- \
  resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
  resfit/rl_finetuning/chunk_residual/offline_stage_replay.py
git -C /mnt/mnt/data/data2/RL/RISE status --short
```

- [ ] Review every implementation commit and request code review before
production launch.

- [ ] Run the opt-in two-episode live cache smoke, then rerun it to verify
cache hits.

- [ ] Only after all gates pass, launch Arm C with the production script.
Keep Arm A (`offline_fraction=0`) and Arm B (`offline_fraction=0.5`,
`demo_bc_coef=0`) as separate three-seed comparisons; do not overwrite their
output directories or W&B names.
