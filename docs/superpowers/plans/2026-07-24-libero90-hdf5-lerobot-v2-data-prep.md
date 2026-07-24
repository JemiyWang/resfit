# LIBERO-90 HDF5 to LeRobot v2.0 Data Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download the four selected LIBERO-90 demonstrations at a fixed upstream revision and convert each task into an independent LeRobot v2.0 dataset that SHORE can consume without modifying the existing 40-task `physical-intelligence/libero` dataset.

**Architecture:** A focused converter reads one official LIBERO HDF5 file, validates its schema, converts every trajectory frame into the existing SHORE-compatible observation/action schema, writes to a sibling `.partial` directory, validates the finished dataset, and atomically renames it into place. Raw HDF5 files and converted datasets live below new subdirectories of the current cache root; the existing `data/` and `meta/` directories are read-only inputs.

**Tech Stack:** Python 3.11, h5py, NumPy, Pillow, LeRobot 0.1.0/v2.0 dataset format, pytest, curl, SHA-256.

## Global Constraints

- Run Python with `/mnt/mnt/data/chj/openpi/.venv/bin/python`.
- Do not modify `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/data` or its existing `/meta`.
- Do not start SHORE training in this plan and do not stop or reuse GPU1.
- Preserve all HDF5 frames; do not temporally subsample.
- Use the exact upstream revision `cf16e484f5a2a556f3e7e5ad8ec3d97d3dcfe498`.
- Use 10 FPS, 256×256 RGB images, 8-D state, and 7-D actions.
- Rotate both source cameras by 180 degrees before bilinear resizing.
- Install a byte-identical copy of the existing global `meta/stats.json` as each converted dataset's canonical `meta/stats.json`; retain task-local generated statistics as `meta/stats.task_local.json`.

---

## Task 1: Add Unit-Tested LIBERO HDF5 Parsing and Frame Conversion

**Files:**

- Create: `resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py`
- Create: `resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py`

- [ ] **Step 1: Write failing tests for numeric demo sorting, schema validation, and frame conversion**

Create a small synthetic HDF5 fixture and tests equivalent to:

```python
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from resfit.lerobot.dataset.convert_libero_hdf5_to_lerobot_v2 import (
    convert_frame,
    get_demo_keys,
    validate_demo,
)


def _write_demo(group: h5py.Group, length: int = 2) -> None:
    group.create_dataset("actions", data=np.arange(length * 7, dtype=np.float32).reshape(length, 7))
    obs = group.create_group("obs")
    image = np.zeros((length, 2, 3, 3), dtype=np.uint8)
    image[0, 0, 0] = [1, 2, 3]
    image[0, -1, -1] = [9, 8, 7]
    obs.create_dataset("agentview_rgb", data=image)
    obs.create_dataset("eye_in_hand_rgb", data=image + 1)
    obs.create_dataset("ee_states", data=np.arange(length * 6, dtype=np.float32).reshape(length, 6))
    obs.create_dataset("gripper_states", data=np.arange(length * 2, dtype=np.float32).reshape(length, 2))


def test_get_demo_keys_uses_numeric_order(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5:
        data = h5.create_group("data")
        _write_demo(data.create_group("demo_10"))
        _write_demo(data.create_group("demo_2"))
    with h5py.File(source, "r") as h5:
        assert get_demo_keys(h5) == ["demo_2", "demo_10"]


def test_validate_demo_rejects_mismatched_lengths(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5:
        demo = h5.create_group("data").create_group("demo_0")
        _write_demo(demo)
        del demo["obs/gripper_states"]
        demo["obs"].create_dataset("gripper_states", data=np.zeros((1, 2), dtype=np.float32))
    with h5py.File(source, "r") as h5:
        with pytest.raises(ValueError, match="length"):
            validate_demo(h5["data/demo_0"], "demo_0")


def test_convert_frame_rotates_resizes_and_concatenates_state(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5:
        demo = h5.create_group("data").create_group("demo_0")
        _write_demo(demo)
        frame = convert_frame(demo, frame_index=0, image_size=4)
    assert frame["image"].shape == (4, 4, 3)
    assert frame["wrist_image"].shape == (4, 4, 3)
    np.testing.assert_array_equal(frame["image"][-1, -1], [1, 2, 3])
    np.testing.assert_array_equal(
        frame["state"],
        np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(frame["actions"], np.arange(7, dtype=np.float32))
```

- [ ] **Step 2: Run the focused tests and confirm the expected import failure**

Run:

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python -m pytest \
  resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py -q
```

Expected: FAIL because `convert_libero_hdf5_to_lerobot_v2` does not exist.

- [ ] **Step 3: Implement the minimal parsing and frame-conversion helpers**

Implement these public helpers in the new converter:

```python
REQUIRED_DATASETS = (
    "actions",
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
    "obs/ee_states",
    "obs/gripper_states",
)


def get_demo_keys(h5: h5py.File) -> list[str]:
    if "data" not in h5:
        raise ValueError("missing HDF5 group: data")
    keys = list(h5["data"].keys())
    try:
        return sorted(keys, key=lambda key: int(key.removeprefix("demo_")))
    except ValueError as exc:
        raise ValueError("all trajectory keys must have the form demo_N") from exc


def validate_demo(demo: h5py.Group, demo_key: str) -> int:
    missing = [key for key in REQUIRED_DATASETS if key not in demo]
    if missing:
        raise ValueError(f"{demo_key}: missing datasets: {missing}")
    expected_shapes = {
        "actions": (7,),
        "obs/agentview_rgb": (128, 128, 3),
        "obs/eye_in_hand_rgb": (128, 128, 3),
        "obs/ee_states": (6,),
        "obs/gripper_states": (2,),
    }
    lengths = {key: demo[key].shape[0] for key in REQUIRED_DATASETS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{demo_key}: inconsistent trajectory length: {lengths}")
    for key, trailing_shape in expected_shapes.items():
        actual = demo[key].shape[1:]
        if actual != trailing_shape:
            raise ValueError(f"{demo_key}: {key} has shape {actual}, expected {trailing_shape}")
    return next(iter(lengths.values()))


def _convert_image(image: np.ndarray, image_size: int) -> np.ndarray:
    rotated = np.ascontiguousarray(image[::-1, ::-1])
    return np.asarray(
        Image.fromarray(rotated).resize((image_size, image_size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def convert_frame(demo: h5py.Group, frame_index: int, image_size: int = 256) -> dict[str, np.ndarray]:
    obs = demo["obs"]
    return {
        "image": _convert_image(obs["agentview_rgb"][frame_index], image_size),
        "wrist_image": _convert_image(obs["eye_in_hand_rgb"][frame_index], image_size),
        "state": np.concatenate(
            (obs["ee_states"][frame_index], obs["gripper_states"][frame_index])
        ).astype(np.float32),
        "actions": np.asarray(demo["actions"][frame_index], dtype=np.float32),
    }
```

Permit non-128 synthetic images in tests by making `validate_demo` accept an optional `source_image_shape=(128, 128, 3)` argument; production calls use the default.

- [ ] **Step 4: Run the unit tests**

Run:

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python -m pytest \
  resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the unit-tested helpers**

```bash
cd /mnt/mnt/data/resfit
git add \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py \
  resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py
git commit -m "feat: parse LIBERO HDF5 demonstrations"
```

---

## Task 2: Implement Atomic End-to-End LeRobot v2 Conversion

**Files:**

- Modify: `resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py`
- Modify: `resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py`

- [ ] **Step 1: Add a failing end-to-end conversion test**

Extend the test fixture so `validate_demo(..., source_image_shape=(2, 3, 3))` can be used, create two demos, and test:

```python
def test_convert_dataset_writes_independent_v2_dataset_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "converted" / "task63"
    canonical_stats = tmp_path / "global-stats.json"
    canonical_stats.write_text('{"state":{"mean":[1]}}\\n')
    with h5py.File(source, "w") as h5:
        data = h5.create_group("data")
        _write_demo(data.create_group("demo_1"), length=2)
        _write_demo(data.create_group("demo_0"), length=3)

    summary = convert_dataset(
        source=source,
        output=output,
        repo_id="physical-intelligence/libero-task63",
        task="stack the left bowl on the right bowl and place them in the tray",
        canonical_stats=canonical_stats,
        source_image_shape=(2, 3, 3),
        image_size=4,
    )

    assert summary == {"episodes": 2, "frames": 5}
    assert output.is_dir()
    assert not output.with_name("task63.partial").exists()
    assert (output / "meta/stats.json").read_bytes() == canonical_stats.read_bytes()
    assert (output / "meta/stats.task_local.json").is_file()
    info = json.loads((output / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.0"
    assert info["robot_type"] == "panda"
    assert info["fps"] == 10
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 5
    assert info["total_tasks"] == 1
    task_record = json.loads((output / "meta/tasks.jsonl").read_text())
    assert task_record == {
        "task_index": 0,
        "task": "stack the left bowl on the right bowl and place them in the tray",
    }
```

Also add tests that:

- refuse an already complete output without `overwrite=True`;
- refuse an existing `.partial` directory rather than merging into it;
- preserve a task-local stats file before canonical stats are copied;
- raise if the canonical stats file is absent;
- load the converted dataset through `LeRobotDataset(repo_id=..., root=output)` and verify the first state/action values.

- [ ] **Step 2: Run the focused tests and confirm failure**

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python -m pytest \
  resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py -q
```

Expected: FAIL because `convert_dataset` and the CLI do not exist.

- [ ] **Step 3: Add SHORE-compatible features, conversion, validation, and CLI**

Use this feature schema:

```python
FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}
```

Implement the conversion around `LeRobotDataset.create`:

```python
def convert_dataset(
    source: Path,
    output: Path,
    repo_id: str,
    task: str,
    canonical_stats: Path,
    *,
    fps: int = 10,
    image_size: int = 256,
    source_image_shape: tuple[int, int, int] = (128, 128, 3),
    overwrite: bool = False,
) -> dict[str, int]:
    source = source.resolve()
    output = output.resolve()
    partial = output.with_name(f"{output.name}.partial")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not canonical_stats.is_file():
        raise FileNotFoundError(canonical_stats)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if partial.exists():
        raise FileExistsError(partial)
    if output.exists():
        shutil.rmtree(output)

    features = {
        key: {**value, "shape": (image_size, image_size, 3)}
        if value["dtype"] == "image"
        else value
        for key, value in FEATURES.items()
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=partial,
        robot_type="panda",
        features=features,
        use_videos=False,
        image_writer_threads=4,
    )
    total_frames = 0
    try:
        with h5py.File(source, "r") as h5:
            demo_keys = get_demo_keys(h5)
            if not demo_keys:
                raise ValueError("source contains no demonstrations")
            for demo_key in demo_keys:
                demo = h5[f"data/{demo_key}"]
                length = validate_demo(demo, demo_key, source_image_shape)
                for frame_index in range(length):
                    dataset.add_frame(
                        {
                            **convert_frame(demo, frame_index, image_size),
                            "task": task,
                        }
                    )
                dataset.save_episode()
                total_frames += length
        dataset.stop_image_writer()
        local_stats = partial / "meta/stats.json"
        if not local_stats.is_file():
            from lerobot.common.datasets.utils import write_stats
            write_stats(dataset.meta.stats, partial)
        local_stats.replace(partial / "meta/stats.task_local.json")
        shutil.copyfile(canonical_stats, partial / "meta/stats.json")
        validate_converted_dataset(
            root=partial,
            repo_id=repo_id,
            task=task,
            expected_episodes=len(demo_keys),
            expected_frames=total_frames,
            canonical_stats=canonical_stats,
            image_size=image_size,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.rename(output)
    except BaseException:
        dataset.stop_image_writer()
        raise
    return {"episodes": len(demo_keys), "frames": total_frames}
```

The implementation may replace an existing output only when `--overwrite` is explicit, but must never automatically delete a stale `.partial`; report it so the operator can inspect it. `validate_converted_dataset` must verify metadata, exact language, episode/frame counts, feature dtypes/shapes, canonical stats byte equality, local stats presence, and successful `LeRobotDataset` loading.

Add a CLI with required `--source`, `--output`, `--repo-id`, `--task`, and `--canonical-stats`, plus optional `--overwrite`. Print a JSON summary to stdout on success.

- [ ] **Step 4: Run focused and neighboring dataset tests**

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python -m pytest \
  resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py -q
/mnt/mnt/data/chj/openpi/.venv/bin/python -m py_compile \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py
```

Expected: all tests PASS and compilation exits 0.

- [ ] **Step 5: Commit end-to-end conversion**

```bash
cd /mnt/mnt/data/resfit
git add \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py \
  resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py
git commit -m "feat: convert LIBERO HDF5 to LeRobot v2"
```

---

## Task 3: Convert and Validate the Existing Task63 Download

**Files:**

- Read: `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo.hdf5`
- Create: `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task63`

- [ ] **Step 1: Verify task63 source integrity**

```bash
sha256sum /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo.hdf5
```

Expected:

```text
6e5dd6cb435e94cd90886e2ae3a5f7145d3034710d7c1605bad1c97d56acb2b3
```

Also verify the exact size is `1073326526` bytes.

- [ ] **Step 2: Record the original dataset metadata hashes before conversion**

```bash
cd /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero
sha256sum meta/info.json meta/tasks.jsonl meta/stats.json
```

Save this output in the execution log for comparison in Task 6.

- [ ] **Step 3: Run the converter**

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py \
  --source /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo.hdf5 \
  --output /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task63 \
  --repo-id physical-intelligence/libero90-task63 \
  --task "stack the left bowl on the right bowl and place them in the tray" \
  --canonical-stats /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json
```

Expected JSON: 50 episodes and 10,771 frames.

- [ ] **Step 4: Independently validate task63**

Run the converter's validation entry point or a short read-only Python inspection and confirm:

- 50 Parquet episode files;
- 10,771 frames;
- exactly one task at task index 0;
- images are `uint8` 256×256×3;
- state is float32[8], actions float32[7];
- `meta/stats.task_local.json` exists;
- `meta/stats.json` has the same SHA-256 as the original global stats;
- no `task63.partial` directory remains.

---

## Task 4: Download and Verify Tasks 57, 60, and 64

**Files:**

- Create: three HDF5 files below `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90`

- [ ] **Step 1: Confirm sufficient disk space**

```bash
df -h /mnt/mnt/data
```

Require at least 5 GiB free for the remaining downloads plus conversion working space.

- [ ] **Step 2: Download task57 using a resumable partial file**

```bash
curl -L --fail --retry 3 --continue-at - \
  -o /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray_demo.hdf5.partial \
  "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/cf16e484f5a2a556f3e7e5ad8ec3d97d3dcfe498/libero_90/LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray_demo.hdf5?download=true"
```

Verify size `768466210` and SHA-256 `1702b5b6c81385edfe6e801372e20c5e16d1774faf030f5c0f67f159bd1eeead`, then rename `.partial` to `.hdf5`.

- [ ] **Step 3: Download task60 using a resumable partial file**

```bash
curl -L --fail --retry 3 --continue-at - \
  -o /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_demo.hdf5.partial \
  "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/cf16e484f5a2a556f3e7e5ad8ec3d97d3dcfe498/libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_demo.hdf5?download=true"
```

Verify size `617164176` and SHA-256 `1c6bd7024ecbac964971669e18425f61da135663b0b2a7ade58c02dd5bce7e93`, then rename `.partial` to `.hdf5`.

- [ ] **Step 4: Download task64 using a resumable partial file**

```bash
curl -L --fail --retry 3 --continue-at - \
  -o /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray_demo.hdf5.partial \
  "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/cf16e484f5a2a556f3e7e5ad8ec3d97d3dcfe498/libero_90/LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray_demo.hdf5?download=true"
```

Verify size `1168927388` and SHA-256 `85012c80aa040f1d35a93b2ffedf30136d794693dac6ac56f24abed247d7d972`, then rename `.partial` to `.hdf5`.

- [ ] **Step 5: Validate all four HDF5 files**

Open each file with h5py and run `get_demo_keys` plus `validate_demo` for every demonstration. Report per-task episode and frame totals before conversion.

---

## Task 5: Convert and Validate Tasks 57, 60, and 64

**Files:**

- Create: `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task57`
- Create: `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task60`
- Create: `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task64`

- [ ] **Step 1: Convert task57**

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py \
  --source /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray_demo.hdf5 \
  --output /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task57 \
  --repo-id physical-intelligence/libero90-task57 \
  --task "pick up the cream cheese and put it in the tray" \
  --canonical-stats /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json
```

- [ ] **Step 2: Convert task60**

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py \
  --source /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_demo.hdf5 \
  --output /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task60 \
  --repo-id physical-intelligence/libero90-task60 \
  --task "pick up the black bowl on the left and put it in the tray" \
  --canonical-stats /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json
```

- [ ] **Step 3: Convert task64**

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py \
  --source /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/raw/libero_90/LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray_demo.hdf5 \
  --output /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task64 \
  --repo-id physical-intelligence/libero90-task64 \
  --task "stack the right bowl on the left bowl and place them in the tray" \
  --canonical-stats /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json
```

- [ ] **Step 4: Validate all converted datasets independently**

For each task, verify the converter summary against the raw HDF5 totals, load it through `LeRobotDataset`, sample the first and last frame, and assert:

- total episodes and frames match the source exactly;
- exactly one exact language instruction exists;
- all feature shapes/dtypes match the existing dataset;
- every task has both task-local and canonical stats;
- every canonical stats hash matches the existing global stats;
- no `.partial` directory remains.

---

## Task 6: Final Regression and Preservation Gate

**Files:**

- Verify: converter and tests
- Verify: existing dataset metadata
- Verify: four raw and four converted task datasets

- [ ] **Step 1: Run the converter test suite**

```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/openpi/.venv/bin/python -m pytest \
  resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py -q
/mnt/mnt/data/chj/openpi/.venv/bin/python -m py_compile \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py
```

Expected: all tests PASS and compilation exits 0.

- [ ] **Step 2: Prove the original dataset metadata was not changed**

```bash
cd /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero
sha256sum meta/info.json meta/tasks.jsonl meta/stats.json
```

Expected: hashes exactly match the values recorded before Task 3.

- [ ] **Step 3: Record final artifact inventory**

Report:

- exact raw HDF5 path, byte size, SHA-256, episode count, and frame count for tasks 57/60/63/64;
- exact converted root, episode count, frame count, and canonical stats SHA-256 for each task;
- any retained `.partial` directories as failures requiring operator attention;
- confirmation that training has not started and GPU1 was not touched.

- [ ] **Step 4: Commit any final test-only corrections**

Only if Task 6 exposed a converter/test defect:

```bash
cd /mnt/mnt/data/resfit
git add \
  resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py \
  resfit/lerobot/dataset/tests/test_convert_libero_hdf5_to_lerobot_v2.py
git commit -m "test: harden LIBERO conversion validation"
```

