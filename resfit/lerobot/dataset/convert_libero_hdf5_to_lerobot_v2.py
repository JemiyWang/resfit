#!/usr/bin/env python3
"""Convert official LIBERO HDF5 demonstrations to LeRobot v2.0."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_info, write_stats
from PIL import Image


REQUIRED_DATASETS = (
    "actions",
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
    "obs/ee_states",
    "obs/gripper_states",
)


def get_demo_keys(h5: h5py.File) -> list[str]:
    """Return HDF5 demonstration keys in numeric rather than lexical order."""
    if "data" not in h5:
        raise ValueError("missing HDF5 group: data")

    keys = list(h5["data"].keys())
    if any(not key.startswith("demo_") for key in keys):
        raise ValueError("all trajectory keys must have the form demo_N")
    try:
        return sorted(keys, key=lambda key: int(key.removeprefix("demo_")))
    except ValueError as exc:
        raise ValueError("all trajectory keys must have the form demo_N") from exc


def validate_demo(
    demo: h5py.Group,
    demo_key: str,
    source_image_shape: tuple[int, int, int] = (128, 128, 3),
) -> int:
    """Validate one official LIBERO trajectory and return its frame count."""
    missing = [key for key in REQUIRED_DATASETS if key not in demo]
    if missing:
        raise ValueError(f"{demo_key}: missing datasets: {missing}")

    expected_shapes = {
        "actions": (7,),
        "obs/agentview_rgb": source_image_shape,
        "obs/eye_in_hand_rgb": source_image_shape,
        "obs/ee_states": (6,),
        "obs/gripper_states": (2,),
    }
    lengths = {key: demo[key].shape[0] for key in REQUIRED_DATASETS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{demo_key}: inconsistent trajectory length: {lengths}")

    for key, expected_shape in expected_shapes.items():
        actual_shape = demo[key].shape[1:]
        if actual_shape != expected_shape:
            raise ValueError(
                f"{demo_key}: {key} has shape {actual_shape}, "
                f"expected {expected_shape}"
            )

    return next(iter(lengths.values()))


def _convert_image(image: np.ndarray, image_size: int) -> np.ndarray:
    rotated = np.ascontiguousarray(image[::-1, ::-1])
    resized = Image.fromarray(rotated).resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.uint8)


def convert_frame(
    demo: h5py.Group,
    frame_index: int,
    image_size: int = 256,
) -> dict[str, Any]:
    """Convert a single LIBERO frame to the SHORE-compatible feature schema."""
    obs = demo["obs"]
    return {
        "image": _convert_image(obs["agentview_rgb"][frame_index], image_size),
        "wrist_image": _convert_image(
            obs["eye_in_hand_rgb"][frame_index],
            image_size,
        ),
        "state": np.concatenate(
            (
                obs["ee_states"][frame_index],
                obs["gripper_states"][frame_index],
            )
        ).astype(np.float32),
        "actions": np.asarray(
            demo["actions"][frame_index],
            dtype=np.float32,
        ),
    }



def _feature_schema(image_size: int) -> dict[str, dict[str, Any]]:
    return {
        "image": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_converted_dataset(
    root: Path,
    repo_id: str,
    task: str,
    expected_episodes: int,
    expected_frames: int,
    canonical_stats: Path,
    image_size: int = 256,
) -> None:
    """Validate a completed task-local LeRobot dataset before publishing it."""
    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.jsonl"
    local_stats_path = root / "meta/stats.task_local.json"
    installed_stats_path = root / "meta/stats.json"
    required_paths = (
        info_path,
        tasks_path,
        local_stats_path,
        installed_stats_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"converted dataset is missing files: {missing}")

    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    expected_info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "fps": 10,
        "total_episodes": expected_episodes,
        "total_frames": expected_frames,
        "total_tasks": 1,
        "total_videos": 0,
    }
    mismatches = {
        key: (info.get(key), expected)
        for key, expected in expected_info.items()
        if info.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"converted metadata mismatch: {mismatches}")

    expected_features = _feature_schema(image_size)
    for key, expected in expected_features.items():
        actual = info["features"].get(key)
        if actual is None:
            raise ValueError(f"converted metadata is missing feature: {key}")
        if actual["dtype"] != expected["dtype"]:
            raise ValueError(f"{key} dtype is {actual['dtype']}, expected {expected['dtype']}")
        if tuple(actual["shape"]) != tuple(expected["shape"]):
            raise ValueError(f"{key} shape is {actual['shape']}, expected {expected['shape']}")

    task_records = _read_jsonlines(tasks_path)
    expected_task_records = [{"task_index": 0, "task": task}]
    if task_records != expected_task_records:
        raise ValueError(
            f"task metadata is {task_records}, expected {expected_task_records}"
        )

    if installed_stats_path.read_bytes() != canonical_stats.read_bytes():
        raise ValueError("installed stats.json differs from canonical stats")

    parquet_files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if len(parquet_files) != expected_episodes:
        raise ValueError(
            f"found {len(parquet_files)} episode files, expected {expected_episodes}"
        )

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    if dataset.num_episodes != expected_episodes or len(dataset) != expected_frames:
        raise ValueError(
            "loaded dataset counts differ from source: "
            f"episodes={dataset.num_episodes}, frames={len(dataset)}"
        )
    first = dataset[0]
    if first["task"] != task:
        raise ValueError(f"loaded task is {first['task']!r}, expected {task!r}")
    if tuple(first["state"].shape) != (8,):
        raise ValueError(f"loaded state shape is {tuple(first['state'].shape)}")
    if tuple(first["actions"].shape) != (7,):
        raise ValueError(f"loaded action shape is {tuple(first['actions'].shape)}")
    if tuple(first["image"].shape) != (3, image_size, image_size):
        raise ValueError(f"loaded image shape is {tuple(first['image'].shape)}")
    if tuple(first["wrist_image"].shape) != (3, image_size, image_size):
        raise ValueError(
            f"loaded wrist image shape is {tuple(first['wrist_image'].shape)}"
        )


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
    """Convert one LIBERO HDF5 task into an independent LeRobot v2 dataset."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    canonical_stats = Path(canonical_stats).resolve()
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

    dataset: LeRobotDataset | None = None
    demo_keys: list[str] = []
    total_frames = 0
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=partial,
            robot_type="panda",
            features=_feature_schema(image_size),
            use_videos=False,
            image_writer_threads=4,
        )
        with h5py.File(source, "r") as h5:
            demo_keys = get_demo_keys(h5)
            if not demo_keys:
                raise ValueError("source contains no demonstrations")
            for episode_index, demo_key in enumerate(demo_keys):
                demo = h5[f"data/{demo_key}"]
                length = validate_demo(
                    demo,
                    demo_key,
                    source_image_shape=source_image_shape,
                )
                if length == 0:
                    raise ValueError(f"{demo_key}: empty trajectory")
                for frame_index in range(length):
                    dataset.add_frame(
                        {
                            **convert_frame(demo, frame_index, image_size),
                            "task": task,
                        }
                    )
                dataset.save_episode()
                total_frames += length
                print(
                    f"[{episode_index + 1}/{len(demo_keys)}] "
                    f"saved {demo_key}: {length} frames",
                    flush=True,
                )

        dataset.stop_image_writer()
        dataset.meta.info["codebase_version"] = "v2.0"
        write_info(dataset.meta.info, partial)
        write_stats(dataset.meta.stats, partial)
        task_local_stats = partial / "meta/stats.task_local.json"
        (partial / "meta/stats.json").replace(task_local_stats)
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
        if dataset is not None:
            dataset.stop_image_writer()
        raise

    return {"episodes": len(demo_keys), "frames": total_frames}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--canonical-stats", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = convert_dataset(
        source=args.source,
        output=args.output,
        repo_id=args.repo_id,
        task=args.task,
        canonical_stats=args.canonical_stats,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
