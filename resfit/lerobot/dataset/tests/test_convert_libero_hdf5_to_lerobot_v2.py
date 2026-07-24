from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from resfit.rl_finetuning.chunk_residual.libero_offline import (
    find_demo_episodes,
    read_libero_demo,
)
from resfit.lerobot.dataset.convert_libero_hdf5_to_lerobot_v2 import (
    convert_dataset,
    convert_frame,
    get_demo_keys,
    validate_demo,
)


def _write_demo(group: h5py.Group, length: int = 2) -> None:
    group.create_dataset(
        "actions",
        data=np.arange(length * 7, dtype=np.float32).reshape(length, 7),
    )
    obs = group.create_group("obs")
    image = np.zeros((length, 2, 3, 3), dtype=np.uint8)
    image[0, 0, 0] = [1, 2, 3]
    image[0, -1, -1] = [9, 8, 7]
    obs.create_dataset("agentview_rgb", data=image)
    obs.create_dataset("eye_in_hand_rgb", data=image + 1)
    obs.create_dataset(
        "ee_states",
        data=np.arange(length * 6, dtype=np.float32).reshape(length, 6),
    )
    obs.create_dataset(
        "gripper_states",
        data=np.arange(length * 2, dtype=np.float32).reshape(length, 2),
    )


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
        demo["obs"].create_dataset(
            "gripper_states", data=np.zeros((1, 2), dtype=np.float32)
        )

    with h5py.File(source, "r") as h5:
        with pytest.raises(ValueError, match="length"):
            validate_demo(
                h5["data/demo_0"],
                "demo_0",
                source_image_shape=(2, 3, 3),
            )


def test_convert_frame_rotates_resizes_and_concatenates_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5:
        demo = h5.create_group("data").create_group("demo_0")
        _write_demo(demo)
        frame = convert_frame(demo, frame_index=0, image_size=4)

    assert frame["image"].shape == (4, 4, 3)
    assert frame["wrist_image"].shape == (4, 4, 3)
    np.testing.assert_array_equal(frame["image"][-1, -1], [1, 2, 3])
    np.testing.assert_array_equal(frame["image"][0, 0], [9, 8, 7])
    np.testing.assert_array_equal(
        frame["state"],
        np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        frame["actions"], np.arange(7, dtype=np.float32)
    )


def test_convert_dataset_writes_loadable_v2_dataset_atomically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "converted" / "task63"
    canonical_stats = tmp_path / "global-stats.json"
    canonical_stats.write_text(
        json.dumps({"state": {"mean": [1.0]}}) + "\n",
        encoding="utf-8",
    )
    with h5py.File(source, "w") as h5:
        data = h5.create_group("data")
        _write_demo(data.create_group("demo_1"), length=2)
        _write_demo(data.create_group("demo_0"), length=3)

    task = "stack the left bowl on the right bowl and place them in the tray"
    summary = convert_dataset(
        source=source,
        output=output,
        repo_id="physical-intelligence/libero90-task63",
        task=task,
        canonical_stats=canonical_stats,
        source_image_shape=(2, 3, 3),
        image_size=4,
    )

    assert summary == {"episodes": 2, "frames": 5}
    assert output.is_dir()
    assert not output.with_name("task63.partial").exists()
    assert (
        output / "meta/stats.json"
    ).read_bytes() == canonical_stats.read_bytes()
    assert (output / "meta/stats.task_local.json").is_file()

    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.0"
    assert info["robot_type"] == "panda"
    assert info["fps"] == 10
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 5
    assert info["total_tasks"] == 1
    assert info["features"]["image"]["shape"] == [4, 4, 3]
    assert info["features"]["state"]["shape"] == [8]
    assert info["features"]["actions"]["shape"] == [7]

    task_record = json.loads(
        (output / "meta/tasks.jsonl").read_text(encoding="utf-8")
    )
    assert task_record == {"task_index": 0, "task": task}

    dataset = LeRobotDataset(
        repo_id="physical-intelligence/libero90-task63",
        root=output,
    )
    assert len(dataset) == 5
    assert dataset.num_episodes == 2
    assert dataset[0]["task"] == task
    np.testing.assert_array_equal(
        dataset[0]["state"].numpy(),
        np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dataset[0]["actions"].numpy(),
        np.arange(7, dtype=np.float32),
    )

    shore_paths = find_demo_episodes(str(output), task)
    assert len(shore_paths) == 2
    shore_demo = read_libero_demo(shore_paths[0])
    assert shore_demo["state"].shape == (3, 8)
    assert shore_demo["state"].dtype == np.float32
    assert shore_demo["action"].shape == (3, 7)
    assert shore_demo["action"].dtype == np.float32
    assert shore_demo["agentview"].shape == (3, 4, 4, 3)
    assert shore_demo["wrist"].shape == (3, 4, 4, 3)


def test_convert_dataset_refuses_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "task63"
    output.mkdir()
    canonical_stats = tmp_path / "stats.json"
    canonical_stats.write_text("{}\n", encoding="utf-8")
    with h5py.File(source, "w") as h5:
        _write_demo(h5.create_group("data").create_group("demo_0"))

    with pytest.raises(FileExistsError, match="task63"):
        convert_dataset(
            source=source,
            output=output,
            repo_id="physical-intelligence/libero90-task63",
            task="task",
            canonical_stats=canonical_stats,
            source_image_shape=(2, 3, 3),
            image_size=4,
        )


def test_convert_dataset_refuses_stale_partial_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "task63"
    output.with_name("task63.partial").mkdir()
    canonical_stats = tmp_path / "stats.json"
    canonical_stats.write_text("{}\n", encoding="utf-8")
    with h5py.File(source, "w") as h5:
        _write_demo(h5.create_group("data").create_group("demo_0"))

    with pytest.raises(FileExistsError, match="partial"):
        convert_dataset(
            source=source,
            output=output,
            repo_id="physical-intelligence/libero90-task63",
            task="task",
            canonical_stats=canonical_stats,
            source_image_shape=(2, 3, 3),
            image_size=4,
        )


def test_convert_dataset_requires_canonical_stats(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5:
        _write_demo(h5.create_group("data").create_group("demo_0"))

    with pytest.raises(FileNotFoundError, match="missing-stats"):
        convert_dataset(
            source=source,
            output=tmp_path / "task63",
            repo_id="physical-intelligence/libero90-task63",
            task="task",
            canonical_stats=tmp_path / "missing-stats.json",
            source_image_shape=(2, 3, 3),
            image_size=4,
        )

