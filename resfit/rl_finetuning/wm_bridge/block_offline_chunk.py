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


def plan_episode_chunks(
    num_frames: int,
    chunk_length: int = CHUNK_LENGTH,
) -> tuple[ChunkSlice, ...]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if num_frames < chunk_length + 1:
        return ()
    terminal_start = num_frames - chunk_length - 1
    starts = list(range(0, terminal_start + 1, chunk_length))
    if starts[-1] != terminal_start:
        starts.append(terminal_start)
    return tuple(
        ChunkSlice(
            start=start,
            end=start + chunk_length,
            terminal=(start == terminal_start),
        )
        for start in starts
    )


def _episode_paths(root: str, episode_id: int) -> tuple[str, dict[str, str]]:
    chunk_id = episode_id // _EPISODES_PER_CHUNK
    parquet = os.path.join(
        root, f"data/chunk-{chunk_id:03d}/episode_{episode_id:06d}.parquet"
    )
    videos = {
        key: os.path.join(
            root,
            f"videos/chunk-{chunk_id:03d}/{key}/episode_{episode_id:06d}.mp4",
        )
        for key in CAMERA_KEYS
    }
    return parquet, videos


def catalog_episodes(
    dataset_root: str,
    num_demos: int | None = None,
) -> tuple[EpisodeRef, ...]:
    paths = sorted(
        glob.glob(
            os.path.join(dataset_root, "data", "chunk-*", "episode_*.parquet")
        )
    )
    if num_demos is not None:
        paths = paths[:num_demos]
    result = []
    for parquet_path in paths:
        episode_id = int(
            os.path.basename(parquet_path)
            .removeprefix("episode_")
            .removesuffix(".parquet")
        )
        expected_parquet, videos = _episode_paths(dataset_root, episode_id)
        if os.path.abspath(parquet_path) != os.path.abspath(expected_parquet):
            raise ValueError(f"unexpected parquet layout: {parquet_path}")
        num_frames = len(pd.read_parquet(parquet_path, columns=["frame_index"]))
        result.append(
            EpisodeRef(dataset_root, episode_id, parquet_path, videos, num_frames)
        )
    if not result:
        raise ValueError(f"no episodes found under {dataset_root}")
    return tuple(result)


def count_block_chunk_transitions(
    dataset_root: str,
    num_demos: int | None = None,
) -> int:
    return sum(
        len(plan_episode_chunks(episode.num_frames))
        for episode in catalog_episodes(dataset_root, num_demos)
    )
