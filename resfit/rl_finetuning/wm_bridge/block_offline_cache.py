from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence
import uuid

import numpy as np

from resfit.rl_finetuning.wm_bridge.block_offline_chunk import EpisodeRef


ENDPOINT_SCHEMA = 1
REPLAY_SCHEMA = 1


@dataclass(frozen=True)
class EndpointRecord:
    frame_indices: np.ndarray
    base_actions: np.ndarray
    prefix_features: np.ndarray
    proprio: np.ndarray


def _stable_sha(payload: dict) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()
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
    if not episodes:
        raise ValueError("dataset manifest requires at least one episode")
    return {
        "root": os.path.realpath(episodes[0].root),
        "episodes": [
            {
                "id": episode.episode_id,
                "frames": episode.num_frames,
                "parquet": _file_identity(
                    episode.parquet_path, hash_content=True),
                "videos": {
                    key: _file_identity(path, hash_content=False)
                    for key, path in sorted(episode.video_paths.items())
                },
            }
            for episode in episodes
        ],
    }


def endpoint_fingerprint(
    *,
    manifest,
    pi0_serve_ckpt_id,
    prompt,
    camera_keys,
    chunk_length=50,
    stride=50,
):
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


def replay_fingerprint(
    *,
    endpoint_fp,
    value_sha256,
    gamma,
    num_demos,
    stats_sha256,
    action_scale,
    min_range_per_dim,
    image_size,
    image_keys,
    n_step,
):
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


def _validate_record(record: EndpointRecord) -> None:
    n = len(record.frame_indices)
    valid = (
        record.frame_indices.ndim == 1
        and np.issubdtype(record.frame_indices.dtype, np.integer)
        and np.all(np.isfinite(record.frame_indices))
        and record.base_actions.shape == (n, 50, 16)
        and record.prefix_features.ndim == 2
        and record.prefix_features.shape[0] == n
        and record.proprio.shape == (n, 16)
        and all(np.all(np.isfinite(value)) for value in (
            record.base_actions,
            record.prefix_features,
            record.proprio,
        ))
    )
    if not valid:
        raise ValueError("invalid endpoint cache record")


class EndpointStore:
    def __init__(
        self,
        root: str,
        fingerprint: str,
        force_rebuild: bool = False,
    ):
        self.directory = Path(root) / "endpoints" / fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        self.force_rebuild = force_rebuild

    def _path(self, episode_id: int) -> Path:
        return self.directory / f"episode_{episode_id:06d}.npz"

    def load_episode(self, episode_id: int) -> EndpointRecord | None:
        path = self._path(episode_id)
        if self.force_rebuild or not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            record = EndpointRecord(
                data["frame_indices"],
                data["base_actions"],
                data["prefix_features"],
                data["proprio"],
            )
        _validate_record(record)
        return record

    def save_episode(
        self,
        episode_id: int,
        record: EndpointRecord,
    ) -> None:
        _validate_record(record)
        final_path = self._path(episode_id)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.directory,
                prefix=f"{final_path.stem}.",
                suffix=".npz.tmp",
                delete=False,
            ) as stream:
                temp_path = Path(stream.name)
                np.savez_compressed(
                    stream,
                    frame_indices=record.frame_indices,
                    base_actions=record.base_actions,
                    prefix_features=record.prefix_features,
                    proprio=record.proprio,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, final_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def resolve_replay_generation(
    cache_root: str,
    replay_fp: str,
    force_rebuild: bool,
) -> str:
    parent = Path(cache_root) / "replay" / replay_fp
    parent.mkdir(parents=True, exist_ok=True)
    complete = sorted(
        path
        for path in parent.glob("gen-*")
        if (path / "buffer_meta.json").is_file()
        and (path / "bridge_meta.json").is_file()
    )
    if complete and not force_rebuild:
        return str(complete[-1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    created = parent / f"gen-{stamp}-{uuid.uuid4().hex[:8]}"
    created.mkdir()
    return str(created)
