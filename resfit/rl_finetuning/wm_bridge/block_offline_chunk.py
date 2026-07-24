from __future__ import annotations

from dataclasses import dataclass
import glob
import os
import zipfile

import cv2
import numpy as np
import pandas as pd
from tensordict import TensorDict
import torch

from resfit.rl_finetuning.wm_bridge.wm_driver import (
    CAMERA_KEYS,
    CHUNK_LENGTH,
    frames_to_obs_images,
)

_EPISODES_PER_CHUNK = 1000
ENDPOINT_CACHE_READ_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    EOFError,
    zipfile.BadZipFile,
)


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


class BlockEpisodeReader:
    def __init__(self, episode: EpisodeRef):
        self.episode = episode
        frame = pd.read_parquet(
            episode.parquet_path,
            columns=["action", "observation.state"],
        )
        self.actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        self.states = np.stack(
            frame["observation.state"].to_numpy()).astype(np.float32)[:, :16]

    def native_frames(self, frame_index: int) -> np.ndarray:
        views = [
            _read_rgb_frame(self.episode.video_paths[key], frame_index)
            for key in CAMERA_KEYS
        ]
        return np.stack(views).astype(np.float32) / 127.5 - 1.0


def _read_rgb_frame(path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    ok = False
    bgr = None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"failed to read {path} at frame {frame_index}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb, (2, 0, 1)).copy()


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} expected finite shape {shape}, got {array.shape}")
    return array


def _finite_vector(value, name):
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def collect_episode_endpoints(
    reader,
    *,
    episode_id,
    slices,
    base_policy,
):
    from resfit.rl_finetuning.wm_bridge.block_offline_cache import EndpointRecord

    indices = sorted({
        index for item in slices for index in (item.start, item.end)
    })
    bases, features, states = [], [], []
    for frame_index in indices:
        print(
            f"offline_endpoint_progress episode={episode_id} "
            f"frame={frame_index}")
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
    expert_norm_mean: float
    base_norm_mean: float
    residual_norm_mean: float
    expert_saturation_fraction: float
    base_saturation_fraction: float


def _agent_images(native_frames):
    array = np.asarray(native_frames, dtype=np.float32)
    if array.ndim != 4 or not np.all(np.isfinite(array)):
        raise ValueError("native_frames must be a finite VCHW array")
    window = torch.from_numpy(array[:, :, None])
    images = frames_to_obs_images(window, t_index=0)
    return {
        key: value[0].mul(255).round().to(torch.uint8)
        for key, value in images.items()
    }


def _endpoint_at(record, frame_index):
    positions = np.flatnonzero(record.frame_indices == frame_index)
    if len(positions) != 1:
        raise ValueError(f"endpoint {frame_index} missing or duplicated")
    position = int(positions[0])
    return (
        record.base_actions[position],
        record.prefix_features[position],
        record.proprio[position],
    )


def endpoint_record_covers_slices(record, slices):
    indices = np.asarray(record.frame_indices)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        return False
    required = {
        frame_index
        for item in slices
        for frame_index in (item.start, item.end)
    }
    return required.issubset(set(indices.tolist()))


def _finite_tensor(value, shape, name):
    tensor = torch.as_tensor(value).detach().float().cpu()
    if tuple(tensor.shape) != tuple(shape) or not bool(
        torch.isfinite(tensor).all()
    ):
        raise ValueError(
            f"{name} expected finite shape {tuple(shape)}, "
            f"got {tuple(tensor.shape)}")
    return tensor


def _replay_obs(
    native_frames,
    raw_state,
    scaled_base,
    state_standardizer,
):
    obs = _agent_images(native_frames)
    standardized = state_standardizer.standardize(
        torch.from_numpy(np.asarray(raw_state, dtype=np.float32)))
    obs["observation.state"] = _finite_tensor(
        standardized, (16,), "standardized state")
    obs["observation.base_action"] = _finite_tensor(
        scaled_base, (CHUNK_LENGTH * 16,), "scaled base action")
    obs["observation.stage_id"] = torch.tensor([0.0], dtype=torch.float32)
    return obs


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _fraction(masks):
    if not masks:
        return 0.0
    return float(np.mean(np.concatenate(masks)))


def build_block_offline_buffer(
    offline_rb,
    dataset_root,
    *,
    action_scaler,
    state_standardizer,
    image_keys,
    gamma,
    num_demos,
    base_policy,
    scorer,
    endpoint_store,
    episodes=None,
    reader_factory=BlockEpisodeReader,
    **_ignored,
) -> BuildStats:
    if tuple(image_keys) != tuple(CAMERA_KEYS):
        raise ValueError(
            f"image_keys must exactly match fixed cameras {tuple(CAMERA_KEYS)}")
    if episodes is None:
        selected = catalog_episodes(dataset_root, num_demos)
    else:
        selected = tuple(episodes)
        if num_demos is not None:
            selected = selected[:num_demos]

    skipped = 0
    endpoint_hits = 0
    endpoint_misses = 0
    transition_count = 0
    rewards = []
    potential_deltas = []
    expert_norms = []
    base_norms = []
    residual_norms = []
    expert_saturation = []
    base_saturation = []

    for episode in selected:
        slices = plan_episode_chunks(episode.num_frames)
        if not slices:
            skipped += 1
            continue

        reader = reader_factory(episode=episode)
        try:
            endpoint_record = endpoint_store.load_episode(episode.episode_id)
        except ENDPOINT_CACHE_READ_ERRORS as exc:
            print(
                f"offline_endpoint_cache_invalid episode={episode.episode_id} "
                f"error={exc}")
            endpoint_record = None
        if endpoint_record is not None and not endpoint_record_covers_slices(
            endpoint_record, slices
        ):
            print(
                f"offline_endpoint_cache_invalid episode={episode.episode_id} "
                "error=missing_or_duplicate_required_endpoints")
            endpoint_record = None
        if endpoint_record is None:
            endpoint_misses += 1
            endpoint_record = collect_episode_endpoints(
                reader,
                episode_id=episode.episode_id,
                slices=slices,
                base_policy=base_policy,
            )
            endpoint_store.save_episode(episode.episode_id, endpoint_record)
        else:
            endpoint_hits += 1

        for item in slices:
            print(
                f"offline_build_progress episode={episode.episode_id} "
                f"frame={item.start}->{item.end}")
            base, feature, raw_state = _endpoint_at(
                endpoint_record, item.start)
            next_base, next_feature, next_raw_state = _endpoint_at(
                endpoint_record, item.end)
            base = _finite_array(base, (CHUNK_LENGTH, 16), "base_action")
            next_base = _finite_array(
                next_base, (CHUNK_LENGTH, 16), "next_base_action")
            feature = _finite_vector(feature, "prefix_feature")
            next_feature = _finite_vector(
                next_feature, "next_prefix_feature")
            raw_state = _finite_array(raw_state, (16,), "proprio")
            next_raw_state = _finite_array(
                next_raw_state, (16,), "next_proprio")
            expert = _finite_array(
                reader.actions[item.start:item.end],
                (CHUNK_LENGTH, 16),
                "expert_action",
            )

            expert_scaled = _finite_tensor(
                action_scaler.scale(torch.from_numpy(expert)).reshape(-1),
                (CHUNK_LENGTH * 16,),
                "scaled expert action",
            )
            base_scaled = _finite_tensor(
                action_scaler.scale(torch.from_numpy(base)).reshape(-1),
                (CHUNK_LENGTH * 16,),
                "scaled base action",
            )
            next_base_scaled = _finite_tensor(
                action_scaler.scale(torch.from_numpy(next_base)).reshape(-1),
                (CHUNK_LENGTH * 16,),
                "scaled next base action",
            )
            current_obs = _replay_obs(
                reader.native_frames(item.start),
                raw_state,
                base_scaled,
                state_standardizer,
            )
            next_obs = _replay_obs(
                reader.native_frames(item.end),
                next_raw_state,
                next_base_scaled,
                state_standardizer,
            )

            phi_current = float(scorer.phi(feature, raw_state))
            phi_next = float(scorer.phi(next_feature, next_raw_state))
            reward = float(gamma) * phi_next - phi_current
            if not np.all(np.isfinite([phi_current, phi_next, reward])):
                raise ValueError("potential and reward must be finite")
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            if not bool(torch.isfinite(reward_tensor)):
                raise ValueError("reward must be finite float32")

            transition = TensorDict({
                "obs": TensorDict(current_obs, batch_size=[]),
                "next": TensorDict({
                    "obs": TensorDict(next_obs, batch_size=[]),
                    "done": torch.tensor(item.terminal, dtype=torch.bool),
                    "reward": reward_tensor,
                }, batch_size=[]),
                "action": expert_scaled,
                "max_stage": torch.tensor(0.0),
                "_priority": torch.tensor(10.0),
            }, batch_size=[]).unsqueeze(0)
            offline_rb.add(transition)

            transition_count += 1
            rewards.append(float(reward_tensor.item()))
            potential_deltas.append(phi_next - phi_current)
            expert_norms.append(float(torch.linalg.vector_norm(
                expert_scaled).item()))
            base_norms.append(float(torch.linalg.vector_norm(
                base_scaled).item()))
            residual_norms.append(float(torch.linalg.vector_norm(
                expert_scaled - base_scaled).item()))
            expert_saturation.append(
                expert_scaled.abs().ge(0.999).numpy())
            base_saturation.append(base_scaled.abs().ge(0.999).numpy())

    reward_mean, reward_std = _mean_std(rewards)
    delta_mean, delta_std = _mean_std(potential_deltas)
    expert_mean, _ = _mean_std(expert_norms)
    base_mean, _ = _mean_std(base_norms)
    residual_mean, _ = _mean_std(residual_norms)
    return BuildStats(
        episodes=len(selected),
        skipped_short_episodes=skipped,
        transitions=transition_count,
        endpoint_hits=endpoint_hits,
        endpoint_misses=endpoint_misses,
        reward_mean=reward_mean,
        reward_std=reward_std,
        potential_delta_mean=delta_mean,
        potential_delta_std=delta_std,
        expert_norm_mean=expert_mean,
        base_norm_mean=base_mean,
        residual_norm_mean=residual_mean,
        expert_saturation_fraction=_fraction(expert_saturation),
        base_saturation_fraction=_fraction(base_saturation),
    )


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
        ),
        key=lambda path: int(
            os.path.basename(path).removeprefix("episode_").removesuffix(".parquet")
        ),
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
