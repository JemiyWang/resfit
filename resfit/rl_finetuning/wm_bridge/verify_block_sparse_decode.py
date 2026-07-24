from __future__ import annotations

import argparse
import json
import multiprocessing
import resource
import time

import numpy as np

from resfit.rl_finetuning.wm_bridge.block_offline_chunk import (
    BlockEpisodeReader,
    CAMERA_KEYS,
    catalog_episodes,
    plan_episode_chunks,
    read_rgb_frame_opencv,
    read_teleavatar_episode_frames_opencv_sequential,
)


def required_indices(episode):
    return sorted({
        index
        for item in plan_episode_chunks(episode.num_frames)
        for index in (item.start, item.end)
    })


def _decode_legacy(episode, indices):
    return {
        camera: np.stack([
            read_rgb_frame_opencv(
                episode.video_paths[camera], frame_index)
            for frame_index in indices
        ])
        for camera in CAMERA_KEYS
    }


def _decode_sparse(episode, indices):
    return read_teleavatar_episode_frames_opencv_sequential(
        episode.root,
        episode.episode_id,
        list(CAMERA_KEYS),
        indices,
    )


def _max_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _reader_peak_worker(episode, sender):
    try:
        before_mib = _max_rss_mib()
        reader = BlockEpisodeReader(episode)
        indices = required_indices(episode)
        reader.prepare_native_frames(indices)
        for frame_index in indices:
            reader.native_frames(frame_index)
        after_mib = _max_rss_mib()
        sender.send(("ok", max(0.0, after_mib - before_mib)))
    except Exception as exc:
        sender.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        sender.close()


def measure_reader_peak_mib(episode):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_reader_peak_worker,
        args=(episode, sender),
    )
    process.start()
    sender.close()
    process.join()
    try:
        if process.exitcode != 0:
            raise RuntimeError(
                f"reader memory child exited with code {process.exitcode}")
        if not receiver.poll():
            raise RuntimeError("reader memory child returned no result")
        status, payload = receiver.recv()
        if status != "ok":
            raise RuntimeError(f"reader memory child failed: {payload}")
        return float(payload)
    finally:
        receiver.close()
        process.close()


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def verify_episode(
    episode,
    *,
    legacy_decoder=None,
    sparse_decoder=None,
    peak_measure=None,
    clock=None,
):
    indices = required_indices(episode)
    legacy_decoder = legacy_decoder or _decode_legacy
    sparse_decoder = sparse_decoder or _decode_sparse
    peak_measure = peak_measure or measure_reader_peak_mib
    clock = clock or time.perf_counter

    legacy_decoder(episode, indices)
    sparse_decoder(episode, indices)

    sparse_started = clock()
    sparse = sparse_decoder(episode, indices)
    sparse_seconds = clock() - sparse_started

    legacy_started = clock()
    legacy = legacy_decoder(episode, indices)
    legacy_seconds = clock() - legacy_started

    pixel_equal = all(np.array_equal(
        legacy[camera],
        _numpy(sparse[camera]),
    ) for camera in CAMERA_KEYS)
    return {
        "episode": episode.episode_id,
        "decoder_backend": "opencv_sequential_sparse",
        "frames": len(indices),
        "pixel_equal": bool(pixel_equal),
        "legacy_seconds": legacy_seconds,
        "sparse_seconds": sparse_seconds,
        "peak_extra_mib": peak_measure(episode),
    }


def check_report(report):
    pixel_equal = report["pixel_equal"]
    if not isinstance(pixel_equal, (bool, np.bool_)):
        raise RuntimeError("pixel_equal must be boolean")
    if not pixel_equal:
        raise RuntimeError("pixel equality gate failed")

    metrics = {}
    for name in ("legacy_seconds", "sparse_seconds", "peak_extra_mib"):
        try:
            value = float(report[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{name} must be finite") from exc
        if not np.isfinite(value):
            raise RuntimeError(f"{name} must be finite")
        metrics[name] = value
    if metrics["legacy_seconds"] <= 0.0:
        raise RuntimeError("legacy_seconds must be positive")
    if metrics["sparse_seconds"] <= 0.0:
        raise RuntimeError("sparse_seconds must be positive")
    if metrics["peak_extra_mib"] < 0.0:
        raise RuntimeError("peak_extra_mib must be non-negative")

    speedup = metrics["legacy_seconds"] / metrics["sparse_seconds"]
    if speedup < 3.0:
        raise RuntimeError(f"speedup {speedup:.3f} is below 3.0x")
    if metrics["peak_extra_mib"] >= 512.0:
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
        "decoder_backend": "opencv_sequential_sparse",
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
