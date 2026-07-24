# Block Offline Replay Sparse Decode Design

**Date:** 2026-07-24

## Context

`block_shore_mixed50_seed0` must build 6,989 offline chunk transitions
before RL updates begin. The first production run processed roughly 900
transitions in 35 minutes, or about 2.3 seconds per transition.

The bottleneck is `BlockEpisodeReader.native_frames()`. Each requested frame
currently opens three MP4 files, seeks to the requested index, decodes one
frame, and closes the files. Endpoint collection reads one three-camera frame
for each unique chunk endpoint. Replay construction then reads both endpoints
again for every transition. This produces roughly nine MP4 open/seek/decode
operations per transition, or about 63,000 operations for the full dataset.

Measured on the production dataset:

- one three-camera OpenCV frame read takes 0.25-1.08 seconds;
- one pi0 endpoint query takes about 0.255 seconds;
- GPU 7 is mostly idle because this phase is CPU/video-I/O bound.

The repository already uses TorchCodec batch decoding for TeleAvatar data, but
the new block offline replay reader does not reuse that path.

## Goals

1. Decode every required image frame at most once per episode.
2. Reuse the same decoded pixels in endpoint collection and replay creation.
3. Preserve the exact transition count, tensor schema, actions, features,
   rewards, and terminal semantics.
4. Reuse valid endpoint cache files produced by the interrupted run.
5. Achieve at least a 3x wall-clock speedup on representative real episodes.
6. Keep memory bounded to the sparse frames of the current episode.

## Non-goals

- Parallelizing pi0 inference.
- Changing chunk boundaries, reward definitions, image resolution, or
  normalization.
- Changing the endpoint or replay fingerprint for a performance-only,
  pixel-equivalent decoder change.
- Reusing the incomplete in-memory replay buffer from the interrupted process.
- Optimizing online imagination or RL updates.

## Considered Approaches

### 1. Persistent OpenCV handles

Keep one `cv2.VideoCapture` per camera open for the duration of an episode.
This removes repeated open/close overhead but retains random seeking and
duplicate endpoint decoding. It is a low-risk fallback, not the primary
design.

### 2. Sparse TorchCodec batch decode

Compute all required endpoint indices for an episode, then call
`VideoDecoder.get_frames_at(indices)` once per camera. Store only those frames
and serve every later lookup from memory.

This is the recommended approach. It directly removes the measured
bottleneck, uses an existing repository dependency, and requires only tens of
megabytes per episode rather than caching the full video.

### 3. Multi-process episode construction

Build several episodes concurrently and serialize results into the replay
buffer. This complicates ordering, cache writes, pi0 client access, and replay
buffer ownership. Once video decoding is fixed, sequential pi0 inference is
expected to become the dominant lower bound, so this approach is deferred.

## Architecture

### Sparse decoder

Add a reusable sparse TeleAvatar frame reader alongside
`read_teleavatar_episode_batched()`:

```python
read_teleavatar_episode_frames(
    root,
    episode_id,
    cameras,
    frame_indices,
) -> dict[str, torch.Tensor]
```

The function will:

1. require sorted, unique, non-negative integer indices;
2. open each camera video once with TorchCodec;
3. validate that the largest requested index exists in every camera;
4. call `get_frames_at(frame_indices)` once per camera;
5. return camera tensors in the same order as `frame_indices`;
6. reject missing, duplicate, misaligned, or non-`uint8` output.

The existing full-episode reader remains unchanged for callers that need every
frame.

### Episode-local frame cache

`BlockEpisodeReader` will gain a preparation operation that accepts the
required frame indices. It will convert the three camera tensors into the
existing `VCHW float32 [-1, 1]` representation once and retain a mapping from
frame index to frame array.

`native_frames(frame_index)` will become a pure cache lookup. A missing index
will be an error rather than silently triggering another decode.

The cache lives only as long as the current `BlockEpisodeReader`. When the
episode finishes, its frames are released before the next episode is decoded.
Typical episodes need about 20-35 endpoint frames, so the cache remains in the
tens-of-megabytes range.

### Offline build flow

For each episode:

1. call `plan_episode_chunks()`;
2. derive the sorted union of every chunk's start and end index;
3. construct the reader and prepare those frames once;
4. load the endpoint cache;
5. on an endpoint miss, query pi0 using the prepared frames and persist the
   endpoint record atomically;
6. build every replay transition using the same prepared frames;
7. release the reader and continue to the next episode.

Endpoint cache hits skip pi0 queries exactly as they do today. They still
decode the sparse image set because replay observations contain images.

## Endpoint Cache Compatibility

The current endpoint fingerprint does not include the decoder backend. Reusing
existing endpoint files is safe only if the new decoder supplies identical
pixels to the old OpenCV path.

Before restarting the production run, compare OpenCV and TorchCodec output on
at least:

- three episodes of different lengths;
- all three cameras;
- the first, middle, final, aligned, and unaligned-terminal endpoints.

The required criterion is exact `uint8` equality after matching CHW/RGB
layout. Shape equality or visual similarity alone is insufficient.

If TorchCodec is not pixel-identical, do not silently combine old endpoint
features with new replay pixels. Fall back to a single-pass OpenCV episode
cache and run the same equality check. If no optimized path is pixel-identical,
stop and request approval to invalidate and rebuild the endpoint cache with a
new schema fingerprint.

## Error Handling

- Reject out-of-range indices before constructing transitions.
- Treat decoder errors, truncated camera streams, and camera length mismatch
  as episode build failures with the episode, camera, and frame index in the
  message.
- Never write an endpoint cache record until every required endpoint has been
  computed and validated.
- Preserve the existing atomic endpoint `.npz` write behavior.
- Do not fall back to per-frame decoding implicitly; fallback selection must
  happen once during startup and be logged.
- Log decoder backend, number of requested sparse frames, decode duration, pi0
  duration, and replay-build duration per episode.

## Verification

### Unit tests

1. Sparse decoder receives sorted unique endpoint indices and calls one batch
   decode per camera.
2. Repeated `native_frames()` lookups cause no additional decoder calls.
3. Frame order, RGB/CHW layout, dtype, and `[-1, 1]` conversion are correct.
4. Duplicate, negative, missing, and out-of-range indices fail clearly.
5. Endpoint-cache hits issue zero pi0 queries.
6. Existing aligned and unaligned terminal chunk tests remain unchanged.

### Real-data regression

On three representative production episodes:

1. compare old and new decoded frames exactly;
2. compare endpoint records exactly or within the existing floating-point
   tolerances;
3. compare all replay TensorDict fields;
4. verify identical transition counts and done flags;
5. verify no NaN or infinite value is introduced.

### Performance gate

Benchmark the old and new readers on the same three episodes with a fake pi0
base so video work is isolated. The optimized path must:

- decode each required frame once;
- be at least 3x faster end-to-end;
- stay below 512 MiB of additional resident memory.

The production restart must show at least 100 transitions per minute over a
ten-minute window, unless pi0 service time is independently shown to dominate.

## Operational Rollout

1. Record the current process ID, endpoint cache directory, completed endpoint
   file count, and last completed episode.
2. Gracefully interrupt only the seed-0 training wrapper and trainer. Keep the
   WM, pi0, and advantage services running.
3. Do not delete or rewrite existing endpoint cache files.
4. Implement and run unit, regression, equivalence, and performance checks.
5. Restart the exact same seed-0 launch command.
6. Confirm startup reports a partial endpoint cache and that cached episodes
   produce endpoint hits with zero pi0 calls.
7. Monitor the first ten minutes for transition throughput, memory growth,
   decoder errors, and GPU/service health.
8. Continue into RL training only after all 6,989 transitions are built and
   the replay build statistics are complete.

The incomplete replay generation from the interrupted process is not reused;
those transitions are rebuilt through the faster path. Its endpoint cache is
reused.

## Expected Result

Sequential pi0 calls impose a lower bound of roughly 30 minutes for a cold
6,989-transition build. Because the interrupted run has already cached a
subset of episodes, the restarted preprocessing phase is expected to complete
in roughly 30-60 minutes. This estimate is a target, not an acceptance
criterion; correctness and the measured 3x speedup gate take precedence.
