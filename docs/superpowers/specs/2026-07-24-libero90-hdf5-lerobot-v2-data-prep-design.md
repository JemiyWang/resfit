# LIBERO-90 HDF5 to LeRobot v2.0 Data Preparation Design

## Goal

Prepare isolated, training-ready LeRobot v2.0 datasets for LIBERO-90 tasks
57, 60, 63, and 64 without modifying the existing 40-task
`physical-intelligence/libero` dataset.

The four tasks are:

| Task ID | Language |
|---|---|
| 57 | `pick up the cream cheese and put it in the tray` |
| 60 | `pick up the black bowl on the left and put it in the tray` |
| 63 | `stack the left bowl on the right bowl and place them in the tray` |
| 64 | `stack the right bowl on the left bowl and place them in the tray` |

## Source and Version Pinning

All raw files come from the official Hugging Face dataset
`yifengzhu-hf/LIBERO-datasets`, pinned to revision
`cf16e484f5a2a556f3e7e5ad8ec3d97d3dcfe498`.

| Task | HDF5 basename | Bytes | SHA-256 |
|---|---|---:|---|
| 57 | `LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray_demo.hdf5` | 768466210 | `1702b5b6c81385edfe6e801372e20c5e16d1774faf030f5c0f67f159bd1eeead` |
| 60 | `LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_demo.hdf5` | 617164176 | `1c6bd7024ecbac964971669e18425f61da135663b0b2a7ade58c02dd5bce7e93` |
| 63 | `LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo.hdf5` | 1073326526 | `6e5dd6cb435e94cd90886e2ae3a5f7145d3034710d7c1605bad1c97d56acb2b3` |
| 64 | `LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray_demo.hdf5` | 1168927388 | `85012c80aa040f1d35a93b2ffedf30136d794693dac6ac56f24abed247d7d972` |

Downloads use a `.partial` suffix, resume in place, and become visible as
`.hdf5` only after both the byte count and SHA-256 match the manifest.

## Isolation and Directory Layout

The existing dataset remains unchanged:

```text
/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/
├── data/                       # existing 1693 episodes; never modified
├── meta/                       # existing 40-task metadata; never modified
├── raw/libero_90/              # pinned official HDF5 inputs
└── converted/libero_90/
    ├── task57/                 # independent LeRobot v2.0 dataset
    ├── task60/
    ├── task63/
    └── task64/
```

Each converted task directory is a complete dataset root with its own
`data/` and `meta/`. Conversion writes to a sibling `.partial` directory and
renames it only after validation. Existing completed outputs are refused
unless the caller explicitly requests overwrite.

## Conversion Semantics

The source HDF5 contains 50 successful demonstrations under
`data/demo_0` through `data/demo_49`. No frames are dropped or temporally
downsampled.

For every frame:

- `image`: source `obs/agentview_rgb`.
- `wrist_image`: source `obs/eye_in_hand_rgb`.
- Both source images are rotated 180 degrees with `[::-1, ::-1]`, matching
  OpenPI's LIBERO online preprocessing.
- Both images are resized from 128x128 to 256x256 with bilinear
  interpolation, matching the existing dataset feature schema.
- `state`: concatenate `obs/ee_states` (6) and
  `obs/gripper_states` (2), yielding float32 shape `(8,)`.
- `actions`: source `actions`, converted to float32 shape `(7,)`.
- The exact benchmark language is attached to every frame and episode.

LeRobot-generated bookkeeping fields (`timestamp`, `frame_index`,
`episode_index`, `index`, and `task_index`) use the standard v2.0 writer.
Metadata uses `robot_type="panda"` and `fps=10`, matching the existing
dataset. Each isolated dataset contains one local task with `task_index=0`;
ResFiT selects demonstrations by the exact language string rather than by
this local index.

## Statistics

The converter preserves the task-local statistics produced by LeRobot as
`meta/stats.task_local.json`. It then installs an exact copy of the existing
40-task `meta/stats.json` as the converted dataset's `meta/stats.json`.

This is intentional: SHORE's current LIBERO recipe uses the frozen
`pi0_libero` base and its established global state/action normalization.
Using task-local statistics would silently change action scaling relative to
the paper's task8 recipe. The copied global file is verified byte-for-byte.

## Implementation Boundary

Add one focused converter:

`resfit/lerobot/dataset/convert_libero_hdf5_to_lerobot_v2.py`

It accepts one source HDF5, task ID, exact language, output directory, and
canonical stats path. It performs source validation, conversion, metadata
validation, statistics preservation, and atomic publication. Downloading is
performed with pinned `curl` commands so partial transfers can be resumed and
verified against the manifest.

No training code, LIBERO environment code, existing dataset metadata, or
base-policy checkpoint is changed.

## Validation

Automated tests use a small synthetic HDF5 and verify:

1. Numeric ordering of `demo_N` keys.
2. Exact state concatenation and float32 action conversion.
3. 180-degree image rotation and 256x256 output shape.
4. Exact language propagation into `episodes.jsonl`.
5. LeRobot v2.0 feature names and shapes.
6. Preservation of task-local statistics and exact installation of canonical
   global statistics.
7. Refusal of a source whose byte count or SHA-256 does not match.

Each real converted dataset must then pass:

- 50 episodes in `meta/episodes.jsonl`.
- The expected language on every episode.
- Source and converted frame counts agree.
- All referenced parquet files exist and can be read.
- One decoded episode has `state (T,8)`, `actions (T,7)`, and two
  `uint8 (T,256,256,3)` image streams.
- `meta/info.json` reports `codebase_version: "v2.0"`.
- `meta/stats.json` hashes identically to the canonical existing file.

