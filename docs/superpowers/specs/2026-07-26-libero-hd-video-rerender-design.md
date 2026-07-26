# LIBERO HD Task Video Rerender Design

## Goal

Replace the existing LIBERO-10 Task 8 and LIBERO-90 Task 57 task-demo
videos with clearer versions suitable for paper figures.

## Inputs

- LIBERO-10 Task 8:
  `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5`
- LIBERO-90 Task 57:
  `LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray_demo.hdf5`

Both inputs contain full MuJoCo simulator states. The renderer will reconstruct
each recorded state instead of enlarging the existing 256×256 raster frames.

## Output

- Render the `agentview` camera natively at 512×512.
- Preserve the dataset playback rate of 10 FPS.
- Encode as H.264 with `yuv420p` pixel format.
- Keep the current output paths:
  - `paper/task_videos/libero10_task8_moka_pots.mp4`
  - `paper/task_videos/libero90_task57_cream_cheese.mp4`

## Replacement Safety

Each video will first be written to a sibling temporary path. The existing
video will be replaced only after the new file:

1. decodes successfully;
2. reports 512×512 resolution, 10 FPS, H.264, and `yuv420p`;
3. contains the same number of frames as its selected HDF5 demonstration; and
4. passes visual inspection of representative beginning, middle, and ending
   frames.

The process will not use interpolation or generative super-resolution and will
not alter task geometry, objects, robot pose, or camera viewpoint.

