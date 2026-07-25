# LIBERO-90 Task 57 Multi-Seed Run Design

## Goal

Stop the running LIBERO-90 task 60, 63, and 64 experiments, keep task 57
seed 0 running on GPU2, and start two additional task 57 experiments with
seeds 1 and 2.

## Run Mapping

- Existing task 57 seed 0 remains unchanged on GPU2.
- New task 57 seed 1 runs on GPU3.
- New task 57 seed 2 runs on GPU4.
- GPU5 remains unused after task 64 stops.

## Configuration Invariants

The seed 1 and seed 2 commands must match the live task 57 seed 0 command in
all training parameters except:

- `CUDA_VISIBLE_DEVICES`;
- `MUJOCO_EGL_DEVICE_ID`;
- `--seed`;
- `--output_dir`;
- `--wandb_name`.

Both new runs reuse the task 57 data, statistics, PI0 feature cache, offline
replay cache, goal-conditioned value checkpoint, and high-level actor
checkpoint. These shared inputs are read-only during training.

## Output Isolation

The new runs use distinct names so they cannot overwrite seed 0 or each other:

- seed 1 output:
  `outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_seed1`
- seed 2 output:
  `outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_seed2`
- seed 1 W&B name:
  `libero90_task57_pi0feat_bp_bc01_h10_seed1`
- seed 2 W&B name:
  `libero90_task57_pi0feat_bp_bc01_h10_seed2`

Their console logs use matching seed-specific filenames under
`logs/libero90_shore/`.

## Stop and Start Procedure

1. Send `Ctrl-C` to tmux sessions `libero90_shore_t60`,
   `libero90_shore_t63`, and `libero90_shore_t64`.
2. Verify the three trainer process trees have exited and GPU3-GPU5 memory is
   released. Preserve all existing outputs and checkpoints.
3. Start task 57 seed 1 in tmux session `libero90_shore_t57_s1` on GPU3.
4. Start task 57 seed 2 in tmux session `libero90_shore_t57_s2` on GPU4.
5. Leave GPU5 free.

## Verification

For each new run:

- the tmux pane is alive;
- the trainer command contains task ID 57 and the intended seed;
- every non-seed training parameter matches the seed 0 command;
- the output directory and W&B name are seed-specific;
- GPU assignment and EGL device assignment agree;
- the log reaches offline-cache loading and training initialization without a
  traceback, OOM, missing asset, or port error.

The existing task 57 seed 0 process and the shared PI0 service on port 8000
must remain alive throughout the operation.
