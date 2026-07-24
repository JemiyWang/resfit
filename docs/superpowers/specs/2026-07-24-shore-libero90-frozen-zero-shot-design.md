# Frozen SHORE Zero-Shot Evaluation on LIBERO-90

Date: 2026-07-24

## Objective

Evaluate one previously trained SHORE policy on four unseen LIBERO-90 tasks
without using LIBERO-90 demonstrations and without updating any model
parameters. Run one task per GPU, preserve per-episode results, and produce a
four-panel success-rate figure.

This experiment measures frozen-weight cross-task transfer. It is not an online
adaptation experiment and its horizontal axis is evaluation episode, not
environment training steps.

## Frozen source bundle

All four tasks use exactly the same source-task bundle:

- Base policy:
  `/mnt/mnt/data/chj/openpi/checkpoints/pi0_libero/pi0_libero_is-dceq77jzghdxvjj2-devmachine-0_20260523_220232/29999`
- Residual SHORE agent:
  `/mnt/mnt/data/resfit/outputs_chunk/libero10_task8_pi0feat_bp_bc01_h10/best.pt`
- Goal-conditioned value:
  `/mnt/mnt/data/resfit/outputs_chunk/libero10_t8moka_pi0_feat_gc_value.pt`
- High-level actor:
  `/mnt/mnt/data/resfit/outputs_chunk/libero10_t8moka_pi0_feat_high_actor.pt`
- Source-task PI0 feature cache:
  `/mnt/mnt/data/resfit/outputs_chunk/libero10_t8moka_pi0_feat.npz`

The task8 feature cache remains the source of the representative goal. No
feature cache, goal, value model, high-level actor, or residual policy is
recomputed from LIBERO-90 data.

## Evaluation tasks

Use suite `libero_90` with the following zero-based task IDs:

| Task ID | Instruction |
| --- | --- |
| 3 | Put the butter at the back in the top drawer of the cabinet and close it |
| 8 | Open the top drawer of the cabinet and put the bowl in it |
| 21 | Turn on the stove and put the frying pan on it |
| 63 | Stack the left bowl on the right bowl and place them in the tray |

The PI0 prompt changes to the current task instruction. Every SHORE parameter
and source-task asset remains unchanged.

## Non-goals

- Do not download or read LIBERO-90 demonstrations.
- Do not train, fine-tune, adapt, relabel, or update replay buffers.
- Do not run a PI0-only baseline in this first evaluation.
- Do not overwrite the existing DSRL figure or its data.
- Do not claim the resulting curve is a learning curve.

## Architecture

### Eval-only driver

Add a dedicated frozen evaluation entry point rather than routing evaluation
through `train_chunk_residual.py`. The driver will:

1. resolve and validate the task and all source-bundle paths;
2. connect to the task-local PI0 feature server;
3. reconstruct the SHORE agent from the saved run configuration;
4. load the residual checkpoint strictly;
5. load the source-task GC value, high actor, and representative goal;
6. freeze all modules and run them under inference mode;
7. construct the LIBERO-90 environment for the current task;
8. execute exactly 50 episodes;
9. append one durable record after each completed episode; and
10. write a final summary.

The driver must not create a replay buffer, load optimizer state, initialize
Weights & Biases, or call any update method.

### Inference-only QAgent construction

`QAgent` currently creates three optimizers unconditionally. Add a
backward-compatible `inference_only=False` constructor flag. When true, it
builds the same neural modules but skips optimizers and learning-rate
schedulers. Training behavior and checkpoint shapes remain unchanged when the
flag is omitted.

After strict checkpoint loading, the evaluator sets every parameter's
`requires_grad` to false and verifies this invariant before the first rollout.

### Existing rollout core

Extend `run_libero_evaluation` with backward-compatible optional episode
recording. Existing training callers retain their current aggregate return
value and mode-restoration behavior. The frozen driver requests:

- per-episode success, return, length, and environment slot;
- no restoration to training mode; and
- a callback invoked after each episode for durable JSONL writes.

Residual actions use `eval_mode=True` and `stddev=0.0` under
`torch.inference_mode()`.

### Fixed initial-state coverage

Current `LiberoGymWrapper` repeats one fixed init state. Add an opt-in,
deterministic init-state schedule for frozen evaluation while preserving the
current default behavior for training.

Across the vector environments, reset order must cover init-state IDs 0 through
49 exactly once. The episode record stores the actual init-state ID. The
evaluator rejects a completed run if an ID is missing or duplicated.

## Four-GPU execution

Run one independent process group per task:

| GPU | Task | PI0 port |
| --- | --- | --- |
| 0 | 3 | 8000 |
| 1 | 8 | 8001 |
| 2 | 21 | 8002 |
| 3 | 63 | 8003 |

Each group contains one PI0 feature server and one frozen SHORE evaluator on
the same visible GPU. The evaluator uses task-local paths and never shares an
action queue with another task.

A launcher starts the four groups, records PIDs and logs, and does not terminate
healthy tasks if one group fails. It performs a GPU-memory preflight before the
formal run.

## Data products

Use a timestamped run root:

`outputs_eval/shore_libero90_zero_shot/<run_id>/`

Each task directory contains:

- `config.json`: task identity, task language, paths, hashes, seed, port, and
  runtime versions;
- `episodes.jsonl`: one flushed and fsynced record per completed episode;
- `summary.json`: successes, total episodes, final success rate, return and
  length summaries;
- `stdout.log` and `server.log`; and
- a completion marker written only after all validation checks pass.

The combined artifacts are:

- `paper/data/fig_shore_libero90_zero_shot.json`
- `paper/data/fig_shore_libero90_zero_shot.npz`
- `paper/figure/fig_shore_libero90_zero_shot.pdf`

No existing DSRL data or figure is overwritten.

### Episode schema

Each JSONL row contains:

- `task_id`
- `task_language`
- `episode_index`
- `init_state_id`
- `success`
- `return`
- `length`
- `elapsed_seconds`
- `seed`
- `source_checkpoint_sha256`

The final success rate is the arithmetic mean of the 50 binary success values.

## Plot

Produce a 2-by-2 panel figure with one task per panel:

- x-axis: `Evaluation episode`;
- y-axis: `10-episode rolling success rate`;
- one SHORE curve only;
- y-range fixed to `[0, 1]`;
- final annotation shown as `successes / 50`;
- task ID and concise instruction in each title; and
- caption/note explicitly stating frozen task8 SHORE weights and zero-shot
  LIBERO-90 evaluation.

The raw binary outcomes remain available in JSON and NPZ so the rolling curve
does not obscure the underlying sample count.

## Failure handling and resumability

Fail before rollout when:

- a source-bundle file is absent;
- strict checkpoint loading reports missing or unexpected keys;
- the saved SHORE architecture is incompatible with the reconstructed agent;
- the PI0 server does not return `prefix_feat`;
- subgoal feature dimensions disagree;
- the environment task language disagrees with the expected task ID; or
- any supposedly frozen parameter remains trainable.

Write episode rows atomically as episodes finish. On restart, validate the
existing config and checkpoint hash, then continue only the missing init-state
IDs. Never silently reuse records produced by another checkpoint or task.

## Verification

### Automated tests

Add tests for:

1. inference-only `QAgent` creates no optimizer or scheduler;
2. existing training-mode construction remains unchanged;
3. strict checkpoint reconstruction and frozen-parameter checks;
4. deterministic 0-through-49 init-state coverage;
5. per-episode evaluator records and callback behavior;
6. resume logic rejects mismatched task/checkpoint metadata;
7. combined-data validation rejects missing or duplicate init states; and
8. plot generation from a small fixture.

### Live checks

1. Start one PI0 server and run task63 for one episode.
2. Confirm prefix features, SHORE action inference, environment termination,
   and durable episode output.
3. Check that no gradient, optimizer, replay-buffer, or W&B activity occurred.
4. Inspect GPU memory before starting the four formal process groups.
5. Run all four tasks for 50 episodes.
6. Validate 200 total unique `(task_id, init_state_id)` pairs before plotting.

## Acceptance criteria

The evaluation is complete when:

- all four tasks used the same frozen source-bundle hashes;
- no LIBERO-90 demonstration was accessed;
- no model parameter was updated;
- every task has exactly 50 unique fixed-init episodes;
- all raw and summary data validate;
- the combined JSON and NPZ files are present; and
- the four-panel PDF is reproducibly generated from the saved raw data.
