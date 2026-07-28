# Supplement Evidence-Bounded Completion Design

Date: 2026-07-28

## Goal

Update `paper/aaai2027-unified-supp.tex` so that every experimentally resolved
statement is filled from repository evidence, while every quantity that cannot
be reconstructed from code, run configuration, logs, or recorded aggregate
results remains explicitly marked for later completion.

## Scope

The implementation modifies only
`paper/aaai2027-unified-supp.tex`. Existing unrelated changes in the supplement
and all other working-tree files must be preserved. `paper/main.tex`, plotting
scripts, experiment code, generated figures, and result files are read-only
evidence sources for this revision.

Adjacent prose may be corrected when a placeholder is based on a premise that
does not match the executed pipeline. This applies especially to Supplement
Sec. C.5, whose current periodic \(K/M/H\) replay-generation description is not
the synchronous imagination environment implemented and launched in the
repository.

## Evidence Hierarchy

Use the following source priority:

1. Executed launch scripts, persisted `bridge_run_config.json` files, completed
   checkpoints, and terminal run logs.
2. The implementation that produced those artifacts, especially
   `wm_bridge/imagination_env.py`, `wm_bridge/builder.py`,
   `wm_bridge/block_offline_chunk.py`, and
   `chunk_residual/train_chunk_residual.py`.
3. Aggregate physical-robot results in
   `paper/plot_real_robot_curves.py` and the matching result statements in
   `paper/main.tex`.
4. Existing supplementary prose and older design documents.

Do not infer paired outcomes, failure categories, session metadata, validation
metrics, or physical randomization details from aggregate success rates.

## Resolved Revisions

### C.2 Tasks and timing

- Report the 30 Hz duration range separately from the 20 Hz paper-roll task.
  For the stated 800--2,500-frame range, 30 Hz corresponds to approximately
  27--83 seconds and 20 Hz to 40--125 seconds.
- Align the high-level task names and descriptions with the main paper:
  Paper-Roll Placement, Block Assembly, and Cup Stacking.
- Keep the dataset captions verbatim because they condition the dynamics model:
  `put the paper roll on the holder`, `build block`, and `pick cup`.
- Do not assert an arm handover, an exact paper-roll insertion tolerance, the
  number/order of blocks, arm assignments, the exact cup start pose, or a
  physical randomization region without direct records. Keep explicit
  placeholders for those facts.

### C.3 data inventory

- Treat Table `tab:realdata` as the exact recorded inventory:
  295/50 block expert/rollout episodes, 207/51 cup episodes, and 849/53
  paper-roll episodes.
- Remove the obsolete request to reconcile these exact values with the older
  “approximately 300 per task” wording. The supplement will state that its
  table reports the exact task-specific counts used by the pipeline.
- Do not change `paper/main.tex` in this task.

### C.4 dynamics-model training

- Identify `RISE_Hi/ckpt/step_18000` as the checkpoint used by the residual
  pipeline. The repository contains no step-20,000 checkpoint.
- Change the table from “20,000 configured, latest 18,000” to an executed
  18,000-step model.
- Retain an explicit placeholder for the formal dynamics-training hardware and
  wall-clock duration because neither is recorded in the available artifacts.
- Preserve the three-axis validation placeholders, including the missing held-
  out split size, rollout metric, controllability/negative-probe result,
  potential-consistency result, and replacement figure.

### C.5 actual imagination training

Replace the periodic buffer-refresh account with the executed synchronous
environment:

- One residual decision emits a 50-step, 16-dimensional action chunk.
- The dynamics model consumes four history frames and the 50-step chunk and
  predicts 25 frames for each of three camera views.
- One generated endpoint is one replay transition.
- Each real seed is recursively advanced for at most two generated segments;
  the artificial second-segment boundary retains the generated final
  observation and bootstraps rather than turning it into a true terminal.
- New seeds are sampled from both the task's successful demonstrations and
  frozen-base failure rollouts.
- The online replay contains imagined transitions. A separate offline replay
  is built from successful demonstrations; base-policy failure recordings are
  used as imagination seeds rather than inserted as recorded transitions.
- Training batches contain 128 imagined and 128 demonstration transitions
  (`batch_size=256`, `offline_fraction=0.5`).
- Use `UTD=4`, online replay capacity 200,000, learning start at 10,000 raw
  control steps, a 10,000-update critic-only warmup, one-step TD targets,
  \(\gamma=0.995\), residual scale 0.2, and demonstration-BC coefficient 0.1.
- State the exact successful-demonstration replay sizes only where persisted
  evidence exists: 6,989 block transitions and 4,237 cup transitions. Do not
  invent a paper-roll transition count if no completed run metadata exists.
- Explain reward and terminal semantics directly from the implementation:
  both recorded successful chunks and imagined chunks store
  \(\gamma\Phi(e')-\Phi(e)\); recorded final chunks have `done=true`, whereas
  artificial imagination boundaries use truncation with endpoint bootstrap.
- The executed real-robot launch scripts do not enable
  `--subgoal_conditioned` and do not pass goal-value/navigator checkpoints.
  Therefore the supplement must not claim that the real-robot residual actor or
  critic was waypoint-conditioned or that a navigator was jointly finetuned.

### C.6--C.7 aggregate physical results

Fill only statistics recoverable from the 50-trial aggregate checkpoints:

| Task | Frozen base | SHORE-RL at 500k | Absolute change |
|---|---:|---:|---:|
| Paper-Roll Placement | 11/50 (.22) | 21/50 (.42) | +.20 |
| Block Assembly | 5/50 (.10) | 14/50 (.28) | +.18 |
| Cup Stacking | 6/50 (.12) | 16/50 (.32) | +.20 |

Compute and report Wilson 95% intervals for each individual binomial rate.
Rename the result-table difference column from “paired \(\Delta\)” to
“absolute \(\Delta\)” because the repository lacks paired trial labels.

Retain explicit placeholders for:

- discordant-pair counts, paired confidence intervals, and an exact McNemar
  test;
- session count, evaluation dates, invalid/aborted-trial handling, and
  confirmation of the initial-state protocol;
- per-task failure-mode definitions and counts.

Do not populate the failure-mode table from aggregate successes.

## Consistency Corrections

Only adjacent claims required to make the resolved sections truthful may be
changed. In particular:

- do not describe the real-robot pipeline as periodically generating \(M\)
  rollouts every \(K\) updates;
- do not call failure-rollout recordings a third replay-batch component;
- do not claim real-robot waypoint conditioning when the launch configuration
  omitted it;
- do not claim paired statistical estimates when only marginal success counts
  are available.

All simulation sections and unrelated real-robot prose remain unchanged.

## Verification

1. Inspect the final diff and confirm it changes only
   `paper/aaai2027-unified-supp.tex` plus this design/plan documentation.
2. Recompute all six Wilson intervals from the integer successes and \(n=50\).
3. Verify every reported C.5 hyperparameter against launch scripts, parser
   defaults, persisted run metadata, and completed logs.
4. Search the supplement for all remaining `\todo{}` instances and confirm
   each corresponds to one of the explicitly unresolved items above.
5. Compile the supplement with BibTeX and sufficient pdfLaTeX passes.
6. Check the log for fatal errors, undefined references/citations, and new
   overfull boxes in the edited blocks.
7. Extract the compiled PDF text and verify that the C.5 settings and all six
   real-robot success counts are present.

## Acceptance Criteria

- Every newly asserted number has a traceable repository source.
- The C.5 description matches the executed synchronous imagination pipeline.
- The physical-result table agrees with the plotting source and main-paper
  500k checkpoint values.
- Unsupported paired statistics, validation metrics, task geometry, and
  failure-mode counts remain visibly unresolved.
- The supplement compiles successfully without altering unrelated user work.
