# Paper RISE–ResFiT Aligned Experiment Design

Date: 2026-07-25

## Goal

Launch a Paper real-robot imagination-space residual-RL experiment aligned with
W&B run
`674575221-beijing-institute-of-technology/dexmg-chunk-residual/eegyfzmz`
(`block_shore_mixed50_seed0`), while replacing every task-specific Block
dependency with a verified Paper dependency.

The target formal run is:

- W&B project: `dexmg-chunk-residual`
- W&B name: `paper_shore_mixed50_seed0`
- seed: `0`
- total environment steps: `500000`

Existing jobs on GPUs 0, 2, 5, and 7 must not be stopped.

## Confirmed Inputs

### Reference Block run

The reference run is finished at 500k and uses the following relevant trainer
configuration:

- actor: `raw`
- actor learning rate: `1e-6`
- action scale: `0.2`
- base policy type: `pi05`
- base action mode: `replan`
- chunk length: `50`
- PI0 execute horizon: `10`
- offline fraction: `0.5`
- offline base compatibility mode: `gt`
- demo BC coefficient: `0.1`
- stage-balanced replay: disabled
- reward shaping: `none`
- critic warmup: `10000`
- learning starts: `10000`
- UTD: `4`
- gamma: `0.995`
- n-step: `1`
- batch size: `256`
- evaluation interval: `50000`
- evaluation episodes: `50`
- evaluation environments: `1`

The W&B trainer config does not contain the AWBC filesystem path. The Block
launcher maps its checkpoint identity to the local checkpoint.

### Paper data

- success:
  `/mnt/mnt/data/domains_rise/paper/paper_success`
- failure:
  `/mnt/mnt/data/domains_rise/paper/paper_fail`
- episodes: `849` success + `53` failure
- FPS: `20`
- task prompt: `put the paper roll on the holder`
- dataset state: 16 dimensions
- dataset action: 16 dimensions

The dataset state order is:

1. seven left-arm joints,
2. left gripper,
3. seven right-arm joints,
4. right gripper.

### Paper AWBC

- checkpoint:
  `/mnt/mnt/data/data2/kai0/checkpoints/paper/19999`
- checkpoint step: `19999`
- proposed serve identity: `pi05_paper_awbc_19999`
- asset ID: `pick_paper_all_merged`
- asset stats:
  `assets/pick_paper_all_merged/norm_stats.json`
- policy state dimension: `14`
- policy action dimension: `16`
- checkpoint size: approximately `12 GB`

The checkpoint statistics match Paper data after dropping the two gripper
coordinates from the dataset state. The maximum mean differences observed were
approximately:

- state: `3.5e-4`
- action: `7.5e-3`

This is strong evidence that the checkpoint and the local Paper dataset are from
the same task distribution.

### Paper value and world model

- Paper RISE monitoring value:
  `/mnt/mnt/data/resfit/RISE_Hi/policy_and_value/`
  `policy_offline_and_value/checkpoints/value_paper/value_paper/20000/`
  `model.safetensors`
- value config: `value_paper`
- shared three-domain dynamics checkpoint:
  `/mnt/mnt/data/resfit/RISE_Hi/ckpt/step_18000/`
  `diffusion_pytorch_model.safetensors`
- dynamics inference config:
  `/mnt/mnt/data/resfit/RISE_Hi/ckpt/block_infer.yaml`

The large `value_paper` model is only an imagined-rollout progress monitor. It
does not replace the compact `pi0_feat` HIQL potential used by the residual
learner.

## Approach Decision

### Selected: Paper-native clock with aligned numeric hyperparameters

Keep the Paper data at its native 20 Hz and retain the reference run's numeric
`chunk_length=50`.

This gives a Paper chunk duration of 2.5 seconds, compared with 1.667 seconds for
Block at 30 Hz. The Paper world-model training data uses the same native Paper
clock, so the action/video relation remains internally task-consistent.

The task clock must be recorded in run metadata. The experiment may be described
as numerically aligned with the reference run, but not as having an identical
physical-time horizon.

### Rejected: approximately 33 Paper actions per chunk

This would more closely match Block's 1.667-second physical horizon, but the
current bridge and dynamics interface are explicitly built around 50 actions
mapped to 25 tokens. It would require a larger temporal-interface rewrite and
would no longer match the reference trainer configuration.

### Rejected: resample the entire Paper stack to 30 Hz

This would require resampling the Paper datasets and retraining the world model,
Paper value model, and AWBC. It is substantially more expensive and unnecessary
for the requested experiment.

## Task Profile and State Mapping

Paper support must be task-configured, not implemented as a copy of the
Block-only code.

The Paper task profile contains:

- task key: `paper`
- task label: `pick_paper_roll`
- prompt: `put the paper roll on the holder`
- data FPS: `20`
- success dataset: `paper_success`
- failure dataset: `paper_fail`
- AWBC checkpoint directory
- AWBC checkpoint identity
- AWBC asset ID
- policy state dimension: `14`
- policy action dimension: `16`
- feature pooling: `mean`
- monitoring value checkpoint and config

The one permitted state conversion at the policy boundary is:

```text
[left_joint_1..7, left_gripper, right_joint_1..7, right_gripper]
    ->
[left_joint_1..7, right_joint_1..7]
```

The original 16-dimensional proprioception remains in:

- the imagination environment state tracker,
- the residual actor and critic observations,
- the offline replay,
- the compact-value feature vector appended to `prefix_feat`.

Only the state sent to Paper AWBC inference is converted to 14 dimensions.

The same conversion function must be used by:

1. offline Paper feature-cache construction, and
2. online Paper base-policy inference in imagination.

This prevents offline/online feature-source drift.

## Bridge Generalization

### AWBC serve

Generalize the existing AWBC feature server so it can receive:

- checkpoint directory,
- asset ID,
- task prompt,
- config/identity label,
- dataset/repo path used to construct transforms,
- feature pooling mode.

The constructed OpenPI config must use:

- PI0.5 model architecture,
- `LerobotTeleAvatarDataConfig`,
- absolute actions,
- the checkpoint's `pick_paper_all_merged` asset ID.

Block defaults must remain unchanged.

### Prompt plumbing

Remove the hard-coded `BLOCK_CAPTION` behavior from start-state sampling.

Use one authoritative task prompt and thread it through:

- start-state captions,
- world-model requests,
- Paper AWBC requests,
- Paper monitoring-value requests,
- offline feature-cache signatures,
- compact-value checkpoint metadata.

Startup must fail if independently supplied prompt values disagree.

### Policy-state plumbing

Add an explicit policy-state-dimension setting for TeleAvatar policy requests.

- `16`: preserve the full TeleAvatar state.
- `14`: drop indices 7 and 15.
- any other value: fail before serving or cache construction.

The serve response must still contain exactly `(50, 16)` actions.

### Provenance metadata

`bridge_run_config.json` and replay-cache metadata must include:

- source W&B run: `eegyfzmz`
- task key and task label
- task prompt
- dataset FPS
- numeric chunk length
- derived physical chunk duration
- success/failure dataset paths
- AWBC checkpoint identity and directory
- AWBC asset ID
- policy state/action dimensions
- pooling
- compact-value SHA-256
- dataset stats SHA-256
- Paper monitoring-value checkpoint and config

The W&B trainer config and local bridge metadata jointly define the full run.
Bridge-only arguments are not currently preserved by the trainer's W&B config,
so the local metadata file is authoritative for those fields.

## Offline Feature and Compact-Value Pipeline

Use the four currently free GPUs for temporary Paper AWBC feature servers.

Feature-cache construction:

1. Run a one-frame, real-checkpoint serve smoke.
2. Split `paper_success` across four non-overlapping shards.
3. Split `paper_fail` across the same four shard indices.
4. Save each shard independently.
5. Validate all shard signatures.
6. Train a compact `pi0_feat` HIQL value from the eight shards.

Expected outputs:

- shard cache:
  `/mnt/mnt/data/resfit/outputs_chunk/paper_pi0feat_shards/`
- shard logs:
  `/mnt/mnt/data/resfit/outputs_chunk/paper_pi0feat_logs/`
- compact value:
  `/mnt/mnt/data/resfit/outputs_chunk/paper_value_pi0feat.pt`

The compact value must record:

- state mode: `pi0_feat`
- Paper serve checkpoint identity
- image keys
- proprio key
- prompt
- pooling
- success/failure supervision mode
- feature normalization statistics

Completed shards are immutable inputs. A failed shard is rerun alone; successful
shards are not deleted or overwritten.

## Formal Runtime

After compact-value production, use four free GPUs for:

| Component | Port | Task-specific input |
|---|---:|---|
| RISE dynamics serve | 9000 | shared step-18000 checkpoint, Paper prompt |
| Paper AWBC feature/action serve | 8001 | `paper/19999`, mean pooling |
| Paper RISE value monitor | 8002 | `value_paper`, step 20000 |
| Residual RL | n/a | Paper aligned trainer configuration |

Exact GPU indices may be selected from the free set at launch time. A resource
recheck is required immediately before starting services. Existing training
processes are never preempted.

The formal launcher uses:

- `--task pick_paper_roll`
- `--dataset paper_success`
- `HF_LEROBOT_HOME=/mnt/mnt/data/domains_rise/paper`
- Paper success/failure init-state datasets
- Paper success offline-chunk dataset
- Paper-specific cache and output paths
- Paper prompt for PI0 and advantage monitoring
- Paper compact value
- Paper serve checkpoint identity

The legacy `base_wandb_id` trainer field is compatibility metadata on this
imagination path. The actual base policy is the Paper AWBC websocket serve and
must be identified in bridge metadata.

## Output Layout

- offline mixed replay:
  `/mnt/mnt/data/resfit/cache/paper_mixed_replay/`
- formal output:
  `/mnt/mnt/data/resfit/outputs_imagination/`
  `paper_shore_mixed50_seed0/`
- formal log:
  `/mnt/mnt/data/resfit/logs/paper_shore_mixed50_seed0.log`
- service logs:
  `/mnt/mnt/data/resfit/logs/paper_{wm,awbc,adv}.log`

No Paper artifact may reuse a Block cache/output directory.

## Validation Gates

### Unit and contract tests

Tests must cover:

- 16-to-14 Paper state conversion;
- 16-dimensional Block state identity behavior;
- invalid policy state dimensions;
- the same conversion in offline and online request construction;
- configurable start-state caption;
- prompt mismatch rejection;
- Paper launch-script paths and aligned numeric hyperparameters;
- preservation of existing Block behavior;
- feature-cache signature inclusion of Paper provenance;
- replay-cache separation between Block and Paper.

### Live checkpoint smoke

Before feature extraction, a real Paper serve request must verify:

- checkpoint loads without missing asset stats;
- request state entering transforms is 14-dimensional;
- three camera inputs are accepted;
- response actions have shape `(50,16)`;
- actions are finite;
- `prefix_feat` exists, is one-dimensional, and is finite;
- server metadata identifies `pi05_paper_awbc_19999` and pooling `mean`.

### Compact-value gate

Before formal RL:

- all success and failure shard files exist;
- every shard signature matches Paper;
- the compact value loads;
- its state mode is `pi0_feat`;
- its feature dimension matches the live Paper serve;
- its recorded checkpoint identity equals the live server identity.

### Service gate

Before formal RL:

- ports 9000, 8001, and 8002 are listening;
- each server returns the expected metadata;
- one disabled-W&B end-to-end smoke completes;
- no service log contains a traceback, OOM, or signature mismatch.

### W&B gate

After formal launch:

- run name is exactly `paper_shore_mixed50_seed0`;
- project is `dexmg-chunk-residual`;
- state is `running`;
- seed and aligned trainer parameters match the reference;
- the first metrics appear;
- the W&B step begins increasing;
- local bridge metadata records all Paper-only provenance.

## Failure and Recovery

- Checkpoint, asset, prompt, pooling, dimension, or signature mismatch: fail
  before formal training.
- Missing/invalid feature shard: rebuild only that shard.
- Compact-value failure: retain verified feature shards.
- Service startup failure: stop only services started for this Paper attempt.
- Formal smoke failure: do not create an online W&B run.
- Formal run failure after creation: preserve logs, output checkpoint, W&B ID,
  and bridge metadata for diagnosis; do not silently create a replacement run.
- Existing unrelated experiments remain untouched in all failure paths.

## Acceptance Criteria

The task is complete when:

1. Paper bridge support and regression tests pass.
2. The Paper checkpoint passes the live action/feature smoke.
3. All Paper feature shards and `paper_value_pi0feat.pt` are valid.
4. WM, AWBC, and Paper value services pass health checks.
5. The disabled-W&B end-to-end smoke succeeds.
6. `paper_shore_mixed50_seed0` is visible in W&B with aligned trainer config.
7. The run is confirmed to be progressing beyond initialization.
