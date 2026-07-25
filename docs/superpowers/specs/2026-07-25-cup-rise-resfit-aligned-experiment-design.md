# Cup RISE + ResFiT Aligned Mixed-Replay Experiment Design

## 1. Goal

Run a Cup residual-RL experiment that is directly comparable to W&B run
`eegyfzmz` (`block_shore_mixed50_seed0`).

Only task-specific inputs may change:

- task prompt;
- success and failure datasets;
- task-specific feature/value artifacts;
- task-specific RISE advantage estimator;
- cache, output, and W&B names.

The RL algorithm, sampling ratio, optimization parameters, evaluation cadence,
and seed remain aligned with the Block reference.

## 2. Chosen Approach

Use the same `pi0_feat` HIQL potential pipeline as the Block reference.
Do not use the large RISE `value_cup` checkpoint as the main shaping potential:
that model has a different architecture, input contract, and value scale. It is
used only for the imagined-advantage monitoring path, matching the role of
`value_block` in the reference experiment.

Before formal RL:

1. extract Cup kai0 prefix features for success and failure demonstrations;
2. train a Cup `pi0_feat` HIQL value checkpoint;
3. build the Cup offline chunk replay;
4. switch the monitoring service from `value_block` to `value_cup`;
5. launch the aligned 500k mixed-replay run.

## 3. Runtime Architecture

The production allocation is:

| GPU | Component | Port | Task-specific configuration |
|---|---|---:|---|
| 0 | RISE world-model D-serve | 9000 | Request prompt `pick cup` |
| 1 | kai0 AWBC feature/action serve | 8001 | Cup export `checkpoints/cup`, pooling `mean` |
| 6 | RISE advantage monitoring serve | 8002 | Cup `value_cup` checkpoint/config |
| 7 | ResFiT residual RL | n/a | Cup aligned run |

The existing three-domain world-model checkpoint is reused. The kai0 checkpoint
at `checkpoints/cup` was exported from the original pick-cup AWBC pipeline and is
therefore retained for Cup.

During feature preprocessing, before formal RL occupies GPU 7, GPUs 1, 6, and 7
may each host a temporary kai0 feature serve on distinct ports. Success and
failure episodes are divided into three non-overlapping shards. After extraction,
the temporary services on GPUs 6 and 7 are stopped; GPU 6 starts `value_cup` and
GPU 7 is left free for RL.

## 4. Bridge Generalization

Generalize the existing Block-only bridge rather than duplicating it.

The bridge receives or derives a task profile containing:

- task name;
- prompt;
- success-dataset basename;
- action dimension;
- kai0 serve checkpoint identity.

The following Block hard-coding becomes task-aware:

- `build block` in offline runtime translation and mixed-replay validation;
- the `block_success` source restriction;
- start-sampler captions;
- bridge metadata `offline_source`;
- Block-specific user-facing error text.

Block defaults remain unchanged, so existing Block launch commands and tests
continue to work. Cup supplies:

- prompt: `pick cup`;
- success source: `cup_success`;
- start-state sources: `cup_success` and `cup_fail`;
- action dimension: 16;
- kai0 identity: `pi05_pick_cup_awbc_49999`.

The mixed-replay contract must still reject:

- a prompt inconsistent with the selected task;
- a dataset name inconsistent with the resolved success source;
- a scorer whose recorded kai0 identity differs from the live serve identity;
- missing or dummy scorers in production mixed replay.

## 5. Cup Feature and Value Artifacts

### 5.1 Feature-cache signature

All success and failure shards use:

- image keys: `top_head`, `hand_left`, `hand_right`;
- proprio key: `observation.state`;
- prompt: `pick cup`;
- pooling: `mean`;
- serve checkpoint identity: `pi05_pick_cup_awbc_49999`.

Success shards use dataset ID `cup_success`; failure shards use `cup_fail`.
Dataset ID may differ across labeled groups, while the five feature-source
fields above must match exactly.

The expected output directory is:

`outputs_chunk/cup_pi0feat_shards/`

### 5.2 HIQL value training

Train one action-free value model over all Cup shards:

- state mode: `pi0_feat`;
- terminal reward mode: `success_signed`;
- successful terminal reward: +1;
- failed terminal reward: -1;
- success and failure features jointly re-standardized;
- hidden dimension: 256;
- training steps: 50,000;
- batch size: 256;
- value gamma: 0.99;
- expectile: 0.7;
- learning rate: 0.0003;
- EMA: 0.005;
- seed: 0.

The output is:

`outputs_chunk/cup_value_pi0feat.pt`

The saved checkpoint must record the Cup feature signature. The scorer must
report expected anchor `pi05_pick_cup_awbc_49999`.

## 6. Formal RL Configuration

The Cup launcher uses:

- offline fraction: 0.5;
- total batch size: 256;
- effective online batch: 128;
- effective offline batch: 128;
- base policy type: `pi05`;
- base action mode: `replan`;
- chunk length: 50;
- actor: `raw`;
- action scale: 0.2;
- minimum range per dimension: 0.1;
- fixed demo BC coefficient: 0.1;
- no final BC coefficient or BC schedule;
- critic warmup steps: 10,000;
- learning starts: 10,000;
- UTD: 4;
- actor learning rate: 0.000001;
- critic learning rate: 0.0001;
- n-step: 1;
- gamma: 0.995;
- imagination gamma: 0.995;
- stage-balanced sampling: disabled;
- reward shaping: `none`;
- potential source: `stage`;
- total environment steps: 500,000;
- evaluation interval: 50,000 environment steps;
- evaluation environments: 1;
- imagined-advantage evaluation episodes: 10;
- seed: 0.

Task-specific paths and names are:

- offline dataset: `/mnt/mnt/data/domains_rise/cup/cup_success`;
- start-state datasets:
  `/mnt/mnt/data/domains_rise/cup/cup_success` and
  `/mnt/mnt/data/domains_rise/cup/cup_fail`;
- replay cache root: `/mnt/mnt/data/resfit/cache/cup_mixed_replay`;
- output directory:
  `/mnt/mnt/data/resfit/outputs_imagination/cup_shore_mixed50_seed0`;
- W&B project: `dexmg-chunk-residual`;
- W&B run name: `cup_shore_mixed50_seed0`.

## 7. Validation Gates

### 7.1 Automated tests

Tests must prove:

- existing Block defaults and contracts remain valid;
- Cup prompt reaches the sampler and runtime metadata;
- Cup success source is accepted;
- mismatched prompt, source, or scorer anchor fails fast;
- offline metadata records Cup paths and identities;
- mixed sampling remains 128 online plus 128 offline.

### 7.2 Live smoke test

Before full preprocessing:

1. read a small number of Cup success and failure episodes;
2. obtain real prefix features from the kai0 serve;
3. train and load a temporary small Cup value;
4. build a small Cup offline replay;
5. execute a real world-model rollout;
6. assert expected shapes and finite actions, features, rewards, and values.

The smoke run uses disabled W&B logging and separate temporary paths.

### 7.3 Full-preprocessing validation

Before training the final value:

- verify all intended episodes are covered once across shards;
- verify no empty or overlapping shards;
- verify every cache has finite features and matching feature-source signature;
- verify success and failure labels come from explicit CLI groups;
- verify the final value checkpoint loads and has the expected Cup anchor.

### 7.4 Formal launch gate

Start the 500k run only when:

- ports 9000, 8001, and 8002 pass health checks;
- port 8002 is backed by `value_cup`;
- GPU 7 has no conflicting training process;
- the scorer anchor matches `pi05_pick_cup_awbc_49999`;
- the startup banner reports `offline_source=cup_success`;
- the startup banner reports batch `128+128`;
- W&B creates `cup_shore_mixed50_seed0`;
- initial metrics contain no NaN or Inf;
- BC coefficient is logged as fixed 0.1.

On any failed gate, stop at that stage and do not launch or continue the formal
experiment with a degraded configuration.

## 8. Service Transition and Recovery

The completed Block RL process is not restarted or modified.

After smoke validation:

1. stop the current GPU 6 Block advantage service;
2. use GPU 6 temporarily for Cup feature extraction if needed;
3. stop temporary feature services on GPUs 6 and 7;
4. start Cup `value_cup` monitoring on GPU 6 and verify port 8002;
5. launch Cup RL on GPU 7.

Keep the prior Block service command and log path available so the monitoring
service can be restored if the Cup service fails before formal RL starts.

## 9. Completion Criteria

The task is complete when:

- generalized bridge code and targeted tests pass;
- Cup feature shards and final Cup value checkpoint exist and validate;
- Cup offline replay is built or selected by a valid fingerprint;
- the Cup RISE monitoring service is healthy;
- the formal W&B run is visible and actively logging;
- runtime logs prove 50/50 mixed sampling and fixed BC coefficient 0.1.
