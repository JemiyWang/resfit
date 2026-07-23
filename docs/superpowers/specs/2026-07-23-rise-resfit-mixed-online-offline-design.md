# RISE × ResFiT Online/Offline Mixed Replay Design

**Date:** 2026-07-23
**Status:** Approved in conversation; ready for implementation planning
**Scope:** `/mnt/mnt/data/resfit/resfit/rl_finetuning/wm_bridge`

## 1. Context

The current combined RISE/ResFiT path runs through:

- `resfit/rl_finetuning/wm_bridge/launch_imagination.py`;
- `resfit/rl_finetuning/wm_bridge/ImaginationVecEnv`;
- the existing ResFiT `train_chunk_residual.py` trainer.

The current `launch_block_imagination.sh` does not enable
`--offline_fraction`, so the existing combined run trains only from the
imagined-online replay buffer.

The original RISE release does use real/offline data during its released
online improvement configuration:

- `rl_release.yaml` sets `offline_rl: True` and `train_mode: "IL"`;
- `huggingface_worker.py` hard-codes `use_expert_data_prob = 0.6`;
- that probability controls whether a generated training item comes directly
  from the LeRobot/offline path or from a world-model rollout.

This is not a fixed two-buffer RLPD minibatch. RISE mixes at rollout/data
generation time, whereas ResFiT already supports sampling independent online
and offline replay buffers and concatenating them before an update.

The integration will preserve the useful RISE intent—retain successful real
data while learning from imagination—but implement it with ResFiT's native,
fixed-ratio replay semantics.

## 2. Goals

1. Every RL update uses exactly 50% imagined-online and 50% real-offline
   transitions.
2. Online and offline transitions have identical 50-environment-step temporal
   semantics.
3. Offline `base_action` is produced by the same frozen kai0 policy used by
   imagination.
4. Offline reward uses the same frozen HIQL potential and endpoint PBRS formula
   as imagination.
5. The original RISE repository remains unchanged.
6. ResFiT's generic HDF5/LeRobot offline builders remain unchanged.
7. All block-specific integration behavior is isolated under `wm_bridge`.
8. Expensive kai0 endpoint inference is resumable and reusable.

## 3. Non-goals

- Do not train on `block_fail` in the first offline replay version.
- Do not finetune kai0, HIQL value, or the world model.
- Do not introduce subgoal conditioning, stage conditioning, relabeling, OAC,
  or online HIQL finetuning in the first version.
- Do not reproduce RISE's hard-coded 0.6 data-generation probability.
- Do not convert the generic ResFiT offline builder into a block-specific
  builder.
- Do not use per-frame offline transitions or stride-1 sliding windows.

## 4. Chosen Architecture

Use a bridge-owned chunk adapter and cache:

```text
block_success
    |
    v
wm_bridge block offline adapter
    |- slice real trajectories into 50-step chunks
    |- query frozen kai0 at chunk endpoints
    |- compute frozen-HIQL endpoint PBRS
    `- construct chunk-level TensorDict transitions
                    |
                    v
              offline replay
                    | 128
                    +------------------+
                                       v
ImaginationVecEnv -> online replay -> concat -> TD3/ResFiT update
                                  128            batch size 256
```

The bridge injects block-specific implementations of:

- `count_offline_transitions`;
- `build_offline_buffer`.

The existing trainer remains responsible for:

- constructing the two replay buffers;
- loading and saving the TensorDict memmap;
- sampling the online and offline halves independently;
- concatenating the halves;
- critic warmup, TD3 updates, target updates, and optional demo BC.

This option was selected over:

1. adding a generic offline-source interface to the core trainer, which would
   have a larger regression surface; and
2. preloading offline data into `online_rb`, which cannot guarantee a fixed
   50/50 ratio after online collection begins.

## 5. Data Sources

### 5.1 Offline replay

Use only:

```text
/mnt/mnt/data/domains_rise/block/block_success
```

The dataset currently contains 295 successful episodes and approximately
342,957 frames. The actual episode task is `build block`.

### 5.2 Imagination reset states

Continue sampling reset windows from both:

```text
/mnt/mnt/data/domains_rise/block/block_success
/mnt/mnt/data/domains_rise/block/block_fail
```

`block_fail` therefore contributes state coverage to imagination but does not
become an offline critic/BC target in the first version.

## 6. Exact Offline Transition Semantics

For an episode with frames `0 ... T-1`, a transition beginning at frame `t`
is:

```text
obs       = frame[t]
expert    = action[t : t+50]       # 50 x 16 physical actions
next_obs  = frame[t+50]
base      = kai0(obs)              # 50 x 16 physical actions
next_base = kai0(next_obs)         # 50 x 16 physical actions
reward    = gamma * Phi(next_obs) - Phi(obs)
```

The current observation is aligned with actions `t ... t+49`; the endpoint
observation is frame `t+50`.

### 6.1 Chunk start indices

For each episode:

1. add regular starts `0, 50, 100, ...` while `t + 50 <= T - 1`;
2. compute `t_terminal = T - 51`;
3. if the last regular chunk does not already end at `T - 1`, add the
   terminal-aligned chunk beginning at `t_terminal`;
4. sort starts chronologically and remove duplicates;
5. skip episodes shorter than 51 frames and report them.

The terminal-aligned chunk may partially overlap the preceding regular chunk.
This is intentional: it preserves a genuine successful terminal transition
without inventing padding or discarding the episode ending.

### 6.2 Done semantics

- Regular chunks: `done=False`.
- The chunk ending at frame `T-1`: `done=True`.

The terminal reward still uses:

```text
gamma * Phi(s_terminal) - Phi(s_start)
```

There is no separate success bonus. `done=True` only prevents critic bootstrap
beyond the successful episode boundary.

### 6.3 Observation and action fields

The offline TensorDict must match an online replay item exactly for all keys
consumed by `QAgent`:

- current and next image observations:
  same three camera keys and same `84 x 84 uint8` representation as online;
- current and next `observation.state`:
  standardized by the trainer's shared `StateStandardizer`;
- current and next `observation.base_action`:
  kai0 physical chunks scaled by the trainer's shared `ActionScaler` and
  flattened to 800 dimensions;
- `action`:
  expert physical chunk scaled by the same `ActionScaler` and flattened to 800
  dimensions;
- `observation.stage_id=0`;
- top-level `max_stage=0`;
- scalar float32 reward;
- scalar boolean done;
- `_priority=10.0`.

The actor's demo residual target remains the existing ResFiT definition:

```text
expert_scaled - kai0_base_scaled
```

The replay stores the expert combined action, not a precomputed residual.

### 6.4 No artificial history

The world model consumes a four-frame window, but the current TD3 observation
and `Kai0ImaginationBase._serve_obs` consume the current endpoint image.
Therefore the offline adapter does not fabricate or pad a four-frame history.

## 7. Reward and Frozen Models

The adapter reuses the bridge's shared `Kai0HiqlScorer` instance.

At every unique endpoint:

1. kai0 returns the physical 50-step base chunk and prefix feature `psi`;
2. the scorer evaluates `Phi(psi, raw_proprio)`;
3. the transition reward is
   `gamma * Phi(next_endpoint) - Phi(current_endpoint)`.

Required invariants:

- `gamma == imagination_gamma`;
- the value checkpoint is frozen;
- `pi0_serve_ckpt_id` must match the scorer's recorded feature source;
- dummy scorers are forbidden for mixed-replay production runs;
- reward, base actions, prefix features, and standardized states must all be
  finite.

## 8. Two-level Cache

### 8.1 Endpoint cache

Layout:

```text
<cache-root>/endpoints/<endpoint-fingerprint>/
    endpoint_meta.json
    episode_000000.npz
    episode_000001.npz
    ...
```

Each episode file contains:

- required endpoint frame indices;
- kai0 base chunks;
- kai0 prefix features;
- raw proprio used by the scorer;
- chunk start/end index metadata.

The endpoint fingerprint includes:

- cache schema version;
- block-success dataset manifest;
- chunk length, stride, and terminal-alignment rule;
- camera mapping;
- prompt;
- action dimension;
- `pi0_serve_ckpt_id`.

The selected demo count is deliberately not part of the endpoint fingerprint.
Endpoint files are lazy per-episode records under the full dataset identity, so
a two-episode smoke build can be resumed and extended by the full 295-episode
build. The selected demo count remains part of the replay fingerprint because
it changes the final replay contents.

Dataset manifesting hashes metadata and parquet identity/content metadata and
records video path, size, and nanosecond mtime. Full video hashing is avoided
because of its cost. `--offline_rebuild` remains available when external tools
modify video content without updating size or mtime.

Endpoint files are written per episode through a temporary file followed by an
atomic rename. An interrupted run only repeats the incomplete episode.

Shared endpoints are queried once. With stride 50, the current dataset should
require on the order of seven thousand endpoint queries, not hundreds of
thousands.

### 8.2 Replay cache

Layout:

```text
<cache-root>/replay/<replay-fingerprint>/
    storage/
    buffer_meta.json
    bridge_meta.json
```

The replay fingerprint includes:

- endpoint fingerprint;
- HIQL value checkpoint SHA-256;
- `gamma`;
- selected demo count;
- action and state statistics hashes;
- `action_scale` and `min_range_per_dim`;
- image size and keys;
- `n_step`;
- replay schema version.

Consequences:

- changing only HIQL/gamma/scaling rebuilds replay without re-querying kai0;
- changing kai0, prompt, chunk rules, cameras, or source data rebuilds both
  levels;
- a new fingerprint creates a new directory rather than deleting an older
  cache.

The bridge resolves the fingerprinted replay directory before executing the
trainer and passes that directory as the trainer's
`--offline_buffer_cache`.

## 9. Trainer Compatibility

The current trainer has a local validation rule:

```text
offline_base_mode=base_policy
    requires base_action_mode=queue and chunk_length=1
```

That rule describes its generic per-frame offline builder and cannot represent
the bridge's 50-step kai0 chunk builder. Because the validator is defined
inside the trainer executed through `runpy`, it cannot be cleanly replaced by
the existing module-injection mechanism.

To retain zero upstream modifications, the bridge uses:

```text
offline_base_mode=gt
```

strictly as a trainer compatibility marker. The bridge builder ignores the
generic GT-as-base interpretation and always writes the actual frozen kai0
chunk.

This otherwise misleading compatibility detail must be made explicit:

- startup prints both
  `trainer_compat_mode=gt` and `actual_base_mode=kai0_chunk`;
- `bridge_run_config.json` records the actual semantics;
- `bridge_meta.json` includes kai0 identity and the full bridge fingerprint;
- the cache path is selected by the stronger bridge fingerprint, not by the
  generic trainer signature;
- the bridge rejects incompatible combinations rather than falling back to
  GT-as-base.

The generic field in the trainer/W&B config may still display `gt`; the
bridge-specific config file and startup banner are authoritative for this
integration.

## 10. Bridge Arguments and Launch Translation

Add bridge-owned arguments:

```text
--offline_chunk_dataset
--offline_chunk_cache_root
--offline_rebuild
```

The bridge translates them into trainer passthrough arguments, including:

```text
--offline_dataset_path <offline_chunk_dataset>
--offline_buffer_cache <cache_root/replay/replay-fingerprint>
--offline_base_mode gt
```

When `--offline_chunk_dataset` is enabled, the production contract rejects any
run that violates:

```text
offline_fraction = 0.5
batch_size       = even
chunk_length     = 50
base_action_mode = replan
base_policy_type = pi05
n_step           = 1
gamma            = imagination_gamma
offline source   = block_success
```

`n_step=1` is mandatory because one transition already represents 50 original
environment steps. Generic multi-step aggregation must not cross episode
boundaries or the deliberately overlapping terminal chunk.

## 11. Recommended Training Configuration

First production run:

```bash
--offline_chunk_dataset /mnt/mnt/data/domains_rise/block/block_success
--offline_chunk_cache_root /mnt/mnt/data/resfit/cache/block_mixed_replay
--offline_fraction 0.5
--batch_size 256
--chunk_length 50
--n_step 1
--gamma 0.995
--imagination_gamma 0.995
--base_policy_type pi05
--base_action_mode replan
--actor raw
--action_scale 0.2
--demo_bc_coef 0.1
--bc_coef_final 0.01
--critic_warmup_steps 10000
--learning_starts 10000
--utd 4
--no_stage_balanced
--reward_shaping none
--potential_source stage
--total_env_steps 500000
```

With batch size 256 and offline fraction 0.5, the trainer computes:

```text
online_batch_size  = 128
offline_batch_size = 128
```

### 11.1 Demo BC

Mixed replay and demo BC are independent:

- mixed replay anchors critic learning and exposes ordinary actor RL updates to
  offline states;
- demo BC explicitly anchors the residual actor toward
  `expert_scaled - base_scaled`.

Use `0.1 -> 0.01` linear decay for the recommended stable run. Keep a
`demo_bc_coef=0` arm to isolate the effect of mixed replay itself.

### 11.2 Critic warmup

After `learning_starts=10000` raw environment steps, the online replay contains
approximately 200 50-step transitions. Run 10,000 critic-only updates, each
with 128 online and 128 offline items, before allowing actor updates. This
matches the critic-warmup convention already present in the ResFiT repository.

## 12. Diagnostics and Output

Startup must report:

```text
offline_source=block_success
offline_episodes=<count>
offline_transitions=<count>
online_batch_size=128
offline_batch_size=128
trainer_compat_mode=gt
actual_offline_base=kai0_chunk
offline_reward=gamma_phi_next_minus_phi
endpoint_cache=<hit|partial|miss>
replay_cache=<hit|miss>
```

Cache construction progress must include episode/frame context. A kai0 error,
decode error, shape mismatch, or non-finite value must stop construction while
preserving already completed atomic episode endpoint files.

Record at least:

- offline and online reward mean/std;
- potential delta mean/std;
- expert/base/residual chunk norms;
- action clamp/saturation ratio;
- critic loss and Q scale;
- online/offline replay sizes;
- cache fingerprint and source model identities.

## 13. Test Strategy

### 13.1 Unit tests

1. **Chunk indices**
   - exact 51-frame episode;
   - aligned and unaligned episode endings;
   - no duplicate terminal chunk;
   - episodes shorter than 51 frames are skipped.
2. **Endpoint reuse**
   - adjacent chunks share a single kai0 endpoint query;
   - terminal-aligned chunks request only missing endpoints.
3. **Reward**
   - deterministic scorer verifies
     `gamma * phi_next - phi_current`;
   - terminal transition receives no bonus.
4. **TensorDict schema**
   - offline and online samples have matching consumed fields, shapes, dtypes,
     and normalization.
5. **Residual target**
   - stored action is expert-scaled;
   - stored base action is kai0-scaled;
   - BC target equals their difference.
6. **Cache invalidation**
   - value/gamma changes invalidate replay only;
   - kai0/prompt/data/chunk-rule changes invalidate endpoint and replay;
   - incomplete episode files are not accepted.
7. **Contract**
   - reject wrong fraction, odd batch size, wrong chunk length, `n_step != 1`,
     mismatched gammas, non-success offline root, or dummy scorer.

### 13.2 Integration smoke

Using two real success episodes and real serves:

1. build endpoint and replay caches;
2. rerun and verify cache hits;
3. collect enough imagined transitions to sample;
4. verify sampling calls request 128 online and 128 offline;
5. verify concatenated batch size 256;
6. run a short critic warmup and at least one actor update;
7. assert finite losses, rewards, Q values, and actions.

### 13.3 Regression

- `offline_fraction=0` retains current pure-imagination behavior.
- Existing generic HDF5/LeRobot offline builder tests remain unchanged.
- Existing `wm_bridge` imagination, truncation bootstrap, evaluator, base
  bridge, and contract tests continue to pass.

## 14. Experiment Matrix

Run at least three seeds with identical kai0, HIQL, world model, initialization
datasets, training budget, and evaluation schedule:

| Arm | Offline fraction | Demo BC | Purpose |
|---|---:|---:|---|
| A | 0.0 | 0 | current imagined-online baseline |
| B | 0.5 | 0 | isolate fixed-ratio offline replay |
| C | 0.5 | 0.1 -> 0.01 | recommended stable mixed training |

Compare:

- imagined advantage proxy;
- endpoint potential improvement;
- reward distributions;
- critic loss and Q scale;
- residual norm and action saturation;
- learning speed;
- cross-seed variance and collapse frequency.

No claim of improvement is made from a single smoke run; the smoke proves
semantic and numerical correctness, while the three-seed matrix evaluates the
learning effect.

## 15. Planned File Changes

New bridge-owned modules:

- `resfit/rl_finetuning/wm_bridge/block_offline_chunk.py`;
- `resfit/rl_finetuning/wm_bridge/block_offline_cache.py`;
- corresponding tests under `resfit/rl_finetuning/wm_bridge/tests/`.

Expected bridge updates:

- `wm_bridge/builder.py`:
  parse offline bridge arguments, share scorer/base state, and create the fake
  offline builder functions;
- `wm_bridge/launch_imagination.py`:
  install the additional injected symbols and translate cache paths;
- `wm_bridge/contract.py`:
  enforce mixed-replay invariants;
- `launch_block_imagination.sh`:
  enable the approved production parameters and use a distinct mixed-run
  output/W&B name.

Explicitly unchanged:

- `/mnt/mnt/data/data2/RL/RISE`;
- ResFiT's generic `offline_stage_replay.py`;
- ResFiT's generic `train_chunk_residual.py` training and two-buffer sampling
  loop.

## 16. Acceptance Criteria

Implementation is ready for a production run only when:

1. all bridge unit and regression tests pass;
2. live two-episode cache build and cache-hit smoke pass;
3. offline replay schema matches online replay schema;
4. every training batch is verified as 128 online + 128 offline;
5. offline rewards match direct scorer calculations;
6. cache fingerprints invalidate at the intended level;
7. no original RISE or generic ResFiT offline/trainer source file is modified;
8. the production launch script records actual kai0-chunk semantics despite
   the documented trainer compatibility marker.
