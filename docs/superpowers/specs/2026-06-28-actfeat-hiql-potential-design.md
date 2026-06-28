# ActFeat HIQL Potential Design

## Goal

Make `--reward_shaping potential --potential_source hiql` support an act_feat scalar value function so the shaping potential uses the same visual feature state family as the HIQL subgoal stack.

The intended experiment口径 is:

- subgoal: `gc_value_actfeat + high_actor_actfeat -> z`
- potential reward: `value_actfeat(image feature + standardized proprio) -> Phi(s)`
- bottom policy: current observation plus `z` trains the residual actor/critic

This replaces the current mismatch where potential HIQL uses `eef` or `eef_piece` low-dimensional value checkpoints while subgoal HIQL uses `act_feat`.

## Current Behavior

`train_hiql_value.py` already supports `--state_mode act_feat`. In that mode it builds or loads an act feature cache, standardizes the frozen ACT encoder feature, and trains a scalar `ValueMLP`.

`train_chunk_residual.py`, `ChunkResidualEnvWrapper`, `offline_hdf5_buffer.py`, and `HiqlPotential` currently only wire online/offline Phi through:

- `eef`: standardized `observation.state`
- `eef_piece`: standardized `observation.state` plus standardized privileged `rel_piece`

This is why the currently launched potential experiments are not act_feat potential experiments:

- lift/pouring use low-dimensional `eef` value checkpoints
- piece/threading use privileged `eef_piece` value checkpoints

## Desired Behavior

When `--potential_source hiql` loads a value checkpoint whose `state_mode` is `act_feat`, the training code must compute Phi from act features, not from low-dimensional state.

Online reward shaping must compute:

```text
phi_start = V_actfeat(f_act(obs_t))
phi_next  = V_actfeat(f_act(obs_t+1))
reward += bonus * (gamma * phi_next - phi_start)
```

Offline reward shaping must compute the same formula from the existing per-demo `act_feat_cache` sequences.

The feature state used for potential must match the one used by the act_feat HIQL subgoal stack:

- same ACT base checkpoint
- same image keys
- same proprio key
- same pooling
- same feature standardization

## Non-Goals

- Do not change the actor/critic observation contract. `observation.state` remains the existing low-dimensional state used by the residual policy.
- Do not train the potential value online. The act_feat value is frozen, as current HIQL potential values are frozen.
- Do not replace `gc_value_actfeat` or `high_actor_actfeat`. They remain responsible for subgoal `z`.
- Do not remove existing `eef` and `eef_piece` potential support. Existing experiments and tests should continue to work.

## Architecture

### HiqlPotential

`HiqlPotential` should support a third state mode:

```text
state_mode = "act_feat"
```

For `act_feat`, the potential model receives standardized act feature vectors. The value checkpoint's `mean/std` must be loaded and used to standardize online features if the online encoder returns raw act features. Existing `eef` and `eef_piece` behavior remains unchanged.

### Online Act Feature Encoder

`ChunkResidualEnvWrapper` should accept an optional `potential_feature_encoder`.

The encoder interface should stay small:

```python
class PotentialFeatureEncoder:
    def encode(self, raw_obs: dict) -> torch.Tensor:
        ...
```

For act_feat potential, `encode(raw_obs)` returns the standardized feature vector expected by `HiqlPotential.phi()`.

Implementation can reuse `ActFeatureExtractor` and value checkpoint stats:

1. Standardize raw `observation.state` with the same `StateStandardizer` already used by the wrapper.
2. Build the ACT feature with the same image keys/proprio key/pooling as the existing act_feat path.
3. Standardize the full act feature with the value checkpoint `mean/std`.

The wrapper should store the start Phi input on reset and after each step, analogous to current `_start_state_std` / `_start_rel_piece`.

### Offline Reward Shaping

Offline reward generation must support act_feat potential without replaying images through ACT during buffer build.

For act_feat mode, use the already materialized `act_feat_cache` sequences:

```text
act_feat_seq[t] -> Phi(t)
```

This keeps offline reward shaping consistent with the scalar value's training data and avoids expensive repeated encoder passes.

The offline builder must fail fast if:

- `potential.state_mode == "act_feat"` and no `act_feat_cache` is available
- the act feature sequence count or per-demo lengths cannot align with the HDF5 demos used to build the buffer
- the feature dimensionality differs from the loaded value checkpoint input dimension

### Train Entry

`train_chunk_residual.py` must detect act_feat potential checkpoints after `HiqlPotential.from_ckpt()`.

For `state_mode=act_feat`:

- require `--act_feat_cache`
- require that the cache can be loaded
- create the online potential feature encoder from the already loaded base ACT policy
- pass act_feat sequences to offline buffer construction
- include act_feat potential identity in the offline buffer signature

For `state_mode=eef` and `state_mode=eef_piece`, keep current behavior.

## Cache And Signature Rules

Offline buffer cache identity must include enough information to prevent cross-contamination between potential types:

- `potential_source`
- `hiql_value_ckpt`
- `potential.state_mode`
- potential scale
- value checkpoint input dimension
- value checkpoint state stats hash or equivalent stable identity
- act_feat cache signature when `state_mode=act_feat`
- act_feat cache ACT weight fingerprint if available

The existing no-shaping and eef/eef_piece potential caches must not be reused for act_feat potential.

## Error Handling

Fail fast with clear messages:

- `--potential_source hiql` with an act_feat checkpoint requires `--reward_shaping potential`.
- act_feat potential requires `--act_feat_cache`.
- act_feat potential requires a value checkpoint with `state_mode=act_feat`.
- online act feature dimensionality must equal `potential.model.state_dim`.
- offline act feature sequence dimensionality must equal `potential.model.state_dim`.
- online/offline feature signatures should be checked when signature metadata is available. If older artifacts lack fingerprints, keep the existing warning behavior rather than blocking.

## Testing

Use TDD and add focused tests before production changes:

1. `HiqlPotential` loads and evaluates a fake `state_mode=act_feat` value checkpoint.
2. `transition_rewards` computes potential reward from `act_feat_seq` and rejects missing state/feature input in act_feat mode.
3. `ChunkResidualEnvWrapper` uses a fake potential feature encoder for `state_mode=act_feat` and does not pass low-dimensional state to Phi.
4. `train_chunk_residual` or its helper validation rejects act_feat potential without `--act_feat_cache`.
5. Offline buffer signature changes when potential mode changes from `eef` to `act_feat` or when the act feature cache identity changes.

Existing tests for `eef` and `eef_piece` potential must continue to pass.

## Rollout

After implementation:

1. Train scalar act_feat value checkpoints per task:
   - `lifttray_value_actfeat_hdf5.pt`
   - `pouring_value_actfeat_hdf5.pt`
   - `three_piece_value_actfeat_hiqlv512.pt` or equivalent
   - `two_arm_threading_value_actfeat_hiqlv512.pt` or equivalent
2. Start new experiments with fresh `_pothiql_actfeat` output directories and offcache paths.
3. Treat previous `_pothiql` runs as separate ablations:
   - lift/pouring: eef potential
   - piece/threading: eef_piece privileged potential

## Open Decisions Resolved

- Use act_feat potential only when the value checkpoint declares `state_mode=act_feat`.
- Keep old `eef` and `eef_piece` potential paths available.
- Do not silently fall back to low-dimensional Phi if an act_feat checkpoint is requested and the feature cache/encoder is missing.
- Reuse existing ACT feature extraction and cache formats; do not add a new image encoder path.
