# ActFeat HIQL Potential Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--reward_shaping potential --potential_source hiql` use frozen act_feat HIQL value checkpoints when the checkpoint declares `state_mode="act_feat"`.

**Architecture:** Keep `HiqlPotential.phi()` as the single scalar Phi API and add act_feat as a third state mode beside `eef` and `eef_piece`. Offline rewards consume already-standardized act feature cache sequences; online rewards use a small encoder adapter around the frozen ACT base policy and checkpoint feature statistics. The residual actor/critic observation contract stays unchanged.

**Tech Stack:** Python, PyTorch, NumPy, TorchRL TensorDict replay buffers, pytest, existing LeRobot ACT policy utilities.

## Global Constraints

- Do not change the actor/critic observation contract; `observation.state` remains the existing low-dimensional residual-policy state.
- Do not train the potential value online; HIQL potential values remain frozen.
- Do not remove existing `eef` and `eef_piece` potential support.
- Offline act_feat potential must use `act_feat_cache` sequences and must not replay images through ACT during buffer build.
- Online act_feat potential must use the same ACT base checkpoint, image keys, proprio key, pooling, and feature standardization as the value checkpoint and cache.
- Fail fast when act_feat potential is requested without `--act_feat_cache`.
- Existing stage, `eef`, and `eef_piece` potential tests must keep passing.
- Preserve user work in the dirty worktree; do not revert unrelated files.

---

## File Structure

- Modify `resfit/rl_finetuning/chunk_residual/hiql_value.py`: persist optional act_feat metadata in scalar value checkpoints.
- Modify `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`: pass act_feat signature and ACT weight fingerprint into `save_value()`.
- Modify `resfit/rl_finetuning/chunk_residual/hiql_potential.py`: load act_feat checkpoint stats and expose feature standardization metadata.
- Modify `resfit/rl_finetuning/chunk_residual/act_feature.py`: add an online potential feature encoder adapter that returns standardized act_feat vectors.
- Modify `resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py`: let transition reward helpers use `act_feat_seq` for act_feat potentials.
- Modify `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`: pass act feature sequences to `transition_fields()` for HDF5 and LeRobot offline buffers.
- Modify `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`: route online potential Phi input through the feature encoder when `state_mode="act_feat"`.
- Modify `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`: load and validate act_feat cache once, wire the online encoder, set correct env state mode, and expand offline cache signatures.
- Test `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`: scalar potential and offline reward unit tests.
- Test `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`: online wrapper feature-encoder routing test.
- Test `resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py`: value checkpoint metadata test if existing helpers make it local and cheap.

---

### Task 1: Value Checkpoint Metadata And ActFeat Potential Core

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_value.py`
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_potential.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`

**Interfaces:**
- Consumes: existing `save_value(path, model, *, v_stats, mean, std, dataset_id, state_mode, rel_piece_stats)`.
- Produces: `save_value(..., act_feat_signature: dict | None = None, act_weight_sha: str | None = None)`.
- Produces: `load_value()` info keys `act_feat_signature`, `act_weight_sha`, `state_dim`.
- Produces: `HiqlPotential.standardize_features(raw_feat) -> torch.Tensor`.
- Produces: `HiqlPotential.feature_mean`, `HiqlPotential.feature_std`, `HiqlPotential.act_feat_signature`, `HiqlPotential.act_weight_sha`.

- [ ] **Step 1: Write the failing act_feat checkpoint test**

Append this test to `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`:

```python
def _make_actfeat_ckpt(tmp_path, vmin=0.0, vmax=4.0):
    m = ValueMLP(state_dim=4, hidden=8)
    p = str(tmp_path / "value_actfeat.pt")
    mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    std = torch.tensor([2.0, 2.0, 4.0, 4.0])
    sig = {
        "act_ckpt_id": "base-act",
        "image_keys": ["observation.images.cam"],
        "proprio_key": "observation.state",
        "pooling": "mean",
    }
    save_value(
        p,
        m,
        v_stats={"min": vmin, "max": vmax, "mean": 0.5 * (vmin + vmax)},
        mean=mean,
        std=std,
        dataset_id="dummy",
        state_mode="act_feat",
        act_feat_signature=sig,
        act_weight_sha="sha-act",
    )
    return p, m, mean, std, sig


def test_hiqlpotential_loads_actfeat_metadata_and_standardizes(tmp_path):
    p, model, mean, std, sig = _make_actfeat_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")

    assert pot.state_mode == "act_feat"
    assert pot.model.state_dim == 4
    assert pot.act_feat_signature == sig
    assert pot.act_weight_sha == "sha-act"
    raw = torch.tensor([[3.0, 6.0, 11.0, 20.0]])
    expected_std = (raw - mean) / std
    assert torch.allclose(pot.standardize_features(raw), expected_std)
    assert torch.allclose(
        pot.phi(expected_std),
        model(expected_std).squeeze(-1) * pot.scale,
        atol=1e-5,
    )
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py::test_hiqlpotential_loads_actfeat_metadata_and_standardizes -q
```

Expected: FAIL because `save_value()` does not accept `act_feat_signature` and `HiqlPotential.standardize_features()` does not exist.

- [ ] **Step 3: Persist metadata in `hiql_value.py`**

Change the `save_value()` signature to:

```python
def save_value(path, model, *, v_stats, mean, std, dataset_id,
               state_mode="eef", rel_piece_stats=None,
               act_feat_signature=None, act_weight_sha=None):
```

Add these fields to the payload:

```python
        "state_dim": model.state_dim,
```

Then add after the `rel_piece_stats` block:

```python
    if act_feat_signature is not None:
        payload["act_feat_signature"] = dict(act_feat_signature)
    if act_weight_sha is not None:
        payload["act_weight_sha"] = str(act_weight_sha)
```

Change `load_value()` info construction to include the dimension and optional metadata:

```python
    info = {k: ckpt[k] for k in ("v_stats", "mean", "std", "dataset_id")}
    info["state_dim"] = ckpt["state_dim"]
    info["state_mode"] = ckpt.get("state_mode", "eef")
    info["rel_piece_mean"] = ckpt.get("rel_piece_mean")
    info["rel_piece_std"] = ckpt.get("rel_piece_std")
    info["act_feat_signature"] = ckpt.get("act_feat_signature")
    info["act_weight_sha"] = ckpt.get("act_weight_sha")
```

- [ ] **Step 4: Pass metadata from `train_hiql_value.py`**

Change the setup line in `main()` from:

```python
    extractor, act_ckpt_id, image_keys, _, _ = setup_act_feat(args)
```

to:

```python
    extractor, act_ckpt_id, image_keys, act_sig, act_sha = setup_act_feat(args)
```

Change the `save_value()` call to:

```python
    save_value(
        args.output,
        model,
        v_stats=v_stats,
        mean=mean,
        std=std,
        dataset_id=args.dataset,
        state_mode=args.state_mode,
        rel_piece_stats=rel_stats,
        act_feat_signature=(act_sig if args.state_mode == "act_feat" else None),
        act_weight_sha=(act_sha if args.state_mode == "act_feat" else None),
    )
```

- [ ] **Step 5: Add act_feat stats to `HiqlPotential`**

Change `HiqlPotential.__init__()` signature to:

```python
    def __init__(self, model, scale, device="cpu",
                 state_mode="eef", rel_piece_mean=None, rel_piece_std=None,
                 feature_mean=None, feature_std=None,
                 act_feat_signature=None, act_weight_sha=None):
```

Inside `__init__()`, after `self.state_mode = state_mode`, add:

```python
        self.feature_mean = None
        self.feature_std = None
        self.act_feat_signature = act_feat_signature
        self.act_weight_sha = act_weight_sha
        if state_mode == "act_feat":
            if feature_mean is None or feature_std is None:
                raise ValueError("state_mode='act_feat' requires feature mean/std in value checkpoint")
            self.feature_mean = torch.as_tensor(np.asarray(feature_mean), dtype=torch.float32, device=device)
            self.feature_std = torch.as_tensor(np.asarray(feature_std), dtype=torch.float32, device=device)
            if int(self.feature_mean.numel()) != int(self.model.state_dim):
                raise ValueError(
                    f"act_feat mean dim {self.feature_mean.numel()} != value state_dim {self.model.state_dim}"
                )
            if int(self.feature_std.numel()) != int(self.model.state_dim):
                raise ValueError(
                    f"act_feat std dim {self.feature_std.numel()} != value state_dim {self.model.state_dim}"
                )
```

Change `from_ckpt()` to pass the new info:

```python
        return cls(model, scale=auto_scale * phi_scale, device=device,
                   state_mode=info["state_mode"],
                   rel_piece_mean=info["rel_piece_mean"],
                   rel_piece_std=info["rel_piece_std"],
                   feature_mean=info["mean"],
                   feature_std=info["std"],
                   act_feat_signature=info.get("act_feat_signature"),
                   act_weight_sha=info.get("act_weight_sha"))
```

Add this method before `phi()`:

```python
    def standardize_features(self, raw_feat):
        if self.state_mode != "act_feat":
            raise ValueError("standardize_features is only valid for state_mode='act_feat'")
        x = torch.as_tensor(raw_feat, dtype=torch.float32, device=self.device)
        return (x - self.feature_mean) / self.feature_std
```

Keep `_value_input()` unchanged for `eef`; for `act_feat`, it should return the already-standardized feature tensor through the existing non-`eef_piece` branch.

- [ ] **Step 6: Run the focused potential tests**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  resfit/rl_finetuning/chunk_residual/hiql_value.py \
  resfit/rl_finetuning/chunk_residual/train_hiql_value.py \
  resfit/rl_finetuning/chunk_residual/hiql_potential.py \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat: load actfeat hiql potential checkpoints"
```

---

### Task 2: Online Potential Act Feature Encoder

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/act_feature.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`

**Interfaces:**
- Consumes: `ActFeatureExtractor.embed_batch(raw_obs) -> torch.Tensor`.
- Consumes: `StateStandardizer.standardize(raw_state) -> torch.Tensor`.
- Consumes: `HiqlPotential.standardize_features(raw_feat) -> torch.Tensor`.
- Produces: `PotentialActFeatureEncoder.encode(raw_obs: dict) -> torch.Tensor`.

- [ ] **Step 1: Write the failing encoder unit test**

Append this test to `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`:

```python
class _FakeStateStandardizer:
    def standardize(self, state):
        return torch.as_tensor(state, dtype=torch.float32) + 10.0


class _FakeExtractor:
    def __init__(self):
        self.seen_state = None
    def embed_batch(self, raw_obs):
        self.seen_state = raw_obs["observation.state"].clone()
        image_scalar = raw_obs["observation.images.cam"].float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        return torch.cat([image_scalar, raw_obs["observation.state"].float()], dim=-1)


def test_potential_act_feature_encoder_standardizes_proprio_then_feature(tmp_path):
    from resfit.rl_finetuning.chunk_residual.act_feature import PotentialActFeatureEncoder

    p, _, mean, std, _ = _make_actfeat_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    extractor = _FakeExtractor()
    encoder = PotentialActFeatureEncoder(
        extractor,
        _FakeStateStandardizer(),
        pot,
        proprio_key="observation.state",
    )
    raw_obs = {
        "observation.state": torch.tensor([[1.0, 2.0, 3.0]]),
        "observation.images.cam": torch.ones(1, 3, 2, 2),
    }

    out = encoder.encode(raw_obs)
    raw_feat = torch.tensor([[1.0, 11.0, 12.0, 13.0]])
    assert torch.allclose(extractor.seen_state, torch.tensor([[11.0, 12.0, 13.0]]))
    assert torch.allclose(out, (raw_feat - mean) / std)
```

- [ ] **Step 2: Run the encoder test and verify it fails**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py::test_potential_act_feature_encoder_standardizes_proprio_then_feature -q
```

Expected: FAIL because `PotentialActFeatureEncoder` is not defined.

- [ ] **Step 3: Add `PotentialActFeatureEncoder`**

Add this class near `ActFeatureExtractor` in `resfit/rl_finetuning/chunk_residual/act_feature.py`:

```python
class PotentialActFeatureEncoder:
    """Online adapter for act_feat HIQL potential Phi inputs."""

    def __init__(self, extractor, state_standardizer, potential,
                 proprio_key="observation.state"):
        if getattr(potential, "state_mode", None) != "act_feat":
            raise ValueError("PotentialActFeatureEncoder requires state_mode='act_feat'")
        self.extractor = extractor
        self.state_standardizer = state_standardizer
        self.potential = potential
        self.proprio_key = proprio_key

    @torch.no_grad()
    def encode(self, raw_obs: dict) -> torch.Tensor:
        obs = dict(raw_obs)
        obs[self.proprio_key] = self.state_standardizer.standardize(raw_obs[self.proprio_key])
        raw_feat = self.extractor.embed_batch(obs)
        feat = self.potential.standardize_features(raw_feat)
        if feat.shape[-1] != self.potential.model.state_dim:
            raise ValueError(
                f"online act_feat dim {feat.shape[-1]} != value state_dim {self.potential.model.state_dim}"
            )
        return feat
```

- [ ] **Step 4: Run encoder and potential tests**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  resfit/rl_finetuning/chunk_residual/act_feature.py \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat: add actfeat potential encoder"
```

---

### Task 3: Offline Reward Shaping From Act Feature Sequences

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py`
- Modify: `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`

**Interfaces:**
- Consumes: `potential.state_mode`.
- Produces: `transition_rewards(..., act_feat_seq=None)`.
- Produces: `transition_fields(..., act_feat_seq=None)`.
- Produces: HDF5 and LeRobot offline replay passing `act_feat_seqs[demo]` when potential is act_feat.

- [ ] **Step 1: Write failing offline reward tests**

Append this to `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`:

```python
class _ActFeatStubPot:
    state_mode = "act_feat"
    def phi(self, state_seq, rel_piece_seq=None):
        assert rel_piece_seq is None
        return torch.as_tensor(state_seq, dtype=torch.float32)[:, 1]


def test_transition_rewards_actfeat_potential_uses_act_feat_seq():
    instant = np.array([0, 0, 0])
    low_state = torch.tensor([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
    act_feat = torch.tensor([[1.0, 10.0], [1.0, 20.0], [1.0, 30.0]])
    r = transition_rewards(
        instant,
        bonus=1.0,
        mode="potential",
        gamma=0.99,
        success=True,
        potential=_ActFeatStubPot(),
        state_seq=low_state,
        act_feat_seq=act_feat,
    )
    exp0 = potential_shaping(10.0, 20.0, bonus=1.0, gamma=0.99, done=False)
    exp1 = 1.0 + potential_shaping(20.0, 30.0, bonus=1.0, gamma=0.99, done=True)
    assert abs(r[0] - exp0) < 1e-5
    assert abs(r[1] - exp1) < 1e-5


def test_transition_rewards_actfeat_requires_act_feat_seq():
    instant = np.array([0, 0, 0])
    with pytest.raises(AssertionError, match="act_feat potential"):
        transition_rewards(
            instant,
            bonus=1.0,
            mode="potential",
            gamma=0.99,
            success=True,
            potential=_ActFeatStubPot(),
            state_seq=torch.zeros(3, 2),
        )
```

- [ ] **Step 2: Run the new offline tests and verify they fail**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py::test_transition_rewards_actfeat_potential_uses_act_feat_seq \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py::test_transition_rewards_actfeat_requires_act_feat_seq -q
```

Expected: FAIL because `transition_rewards()` does not accept `act_feat_seq`.

- [ ] **Step 3: Extend `transition_fields()` and `transition_rewards()`**

Change the signatures in `offline_hdf5_buffer.py` to:

```python
def transition_fields(instant_stages, *, bonus: float, mode: str,
                      gamma: float, success: bool = True,
                      potential=None, state_seq=None, rel_piece_seq=None,
                      act_feat_seq=None) -> dict:
```

and:

```python
def transition_rewards(instant_stages, *, bonus: float, mode: str,
                       gamma: float, success: bool = True,
                       potential=None, state_seq=None, rel_piece_seq=None,
                       act_feat_seq=None) -> np.ndarray:
```

Pass `act_feat_seq` through from `transition_fields()`:

```python
        "reward": transition_rewards(instant, bonus=bonus, mode=mode,
                                     gamma=gamma, success=success,
                                     potential=potential, state_seq=state_seq,
                                     rel_piece_seq=rel_piece_seq,
                                     act_feat_seq=act_feat_seq),
```

Change the potential block in `transition_rewards()` to:

```python
    if potential is not None:
        if getattr(potential, "state_mode", "eef") == "act_feat":
            assert act_feat_seq is not None and len(act_feat_seq) == T, \
                "act_feat potential 模式需 act_feat_seq 且长度=T"
            phi = potential.phi(act_feat_seq, None)
        else:
            assert state_seq is not None and len(state_seq) == T, \
                "potential 模式需 state_seq 且长度=T"
            assert rel_piece_seq is None or len(rel_piece_seq) == T, \
                "rel_piece_seq 长度须 = T"
            phi = potential.phi(state_seq, rel_piece_seq)
```

- [ ] **Step 4: Wire HDF5 offline replay**

In `offline_stage_replay.py`, add:

```python
    _act_feat_potential = potential is not None and getattr(potential, "state_mode", "eef") == "act_feat"
```

Change the `_af_by_ep` setup to run for either act_feat subgoal or act_feat potential:

```python
            if _act_feat_subgoal or _act_feat_potential:
                assert act_feat_seqs is not None, \
                    "act_feat subgoal/potential 的 offline buffer 需 act_feat_seqs(缓存序列)"
                assert len(act_feat_seqs) == len(demos), \
                    f"act_feat_seqs 数({len(act_feat_seqs)}) != demo 数({len(demos)})"
                _af_by_ep = dict(zip(demos, act_feat_seqs))
```

Before `transition_fields()`, add:

```python
                act_feat_seq = None
                if _act_feat_potential:
                    act_feat_seq = torch.as_tensor(_af_by_ep[ep], dtype=torch.float32)
                    assert act_feat_seq.shape[0] == T, \
                        f"act_feat potential 缓存帧数 {act_feat_seq.shape[0]} != demo {T} ({ep})"
                    assert act_feat_seq.shape[1] == potential.model.state_dim, \
                        f"act_feat potential 维度 {act_feat_seq.shape[1]} != value state_dim {potential.model.state_dim}"
```

Then pass it:

```python
                fld = transition_fields(instant, bonus=bonus, mode=mode,
                                        gamma=gamma, success=True,
                                        potential=potential, state_seq=state_n,
                                        rel_piece_seq=rel_seq,
                                        act_feat_seq=act_feat_seq)
```

- [ ] **Step 5: Wire LeRobot offline replay**

In `_build_offline_lerobot()`, derive:

```python
    _act_feat_potential = potential is not None and getattr(potential, "state_mode", "eef") == "act_feat"
```

Change the cache assertion to include potential:

```python
    if subgoal is not None or _act_feat_potential:
        assert act_feat_seqs is not None, \
            "act_feat subgoal/potential 的 lerobot offline buffer 需 act_feat_seqs(缓存序列)"
        assert len(act_feat_seqs) >= n, \
            f"act_feat_seqs 数({len(act_feat_seqs)}) < demo 数({n})"
```

Before `transition_fields()` in the episode loop, add:

```python
        act_feat_seq = None
        if _act_feat_potential:
            act_feat_seq = torch.as_tensor(act_feat_seqs[ep], dtype=torch.float32)
            assert act_feat_seq.shape[0] == T, \
                f"act_feat potential 缓存帧数 {act_feat_seq.shape[0]} != demo {T} (ep{ep})"
            assert act_feat_seq.shape[1] == potential.model.state_dim, \
                f"act_feat potential 维度 {act_feat_seq.shape[1]} != value state_dim {potential.model.state_dim}"
```

Pass the potential and act features:

```python
        fld = transition_fields(
            instant,
            bonus=bonus,
            mode=mode,
            gamma=gamma,
            success=True,
            potential=potential,
            state_seq=state_n,
            rel_piece_seq=None,
            act_feat_seq=act_feat_seq,
        )
```

Change the caller in `build_offline_buffer()` so the LeRobot path passes `potential=potential` into `_build_offline_lerobot()`.

- [ ] **Step 6: Run focused offline tests**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py \
  resfit/rl_finetuning/chunk_residual/offline_stage_replay.py \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat: shape offline rewards with actfeat potential"
```

---

### Task 4: Online Wrapper Uses Feature Encoder For ActFeat Potential

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`

**Interfaces:**
- Consumes: `potential.state_mode`.
- Consumes: `potential_feature_encoder.encode(raw_obs) -> torch.Tensor`.
- Produces: `ChunkResidualEnvWrapper(..., potential_feature_encoder=None)`.
- Produces: wrapper cache fields `_start_phi_state` and `_start_rel_piece`.

- [ ] **Step 1: Write the failing online wrapper test**

Append this test to `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`:

```python
class _ActFeatPotential:
    state_mode = "act_feat"
    def __init__(self):
        self.inputs = []
    def phi(self, state_std, rel_piece_raw=None):
        assert rel_piece_raw is None
        x = torch.as_tensor(state_std, dtype=torch.float32)
        self.inputs.append(x.clone())
        return x[:, 0]


class _SequentialFeatureEncoder:
    def __init__(self):
        self.calls = 0
    def encode(self, raw_obs):
        self.calls += 1
        return torch.tensor([[float(self.calls), 99.0]])


def test_actfeat_potential_uses_feature_encoder_not_lowdim_state():
    env = _FakeVecEnvNoTerm()
    pot = _ActFeatPotential()
    enc = _SequentialFeatureEncoder()
    w = ChunkResidualEnvWrapper(
        env,
        _FakeBase(),
        _IdentityScaler(),
        _IdentityStd(),
        chunk_length=1,
        stage_reward_bonus=1.0,
        reward_shaping_mode="potential",
        gamma=0.5,
        potential=pot,
        potential_feature_encoder=enc,
    )
    w.reset()
    _, reward, _, _, _ = w.step(torch.zeros(1, D))

    assert enc.calls == 3
    assert torch.allclose(pot.inputs[0], torch.tensor([[1.0, 99.0]]))
    assert torch.allclose(pot.inputs[1], torch.tensor([[2.0, 99.0]]))
    assert reward.item() == 0.0
```

The expected reward is `base 0 + (0.5 * 2 - 1)`.

- [ ] **Step 2: Run the wrapper test and verify it fails**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_actfeat_potential_uses_feature_encoder_not_lowdim_state -q
```

Expected: FAIL because `ChunkResidualEnvWrapper.__init__()` does not accept `potential_feature_encoder`.

- [ ] **Step 3: Add constructor argument and Phi input helper**

Change the constructor signature in `chunk_env_wrapper.py` to:

```python
                 base_action_mode: str = "replan", potential=None,
                 potential_feature_encoder=None):
```

Store:

```python
        self.potential_feature_encoder = potential_feature_encoder
        if potential is not None and getattr(potential, "state_mode", "eef") == "act_feat" \
                and potential_feature_encoder is None:
            raise ValueError("act_feat potential requires potential_feature_encoder")
        self._start_phi_state = None
```

Add this method after `_extract_rel()`:

```python
    def _phi_input(self, raw_obs, *, aug_obs=None, info=None):
        if self.potential is None:
            return None, None
        if getattr(self.potential, "state_mode", "eef") == "act_feat":
            return self.potential_feature_encoder.encode(raw_obs), None
        state_std = aug_obs["observation.state"] if aug_obs is not None \
            else self.state_standardizer.standardize(raw_obs["observation.state"])
        return state_std, self._extract_rel(info)
```

- [ ] **Step 4: Cache Phi input on reset**

Change the end of `reset()` from:

```python
        self._start_state_std = aug["observation.state"]
        self._start_rel_piece = self._extract_rel(info)
```

to:

```python
        self._start_state_std = aug["observation.state"]
        self._start_phi_state, self._start_rel_piece = self._phi_input(
            raw_obs, aug_obs=aug, info=info)
```

- [ ] **Step 5: Use cached Phi input in `step()`**

Replace the potential branch in `step()` with:

```python
        else:
            end_phi_state, rel_next = self._phi_input(raw_obs, info=last_info)
            phi_start = self.potential.phi(self._start_phi_state, self._start_rel_piece)
            phi_next = self.potential.phi(end_phi_state, rel_next)
            total_reward = total_reward + potential_shaping(
                phi_start, phi_next, bonus=self.stage_reward_bonus,
                gamma=self.gamma, done=chunk_done)
```

At the end of `step()`, after `aug_obs = self._augment(raw_obs, base_flat)`, replace:

```python
        self._start_state_std = aug_obs["observation.state"]
        self._start_rel_piece = self._extract_rel(last_info)
```

with:

```python
        self._start_state_std = aug_obs["observation.state"]
        self._start_phi_state, self._start_rel_piece = self._phi_input(
            raw_obs, aug_obs=aug_obs, info=last_info)
```

- [ ] **Step 6: Run wrapper tests**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
git commit -m "feat: route online actfeat potential through encoder"
```

---

### Task 5: Train Entry Wiring And Offline Cache Signature

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`

**Interfaces:**
- Consumes: `HiqlPotential.state_mode`, `HiqlPotential.act_feat_signature`, `HiqlPotential.act_weight_sha`.
- Consumes: `load_act_feat_cache(path) -> (seqs, (mean, std), signature, act_weight_sha)`.
- Consumes: `PotentialActFeatureEncoder`.
- Produces: `_env_state_mode_for_training(potential_state_mode, subgoal_state_mode) -> str`.
- Produces: `_validate_actfeat_potential_cache(args, potential, cache_sig, cache_sha, act_feat_seqs) -> None`.
- Produces: `_offline_buffer_signature(..., potential=potential, act_feat_cache_sig=None, act_feat_cache_sha=None)`.

- [ ] **Step 1: Write failing helper tests**

Append these tests to `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`:

```python
def test_env_state_mode_actfeat_potential_keeps_env_eef():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _env_state_mode_for_training
    assert _env_state_mode_for_training("act_feat", None) == "eef"
    assert _env_state_mode_for_training("act_feat", "eef_piece") == "eef_piece"
    assert _env_state_mode_for_training("eef_piece", None) == "eef_piece"


class _CacheArgs:
    act_feat_cache = None
    allow_act_base_mismatch = False


def test_validate_actfeat_potential_requires_cache_path(tmp_path):
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _validate_actfeat_potential_cache
    p, _, _, _, _ = _make_actfeat_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    with pytest.raises(AssertionError, match="act_feat potential"):
        _validate_actfeat_potential_cache(_CacheArgs(), pot, None, None, None)
```

- [ ] **Step 2: Run the helper tests and verify they fail**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py::test_env_state_mode_actfeat_potential_keeps_env_eef \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py::test_validate_actfeat_potential_requires_cache_path -q
```

Expected: FAIL because the helpers do not exist.

- [ ] **Step 3: Add helper functions**

In `train_chunk_residual.py`, add `import hashlib` near the imports.

Add these helpers after `_offline_buffer_signature()` or before `main()`:

```python
def _array_sha256(x) -> str:
    arr = np.asarray(x, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _env_state_mode_for_training(potential_state_mode, subgoal_state_mode):
    if subgoal_state_mode == "eef_piece":
        return "eef_piece"
    if potential_state_mode == "eef_piece":
        return "eef_piece"
    return "eef"


def _validate_actfeat_potential_cache(args, potential, cache_sig, cache_sha, act_feat_seqs):
    if potential is None or getattr(potential, "state_mode", "eef") != "act_feat":
        return
    assert args.act_feat_cache, "act_feat potential 需 --act_feat_cache"
    assert act_feat_seqs is not None, "act_feat potential 需加载 act_feat cache 序列"
    assert cache_sig is not None, "act_feat potential 需 act_feat cache signature"
    first = torch.as_tensor(act_feat_seqs[0], dtype=torch.float32)
    assert first.ndim == 2, f"act_feat cache demo 应为 [T,D], got {tuple(first.shape)}"
    assert first.shape[1] == potential.model.state_dim, (
        f"act_feat cache dim {first.shape[1]} != value state_dim {potential.model.state_dim}"
    )
    value_sig = potential.act_feat_signature or {}
    for k in ("act_ckpt_id", "image_keys", "proprio_key", "pooling"):
        if value_sig.get(k) is not None:
            assert cache_sig.get(k) == value_sig.get(k), (
                f"act_feat cache 与 value 签名不符 [{k}]: {cache_sig.get(k)} vs {value_sig.get(k)}"
            )
    if cache_sha and potential.act_weight_sha and cache_sha != potential.act_weight_sha:
        raise ValueError(
            f"act_feat cache 与 value 权重指纹不符: {cache_sha} vs {potential.act_weight_sha}"
        )
```

- [ ] **Step 4: Expand offline buffer signature**

Change `_offline_buffer_signature()` signature to:

```python
def _offline_buffer_signature(args, image_keys, offline_cap, shaping_mode, potential=None,
                              act_feat_cache_sig=None, act_feat_cache_sha=None):
```

Inside the `if args.potential_source == "hiql":` block add:

```python
        sig["value_state_dim"] = (int(potential.model.state_dim)
                                  if potential is not None else None)
        if potential is not None and getattr(potential, "feature_mean", None) is not None:
            sig["value_state_stats_sha"] = _array_sha256(
                np.concatenate([
                    potential.feature_mean.detach().cpu().numpy(),
                    potential.feature_std.detach().cpu().numpy(),
                ])
            )
        if potential is not None and getattr(potential, "state_mode", None) == "act_feat":
            sig["act_feat_cache"] = (os.path.abspath(args.act_feat_cache)
                                     if args.act_feat_cache else None)
            sig["act_feat_cache_signature"] = act_feat_cache_sig
            sig["act_feat_cache_weight_sha"] = act_feat_cache_sha
            sig["hiql_value_act_feat_signature"] = potential.act_feat_signature
            sig["hiql_value_act_weight_sha"] = potential.act_weight_sha
```

Change call sites to pass `act_feat_cache_sig=_act_feat_cache_sig` and `act_feat_cache_sha=_act_feat_cache_sha`.

- [ ] **Step 5: Load act_feat cache once in `main()`**

Before env construction, initialize:

```python
    _offline_act_feat_seqs = None
    _act_feat_cache_sig = None
    _act_feat_cache_sha = None
```

After `_subgoal_sm` is known, add:

```python
    _needs_act_feat_cache = (
        (potential is not None and potential.state_mode == "act_feat")
        or (args.subgoal_conditioned and _subgoal_sm == "act_feat")
    )
    if _needs_act_feat_cache:
        assert args.act_feat_cache, "act_feat potential/subgoal 需 --act_feat_cache"
        from resfit.rl_finetuning.chunk_residual.act_feat_cache import load_act_feat_cache
        _offline_act_feat_seqs, _act_feat_stats, _act_feat_cache_sig, _act_feat_cache_sha = \
            load_act_feat_cache(args.act_feat_cache)
        _validate_actfeat_potential_cache(
            args, potential, _act_feat_cache_sig, _act_feat_cache_sha, _offline_act_feat_seqs)
```

Remove the later duplicate initialization of `_offline_act_feat_seqs = None`. In the subgoal act_feat branch, reuse `_offline_act_feat_seqs`, `_act_feat_cache_sig`, and `_act_feat_cache_sha` instead of loading the cache again.

- [ ] **Step 6: Set env state mode correctly**

Replace the existing env-state-mode block with:

```python
    _potential_sm = getattr(potential, "state_mode", None) if potential is not None else None
    env_state_mode = _env_state_mode_for_training(_potential_sm, _subgoal_sm)
```

Keep the existing print, but make the comment explicit:

```python
    # act_feat potential uses images/proprio through PotentialActFeatureEncoder;
    # it does not require env state_mode="act_feat".
```

- [ ] **Step 7: Build online potential feature encoder**

Move:

```python
    image_keys = list(base_policy.config.image_features.keys())
```

before constructing the training `ChunkResidualEnvWrapper`.

Before the wrapper construction, add:

```python
    potential_feature_encoder = None
    if potential is not None and potential.state_mode == "act_feat":
        from resfit.rl_finetuning.chunk_residual.act_feature import (
            ActFeatureExtractor, PotentialActFeatureEncoder,
            act_weight_fingerprint, assert_act_base_samesource)
        assert _act_feat_cache_sig is not None, "act_feat potential cache signature missing"
        potential_image_keys = _act_feat_cache_sig.get("image_keys") or image_keys
        potential_proprio_key = _act_feat_cache_sig.get("proprio_key", "observation.state")
        potential_pooling = _act_feat_cache_sig.get("pooling", "mean")
        assert_act_base_samesource(
            gv_sha=potential.act_weight_sha,
            cache_sha=_act_feat_cache_sha,
            base_sha=act_weight_fingerprint(base_policy),
            allow_mismatch=args.allow_act_base_mismatch,
        )
        potential_extractor = ActFeatureExtractor(
            base_policy,
            potential_image_keys,
            proprio_key=potential_proprio_key,
            pooling=potential_pooling,
            proprio_dim=state_standardizer._mean.numel(),
        )
        potential_feature_encoder = PotentialActFeatureEncoder(
            potential_extractor,
            state_standardizer,
            potential,
            proprio_key=potential_proprio_key,
        )
```

Pass it into the training env:

```python
                                  potential=potential,
                                  potential_feature_encoder=potential_feature_encoder)
```

Do not pass it to `eval_env`.

- [ ] **Step 8: Fix potential dimension assertion**

Replace the current `exp_v_dim` assertion with:

```python
    if potential is not None:
        if potential.state_mode == "act_feat":
            assert _offline_act_feat_seqs is not None, "act_feat potential requires loaded act_feat sequences"
            feat_dim = int(torch.as_tensor(_offline_act_feat_seqs[0]).shape[1])
            assert potential.model.state_dim == feat_dim, (
                f"Φ act_feat value 输入维度 {potential.model.state_dim} != act_feat cache dim {feat_dim}")
        else:
            exp_v_dim = state_dim + (12 if env_state_mode == "eef_piece" else 0)
            assert potential.model.state_dim == exp_v_dim, (
                f"Φ value 输入维度 {potential.model.state_dim} != observation.state({state_dim})"
                f"+rel({12 if env_state_mode == 'eef_piece' else 0});value.pt 的 state_mode 与 task 不匹配")
```

- [ ] **Step 9: Run helper and parser tests**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add \
  resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat: wire actfeat hiql potential training"
```

---

### Task 6: End-To-End Verification

**Files:**
- Verify only; no source edits expected.

**Interfaces:**
- Consumes: all tasks above.
- Produces: a verified codebase ready to train new act_feat potential experiments.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py -q
```

Expected: PASS.

- [ ] **Step 2: Syntax-check modified modules**

Run:

```bash
python -m py_compile \
  resfit/rl_finetuning/chunk_residual/hiql_value.py \
  resfit/rl_finetuning/chunk_residual/train_hiql_value.py \
  resfit/rl_finetuning/chunk_residual/hiql_potential.py \
  resfit/rl_finetuning/chunk_residual/act_feature.py \
  resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py \
  resfit/rl_finetuning/chunk_residual/offline_stage_replay.py \
  resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Smoke-check CLI parsing**

Run:

```bash
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly \
  --reward_shaping potential \
  --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/nonexistent_actfeat_value.pt \
  --act_feat_cache outputs_chunk/nonexistent_actfeat_cache.npz \
  --smoke
```

Expected: FAIL quickly with the existing missing checkpoint assertion. This confirms the parser accepts the new combination without requiring environment startup.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --stat
git diff --check
```

Expected: `git diff --check` has no whitespace errors.

- [ ] **Step 5: Report rollout commands**

Report that new experiments require act_feat scalar value checkpoints first, for example:

```bash
python -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --state_mode act_feat \
  --act_feat_cache outputs_chunk/<task>_act_feat_cache.npz \
  --act_base_ckpt <same ACT base dir used by the residual run> \
  --hdf5 <task hdf5> \
  --dataset <lerobot dataset id> \
  --output outputs_chunk/<task>_value_actfeat.pt
```

Then launch residual runs with:

```bash
--reward_shaping potential \
--potential_source hiql \
--hiql_value_ckpt outputs_chunk/<task>_value_actfeat.pt \
--act_feat_cache outputs_chunk/<task>_act_feat_cache.npz
```

- [ ] **Step 6: Final commit**

If Task 6 changed no files, skip this step. If verification required a small fix, commit only that fix:

```bash
git add <fixed files>
git commit -m "fix: verify actfeat hiql potential wiring"
```

---

## Self-Review

- Spec coverage: Task 1 covers checkpoint stats and metadata; Task 2 covers online act feature construction; Task 3 covers offline reward shaping from cached act features; Task 4 covers wrapper Phi routing; Task 5 covers train entry validation, env mode, signatures, and cache reuse; Task 6 covers verification and rollout.
- Placeholder scan: no banned placeholder wording is present in task steps.
- Type consistency: `act_feat_seq` is consistently a `[T, D]` tensor/array; `PotentialActFeatureEncoder.encode()` returns a standardized `[B, D]` tensor; `HiqlPotential.phi()` receives already-standardized act_feat vectors for `state_mode="act_feat"`.
