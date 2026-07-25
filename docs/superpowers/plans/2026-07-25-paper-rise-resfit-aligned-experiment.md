# Paper RISE–ResFiT Aligned Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the committed Block/Cup imagination bridge with a verified Paper task profile, build Paper policy features and compact value, then launch `paper_shore_mixed50_seed0` aligned with W&B run `eegyfzmz`.

**Architecture:** Treat commit `9247b16` as the immutable behavioral baseline and add Paper through the existing task-profile path. Use one shared 16-to-14 TeleAvatar policy-state mapper in both offline feature extraction and online AWBC inference, while keeping the residual learner and compact value on full 16-dimensional proprioception. Gate the formal run on unit tests, a live Paper AWBC smoke, signed feature/value provenance, service health, and a disabled-W&B end-to-end smoke.

**Tech Stack:** Python 3.10, pytest, NumPy, PyTorch, JAX/OpenPI/Orbax, LeRobot v2.1, websocket policy servers, Bash/tmux, W&B.

## Global Constraints

- Design authority: `docs/superpowers/specs/2026-07-25-paper-rise-resfit-aligned-experiment-design.md`.
- Cup baseline: commit `9247b16`; existing Block and Cup behavior and tests must remain green.
- Existing jobs on GPUs 0, 2, 5, and 7 must not be stopped.
- Recheck GPU occupancy immediately before every service allocation.
- Do not delete any file without explicit user approval.
- All non-deleting implementation, testing, service, preprocessing, and launch actions are pre-authorized.
- Paper prompt is exactly `put the paper roll on the holder`.
- Paper AWBC checkpoint is `/mnt/mnt/data/data2/kai0/checkpoints/paper/19999`.
- Paper AWBC asset ID is `pick_paper_all_merged`.
- Paper policy input state is 14-dimensional; dataset/residual state and action are 16-dimensional.
- Paper feature pooling is `mean`; serve identity is `pi05_paper_awbc_19999`.
- Paper data remains native 20 Hz; `chunk_length=50` therefore represents 2.5 seconds.
- Formal W&B name is exactly `paper_shore_mixed50_seed0`; seed is 0; total steps are 500000.
- Work only on files named in the current task and preserve unrelated dirty-worktree changes.

---

### Task 1: Add the Paper profile and shared policy-state mapping

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/teleavatar_policy_state.py`
- Modify: `resfit/rl_finetuning/wm_bridge/task_profiles.py`
- Modify: `resfit/rl_finetuning/wm_bridge/base_bridge.py`
- Modify: `resfit/rl_finetuning/wm_bridge/builder.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_base_bridge.py`

**Interfaces:**
- Produces: `map_teleavatar_policy_state(state, target_dim) -> np.ndarray`.
- Produces: `TaskProfile.policy_state_dim`, `TaskProfile.dataset_fps`, and the `paper` profile.
- Consumes later: offline feature construction, online base inference, metadata, and launch contracts.

- [ ] **Step 1: Write failing mapper/profile tests**

Add tests equivalent to:

```python
from resfit.rl_finetuning.wm_bridge.teleavatar_policy_state import (
    map_teleavatar_policy_state,
)

def test_paper_profile_is_exact():
    paper = get_task_profile("paper")
    assert (
        paper.prompt,
        paper.success_dataset,
        paper.action_dim,
        paper.policy_state_dim,
        paper.dataset_fps,
    ) == (
        "put the paper roll on the holder",
        "paper_success",
        16,
        14,
        20.0,
    )

def test_policy_state_14_drops_only_grippers():
    state = np.arange(16, dtype=np.float32)
    got = map_teleavatar_policy_state(state, 14)
    np.testing.assert_array_equal(
        got,
        np.concatenate([state[:7], state[8:15]]),
    )

def test_policy_state_16_is_identity():
    state = np.arange(16, dtype=np.float32)
    np.testing.assert_array_equal(
        map_teleavatar_policy_state(state, 16),
        state,
    )

@pytest.mark.parametrize("target_dim", [13, 15, 17])
def test_invalid_policy_state_dim_fails(target_dim):
    with pytest.raises(ValueError, match="14 or 16"):
        map_teleavatar_policy_state(np.zeros(16), target_dim)
```

Change the unknown-profile test to use `"nonexistent"` instead of `"paper"`.
Add a base-bridge test asserting `_serve_obs(..., policy_state_dim=14)` sends
`[0..6,8..14]`.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest -q \
  resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py \
  resfit/rl_finetuning/wm_bridge/tests/test_base_bridge.py
```

Expected: failures because the mapper, Paper profile, and
`policy_state_dim` constructor argument do not exist.

- [ ] **Step 3: Implement the mapper and profile**

Create:

```python
from __future__ import annotations

import numpy as np


def map_teleavatar_policy_state(state, target_dim: int) -> np.ndarray:
    vector = np.asarray(state, dtype=np.float32).reshape(-1)
    if vector.shape != (16,):
        raise ValueError(f"TeleAvatar source state must be (16,), got {vector.shape}")
    if target_dim == 16:
        return vector
    if target_dim == 14:
        return np.concatenate([vector[:7], vector[8:15]]).astype(
            np.float32, copy=False
        )
    raise ValueError(f"policy state dim must be 14 or 16, got {target_dim}")
```

Extend `TaskProfile`:

```python
@dataclass(frozen=True)
class TaskProfile:
    name: str
    prompt: str
    success_dataset: str
    action_dim: int = 16
    policy_state_dim: int = 16
    dataset_fps: float = 30.0
    feature_pooling: str = "mean"
```

Add:

```python
"paper": TaskProfile(
    "paper",
    "put the paper roll on the holder",
    "paper_success",
    action_dim=16,
    policy_state_dim=14,
    dataset_fps=20.0,
),
```

Update bridge parser choices to `("block", "cup", "paper")`. Add
`policy_state_dim=profile.policy_state_dim` when constructing
`Kai0ImaginationBase`, and map the state in `Kai0ImaginationBase._serve_obs`.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the Task 1 command again.

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  resfit/rl_finetuning/wm_bridge/teleavatar_policy_state.py \
  resfit/rl_finetuning/wm_bridge/task_profiles.py \
  resfit/rl_finetuning/wm_bridge/base_bridge.py \
  resfit/rl_finetuning/wm_bridge/builder.py \
  resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py \
  resfit/rl_finetuning/wm_bridge/tests/test_base_bridge.py
git commit -m "feat: add paper policy-state profile"
```

### Task 2: Make offline Paper features use the same 16-to-14 mapping

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/build_pi0_feat_cache_via_serve.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache_teleavatar.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_build_pi0_feat_via_serve.py`

**Interfaces:**
- Consumes: `map_teleavatar_policy_state`.
- Produces: `build_main_teleavatar(..., policy_state_dim=16)` and CLI
  `--policy_state_dim {14,16}`.
- Guarantees: AWBC receives mapped state; cached proprio remains full 16D.

- [ ] **Step 1: Write failing offline-mapping tests**

Add a recording stub and assert:

```python
def test_paper_policy_state_is_14_but_cached_proprio_stays_16(
        tmp_path, monkeypatch):
    _patch_lerobot(monkeypatch, n_eps=1, T=4)
    client = _RecordingClient()
    out = str(tmp_path / "paper.npz")
    build_main_teleavatar(
        client,
        lerobot_root="/x",
        repo_id="paper_success",
        prompt="put the paper roll on the holder",
        pooling="mean",
        serve_ckpt_id="pi05_paper_awbc_19999",
        out_cache=out,
        policy_state_dim=14,
    )
    assert all(obs["state"].shape == (14,) for obs in client.observations)
    seqs, _, sig = load_pi0_feat_cache(out)
    assert seqs[0].shape[1] == 8 + 16
    assert sig["policy_state_dim"] == 14
```

Add parser coverage for `--policy_state_dim 14`.

- [ ] **Step 2: Run tests and confirm RED**

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest -q \
  resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache_teleavatar.py \
  resfit/rl_finetuning/chunk_residual/tests/test_build_pi0_feat_via_serve.py
```

Expected: unexpected keyword/unknown CLI failures.

- [ ] **Step 3: Implement mapped serve state and signed provenance**

Add `policy_state_dim` to `build_main_teleavatar`, map only the state passed to
`build_teleavatar_serve_obs`, retain `fr["state"]` in `proprios`, and add
`policy_state_dim` to the TeleAvatar `pi0_feat_signature`.

Add:

```python
ap.add_argument("--policy_state_dim", type=int, choices=(14, 16), default=16)
```

Pass it from `main()` to `build_main_teleavatar`.

- [ ] **Step 4: Run tests and confirm GREEN**

Run the Task 2 command again.

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  resfit/rl_finetuning/chunk_residual/build_pi0_feat_cache_via_serve.py \
  resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache_teleavatar.py \
  resfit/rl_finetuning/chunk_residual/tests/test_build_pi0_feat_via_serve.py
git commit -m "feat: align paper offline policy state"
```

### Task 3: Parameterize the AWBC feature server and expose identity metadata

**Files:**
- Modify: `pi0_serve/feature_policy.py`
- Modify: `pi0_serve/serve_block_awbc.py`
- Test: `pi0_serve/tests/test_feature_policy.py`
- Create: `pi0_serve/tests/test_serve_block_awbc.py`

**Interfaces:**
- Produces: `wrap_with_feature(..., metadata_overrides=None)`.
- Produces CLI flags: `--asset_id`, `--serve_ckpt_id`.
- Server metadata keys: `serve_ckpt_id`, `pooling`, `asset_id`,
  `policy_state_dim`.

- [ ] **Step 1: Write failing metadata/config tests**

Add:

```python
def test_metadata_overrides_are_merged():
    inner = _Inner(metadata={"model": "pi05"})
    wrapped = wrap_with_feature(
        inner,
        pooling="mean",
        prefix_feat_fn=lambda *args: np.zeros(2, np.float32),
        metadata_overrides={"serve_ckpt_id": "paper-19999", "asset_id": "paper"},
    )
    assert wrapped.metadata == {
        "model": "pi05",
        "serve_ckpt_id": "paper-19999",
        "asset_id": "paper",
    }
```

In the serve test, monkeypatch `pick_cup_configs.make_awbc_config` and
`create_trained_policy`; assert `_create_policy` replaces only
`train_config.data.assets.asset_id` with `pick_paper_all_merged`.

- [ ] **Step 2: Run tests and confirm RED**

```bash
/mnt/mnt/data/chj/openpi/.venv/bin/python -m pytest -q \
  pi0_serve/tests/test_feature_policy.py \
  pi0_serve/tests/test_serve_block_awbc.py
```

Expected: missing arguments/functions.

- [ ] **Step 3: Implement additive server parameters**

Extend `FeaturePolicy` with a copied metadata override dictionary and merge it
without mutating `inner.metadata`.

Extend `Args`:

```python
asset_id: str = "inference"
serve_ckpt_id: str = "pi05_block_awbc_49999"
policy_state_dim: int = 16
```

After `make_awbc_config`, replace the nested asset ID:

```python
assets = dataclasses.replace(
    train_config.data.assets,
    asset_id=args.asset_id,
)
train_config = dataclasses.replace(
    train_config,
    data=dataclasses.replace(train_config.data, assets=assets),
)
```

Pass metadata overrides when wrapping:

```python
metadata = {
    "serve_ckpt_id": args.serve_ckpt_id,
    "pooling": args.pooling,
    "asset_id": args.asset_id,
    "policy_state_dim": args.policy_state_dim,
}
policy = wrap_with_feature(
    inner,
    pooling=args.pooling,
    metadata_overrides=metadata,
)
```

Keep all Block defaults backward compatible.

- [ ] **Step 4: Run tests and confirm GREEN**

Run the Task 3 command again.

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  pi0_serve/feature_policy.py \
  pi0_serve/serve_block_awbc.py \
  pi0_serve/tests/test_feature_policy.py \
  pi0_serve/tests/test_serve_block_awbc.py
git commit -m "feat: parameterize teleavatar awbc serve"
```

### Task 4: Add Paper provenance and server-contract gates

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/builder.py`
- Modify: `resfit/rl_finetuning/wm_bridge/contract.py`
- Modify: `resfit/rl_finetuning/wm_bridge/block_offline_cache.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_contract.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_block_offline_cache.py`

**Interfaces:**
- Consumes: profile FPS/state/pooling and server metadata.
- Produces bridge CLI metadata flags and a complete
  `bridge_run_config.json`.
- Produces: `check_pi0_server_metadata(...)`.

- [ ] **Step 1: Write failing provenance and mismatch tests**

Cover:

```python
def test_paper_metadata_records_physical_clock_and_awbc_provenance(...):
    assert meta["task_profile"] == "paper"
    assert meta["dataset_fps"] == 20.0
    assert meta["physical_chunk_seconds"] == 2.5
    assert meta["pi0_policy_state_dim"] == 14
    assert meta["pi0_serve_ckpt_dir"].endswith("/paper/19999")
    assert meta["pi0_asset_id"] == "pick_paper_all_merged"
    assert meta["source_wandb_run"] == "eegyfzmz"

def test_pi0_server_metadata_rejects_wrong_checkpoint():
    with pytest.raises(ContractError, match="serve_ckpt_id"):
        check_pi0_server_metadata(
            {"serve_ckpt_id": "block", "pooling": "mean",
             "policy_state_dim": 14, "asset_id": "pick_paper_all_merged"},
            expected_ckpt_id="pi05_paper_awbc_19999",
            expected_pooling="mean",
            expected_state_dim=14,
            expected_asset_id="pick_paper_all_merged",
        )
```

Add endpoint fingerprint coverage proving `policy_state_dim=14` and `16` produce
different keys.

- [ ] **Step 2: Run tests and confirm RED**

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest -q \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_cache.py
```

Expected: missing fields/check function.

- [ ] **Step 3: Implement provenance and gates**

Add optional bridge flags:

```text
--source_wandb_run
--pi0_serve_ckpt_dir
--pi0_asset_id
--pi0_pooling
--adv_ckpt
--adv_config
```

Use profile fields for `dataset_fps`, `policy_state_dim`, and default pooling.
When checkpoint directory and asset ID are present, require
`assets/<asset_id>/norm_stats.json`, hash it, and record the SHA-256.

Include `policy_state_dim` in the endpoint fingerprint payload. Validate live
server metadata when `WebsocketClientPolicy` is constructed, before the first
environment reset.

- [ ] **Step 4: Run tests and confirm GREEN**

Run the Task 4 command again.

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  resfit/rl_finetuning/wm_bridge/builder.py \
  resfit/rl_finetuning/wm_bridge/contract.py \
  resfit/rl_finetuning/wm_bridge/block_offline_cache.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_offline_cache.py
git commit -m "feat: gate paper bridge provenance"
```

### Task 5: Add Paper preparation, value, and formal launch scripts

**Files:**
- Create: `serve_paper_awbc.sh`
- Create: `prepare_paper_pi0feat.sh`
- Create: `train_paper_pi0feat_value.sh`
- Create: `launch_paper_imagination.sh`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`

**Interfaces:**
- Produces four temporary Paper AWBC servers.
- Produces eight feature shards and `paper_value_pi0feat.pt`.
- Produces formal/SMOKE launch commands.

- [ ] **Step 1: Write failing script tests**

Require exact tokens:

```text
--task_profile paper
--policy_state_dim 14
--pi0_serve_ckpt_id pi05_paper_awbc_19999
--pi0_asset_id pick_paper_all_merged
--pi0_prompt "put the paper roll on the holder"
--dataset paper_success
--task pick_paper_roll
paper_value_pi0feat.pt
paper_shore_mixed50_seed
```

Capture actual launcher argv as the Cup baseline test does and parse it with
both bridge and trainer parsers. Assert seed 0 formal hyperparameters match the
reference run.

- [ ] **Step 2: Run the script tests and confirm RED**

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest -q \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py
```

Expected: missing Paper scripts/profile assertions.

- [ ] **Step 3: Create the scripts**

`serve_paper_awbc.sh <GPU> <PORT>` must run the existing serve entry with:

```text
--dir /mnt/mnt/data/data2/kai0/checkpoints/paper/19999
--repo_id /mnt/mnt/data/domains_rise/paper/paper_success
--config_name pi05_paper_awbc
--default_prompt "put the paper roll on the holder"
--asset_id pick_paper_all_merged
--serve_ckpt_id pi05_paper_awbc_19999
--policy_state_dim 14
--pooling mean
```

`prepare_paper_pi0feat.sh` accepts four ports, uses four non-overlapping shards,
and passes `--policy_state_dim 14` for success and failure.

`train_paper_pi0feat_value.sh` consumes exactly four success and four failure
shards, trains 50000 steps with success-signed targets, and writes
`outputs_chunk/paper_value_pi0feat.pt`.

`launch_paper_imagination.sh` mirrors the aligned Cup launcher but uses Paper
paths/profile/provenance. Support `SMOKE=1` by adding `--smoke` and writing to a
`_smoke` output directory; formal mode keeps the exact W&B name.

- [ ] **Step 4: Run the script tests and confirm GREEN**

Run the Task 5 test command again.

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  serve_paper_awbc.sh \
  prepare_paper_pi0feat.sh \
  train_paper_pi0feat_value.sh \
  launch_paper_imagination.sh \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py
git commit -m "feat: add aligned paper experiment scripts"
```

### Task 6: Run the complete regression gate

**Files:**
- No production changes expected.

**Interfaces:**
- Verifies all Block, Cup, Paper, feature-server, cache, and contract behavior.

- [ ] **Step 1: Run focused complete suites**

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest -q \
  resfit/rl_finetuning/wm_bridge/tests \
  resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache_teleavatar.py \
  resfit/rl_finetuning/chunk_residual/tests/test_build_pi0_feat_via_serve.py \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_value_pi0_feat.py

/mnt/mnt/data/chj/openpi/.venv/bin/python -m pytest -q \
  pi0_serve/tests
```

Expected: zero failures.

- [ ] **Step 2: Run static checks**

```bash
git diff --check
bash -n \
  serve_paper_awbc.sh \
  prepare_paper_pi0feat.sh \
  train_paper_pi0feat_value.sh \
  launch_paper_imagination.sh
```

Expected: exit 0.

- [ ] **Step 3: Record the regression evidence**

Write the exact pass counts and commit hashes to
`outputs_chunk/paper_pi0feat_logs/verification.txt` using a normal command that
does not modify tracked files.

### Task 7: Live-smoke Paper AWBC and build feature shards

**Files produced:**
- `outputs_chunk/paper_pi0feat_shards/*.npz`
- `outputs_chunk/paper_pi0feat_shards/shard*.log`
- `logs/paper_awbc_feature_gpu*.log`

**Interfaces:**
- Consumes the Paper AWBC and four free GPUs.
- Produces eight signed feature shards.

- [ ] **Step 1: Recheck resources and ports**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
ss -ltnp | rg ':(8101|8102|8103|8104)\b'
```

Expected: four GPUs are free and ports 8101–8104 are unused. Do not stop
unrelated processes.

- [ ] **Step 2: Start four temporary Paper serves**

Start one tmux session per free GPU, each running:

```bash
bash /mnt/mnt/data/resfit/serve_paper_awbc.sh <GPU> <PORT>
```

Use ports 8101–8104 and separate logs.

- [ ] **Step 3: Run a real one-episode smoke**

Run the feature builder on one Paper success episode with
`--policy_state_dim 14`, then load the `.npz` and assert:

```python
assert signature["serve_ckpt_id"] == "pi05_paper_awbc_19999"
assert signature["policy_state_dim"] == 14
assert signature["pooling"] == "mean"
assert all(np.isfinite(array).all() for array in seqs)
```

Also issue one direct websocket request and assert actions `(50,16)`, finite
actions, finite one-dimensional `prefix_feat`, and exact server metadata.

- [ ] **Step 4: Launch four-shard extraction**

```bash
bash prepare_paper_pi0feat.sh 8101 8102 8103 8104
```

Expected: eight `.npz` files, four success and four failure, and all shard
processes exit 0.

- [ ] **Step 5: Validate every shard**

Load all eight shards and require identical prompt, pooling, policy-state
dimension, checkpoint identity, image keys, and proprio key. Require total
episode counts 849 success and 53 failure.

### Task 8: Train and validate the Paper compact value

**Files produced:**
- `outputs_chunk/paper_value_pi0feat.pt`
- `outputs_chunk/paper_pi0feat_logs/train_value.log`

- [ ] **Step 1: Train the compact value**

```bash
bash train_paper_pi0feat_value.sh
```

Expected: exit 0 and a nonempty `paper_value_pi0feat.pt`.

- [ ] **Step 2: Validate the checkpoint**

Load with the repository value loader and assert:

```python
assert info["state_mode"] == "pi0_feat"
assert info["pi0_feat_signature"]["serve_ckpt_id"] == \
    "pi05_paper_awbc_19999"
assert info["pi0_feat_signature"]["policy_state_dim"] == 14
assert info["pi0_feat_signature"]["prompt"] == \
    "put the paper roll on the holder"
assert info["pi0_feat_signature"]["pooling"] == "mean"
```

Require finite mean/std and input dimension equal to live prefix dimension + 16.

### Task 9: Start formal services and pass disabled-W&B smoke

**Files produced:**
- `logs/paper_wm.log`
- `logs/paper_awbc.log`
- `logs/paper_adv.log`
- `outputs_imagination/paper_shore_mixed50_seed0_smoke/`

- [ ] **Step 1: Stop only the temporary Paper feature serves**

Terminate only tmux sessions created in Task 7. This does not delete files and
must not affect unrelated experiments.

- [ ] **Step 2: Recheck four free GPUs and formal ports**

Require free GPUs and unused ports 9000, 8001, 8002.

- [ ] **Step 3: Start the three formal services**

Start:

```text
RISE D serve:
  config=/mnt/mnt/data/resfit/RISE_Hi/ckpt/block_infer.yaml
  port=9000

Paper AWBC:
  checkpoint=/mnt/mnt/data/data2/kai0/checkpoints/paper/19999
  port=8001
  pooling=mean

Paper value monitor:
  ckpt=.../value_paper/value_paper/20000/model.safetensors
  config_name=value_paper
  port=8002
```

Wait for each port and inspect logs for traceback/OOM before continuing.

- [ ] **Step 4: Run disabled-W&B end-to-end smoke**

```bash
SMOKE=1 bash launch_paper_imagination.sh <FREE_TRAIN_GPU> 0
```

Expected: exit 0, bridge metadata contains Paper provenance, at least one
environment transition completes, and no online W&B run is created.

### Task 10: Launch and verify the formal W&B experiment

**Files produced:**
- `outputs_imagination/paper_shore_mixed50_seed0/`
- `logs/paper_shore_mixed50_seed0.log`
- W&B run in `dexmg-chunk-residual`

- [ ] **Step 1: Launch in a dedicated tmux session**

```bash
bash launch_paper_imagination.sh <FREE_TRAIN_GPU> 0
```

Use tmux session `paper_shore_mixed50_seed0`.

- [ ] **Step 2: Verify local startup**

Require:

- the process remains alive;
- no traceback/OOM/signature mismatch in the log;
- `bridge_run_config.json` exists and contains Paper provenance;
- W&B initialization prints a run ID.

- [ ] **Step 3: Verify W&B**

Query the API and require:

```text
project=dexmg-chunk-residual
name=paper_shore_mixed50_seed0
state=running
seed=0
total_env_steps=500000
chunk_length=50
offline_fraction=0.5
demo_bc_coef=0.1
gamma=0.995
n_step=1
utd=4
```

- [ ] **Step 4: Verify progress**

Poll the local log and W&B history until the first metric is present and the
step is greater than zero. Record the run ID, display name, GPU allocation,
service tmux sessions, and log paths for handoff.

