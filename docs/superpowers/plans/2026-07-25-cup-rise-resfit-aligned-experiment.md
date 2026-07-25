# Cup RISE + ResFiT Aligned Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the Block-only imagination bridge for Cup, build task-correct Cup feature/value/replay artifacts, and start a 500k Cup run aligned with W&B run `eegyfzmz`.

**Architecture:** Introduce a small immutable task-profile boundary and pass it through the existing sampler, contract, offline-runtime, and imagination-factory paths. Keep the Block defaults backward compatible, add Cup-specific reproducible launch/preprocessing scripts, then gate the real run behind unit tests, a live two-demo smoke test, full feature/value construction, service health checks, and offline replay validation.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, Bash, LeRobot v2.1 datasets, websocket kai0/RISE services, W&B.

## Global Constraints

- Preserve all unrelated tracked, staged, deleted, and untracked user files.
- Make code edits with `apply_patch`.
- Use TDD: failing focused test, minimal implementation, passing focused test, then wider regression.
- Keep Block defaults exactly `build block`, `block_success`, action dimension 16, and `pi05_block_awbc_49999`.
- Cup uses `pick cup`, `cup_success`, `cup_fail`, action dimension 16, and `pi05_pick_cup_awbc_49999`.
- Formal RL uses batch 256 split 128 online plus 128 offline, chunk length 50, fixed BC coefficient 0.1, UTD 4, and 500,000 environment steps.
- Do not launch formal RL with a dummy scorer, mismatched feature anchor, unhealthy service, NaN/Inf smoke output, or an occupied GPU 7.
- Do not terminate a process until its PID, command line, port, and GPU assignment have been resolved read-only.
- The completed Block RL run remains untouched; only the GPU 6 Block monitoring service is authorized for replacement.

---

## File Structure

- Create `resfit/rl_finetuning/wm_bridge/task_profiles.py`: immutable task definitions and profile lookup.
- Modify `resfit/rl_finetuning/wm_bridge/init_states.py`: retain Block compatibility constant while allowing callers to provide a caption.
- Modify `resfit/rl_finetuning/wm_bridge/teleavatar_start_sampler.py`: task-provided caption for sampled start states.
- Modify `resfit/rl_finetuning/wm_bridge/contract.py`: validate mixed replay against the selected task profile rather than Block literals.
- Modify `resfit/rl_finetuning/wm_bridge/builder.py`: parse `--task_profile`, select prompts/sources dynamically, and record task-correct metadata.
- Create `resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py`: focused profile and sampler tests.
- Modify `resfit/rl_finetuning/wm_bridge/tests/test_contract.py`: Cup acceptance and mismatch rejection.
- Modify `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`: Block regression plus Cup runtime/factory assertions.
- Create `prepare_cup_pi0feat.sh`: reproducible three-shard Cup feature extraction against three serve ports.
- Create `train_cup_pi0feat_value.sh`: reproducible final Cup `pi0_feat` value training.
- Create `launch_cup_imagination.sh`: aligned formal Cup RL launcher.
- Test the scripts with `bash -n` and argument/static-content tests.

---

### Task 1: Add the Task-Profile Boundary and Caption Injection

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/task_profiles.py`
- Modify: `resfit/rl_finetuning/wm_bridge/init_states.py`
- Modify: `resfit/rl_finetuning/wm_bridge/teleavatar_start_sampler.py`
- Create: `resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py`

**Interfaces:**
- Produces: `TaskProfile(name: str, prompt: str, success_dataset: str, action_dim: int)`
- Produces: `get_task_profile(name: str) -> TaskProfile`
- Produces: `TeleavatarStartSampler(dataset_roots, rng=None, caption=BLOCK_CAPTION)`
- Consumed later by: `builder.prepare_offline_runtime()` and `builder.build_imagination_factories()`

- [ ] **Step 1: Write failing profile and sampler tests**

```python

import numpy as np
import pandas as pd
import pytest

from resfit.rl_finetuning.wm_bridge.task_profiles import get_task_profile
from resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler import (
    TeleavatarStartSampler,
)


def test_builtin_task_profiles_are_exact():
    block = get_task_profile("block")
    cup = get_task_profile("cup")
    assert (block.prompt, block.success_dataset, block.action_dim) == (
        "build block", "block_success", 16)
    assert (cup.prompt, cup.success_dataset, cup.action_dim) == (
        "pick cup", "cup_success", 16)


def test_unknown_task_profile_fails():
    with pytest.raises(ValueError, match="unknown task profile"):
        get_task_profile("paper")


def test_start_sampler_uses_injected_caption(monkeypatch):
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.glob.glob",
        lambda pattern: [
            "/data/cup_success/data/chunk-000/episode_000000.parquet"
        ],
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.pd.read_parquet",
        lambda path, columns: (
            pd.DataFrame({"frame_index": range(4)})
            if columns == ["frame_index"]
            else pd.DataFrame({
                "observation.state": [
                    np.zeros(16, dtype=np.float32) for _ in range(4)
                ]
            })
        ),
    )

    class Capture:
        def set(self, *args):
            return None

        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            return None

    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.cv2.VideoCapture",
        lambda path: Capture(),
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.cv2.cvtColor",
        lambda image, code: image,
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.cv2.resize",
        lambda image, size: np.zeros((192, 256, 3), dtype=np.uint8),
    )
    sampler = TeleavatarStartSampler(
        ["/data/cup_success"],
        rng=np.random.default_rng(0),
        caption="pick cup",
    )
    assert sampler.sample().caption == "pick cup"
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py -v
```

Expected: collection fails because `task_profiles` does not exist or the sampler
does not accept `caption`.

- [ ] **Step 3: Implement immutable profiles and caption injection**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    name: str
    prompt: str
    success_dataset: str
    action_dim: int = 16


_PROFILES = {
    "block": TaskProfile("block", "build block", "block_success"),
    "cup": TaskProfile("cup", "pick cup", "cup_success"),
}


def get_task_profile(name: str) -> TaskProfile:
    try:
        return _PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown task profile {name!r}; expected one of "
            f"{sorted(_PROFILES)}"
        ) from exc
```

In `TeleavatarStartSampler.__init__`, add:

```python
def __init__(self, dataset_roots, rng=None, caption=BLOCK_CAPTION):
    # existing dataset catalog logic remains unchanged
    self.caption = str(caption)
```

In `sample()`, replace the fixed caption:

```python
return InitState(
    obs_window=obs_window,
    proprio=proprio,
    caption=self.caption,
)
```

Keep `BLOCK_CAPTION = "build block"` in `init_states.py` for API and test
compatibility. Do not rename or delete it.

- [ ] **Step 4: Run focused and existing sampler tests**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py \
  resfit/rl_finetuning/wm_bridge/tests/test_init_states.py -v
```

Expected: all tests pass, including the existing Block caption assertion.

- [ ] **Step 5: Commit the profile boundary**

```bash
git add \
  resfit/rl_finetuning/wm_bridge/task_profiles.py \
  resfit/rl_finetuning/wm_bridge/init_states.py \
  resfit/rl_finetuning/wm_bridge/teleavatar_start_sampler.py \
  resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py
git commit -m "feat: add teleavatar task profiles"
```

---

### Task 2: Make the Mixed-Replay Contract Task-Aware

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/contract.py`
- Modify: `resfit/rl_finetuning/wm_bridge/tests/test_contract.py`

**Interfaces:**
- Consumes: `TaskProfile` fields from Task 1.
- Produces:
  `check_mixed_replay_args(trainer_args, offline_chunk_dataset, *, expected_prompt="build block", expected_source="block_success") -> None`
- Preserves: all current two-positional-argument Block callers.

- [ ] **Step 1: Add failing Cup contract tests**

```python
def test_cup_mixed_replay_contract_accepts_matching_profile():
    args = _mixed_args()
    args.pi0_prompt = "pick cup"
    args.dataset = "cup_success"
    contract.check_mixed_replay_args(
        args,
        "/data/cup_success",
        expected_prompt="pick cup",
        expected_source="cup_success",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pi0_prompt", "build block", "pick cup"),
        ("dataset", "block_success", "cup_success"),
    ],
)
def test_cup_mixed_replay_contract_rejects_cross_task_values(
    field, value, message
):
    args = _mixed_args()
    args.pi0_prompt = "pick cup"
    args.dataset = "cup_success"
    setattr(args, field, value)
    with pytest.raises(contract.ContractError, match=message):
        contract.check_mixed_replay_args(
            args,
            "/data/cup_success",
            expected_prompt="pick cup",
            expected_source="cup_success",
        )


def test_cup_contract_rejects_failure_dataset_as_offline_source():
    args = _mixed_args()
    args.pi0_prompt = "pick cup"
    args.dataset = "cup_fail"
    with pytest.raises(contract.ContractError, match="cup_success"):
        contract.check_mixed_replay_args(
            args,
            "/data/cup_fail",
            expected_prompt="pick cup",
            expected_source="cup_success",
        )
```

- [ ] **Step 2: Run new contract tests and verify failure**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py -k cup -v
```

Expected: FAIL because `check_mixed_replay_args` does not accept the new keyword
arguments and still hard-codes Block.

- [ ] **Step 3: Implement parameterized validation**

Change the signature to:

```python
def check_mixed_replay_args(
    trainer_args,
    offline_chunk_dataset,
    *,
    expected_prompt="build block",
    expected_source="block_success",
) -> None:
```

Replace the prompt check with:

```python
(
    "pi0_prompt",
    getattr(trainer_args, "pi0_prompt", None) == expected_prompt,
    f"must equal {expected_prompt!r}",
),
```

Replace the final source restriction with:

```python
if os.path.basename(source) != expected_source:
    raise ContractError(
        "offline_chunk_dataset must resolve to "
        f"{expected_source}, got {source!r}"
    )
```

Do not relax any algorithmic invariant: fraction 0.5, batch 256, `pi05`,
`replan`, chunk 50, n-step 1, raw actor, no relabel/stage/subgoal/joint value
updates, action dimension 16, and `hdf5` remain mandatory.

- [ ] **Step 4: Run all contract tests**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py -v
```

Expected: all existing Block and new Cup contract tests pass.

- [ ] **Step 5: Commit the task-aware contract**

```bash
git add \
  resfit/rl_finetuning/wm_bridge/contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py
git commit -m "feat: validate mixed replay by task"
```

---

### Task 3: Thread the Selected Task Through the Bridge Runtime

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/builder.py`
- Modify: `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`

**Interfaces:**
- Consumes: `get_task_profile(bridge_args.task_profile)`.
- Produces bridge CLI: `--task_profile {block,cup}`, default `block`.
- Produces task-correct translated trainer args, start-state captions, advantage
  prompt, cache fingerprints, and `bridge_meta["offline_source"]`.

- [ ] **Step 1: Add failing builder/runtime tests**

```python
def test_parse_bridge_args_defaults_to_block_profile():
    args, _ = builder.parse_bridge_args([
        "--value_ckpt", "/tmp/value.pt",
    ])
    assert args.task_profile == "block"


def test_prepare_cup_runtime_translates_task_fields(tmp_path, monkeypatch):
    dataset = _write_success_dataset(tmp_path, name="cup_success")
    value_ckpt = tmp_path / "cup_value.pt"
    value_ckpt.write_bytes(b"value-weights")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    bridge, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--task_profile", "cup",
        "--offline_chunk_dataset", str(dataset),
        "--offline_chunk_cache_root", str(tmp_path / "cache"),
        "--pi0_serve_ckpt_id", "pi05_pick_cup_awbc_49999",
    ])
    runtime, translated = builder.prepare_offline_runtime(
        bridge, _mixed_passthrough(tmp_path / "output"))
    assert translated[translated.index("--pi0_prompt") + 1] == "pick cup"
    assert translated[translated.index("--dataset") + 1] == "cup_success"
    assert runtime.bridge_meta["offline_source"] == "cup_success"
    assert runtime.bridge_meta["pi0_prompt"] == "pick cup"


def test_write_success_dataset_helper_can_create_cup_source(tmp_path):
    dataset = _write_success_dataset(tmp_path, name="cup_success")
    assert dataset.name == "cup_success"
    assert (dataset / "meta/episodes_stats.jsonl").is_file()
```

Change the test helper signature from `_write_success_dataset(parent)` to
`_write_success_dataset(parent, name="block_success")` and construct
`root = parent / name`. Existing calls therefore continue testing Block, while
the Cup test exercises the same real catalog/hash/cache path.

- [ ] **Step 2: Run focused builder tests and verify failure**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py \
  -k "task_profile or cup or translates_task_fields" -v
```

Expected: FAIL because `--task_profile` is unknown and runtime translation is
Block-only.

- [ ] **Step 3: Implement task-aware builder flow**

Add `--task_profile` and replace the existing `--adv_prompt` declaration in `parse_bridge_args`:

```python
p.add_argument(
    "--task_profile",
    choices=("block", "cup"),
    default="block",
    help="teleavatar task profile controlling prompt and success source",
)
p.add_argument("--adv_prompt", default=None)
```

At the start of `prepare_offline_runtime`:

```python
from resfit.rl_finetuning.wm_bridge.task_profiles import get_task_profile

profile = get_task_profile(bridge_args.task_profile)
```

Use:

```python
translated = _replace_option(
    translated, "--pi0_prompt", profile.prompt)
translated = _replace_option(
    translated, "--pi0_action_dim", profile.action_dim)
translated = _replace_option(
    translated, "--dataset", dataset_name)
contract.check_mixed_replay_args(
    parsed,
    dataset_root,
    expected_prompt=profile.prompt,
    expected_source=profile.success_dataset,
)
```

Change metadata to:

```python
"task_profile": profile.name,
"offline_source": dataset_name,
```

In `build_imagination_factories`, resolve the same profile once and pass:

```python
sampler=TeleavatarStartSampler(
    bridge_args.init_state_dataset,
    caption=profile.prompt,
)
```

For advantage monitoring:

```python
adv_prompt = bridge_args.adv_prompt or profile.prompt
adv_scorer = AdvServeClient(
    host=bridge_args.adv_host,
    port=bridge_args.adv_port,
    prompt=adv_prompt,
)
```

Do not change endpoint/replay fingerprint inputs beyond receiving the selected
prompt and task dataset naturally.

- [ ] **Step 4: Run builder, contract, and sampler regression tests**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_task_profiles.py \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit the bridge generalization**

```bash
git add \
  resfit/rl_finetuning/wm_bridge/builder.py \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py
git commit -m "feat: generalize imagination bridge for cup"
```

---

### Task 4: Add Reproducible Cup Preparation and Launch Scripts

**Files:**
- Create: `prepare_cup_pi0feat.sh`
- Create: `train_cup_pi0feat_value.sh`
- Create: `launch_cup_imagination.sh`
- Modify: `resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py`

**Interfaces:**
- `prepare_cup_pi0feat.sh <port0> <port1> <port2>` writes six NPZ shards.
- `train_cup_pi0feat_value.sh` consumes those six shards and writes
  `outputs_chunk/cup_value_pi0feat.pt`.
- `launch_cup_imagination.sh <training_gpu> <seed>` starts the aligned run.

- [ ] **Step 1: Add failing static launcher tests**

```python
def test_cup_launcher_is_aligned():
    text = Path("launch_cup_imagination.sh").read_text()
    required = {
        "--task_profile cup",
        "--offline_fraction 0.5",
        "--batch_size 256",
        "--chunk_length 50",
        "--demo_bc_coef 0.1",
        "--critic_warmup_steps 10000",
        "--learning_starts 10000",
        "--n_step 1",
        "--gamma 0.995",
        '--pi0_prompt "pick cup"',
        "--dataset cup_success",
        "cup_value_pi0feat.pt",
        "cup_shore_mixed50_seed",
    }
    for item in required:
        assert item in text
    assert "--bc_coef_final" not in text


def test_cup_preparation_uses_three_disjoint_shards():
    text = Path("prepare_cup_pi0feat.sh").read_text()
    assert "--num_shards 3" in text
    for shard in range(3):
        assert f"--shard_index {shard}" in text
        assert f"success_shard{shard}.npz" in text
        assert f"fail_shard{shard}.npz" in text
```

- [ ] **Step 2: Run static tests and verify failure**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py \
  -k "cup_launcher or cup_preparation" -v
```

Expected: FAIL because the Cup scripts do not exist.

- [ ] **Step 3: Create the feature-preparation script**

The script must:

```bash
#!/usr/bin/env bash
set -euo pipefail

PORTS=("${1:?port0}" "${2:?port1}" "${3:?port2}")
ROOT=/mnt/mnt/data/domains_rise/cup
OUT=/mnt/mnt/data/resfit/outputs_chunk/cup_pi0feat_shards
PY=/mnt/mnt/data/envs/residual/bin/python
pids=()
mkdir -p "${OUT}"

build_shard() {
  local port=$1
  local shard=$2
  "${PY}" -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host 127.0.0.1 --port "${port}" --data_source teleavatar \
    --lerobot_root "${ROOT}" --repo_id cup_success --prompt "pick cup" \
    --pooling mean --serve_ckpt_id pi05_pick_cup_awbc_49999 \
    --out_cache "${OUT}/success_shard${shard}.npz" \
    --num_shards 3 --shard_index "${shard}"
  "${PY}" -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host 127.0.0.1 --port "${port}" --data_source teleavatar \
    --lerobot_root "${ROOT}" --repo_id cup_fail --prompt "pick cup" \
    --pooling mean --serve_ckpt_id pi05_pick_cup_awbc_49999 \
    --out_cache "${OUT}/fail_shard${shard}.npz" \
    --num_shards 3 --shard_index "${shard}"
}

for shard in 0 1 2; do
  build_shard "${PORTS[$shard]}" "${shard}" \
    >"${OUT}/shard${shard}.log" 2>&1 &
  pids[$shard]=$!
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
```

Add `cd /mnt/mnt/data/resfit`, `PYTHONPATH`, and the existing torchcodec
environment setup before the Python calls.

- [ ] **Step 4: Create the value-training script**

Use the exact command:

```bash
/mnt/mnt/data/envs/residual/bin/python -m \
  resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --dataset cup_success \
  --output /mnt/mnt/data/resfit/outputs_chunk/cup_value_pi0feat.pt \
  --state_mode pi0_feat \
  --terminal_reward_mode success_signed \
  --success_dataset outputs_chunk/cup_pi0feat_shards/success_shard0.npz \
  --success_dataset outputs_chunk/cup_pi0feat_shards/success_shard1.npz \
  --success_dataset outputs_chunk/cup_pi0feat_shards/success_shard2.npz \
  --failure_dataset outputs_chunk/cup_pi0feat_shards/fail_shard0.npz \
  --failure_dataset outputs_chunk/cup_pi0feat_shards/fail_shard1.npz \
  --failure_dataset outputs_chunk/cup_pi0feat_shards/fail_shard2.npz \
  --pi0_serve_ckpt_id pi05_pick_cup_awbc_49999 \
  --pi0_image_keys top_head hand_left hand_right \
  --pi0_proprio_key observation.state \
  --pi0_pooling mean \
  --pi0_prompt "pick cup" \
  --gamma 0.99 --expectile 0.7 --ema 0.005 \
  --lr 0.0003 --batch_size 256 --steps 50000 \
  --value_hidden 256 --seed 0
```

- [ ] **Step 5: Create the aligned Cup launcher**

Copy the structure of `launch_block_imagination.sh`, changing only task-specific
values and adding `--task_profile cup` plus `--adv_prompt "pick cup"`. Retain all
aligned hyperparameters in Global Constraints and write logs to:

```text
/mnt/mnt/data/resfit/outputs_imagination/cup_shore_mixed50_seed${SEED}.log
```

- [ ] **Step 6: Validate scripts and static tests**

Run:

```bash
bash -n prepare_cup_pi0feat.sh
bash -n train_cup_pi0feat_value.sh
bash -n launch_cup_imagination.sh
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py \
  -k "cup_launcher or cup_preparation" -v
```

Expected: all syntax checks and tests pass.

- [ ] **Step 7: Commit the scripts**

```bash
git add \
  prepare_cup_pi0feat.sh \
  train_cup_pi0feat_value.sh \
  launch_cup_imagination.sh \
  resfit/rl_finetuning/wm_bridge/tests/test_block_mixed_launch.py
git commit -m "feat: add aligned cup experiment launchers"
```

---

### Task 5: Run the Complete Automated Regression Gate

**Files:**
- No source changes expected.

**Interfaces:**
- Validates Tasks 1–4 as a single bridge.

- [ ] **Step 1: Run formatting/static checks**

Run:

```bash
git diff --check HEAD~3..HEAD
bash -n launch_block_imagination.sh
bash -n launch_cup_imagination.sh
bash -n prepare_cup_pi0feat.sh
bash -n train_cup_pi0feat_value.sh
```

Expected: no output from `git diff --check`; all shell checks exit 0.

- [ ] **Step 2: Run the full wm_bridge test suite**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/wm_bridge/tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run affected chunk-residual cache/value tests**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache_teleavatar.py \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_value_pi0_feat.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Record verification evidence**

Record commands, pass counts, and elapsed times in:

```text
/mnt/mnt/data/resfit/outputs_imagination/cup_preflight_verification.txt
```

Do not commit the generated verification file.

---

### Task 6: Run a Real Two-Dataset Cup Smoke Test

**Files:**
- Generated only under `/tmp/resfit-cup-smoke-*`.

**Interfaces:**
- Consumes live ports 9000 and 8001.
- Produces a temporary success cache, failure cache, small value, endpoint cache,
  replay cache, and finite one-segment rollout.

- [ ] **Step 1: Verify live services without mutation**

Run:

```bash
ss -ltnp | rg ':9000|:8001'
ps -eo pid,pgid,stat,etime,cmd | rg 'd_serve|serve_block_awbc'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
```

Expected: D-serve listens on 9000 and kai0 checkpoint 49999 with pooling `mean`
listens on 8001.

- [ ] **Step 2: Create one-success and one-failure real feature cache**

Create a unique temporary directory with `mktemp -d`, then run the feature-cache
builder twice with:

```text
--data_source teleavatar
--lerobot_root /mnt/mnt/data/domains_rise/cup
--repo_id cup_success  (then cup_fail)
--prompt "pick cup"
--pooling mean
--serve_ckpt_id pi05_pick_cup_awbc_49999
--num_demos 1
```

Expected: each NPZ contains one nonempty sequence and the exact Cup signature.

- [ ] **Step 3: Train and load a temporary small value**

Run the Task 4 value command against the two temporary caches, overriding:

```text
--steps 20
--output <tmp>/cup_value_smoke.pt
```

Then load it:

```bash
/mnt/mnt/data/envs/residual/bin/python -c \
  "from resfit.rl_finetuning.wm_bridge.scorers import Kai0HiqlScorer; \
s=Kai0HiqlScorer.from_value_ckpt('<tmp>/cup_value_smoke.pt'); \
assert s.expected_psi_anchor == 'pi05_pick_cup_awbc_49999'; print(s.state_dim)"
```

Expected: exits 0 and prints a positive state dimension.

- [ ] **Step 4: Run a minimal real bridge rollout and offline replay build**

Invoke `launch_imagination` directly with the formal Cup arguments, but override:

```text
--value_ckpt <tmp>/cup_value_smoke.pt
--offline_num_demos 2
--offline_chunk_cache_root <tmp>/replay
--total_env_steps 100
--eval_every_env_steps 100000
--wandb_mode disabled
--output_dir <tmp>/run
```

Do not pass `--adv_host` in smoke. Keep fraction 0.5, batch 256, chunk 50,
fixed BC 0.1, and all contract-required values unchanged.

Expected:

- startup banner contains `offline_source=cup_success`;
- startup banner contains `batch=128+128`;
- offline build stats are finite;
- at least one world-model segment completes;
- logs contain no `nan`, `inf`, traceback, contract failure, or shape mismatch.

- [ ] **Step 5: Stop on any smoke failure**

If any assertion fails, preserve the temporary directory and logs, diagnose with
the systematic-debugging skill, and do not proceed to full preprocessing.

---

### Task 7: Build Full Cup Features and the Final Cup Value

**Files:**
- Generated: `outputs_chunk/cup_pi0feat_shards/*.npz`
- Generated: `outputs_chunk/cup_pi0feat_shards/*.log`
- Generated: `outputs_chunk/cup_value_pi0feat.pt`

**Interfaces:**
- Consumes Cup datasets and kai0 checkpoint 49999.
- Produces the scorer checkpoint required by formal RL.

- [ ] **Step 1: Resolve and stop only the authorized GPU 6 service**

Resolve the PID listening on port 8002, confirm its command is
`wm_bridge.adv_serve`, and confirm its GPU UUID maps to physical GPU 6.
Send `SIGTERM` to that exact PID, wait for exit, then verify port 8002 is free.

Do not use `pkill`, broad process patterns, or unresolved shell substitutions.

- [ ] **Step 2: Start temporary Cup kai0 serves**

Retain the existing GPU 1 port-8001 serve. Start two additional instances:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/mnt/mnt/data/data2/kai0/src:/mnt/mnt/data/data2/kai0 \
/mnt/mnt/data/chj/openpi/.venv/bin/python \
  /mnt/mnt/data/resfit/pi0_serve/serve_block_awbc.py \
  --dir /mnt/mnt/data/data2/kai0/checkpoints/49999 \
  --port 8011 --pooling mean \
  --default-prompt "pick cup" \
  --config-name pi05_pick_cup_awbc \
  --repo-id /mnt/mnt/data/domains_rise/cup/cup_success
```

Repeat on GPU 7 and port 8012. Redirect each stdout/stderr to distinct files
under `logs/` and record PIDs.

Expected: ports 8001, 8011, and 8012 listen and each process uses about the same
memory as the known port-8001 serve.

- [ ] **Step 3: Run full three-way extraction**

Run:

```bash
bash prepare_cup_pi0feat.sh 8001 8011 8012
```

Expected: six shard files complete successfully.

- [ ] **Step 4: Validate shard coverage and signatures**

Use `load_pi0_feat_cache` to assert:

- success shard sequence counts sum to 207;
- failure shard sequence counts sum to 51;
- each group has disjoint shard episode coverage by construction;
- every sequence has at least one frame;
- every feature is finite;
- all six caches share serve identity, image keys, proprio key, pooling, and
  prompt;
- success dataset IDs are `cup_success`;
- failure dataset IDs are `cup_fail`.

If actual dataset metadata changed since discovery, stop and report the new
episode counts rather than silently accepting them.

- [ ] **Step 5: Stop temporary GPU 6 and GPU 7 kai0 serves**

Send `SIGTERM` only to the two recorded temporary PIDs. Verify ports 8011 and
8012 are free and GPUs 6 and 7 release their model memory. Leave GPU 1 port 8001
running.

- [ ] **Step 6: Train the final Cup value**

Run:

```bash
bash train_cup_pi0feat_value.sh
```

Expected: `outputs_chunk/cup_value_pi0feat.pt` exists, is nonempty, and training
prints 207 successful plus 51 failed demonstrations.

- [ ] **Step 7: Validate the final scorer checkpoint**

Load with `Kai0HiqlScorer.from_value_ckpt` and assert:

```python
assert scorer.expected_psi_anchor == "pi05_pick_cup_awbc_49999"
assert scorer.state_dim > 16
assert np.isfinite(scorer.mean).all()
assert np.isfinite(scorer.std).all()
assert (scorer.std > 0).all()
```

Record SHA-256 and file size in the preflight verification log.

---

### Task 8: Switch Cup Monitoring, Prebuild Replay, and Launch Formal RL

**Files:**
- Generated: `cache/cup_mixed_replay/`
- Generated: `logs/adv_serve_cup_8002.log`
- Generated: `outputs_imagination/cup_shore_mixed50_seed0.log`
- Generated: W&B run `cup_shore_mixed50_seed0`

**Interfaces:**
- Consumes ports 9000/8001, Cup value checkpoint, Cup datasets, and Cup RISE
  `value_cup`.
- Produces the formal 500k mixed-replay run.

- [ ] **Step 1: Start Cup RISE advantage monitoring on GPU 6**

Start:

```bash
CUDA_VISIBLE_DEVICES=6 \
/mnt/mnt/data/resfit/rise_venv/bin/python \
  -m resfit.rl_finetuning.wm_bridge.adv_serve \
  --ckpt /mnt/mnt/data/resfit/RISE_Hi/policy_and_value/policy_offline_and_value/checkpoints/value_cup/value_cup/10000/model.safetensors \
  --config_name value_cup --port 8002
```

Use the same RISE environment setup as the known-good Block service. Redirect
to `logs/adv_serve_cup_8002.log`.

Expected: log reaches `value 模型就绪` and `up on 0.0.0.0:8002`.

If `value_cup` is unavailable from the active runtime config registry, stop,
diagnose the exact RISE environment/config source, and do not substitute
`value_block`.

- [ ] **Step 2: Exercise all three services**

Perform one metadata/inference request against ports 9000, 8001, and 8002 using
Cup-shaped inputs and prompt `pick cup`.

Expected: finite outputs with the expected dimensions and no server traceback.

- [ ] **Step 3: Prebuild the full Cup offline replay without W&B**

Invoke the Cup launcher command directly with the final value and formal
algorithmic parameters, overriding only:

```text
--total_env_steps 1
--eval_every_env_steps 100000
--wandb_mode disabled
--output_dir /tmp/resfit-cup-replay-prebuild
```

Expected:

- all Cup success episodes are cataloged;
- replay metadata is written below `cache/cup_mixed_replay`;
- startup banner reports `offline_source=cup_success` and `batch=128+128`;
- replay build stats are finite.

- [ ] **Step 4: Re-run the prebuild command and verify cache hit**

Expected:

```text
endpoint_cache=hit
replay_cache=hit
```

and zero new kai0 endpoint queries during offline replay construction.

- [ ] **Step 5: Verify GPU 7 and output identity**

Run:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
ps -eo pid,pgid,stat,etime,cmd | rg 'cup_shore_mixed50_seed0|launch_imagination'
```

Expected: GPU 7 has no conflicting process and no formal Cup run with the same
output/W&B identity already exists.

- [ ] **Step 6: Launch formal RL**

Run the script in a dedicated persistent session:

```bash
bash launch_cup_imagination.sh 7 0
```

Expected: process stays alive and creates the W&B run
`cup_shore_mixed50_seed0`.

- [ ] **Step 7: Verify formal startup evidence**

Tail the log until it proves:

- `offline_source=cup_success`;
- `batch=128+128`;
- endpoint and replay cache hits;
- fixed BC coefficient 0.1;
- first online and offline batch metrics;
- finite actor, critic, BC, reward, Q, and residual metrics;
- no traceback, NaN, Inf, or service disconnect.

Also inspect W&B configuration and confirm all formal parameters match Section 6
of the approved design.

- [ ] **Step 8: Report the running experiment**

Report:

- W&B run URL and run ID;
- OS PID/process group and persistent-session name;
- GPU/service allocation;
- output and log paths;
- cache hit status;
- first observed Online/Offline ratio and BC coefficient;
- any non-fatal warnings, clearly separated from failures.

Do not claim success rate from `eval/success_rate`; in this imagination setup it
is a placeholder. Use the Cup `imagined_adv_eval.jsonl` proxy only as monitoring,
matching the Block reference.
