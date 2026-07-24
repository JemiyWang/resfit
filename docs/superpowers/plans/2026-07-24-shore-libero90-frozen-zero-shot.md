# Frozen SHORE LIBERO-90 Zero-Shot Evaluation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the unchanged task8 SHORE checkpoint on LIBERO-90 tasks 3, 8, 21, and 63, record 50 fixed-init evaluation episodes per task, and generate a four-panel rolling-success PDF plus raw JSON/NPZ data.

**Architecture:** Add an inference-only construction path to the existing residual agent, extend the LIBERO environment and rollout helpers with opt-in deterministic episode metadata, and add a dedicated frozen-evaluation driver. A shell launcher runs one PI0 feature server and one evaluator on each selected GPU; a separate pure plotting script validates and aggregates the durable JSONL records.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, LIBERO, NumPy, Matplotlib, Bash, pytest, OpenPI PI0 feature server.

---

## Fixed experiment contract

Use these exact source assets for every target task:

```text
BASE_CHECKPOINT=/mnt/mnt/data/chj/openpi/checkpoints/pi0_libero/pi0_libero_is-dceq77jzghdxvjj2-devmachine-0_20260523_220232/29999
RESIDUAL_CHECKPOINT=/mnt/mnt/data/resfit/outputs_chunk/libero10_task8_pi0feat_bp_bc01_h10/best.pt
GC_VALUE_CHECKPOINT=/mnt/mnt/data/resfit/outputs_chunk/libero10_t8moka_pi0_feat_gc_value.pt
HIGH_ACTOR_CHECKPOINT=/mnt/mnt/data/resfit/outputs_chunk/libero10_t8moka_pi0_feat_high_actor.pt
PI0_FEATURE_CACHE=/mnt/mnt/data/resfit/outputs_chunk/libero10_t8moka_pi0_feat.npz
PI0_STATS=/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json
```

The target set is `libero_90:{3,8,21,63}`. Each target gets init-state IDs
`0..49` exactly once. The implementation must never construct an optimizer,
replay buffer, W&B run, or training loop in the dedicated evaluation process.
The task prompt is target-specific, but all learned parameters and the
representative task8 goal are unchanged.

Use one sequential LIBERO environment in each evaluator. The four tasks still
run concurrently on four GPUs, while sequential init-state execution makes the
`0..49` coverage invariant direct and auditable.

## Task 1: Add inference-only `QAgent` construction

**Files:**

- Create: `resfit/rl_finetuning/chunk_residual/tests/test_q_agent_inference_only.py`
- Modify: `resfit/rl_finetuning/off_policy/rl/q_agent.py`

### Step 1: Write the failing tests

Create a minimal agent fixture using the same config class as residual
training. Patch optimizer and scheduler constructors so the tests detect any
attempt to initialize training state.

```python
from unittest.mock import patch

from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


def _kwargs():
    cfg = ResidualTD3BoxCleanConfig().agent
    cfg.device = "cpu"
    cfg.lr_warmup_steps = 2
    return {
        "obs_shape": (3, 84, 84),
        "prop_shape": (8,),
        "action_dim": 7,
        "rl_cameras": ["agentview_image"],
        "cfg": cfg,
        "residual_actor": True,
        "stage_conditioned": False,
        "num_stages": 0,
        "subgoal_conditioned": True,
        "subgoal_dim": 16,
    }


def test_inference_only_skips_all_training_state():
    with (
        patch("torch.optim.AdamW", side_effect=AssertionError("optimizer created")),
        patch(
            "torch.optim.lr_scheduler.LinearLR",
            side_effect=AssertionError("scheduler created"),
        ),
    ):
        agent = QAgent(**_kwargs(), inference_only=True)

    assert agent.encoder_opt is None
    assert agent.critic_opt is None
    assert agent.actor_opt is None
    assert agent.encoder_scheduler is None
    assert agent.critic_scheduler is None
    assert agent.actor_scheduler is None


def test_default_constructor_still_creates_training_state():
    agent = QAgent(**_kwargs())
    assert agent.encoder_opt is not None
    assert agent.critic_opt is not None
    assert agent.actor_opt is not None
    assert agent.encoder_scheduler is not None
    assert agent.critic_scheduler is not None
    assert agent.actor_scheduler is not None
```

### Step 2: Confirm the tests fail for the intended reason

Run:

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_q_agent_inference_only.py -q
```

Expected: the first test fails because `QAgent.__init__` does not accept
`inference_only`.

### Step 3: Implement the backward-compatible constructor flag

Add `inference_only: bool = False` to `QAgent.__init__`. Keep all neural-module
construction unchanged, then replace unconditional optimizer/scheduler
creation with:

```python
self.inference_only = bool(inference_only)
self.encoder_opt = None
self.critic_opt = None
self.actor_opt = None
self.encoder_scheduler = None
self.critic_scheduler = None
self.actor_scheduler = None

if not self.inference_only:
    self.encoder_opt = torch.optim.AdamW(
        self.encoders.parameters(), lr=self.cfg.critic_lr
    )
    self.critic_opt = torch.optim.AdamW(
        self.critic.parameters(), lr=self.cfg.critic_lr
    )
    self.actor_opt = torch.optim.AdamW(
        self.actor.parameters(), lr=self.cfg.actor_lr
    )
    if self.cfg.lr_warmup_steps > 0:
        critic_start_factor = max(
            self.cfg.lr_warmup_start / self.cfg.critic_lr, 1e-8
        )
        actor_start_factor = max(
            self.cfg.lr_warmup_start / self.cfg.actor_lr, 1e-8
        )
        self.encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.encoder_opt,
            start_factor=critic_start_factor,
            total_iters=self.cfg.lr_warmup_steps,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.critic_opt,
            start_factor=critic_start_factor,
            total_iters=self.cfg.lr_warmup_steps,
        )
        self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.actor_opt,
            start_factor=actor_start_factor,
            total_iters=self.cfg.lr_warmup_steps,
        )
```

### Step 4: Run focused and nearby tests

Run:

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_q_agent_inference_only.py \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py -q
```

Expected: all tests pass.

### Step 5: Commit

```bash
git add \
  resfit/rl_finetuning/off_policy/rl/q_agent.py \
  resfit/rl_finetuning/chunk_residual/tests/test_q_agent_inference_only.py
git commit -m "feat: support inference-only residual agents"
```

## Task 2: Add deterministic LIBERO init-state scheduling

**Files:**

- Modify: `resfit/rl_finetuning/chunk_residual/libero_env.py`
- Modify: `resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py`

### Step 1: Add failing wrapper tests

Extend the fake LIBERO environment so `set_init_state` records its argument.
Add tests that establish both the new schedule and the old fixed-state default:

```python
def test_init_state_schedule_is_used_in_order(fake_libero_env):
    wrapper = LiberoGymWrapper(
        fake_libero_env,
        init_states=["state0", "state1", "state2"],
        init_state_indices=[2, 0, 1],
        max_steps=5,
    )

    _, info0 = wrapper.reset()
    _, info1 = wrapper.reset()
    _, info2 = wrapper.reset()

    assert [info0["init_state_id"], info1["init_state_id"], info2["init_state_id"]] == [
        2,
        0,
        1,
    ]
    assert fake_libero_env.seen_init_states == ["state2", "state0", "state1"]


def test_step_info_keeps_active_init_state_id(fake_libero_env):
    wrapper = LiberoGymWrapper(
        fake_libero_env,
        init_states=["state0", "state1"],
        init_state_indices=[1, 0],
        max_steps=5,
    )
    wrapper.reset()
    _, _, _, _, info = wrapper.step(fake_libero_env.action_space.sample())
    assert info["init_state_id"] == 1


def test_default_reset_keeps_existing_fixed_init_behavior(fake_libero_env):
    wrapper = LiberoGymWrapper(
        fake_libero_env,
        init_states=["state0", "state1"],
        init_idx=1,
        max_steps=5,
    )
    _, first = wrapper.reset()
    _, second = wrapper.reset()
    assert first["init_state_id"] == 1
    assert second["init_state_id"] == 1
```

### Step 2: Run the test and observe failure

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py -q
```

Expected: schedule-related constructor and metadata assertions fail.

### Step 3: Implement the opt-in schedule

Add a `Sequence[int] | None` argument to `LiberoGymWrapper`. Validate all IDs
against `len(init_states)`, increment the schedule cursor only on reset, and
attach the selected ID to both reset and step info:

```python
self.init_state_indices = (
    None if init_state_indices is None else tuple(int(i) for i in init_state_indices)
)
if self.init_state_indices is not None:
    if not self.init_state_indices:
        raise ValueError("init_state_indices must not be empty")
    invalid = [
        index
        for index in self.init_state_indices
        if index < 0 or index >= len(self.init_states)
    ]
    if invalid:
        raise ValueError(f"invalid LIBERO init-state IDs: {invalid}")
self._init_schedule_cursor = 0
self._active_init_state_id = int(init_idx)
```

The selection performed by `reset` is below. `SAME_STEP` autoreset performs
one reset immediately after the final terminal transition, so an exhausted
schedule must reuse the last ID as a harmless post-run observation rather than
raising inside the final `step`. That reuse is never recorded as an episode.

```python
if self.init_state_indices is None:
    init_state_id = int(self.init_idx)
else:
    schedule_index = min(
        self._init_schedule_cursor, len(self.init_state_indices) - 1
    )
    init_state_id = self.init_state_indices[schedule_index]
    if self._init_schedule_cursor < len(self.init_state_indices):
        self._init_schedule_cursor += 1
self._active_init_state_id = init_state_id
observation = self.env.set_init_state(self.init_states[init_state_id])
info = {"init_state_id": init_state_id}
```

Forward `init_state_indices` through `make_libero_env` and
`create_libero_vectorized_env`. When it is present, require `num_envs == 1`;
this prevents ambiguous multi-process schedule consumption:

```python
if init_state_indices is not None and num_envs != 1:
    raise ValueError("fixed init-state scheduling currently requires num_envs=1")
```

The default `init_idx=0` path must remain unchanged apart from reporting its
metadata.

### Step 4: Run tests

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py -q
```

Expected: all tests pass.

### Step 5: Commit

```bash
git add \
  resfit/rl_finetuning/chunk_residual/libero_env.py \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py
git commit -m "feat: schedule deterministic LIBERO init states"
```

## Task 3: Record individual evaluation episodes without restoring train mode

**Files:**

- Modify: `resfit/rl_finetuning/chunk_residual/libero_eval.py`
- Modify: `resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py`

### Step 1: Add failing compatibility and callback tests

Add a fake vector environment whose terminal `info` contains
`init_state_id`. Verify callback rows and the opt-out from train-mode
restoration:

```python
def test_episode_callback_receives_init_state_metadata():
    rows = []
    agent = FakeAgent()
    result = run_libero_evaluation(
        env=FakeVectorEnv(init_state_ids=[7, 9]),
        agent=agent,
        num_episodes=2,
        episode_callback=rows.append,
        return_episodes=True,
        restore_train_mode=False,
    )

    assert [row["init_state_id"] for row in rows] == [7, 9]
    assert rows == result["eval/episodes"]
    assert all(
        set(row) == {
            "episode_index",
            "env_index",
            "init_state_id",
            "success",
            "return",
            "length",
        }
        for row in rows
    )
    assert agent.training is False


def test_default_result_and_mode_restoration_are_backward_compatible():
    agent = FakeAgent()
    result = run_libero_evaluation(
        env=FakeVectorEnv(), agent=agent, num_episodes=1
    )
    assert "eval/episodes" not in result
    assert agent.training is True
```

### Step 2: Run the test and verify the API is absent

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py -q
```

Expected: the callback test fails on unexpected keyword arguments.

### Step 3: Extend the evaluator

Add these keyword-only arguments:

```python
episode_callback: Callable[[dict[str, Any]], None] | None = None,
return_episodes: bool = False,
restore_train_mode: bool = True,
```

Add helpers that handle scalar, list, NumPy-vector, and Gymnasium
`SAME_STEP` terminal info. A completed episode reads `final_info`, not the reset
info returned for the next episode:

```python
def _value_at(value: Any, env_index: int) -> Any:
    if isinstance(value, np.ndarray):
        item = value[env_index]
        return item.item() if hasattr(item, "item") else item
    if isinstance(value, (list, tuple)):
        return value[env_index]
    return value


def _episode_info_at(info: Mapping[str, Any], env_index: int) -> Mapping[str, Any]:
    final_infos = info.get("final_info")
    if final_infos is not None:
        final_info = _value_at(final_infos, env_index)
        if isinstance(final_info, Mapping):
            return {
                key: _value_at(value, env_index)
                for key, value in final_info.items()
                if not key.startswith("_")
            }
    return {
        key: _value_at(value, env_index)
        for key, value in info.items()
        if not key.startswith("_")
    }
```

On every completed episode construct and synchronously dispatch:

```python
episode_info = _episode_info_at(info, env_index)
episode_return = float(sum(ep_rewards[env_index]))
episode_length = len(ep_rewards[env_index])
row = {
    "episode_index": len(episode_rows),
    "env_index": int(env_index),
    "init_state_id": int(episode_info["init_state_id"]),
    "success": bool(is_success),
    "return": episode_return,
    "length": episode_length,
}
episode_rows.append(row)
if episode_callback is not None:
    episode_callback(dict(row))
```

Only add `result["eval/episodes"] = episode_rows` when `return_episodes` is
true. In the `finally` block call `agent.train(True)` only when
`restore_train_mode` is true. Keep deterministic calls
`agent.act(eval_mode=True, stddev=0.0)` unchanged.

### Step 4: Run evaluator tests

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py -q
```

Expected: all tests pass.

### Step 5: Commit

```bash
git add \
  resfit/rl_finetuning/chunk_residual/libero_eval.py \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py
git commit -m "feat: expose durable LIBERO episode results"
```

## Task 4: Build the frozen SHORE evaluation driver

**Files:**

- Create: `resfit/rl_finetuning/chunk_residual/frozen_shore_libero_eval.py`
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_frozen_shore_libero_eval.py`

### Step 1: Write unit tests for the pure safety helpers

Test checkpoint hashing, resume validation, strict coverage, durable append,
and frozen parameters without starting MuJoCo:

```python
def test_missing_or_duplicate_init_states_are_rejected():
    rows = [_row(i) for i in range(50)]
    validate_episode_coverage(rows, expected_ids=range(50))
    with pytest.raises(ValueError, match="duplicate"):
        validate_episode_coverage(rows + [_row(49)], expected_ids=range(50))
    with pytest.raises(ValueError, match="missing"):
        validate_episode_coverage(rows[:-1], expected_ids=range(50))


def test_resume_requires_same_task_and_checkpoint_hash(tmp_path):
    expected = {"task_id": 63, "source_checkpoint_sha256": "abc"}
    write_json(tmp_path / "config.json", expected)
    validate_resume_config(tmp_path / "config.json", expected)
    with pytest.raises(ValueError, match="source_checkpoint_sha256"):
        validate_resume_config(
            tmp_path / "config.json",
            {"task_id": 63, "source_checkpoint_sha256": "different"},
        )


def test_freeze_module_disables_every_parameter():
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    freeze_and_validate(module, "agent")
    assert module.training is False
    assert not any(parameter.requires_grad for parameter in module.parameters())


def test_jsonl_writer_flushes_one_complete_row(tmp_path):
    path = tmp_path / "episodes.jsonl"
    with EpisodeJsonlWriter(path) as writer:
        writer.append(_row(0))
    assert load_jsonl(path) == [_row(0)]
```

### Step 2: Run tests and verify the module is missing

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_frozen_shore_libero_eval.py -q
```

Expected: collection fails because the new module does not exist.

### Step 3: Implement the driver’s pure helpers and CLI

The module must provide:

```python
TASKS = {
    3: "put the butter at the back in the top drawer of the cabinet and close it",
    8: "open the top drawer of the cabinet and put the bowl in it",
    21: "turn on the stove and put the frying pan on it",
    63: "stack the left bowl on the right bowl and place them in the tray",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_and_validate(module: torch.nn.Module, name: str) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    trainable = [key for key, value in module.named_parameters() if value.requires_grad]
    if trainable:
        raise RuntimeError(f"{name} still has trainable parameters: {trainable}")


def validate_episode_coverage(
    rows: Sequence[Mapping[str, Any]], expected_ids: Iterable[int]
) -> None:
    expected = list(expected_ids)
    actual = [int(row["init_state_id"]) for row in rows]
    duplicate = sorted(key for key, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if duplicate or missing or unexpected or len(actual) != len(expected):
        raise ValueError(
            f"invalid init-state coverage: duplicate={duplicate}, "
            f"missing={missing}, unexpected={unexpected}"
        )
```

`EpisodeJsonlWriter.append` writes one JSON object plus newline, flushes the
Python stream, then calls `os.fsync`. `validate_resume_config` compares at
least suite, task ID, seed, source checkpoint path, and all five source-bundle
hashes.

Expose these CLI arguments with the fixed contract as defaults:

```text
--suite libero_90
--task-id {3,8,21,63}
--seed 1
--num-episodes 50
--device cuda
--host 127.0.0.1
--port required
--run-dir required
--base-checkpoint fixed path
--residual-checkpoint fixed path
--gc-value-checkpoint fixed path
--high-actor-checkpoint fixed path
--pi0-feature-cache fixed path
--pi0-stats fixed path
```

### Step 4: Reconstruct the exact source SHORE model

Load the trusted local residual bundle using:

```python
bundle = torch.load(
    args.residual_checkpoint,
    map_location="cpu",
    mmap=True,
    weights_only=False,
)
source_args = bundle["config"]
```

Build the target-task base policy by reusing the existing PI0 adapter builder,
but override only suite, task ID, host, port, device, and run-only fields. Build
normalization scalers from `source_args.action_scale`,
`source_args.min_range`, and the fixed PI0 stats.

Build the hierarchy using the existing loaders:

```python
_, gc_info = load_gc_value(args.gc_value_checkpoint, map_location="cpu")
sequences, _, cache_signature = load_pi0_feat_cache(args.pi0_feature_cache)
validate_pi0_feature_signature(cache_signature, gc_info["pi0_feat_signature"])
goal = representative_goal(sequences)
subgoal = HiqlSubgoal.from_ckpts(
    args.gc_value_checkpoint,
    args.high_actor_checkpoint,
    goal=goal,
    device=args.device,
    renorm_subgoal=True,
    base_policy=None,
)
```

Construct `QAgent` with the same image keys, state dimensions, action
dimensions, config values, and subgoal dimension used by
`train_chunk_residual.py`, plus `inference_only=True`. Load strictly:

```python
incompatible = agent.load_state_dict(bundle["agent_state_dict"], strict=True)
if incompatible.missing_keys or incompatible.unexpected_keys:
    raise RuntimeError(f"strict load failed: {incompatible}")
freeze_and_validate(agent, "residual_agent")
freeze_and_validate(subgoal.ha, "high_actor")
freeze_and_validate(subgoal.vf, "gc_value")
```

Create the LIBERO environment with one env and only the missing init-state IDs.
Wrap it with the existing chunk residual wrapper and target-specific PI0
prompt. Do a preflight inference and require `prefix_feat` before episode 0.

### Step 5: Implement resume, callback, and completion

Before starting, write or validate `config.json`. Include absolute source
paths, SHA-256 hashes, task language read from LIBERO, seed, port, Python,
Torch, and LIBERO versions.

Load existing JSONL rows and calculate:

```python
completed = {int(row["init_state_id"]) for row in existing_rows}
remaining = [init_id for init_id in range(args.num_episodes) if init_id not in completed]
```

The callback enriches rollout rows with task metadata, elapsed time, seed, and
the residual checkpoint hash before `EpisodeJsonlWriter.append`. Run:

```python
with torch.inference_mode():
    run_libero_evaluation(
        env=env,
        agent=agent,
        num_episodes=len(remaining),
        device=args.device,
        subgoal=subgoal,
        base_policy=base_policy,
        episode_callback=record_episode,
        return_episodes=False,
        restore_train_mode=False,
    )
```

Reload JSONL, validate exactly the expected init IDs, write `summary.json` via
temp-file plus `os.replace`, then create `COMPLETED` only after all checks pass.

### Step 6: Run unit and nearby regression tests

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_frozen_shore_libero_eval.py \
  resfit/rl_finetuning/chunk_residual/tests/test_q_agent_inference_only.py \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py -q
```

Expected: all tests pass.

### Step 7: Commit

```bash
git add \
  resfit/rl_finetuning/chunk_residual/frozen_shore_libero_eval.py \
  resfit/rl_finetuning/chunk_residual/tests/test_frozen_shore_libero_eval.py
git commit -m "feat: add frozen SHORE LIBERO evaluator"
```

## Task 5: Add the four-GPU launcher

**Files:**

- Create: `run_shore_libero90_zero_shot_4gpu.sh`
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_shore_libero_launcher.py`

### Step 1: Write a launcher dry-run test

The test executes the launcher with `DRY_RUN=1`, `GPU_IDS="4 5 6 7"`, and a
temporary run root. Assert that stdout contains four server commands, four
evaluator commands, tasks `3 8 21 63`, ports `8000 8001 8002 8003`, and
one-to-one GPU assignments.

### Step 2: Run the failing test

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_shore_libero_launcher.py -q
```

Expected: failure because the launcher is absent.

### Step 3: Implement the launcher

Use strict Bash mode and configurable GPU IDs:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/mnt/mnt/data/resfit
OPENPI_ROOT=/mnt/mnt/data/chj/openpi
OPENPI_PYTHON=/mnt/mnt/data/chj/openpi/.venv/bin/python
EVAL_PYTHON=/mnt/mnt/data/envs/resfit-libero/bin/python
TASK_IDS=(3 8 21 63)
PORTS=(8000 8001 8002 8003)
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3}"

if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four GPU IDs" >&2
  exit 2
fi

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/outputs_eval/shore_libero90_zero_shot/${RUN_ID}}
mkdir -p "${RUN_ROOT}"
```

For each mapping, create `taskXX`, start
`pi0_serve/serve_with_feat.py --config pi0_libero --dir
/mnt/mnt/data/chj/openpi/checkpoints/pi0_libero/pi0_libero_is-dceq77jzghdxvjj2-devmachine-0_20260523_220232/29999
--port PORT --pooling last` under that GPU, wait for the TCP port with a bounded retry
loop, then start the driver under the same visible GPU. Set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.65` for the server so the residual evaluator
has headroom. Set `MUJOCO_GL=egl` and the repo/LIBERO paths in `PYTHONPATH`.

Run all four process groups concurrently. Save `server.log`, `stdout.log`, and
PID files inside each task directory. A group failure is recorded without
killing the other healthy task groups. The launcher exits nonzero after all
groups settle if any group failed.

When `DRY_RUN=1`, print the fully resolved commands and exit before starting
processes.

### Step 4: Check syntax and dry-run mapping

```bash
bash -n run_shore_libero90_zero_shot_4gpu.sh
DRY_RUN=1 GPU_IDS="4 5 6 7" RUN_ID=plan-check \
  bash run_shore_libero90_zero_shot_4gpu.sh
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_shore_libero_launcher.py -q
```

Expected: syntax succeeds, dry run prints four disjoint groups, and the test
passes.

### Step 5: Commit

```bash
git add \
  run_shore_libero90_zero_shot_4gpu.sh \
  resfit/rl_finetuning/chunk_residual/tests/test_shore_libero_launcher.py
git commit -m "feat: launch four-task SHORE zero-shot evaluation"
```

## Task 6: Add validated aggregation and plotting

**Files:**

- Create: `paper/plot_shore_libero90_zero_shot.py`
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_plot_shore_libero90_zero_shot.py`

### Step 1: Write plot-data tests

Generate four temporary `episodes.jsonl` fixtures with 50 binary outcomes
each. Assert:

- a 10-point rolling rate uses `min_periods=1`;
- the 10th point is the mean of episodes 0 through 9;
- missing and duplicate init IDs fail before any output is written;
- JSON, NPZ, and PDF are created for valid input; and
- combined pairs total 200.

### Step 2: Run the missing-module failure

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_plot_shore_libero90_zero_shot.py -q
```

Expected: collection fails because the plotting module is absent.

### Step 3: Implement pure validation and rolling success

Use:

```python
def rolling_success(successes: Sequence[float], window: int = 10) -> np.ndarray:
    values = np.asarray(successes, dtype=np.float64)
    return np.asarray(
        [
            values[max(0, index - window + 1) : index + 1].mean()
            for index in range(len(values))
        ],
        dtype=np.float64,
    )
```

For each task, sort rows by `init_state_id`, require IDs `0..49`, require
binary success values, and require one common set of source hashes across all
four configs. Write:

```text
paper/data/fig_shore_libero90_zero_shot.json
paper/data/fig_shore_libero90_zero_shot.npz
paper/figure/fig_shore_libero90_zero_shot.pdf
```

The JSON contains experiment metadata plus raw outcomes and rolling values.
The NPZ contains `task_ids`, `init_state_ids`, `successes`,
`rolling_success`, `returns`, and `lengths`.

Render a 2-by-2 figure with fixed y-range `[0, 1]`, x label
`Evaluation episode`, y label `10-episode rolling success rate`, one SHORE
line per panel, concise target instruction titles, and a `successes / 50`
annotation. Add a figure note stating that task8 SHORE weights are frozen and
that the run is zero-shot evaluation rather than training.

### Step 4: Run plot tests

```bash
MPLBACKEND=Agg /mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_plot_shore_libero90_zero_shot.py -q
```

Expected: all tests pass.

### Step 5: Commit

```bash
git add \
  paper/plot_shore_libero90_zero_shot.py \
  resfit/rl_finetuning/chunk_residual/tests/test_plot_shore_libero90_zero_shot.py
git commit -m "feat: plot SHORE LIBERO zero-shot curves"
```

## Task 7: Verify the implementation and run the experiment

**Files produced:**

- `outputs_eval/shore_libero90_zero_shot/$RUN_ID/task03/`
- `outputs_eval/shore_libero90_zero_shot/$RUN_ID/task08/`
- `outputs_eval/shore_libero90_zero_shot/$RUN_ID/task21/`
- `outputs_eval/shore_libero90_zero_shot/$RUN_ID/task63/`
- `paper/data/fig_shore_libero90_zero_shot.json`
- `paper/data/fig_shore_libero90_zero_shot.npz`
- `paper/figure/fig_shore_libero90_zero_shot.pdf`

### Step 1: Run the complete focused test set

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_q_agent_inference_only.py \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py \
  resfit/rl_finetuning/chunk_residual/tests/test_frozen_shore_libero_eval.py \
  resfit/rl_finetuning/chunk_residual/tests/test_shore_libero_launcher.py \
  resfit/rl_finetuning/chunk_residual/tests/test_plot_shore_libero90_zero_shot.py \
  -q
bash -n run_shore_libero90_zero_shot_4gpu.sh
```

Expected: all tests pass and Bash syntax is valid.

### Step 2: Check assets and choose the four idle GPUs

```bash
sha256sum \
  outputs_chunk/libero10_task8_pi0feat_bp_bc01_h10/best.pt \
  outputs_chunk/libero10_t8moka_pi0_feat_gc_value.pt \
  outputs_chunk/libero10_t8moka_pi0_feat_high_actor.pt \
  outputs_chunk/libero10_t8moka_pi0_feat.npz
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
```

Expected: every asset exists and four GPUs have enough free memory for one PI0
server plus one residual evaluator. Record the selected IDs; do not assume
they are `0 1 2 3`.

### Step 3: Run a one-episode task63 smoke check

Start the PI0 feature server on one selected GPU and port 8063. Then invoke the
driver with `--task-id 63 --num-episodes 1` and a distinct smoke run directory.
Verify:

```bash
wc -l outputs_eval/shore_libero90_zero_shot/smoke-task63/task63/episodes.jsonl
sed -n '1p' \
  outputs_eval/shore_libero90_zero_shot/smoke-task63/task63/episodes.jsonl
```

Expected: exactly one complete row with `init_state_id: 0`, a binary success,
and the expected source-checkpoint hash. Logs must show PI0 `prefix_feat`,
strict residual loading, and frozen modules. They must not show optimizer,
replay buffer, W&B, backward, or update activity.

### Step 4: Launch the four formal evaluations

Set `GPU_IDS` to the four IDs selected in Step 2 and keep the printed `RUN_ID`
for validation and resume:

```bash
export GPU_IDS
RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RUN_ID
echo "formal run: GPU_IDS=${GPU_IDS} RUN_ID=${RUN_ID}"
bash run_shore_libero90_zero_shot_4gpu.sh
```

Expected: all four task directories receive 50 flushed rows and a `COMPLETED`
marker. If interrupted, rerun with the same `RUN_ID`; the driver validates the
config and continues only missing init-state IDs.

### Step 5: Validate all raw records before plotting

Run the plotting script in the same shell, where `RUN_ID` remains set:

```bash
MPLBACKEND=Agg /mnt/mnt/data/envs/resfit-libero/bin/python \
  paper/plot_shore_libero90_zero_shot.py \
  --run-root "outputs_eval/shore_libero90_zero_shot/${RUN_ID}" \
  --json-out paper/data/fig_shore_libero90_zero_shot.json \
  --npz-out paper/data/fig_shore_libero90_zero_shot.npz \
  --pdf-out paper/figure/fig_shore_libero90_zero_shot.pdf
```

Expected: the script reports 200 unique `(task_id, init_state_id)` pairs, one
common frozen source bundle, and four final success counts.

### Step 6: Inspect the generated artifacts

```bash
pdfinfo paper/figure/fig_shore_libero90_zero_shot.pdf
ls -lh \
  paper/data/fig_shore_libero90_zero_shot.json \
  paper/data/fig_shore_libero90_zero_shot.npz \
  paper/figure/fig_shore_libero90_zero_shot.pdf
/mnt/mnt/data/envs/resfit-libero/bin/python -c \
  "import json; p='paper/data/fig_shore_libero90_zero_shot.json'; d=json.load(open(p)); assert len(d['tasks']) == 4; assert sum(len(t['successes']) for t in d['tasks']) == 200; print([(t['task_id'], sum(t['successes'])) for t in d['tasks']])"
```

Expected: a valid one-page PDF, nonempty JSON/NPZ files, four tasks, and 200
raw successes. Visually inspect the PDF to confirm four readable panels,
correct labels, and no DSRL curve.

### Step 7: Final regression and status report

```bash
git status --short
git log --oneline -7
```

Report:

- the exact run root and selected GPU mapping;
- per-task successes out of 50;
- source-bundle hashes;
- test results;
- smoke and formal-run status;
- clickable paths to JSON, NPZ, PDF, and task logs; and
- any failed episode or resumed segment.

Do not commit bulky runtime logs. Commit the final JSON, NPZ, and PDF only if
the repository’s existing paper-artifact policy tracks generated figure data;
otherwise leave them saved at the required paths and report them explicitly.
