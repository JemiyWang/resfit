# chunk_residual critic warmup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional critic-only warmup phase to `train_chunk_residual.py` that runs N critic-only gradient updates on the filled buffer before actor training begins, mirroring offpolicy's `_run_critic_warmup`.

**Architecture:** New flag `--critic_warmup_steps` (default 0 = no change). A small isolated module `critic_warmup.py` holds the pure loop `run_critic_warmup(agent, steps, sample_batch_fn)`. `main()` calls it once when `env_steps` first reaches `learning_starts`, before the actor-enabled update loop, using the same batch sampling as the main loop.

**Tech Stack:** Python, PyTorch, torchrl, pytest. Repo: `/mnt/mnt/data/resfit`, branch `chunk-residual-validation`. Env: `/mnt/mnt/data/envs/residual` (run pytest with that env's python).

## Global Constraints

- Default `--critic_warmup_steps 0` MUST be behavior-identical to current code: `run_critic_warmup` is only called when steps > 0, so no extra RNG draws / no path change at default.
- During critic warmup: actor frozen (`update_actor=False`), no bc loss (`bc_batch=None`), `stddev=0.0` for target smoothing (aligns offpolicy), and the environment is NOT stepped.
- `agent.update` signature is fixed: `update(batch, stddev, update_actor, bc_batch=None, ref_agent=None)` (`resfit/rl_finetuning/off_policy/rl/q_agent.py:665`).
- Only touch `train_chunk_residual.py` + one new module + two new test files. Do not modify `agent.update`, buffers, or offpolicy repo.
- Run tests with: `/mnt/mnt/data/envs/residual/bin/python -m pytest <path> -v` from repo root `/mnt/mnt/data/resfit` (PYTHONPATH=/mnt/mnt/data/resfit).

---

### Task 1: Add `--critic_warmup_steps` flag

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` (in `build_parser()`, near line 585 after `--learning_starts`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup_cli.py` (create)

**Interfaces:**
- Produces: argparse arg `critic_warmup_steps: int` (default `0`) on the namespace returned by `build_parser().parse_args(...)`.

- [ ] **Step 1: Write the failing test**

Create `resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup_cli.py`:
```python
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def _base_args():
    # Minimal args build_parser needs; --task/--dataset are the common required-ish flags.
    return ["--task", "TwoArmPouring", "--dataset", "d"]


def test_critic_warmup_steps_default_zero():
    a = build_parser().parse_args(_base_args())
    assert a.critic_warmup_steps == 0


def test_critic_warmup_steps_parses_value():
    a = build_parser().parse_args(_base_args() + ["--critic_warmup_steps", "10000"])
    assert a.critic_warmup_steps == 10000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup_cli.py -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'critic_warmup_steps'`.

- [ ] **Step 3: Add the flag**

In `train_chunk_residual.py` `build_parser()`, immediately after the `--learning_starts` line (line 585):
```python
    p.add_argument("--critic_warmup_steps", type=int, default=0,
                   help="critic-only updates on the filled buffer before actor training "
                        "(mirrors offpolicy _run_critic_warmup). 0 = disabled (default).")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup_cli.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
        resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup_cli.py
git commit -m "feat(chunk_residual): add --critic_warmup_steps flag (default 0)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `run_critic_warmup` helper (isolated, unit-tested)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/critic_warmup.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup.py` (create)

**Interfaces:**
- Produces: `run_critic_warmup(agent, steps, sample_batch_fn, *, log_every=1000) -> last_metrics | None`.
  - Calls `agent.update(batch, 0.0, False, bc_batch=None, ref_agent=None)` exactly `steps` times, each on `batch = sample_batch_fn()`.
  - Returns the last metrics dict `agent.update` returned, or `None` if `steps <= 0`.
  - Does not import torch or any RL machinery (pure orchestration) so it stays fast to import in tests.

- [ ] **Step 1: Write the failing test**

Create `resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup.py`:
```python
from resfit.rl_finetuning.chunk_residual.critic_warmup import run_critic_warmup


class _MockAgent:
    def __init__(self):
        self.calls = []

    def update(self, batch, stddev, update_actor, bc_batch=None, ref_agent=None):
        self.calls.append(dict(stddev=stddev, update_actor=update_actor,
                               bc_batch=bc_batch, ref_agent=ref_agent))
        return {"critic_loss": 0.5}


def test_runs_critic_only_n_times():
    agent = _MockAgent()
    sampled = {"n": 0}

    def sample_fn():
        sampled["n"] += 1
        return object()

    out = run_critic_warmup(agent, 5, sample_fn, log_every=0)
    assert len(agent.calls) == 5
    assert sampled["n"] == 5
    assert all(c["update_actor"] is False for c in agent.calls)
    assert all(c["stddev"] == 0.0 for c in agent.calls)
    assert all(c["bc_batch"] is None and c["ref_agent"] is None for c in agent.calls)
    assert out == {"critic_loss": 0.5}


def test_zero_steps_is_noop():
    agent = _MockAgent()
    out = run_critic_warmup(agent, 0, lambda: object(), log_every=0)
    assert agent.calls == []
    assert out is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...critic_warmup'`.

- [ ] **Step 3: Write minimal implementation**

Create `resfit/rl_finetuning/chunk_residual/critic_warmup.py`:
```python
"""Critic-only warmup for chunk_residual RL (aligns offpolicy _run_critic_warmup).

Runs `steps` critic-only gradient updates (actor frozen) on batches produced by
`sample_batch_fn`, WITHOUT stepping the environment. Intended to run once, right
after the online buffer is filled and before actor training begins.
"""


def run_critic_warmup(agent, steps, sample_batch_fn, *, log_every=1000):
    """Do `steps` critic-only updates and return the last metrics dict (or None).

    agent.update is called as agent.update(batch, 0.0, False, bc_batch=None,
    ref_agent=None): stddev=0.0 target smoothing, update_actor=False (critic only),
    no BC. `sample_batch_fn()` must return a fresh training batch per call.
    """
    metrics = None
    for i in range(steps):
        batch = sample_batch_fn()
        metrics = agent.update(batch, 0.0, False, bc_batch=None, ref_agent=None)
        if log_every and (i + 1) % log_every == 0:
            cl = metrics.get("critic_loss") if isinstance(metrics, dict) else None
            tail = f" critic_loss={cl:.4f}" if isinstance(cl, (int, float)) else ""
            print(f"[critic-warmup] {i + 1}/{steps}{tail}")
    return metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/critic_warmup.py \
        resfit/rl_finetuning/chunk_residual/tests/test_critic_warmup.py
git commit -m "feat(chunk_residual): add run_critic_warmup helper + unit tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire critic warmup into the training loop

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` (import at top ~line 44; `main()` loop, the `if env_steps >= args.learning_starts ...` block at line 1225 and just before it)

**Interfaces:**
- Consumes: `run_critic_warmup` (Task 2); `critic_warmup_steps` arg (Task 1); existing in-scope names `online_rb, offline_rb, online_batch_size, offline_batch_size, args.stage_balanced, sample_gen, args.device, sample_stage_balanced, concat_mixed_batch`.

**Context — the block being modified** (`train_chunk_residual.py:1225-1236`, current):
```python
        if env_steps >= args.learning_starts and len(online_rb) > online_batch_size:
            for i in range(args.utd):
                if args.stage_balanced:
                    online_batch = sample_stage_balanced(online_rb, online_batch_size, generator=sample_gen)
                else:
                    online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(args.device, non_blocking=True)
                if offline_rb is not None:
                    offline_batch = offline_rb.sample(offline_batch_size).to(args.device, non_blocking=True)
                    batch = concat_mixed_batch(online_batch, offline_batch)
                else:
                    batch = online_batch
                update_actor = ((i + 1) % args.utd == 0)
                ...
```

- [ ] **Step 1: Add the import**

Near the other chunk_residual imports (after line 44 `from ...stage_replay import sample_stage_balanced`):
```python
from resfit.rl_finetuning.chunk_residual.critic_warmup import run_critic_warmup
```

- [ ] **Step 2: Initialize the one-shot guard before the loop**

Just before the main loop (`while env_steps <= total:`, line 1180) — beside the other pre-loop inits like `next_eval = 0` (line 1174):
```python
    did_critic_warmup = False
```

- [ ] **Step 3: Factor sampling into a closure and insert the warmup call**

Replace the body of the `if env_steps >= args.learning_starts and len(online_rb) > online_batch_size:` block (lines 1225-1234) so the per-batch sampling becomes a local closure used by BOTH the warmup and the utd loop:
```python
        if env_steps >= args.learning_starts and len(online_rb) > online_batch_size:
            def _sample_train_batch():
                if args.stage_balanced:
                    online_batch = sample_stage_balanced(online_rb, online_batch_size, generator=sample_gen)
                else:
                    online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(args.device, non_blocking=True)
                if offline_rb is not None:
                    offline_batch = offline_rb.sample(offline_batch_size).to(args.device, non_blocking=True)
                    return concat_mixed_batch(online_batch, offline_batch)
                return online_batch

            if args.critic_warmup_steps > 0 and not did_critic_warmup:
                print(f"[critic-warmup] starting {args.critic_warmup_steps} critic-only "
                      f"updates at env_steps={env_steps} (actor frozen, no env stepping)…")
                run_critic_warmup(agent, args.critic_warmup_steps, _sample_train_batch)
                did_critic_warmup = True

            for i in range(args.utd):
                batch = _sample_train_batch()
                update_actor = ((i + 1) % args.utd == 0)
```
(Everything from `update_actor = ((i + 1) % args.utd == 0)` onward — the `bc_batch` handling, `agent.update(...)`, diagnostics — stays exactly as it currently is at lines 1237+.)

- [ ] **Step 4: Regression check — default path unchanged**

Run the existing chunk_residual test suite to confirm no regression from the refactor:
Run: `/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: PASS (all previously-passing tests still pass; the 4 new tests from Tasks 1-2 pass).

Also confirm by inspection: with `--critic_warmup_steps 0`, the `if args.critic_warmup_steps > 0` guard is false → `run_critic_warmup` is never called → the utd loop samples exactly as before (same calls, same order, same RNG) → behavior-identical.

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat(chunk_residual): run critic warmup before actor training

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Manual end-to-end smoke (requires GPU + env; run when a card is free)**

This is a real-env check, not a pytest. From repo root, source the env and run a tiny smoke with the flag on a free GPU (pick an idle card):
```bash
cd /mnt/mnt/data/resfit
CUDA_VISIBLE_DEVICES=<free_gpu> MUJOCO_GL=egl PYTHONPATH=/mnt/mnt/data/resfit \
  /mnt/mnt/data/envs/residual/bin/python -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --smoke --learning_starts 4 --critic_warmup_steps 20 \
  <plus the same base/offcache/value/gc_value/high_actor/act_feat flags as an existing pouring run>
```
Expected in stdout: `[critic-warmup] starting 20 critic-only updates at env_steps=...` appears once, before the normal per-step training logs, and training then proceeds without error. (Reuse an existing pouring run's flags for the base/ckpt/offcache paths; the offcache will be reused so no long rebuild.)

---

## Self-Review

**1. Spec coverage:**
- Flag `--critic_warmup_steps` default 0 → Task 1. ✓
- Faithful critic-only block, static buffer, no env stepping → Task 2 (`run_critic_warmup`) + Task 3 (called once at boundary). ✓
- actor frozen / bc off / stddev=0.0 → Task 2 (`agent.update(batch, 0.0, False, bc_batch=None, ref_agent=None)`). ✓
- HIQL value/high_actor auto-frozen → guaranteed by design: `run_critic_warmup` never calls `finetuner.on_step` and no env stepping; noted, no code needed. ✓
- Reuse main-loop sampling (stage_balanced + concat_mixed_batch) → Task 3 `_sample_train_batch` closure used by both. ✓
- Default 0 byte-identical → Task 3 Step 4 (guard + regression). ✓
- Runs once → `did_critic_warmup` guard. ✓
- Logging every ~1000 → `log_every` in `run_critic_warmup`. ✓
- Testing (actor unchanged / only critic) → unit test asserts `update_actor=False` for every call; end-to-end smoke in Task 3 Step 6. ✓ (Note: the "actor params byte-identical before/after" assertion from the spec is covered structurally by asserting every update is `update_actor=False`; a full param-snapshot test would need a real agent and is deferred to the manual smoke.)

**2. Placeholder scan:** No TBD/TODO; all code shown. The smoke command has one intentional `<free_gpu>` / `<...flags...>` placeholder because those are runtime/experiment-specific (documented as such). ✓

**3. Type consistency:** `run_critic_warmup(agent, steps, sample_batch_fn, *, log_every=1000)` — same name/signature in Task 2 definition, Task 2 test, and Task 3 call site. `agent.update(batch, 0.0, False, bc_batch=None, ref_agent=None)` matches the real signature at `q_agent.py:665`. ✓
