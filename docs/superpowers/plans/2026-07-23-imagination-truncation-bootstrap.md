# Imagination Truncation Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the two-segment SAME_STEP imagination reset while storing the second segment as a nonterminal TD3 transition whose replay successor is the generated endpoint `s2`.

**Architecture:** `ImaginationVecEnv` captures raw `s2` and the already-queried `base_chunk(s2)` before resetting to real seed `B`. `ChunkResidualEnvWrapper` augments both observations independently, and the trainer uses `B` for control flow but `s2` with `done=False` for replay. Existing `MultiStepTransform` and `QAgent` then bootstrap from `s2` without any critic implementation change.

**Tech Stack:** Python 3, PyTorch, NumPy, TensorDict/TorchRL replay transforms, pytest

## Global Constraints

- Preserve `max_segments=2` behavior: truncation sequence remains `False, True, False, True`.
- Preserve SAME_STEP autoreset: the observation returned by the truncating step is reset seed `B`.
- The truncating transition's replay successor is fully augmented `s2`, not `B`.
- The truncating transition uses `replay_done=False` when `terminated=False`.
- Do not perform a third world-model rollout from `s2`.
- Do not add a kai0 / π0.5 query for `base_chunk(s2)`; reuse the endpoint query already needed for `Phi(s2)`.
- Preserve PBRS reward exactly as `imagination_gamma * Phi(s2) - Phi(s1)`.
- Keep normal real-environment terminated/truncated replay semantics unchanged.
- Keep `n_step=1` and trainer `gamma == imagination_gamma`.
- Reject imagination with `subgoal_conditioned=True` until an endpoint-subgoal protocol exists.
- Do not modify `resfit/rl_finetuning/off_policy/rl/q_agent.py`.
- The worktree contains unrelated edits; stage and commit only the exact files named by each task.

---

## File Structure

- Modify `resfit/rl_finetuning/wm_bridge/imagination_env.py`
  - Validate the segment horizon and expose the raw endpoint bootstrap payload before autoreset.
- Modify `resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py`
  - Verify endpoint identity, saved base chunk, query count, two-segment reset, and PBRS behavior.
- Modify `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`
  - Turn the raw endpoint payload into the same augmented observation schema consumed by TD3.
- Modify `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`
  - Verify independent augmentation of generated endpoint `s2` and reset state `B`, plus failure contracts.
- Modify `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
  - Separate control-loop completion from replay terminal semantics.
- Create `resfit/rl_finetuning/chunk_residual/tests/test_imagination_replay_boundary.py`
  - Verify replay selection, `MultiStepTransform.nonterminal`, and the resulting bootstrapped scalar target.
- Modify `resfit/rl_finetuning/wm_bridge/contract.py`
  - Reject unsupported imagination/subgoal combinations in both direct and passthrough argument checks.
- Modify `resfit/rl_finetuning/wm_bridge/tests/test_contract.py`
  - Cover the new launch contract.

---

### Task 1: Preserve the World-Model Endpoint Before SAME_STEP Reset

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/imagination_env.py:34-164`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py:43-186`

**Interfaces:**
- Consumes: `base.query(endpoint_obs) -> tuple[np.ndarray, np.ndarray]`, where the first value has shape `(CHUNK_LENGTH, ACTION_DIM)`.
- Produces: on imagination truncation, `info["final_observation"]`, `info["_wm_final_base_chunk"]`, and `info["bootstrap_on_truncation"]`.

- [ ] **Step 1: Write failing horizon and endpoint-payload tests**

Add a tagged base-policy fake and these tests to `test_imagination_env.py`:

```python
class _TaggedBase(_StubBase):
    def query(self, raw_obs):
        self.n += 1
        actions = np.full(
            (CHUNK_LENGTH, ACTION_DIM), float(self.n), dtype=np.float32)
        psi = np.full(8, float(self.n), dtype=np.float32)
        return actions, psi


def test_max_segments_must_be_positive():
    with pytest.raises(ValueError, match="max_segments"):
        _env(max_segments=0)


def test_truncation_exports_endpoint_base_chunk_before_reset_query():
    base = _TaggedBase()
    env = _env(base=base, max_segments=2)
    env.reset()                                      # query 1: initial seed

    for _ in range(CHUNK_LENGTH):
        env.step(_act())                             # query 2: s1
    for _ in range(CHUNK_LENGTH):
        reset_obs, _, terminated, truncated, info = env.step(_act())
                                                     # query 3: s2; query 4: B

    assert not terminated.item()
    assert truncated.item()
    assert info["bootstrap_on_truncation"] is True
    assert info["final_observation"]["_wm_window_token"] < reset_obs["_wm_window_token"]
    np.testing.assert_array_equal(
        info["_wm_final_base_chunk"],
        np.full((CHUNK_LENGTH, ACTION_DIM), 3.0, dtype=np.float32),
    )
    assert base.n == 4
```

Also extend `test_second_chunk_truncates_and_same_step_resets_to_real_seed`:

```python
assert info["bootstrap_on_truncation"] is True
assert info["_wm_final_base_chunk"].shape == (CHUNK_LENGTH, ACTION_DIM)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest -q \
  resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py::test_max_segments_must_be_positive \
  resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py::test_truncation_exports_endpoint_base_chunk_before_reset_query
```

Expected: both tests fail because nonpositive `max_segments` is accepted and the truncation info lacks the endpoint base chunk/marker.

- [ ] **Step 3: Implement endpoint capture without an extra base query**

In `ImaginationVecEnv.__init__`, validate the converted value:

```python
self.max_segments = int(max_segments)
if self.max_segments <= 0:
    raise ValueError(
        f"max_segments must be a positive integer, got {self.max_segments}")
```

In `ImaginationVecEnv.step`, retain the first result of the endpoint query:

```python
endpoint_obs = self._obs_from_window()
endpoint_base_chunk, psi = self.base.query(endpoint_obs)
phi_next = self.scorer.phi(psi, self._tracker.proprio)
```

Populate the payload before calling `reset()`:

```python
if truncated:
    final_base_chunk = np.asarray(
        endpoint_base_chunk, dtype=np.float32).copy()
    if final_base_chunk.shape != (CHUNK_LENGTH, ACTION_DIM):
        raise ValueError(
            "endpoint base chunk must have shape "
            f"({CHUNK_LENGTH}, {ACTION_DIM}), got {final_base_chunk.shape}")

    obs, reset_info = self.reset()
    info = dict(reset_info)
    info["final_observation"] = endpoint_obs
    info["_wm_final_base_chunk"] = final_base_chunk
    info["bootstrap_on_truncation"] = True
```

Update the module comment to state that `final_observation` is a replay bootstrap state even though the returned observation is the reset state.

- [ ] **Step 4: Run all imagination-env tests and confirm GREEN**

Run:

```bash
pytest -q resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py
```

Expected: all tests pass, including PBRS telescoping and the `False, True, False, True` reset sequence.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add \
  resfit/rl_finetuning/wm_bridge/imagination_env.py \
  resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py
git diff --cached --check
git commit -m "feat: expose imagination bootstrap endpoint"
```

---

### Task 2: Augment `s2` Independently From Reset State `B`

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py:157-231`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`

**Interfaces:**
- Consumes: Task 1 info keys `final_observation: dict`, `_wm_final_base_chunk: np.ndarray`, and `bootstrap_on_truncation: bool`.
- Produces: `info["final_observation"]` with `observation.state`, `observation.base_action`, and `observation.stage_id` in the standard augmented schema.

- [ ] **Step 1: Write a failing SAME_STEP dual-observation test**

Add these fakes and test to `test_chunk_env_wrapper.py`:

```python
class _BootstrapTruncationEnv:
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(D,), dtype=np.float32)

    @staticmethod
    def _obs(value):
        return {
            "observation.state": torch.full((1, 3), float(value)),
            "observation.images.cam": torch.full(
                (1, 3, 4, 4), float(value)),
        }

    def reset(self, **kwargs):
        return self._obs(0), {}

    def step(self, action):
        info = {
            "final_observation": self._obs(10),
            "_wm_final_base_chunk": np.full((1, D), 0.25, np.float32),
            "bootstrap_on_truncation": True,
        }
        return (
            self._obs(20),                         # reset state B
            torch.tensor([1.0]),
            torch.tensor([False]),
            torch.tensor([True]),
            info,
        )


class _CountingBase(_FakeBase):
    def __init__(self):
        self.chunk_calls = 0

    def get_action_chunk(self, raw_obs, chunk_length):
        self.chunk_calls += 1
        return super().get_action_chunk(raw_obs, chunk_length)


def test_imagination_truncation_augments_s2_but_returns_augmented_reset_seed():
    base = _CountingBase()
    wrapper = ChunkResidualEnvWrapper(
        _BootstrapTruncationEnv(), base, _IdentityScaler(), _IdentityStd(),
        chunk_length=1, base_action_mode="replan",
    )
    wrapper.reset()

    control_next_obs, _, terminated, truncated, info = wrapper.step(
        torch.zeros(1, D))
    replay_next_obs = info["final_observation"]

    assert not terminated.item()
    assert truncated.item()
    assert control_next_obs["observation.state"][0, 0].item() == 20.0
    assert replay_next_obs["observation.state"][0, 0].item() == 10.0
    assert torch.allclose(
        control_next_obs["observation.base_action"],
        torch.full((1, D), 0.5),
    )
    assert torch.allclose(
        replay_next_obs["observation.base_action"],
        torch.full((1, D), 0.25),
    )
    assert "observation.stage_id" in replay_next_obs
    assert base.chunk_calls == 2                 # reset state 0, then B; never s2
```

- [ ] **Step 2: Write failing payload-validation tests**

Add a configurable info fake:

```python
class _InvalidBootstrapEnv(_BootstrapTruncationEnv):
    def __init__(self, info, *, terminated=False, truncated=True):
        super().__init__()
        self.info = info
        self.terminated = terminated
        self.truncated = truncated

    def step(self, action):
        return (
            self._obs(20),
            torch.tensor([0.0]),
            torch.tensor([self.terminated]),
            torch.tensor([self.truncated]),
            dict(self.info),
        )


@pytest.mark.parametrize("missing_key", [
    "final_observation",
    "_wm_final_base_chunk",
])
def test_bootstrap_marker_requires_complete_endpoint_payload(missing_key):
    info = {
        "final_observation": _BootstrapTruncationEnv._obs(10),
        "_wm_final_base_chunk": np.full((1, D), 0.25, np.float32),
        "bootstrap_on_truncation": True,
    }
    del info[missing_key]
    wrapper = ChunkResidualEnvWrapper(
        _InvalidBootstrapEnv(info), _FakeBase(),
        _IdentityScaler(), _IdentityStd(), chunk_length=1)
    wrapper.reset()

    with pytest.raises(RuntimeError, match=missing_key):
        wrapper.step(torch.zeros(1, D))


def test_bootstrap_marker_rejects_true_termination():
    info = {
        "final_observation": _BootstrapTruncationEnv._obs(10),
        "_wm_final_base_chunk": np.full((1, D), 0.25, np.float32),
        "bootstrap_on_truncation": True,
    }
    wrapper = ChunkResidualEnvWrapper(
        _InvalidBootstrapEnv(info, terminated=True), _FakeBase(),
        _IdentityScaler(), _IdentityStd(), chunk_length=1)
    wrapper.reset()

    with pytest.raises(RuntimeError, match="terminated"):
        wrapper.step(torch.zeros(1, D))
```

Add `import pytest` at module scope.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
pytest -q \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_imagination_truncation_augments_s2_but_returns_augmented_reset_seed \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_bootstrap_marker_requires_complete_endpoint_payload \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_bootstrap_marker_rejects_true_termination
```

Expected: failures because `final_observation` remains raw and no payload validation exists.

- [ ] **Step 4: Add a focused endpoint augmentation helper**

Add this method to `ChunkResidualEnvWrapper`:

```python
def _augment_bootstrap_final_observation(
        self, info: dict, terminated: torch.Tensor,
        truncated: torch.Tensor) -> dict:
    info = dict(info)
    if not bool(info.get("bootstrap_on_truncation", False)):
        return info
    if bool(terminated.any()):
        raise RuntimeError(
            "bootstrap_on_truncation cannot be combined with terminated=True")
    if not bool(truncated.any()):
        raise RuntimeError(
            "bootstrap_on_truncation requires truncated=True")
    if "final_observation" not in info:
        raise RuntimeError(
            "bootstrap_on_truncation requires final_observation")
    if "_wm_final_base_chunk" not in info:
        raise RuntimeError(
            "bootstrap_on_truncation requires _wm_final_base_chunk")

    raw_final_obs = info["final_observation"]
    final_state = raw_final_obs["observation.state"]
    final_base_chunk = torch.as_tensor(
        info["_wm_final_base_chunk"],
        dtype=torch.float32,
        device=final_state.device,
    )
    if final_base_chunk.ndim == 2:
        final_base_chunk = final_base_chunk.unsqueeze(0)
    expected = (self.num_envs, self.chunk_length, self.action_dim)
    if tuple(final_base_chunk.shape) != expected:
        raise RuntimeError(
            f"_wm_final_base_chunk must have shape {expected}, "
            f"got {tuple(final_base_chunk.shape)}")

    final_base_flat = self.action_scaler.scale(final_base_chunk).reshape(
        self.num_envs, self.flat_dim)
    info["final_observation"] = self._augment(
        raw_final_obs, final_base_flat)
    return info
```

In `step()`, call it after reward computation but before resetting `_stage`/`_stage_now`:

```python
info = self._augment_bootstrap_final_observation(
    last_info, terminated, truncated)

if bool((terminated | truncated).any()):
    self.base_policy.reset()
    self._stage = 0
    self._stage_now = 0
```

At the end of `step()`, remove the second `info = dict(last_info)` assignment and append `scaled_action`/`max_stage_in_chunk` to the already processed `info`.

Replace the old docstring claim that final observations are never exposed with:

```python
"""执行一段 chunk(开环逐步),返回 chunk 级 transition。

普通 done 仍返回 autoreset 后的 augmented observation。带
bootstrap_on_truncation 标记的 imagination 边界还会在 info 中保留独立的、
完整 augmented final_observation，供 replay 从生成终点 bootstrap。
"""
```

- [ ] **Step 5: Run wrapper tests and confirm GREEN**

Run:

```bash
pytest -q resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
```

Expected: all wrapper tests pass; `s2` and `B` retain different state/base-action payloads and no endpoint base-policy call is added.

- [ ] **Step 6: Commit only Task 2 files**

```bash
git add \
  resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
git diff --cached --check
git commit -m "feat: augment imagination bootstrap state"
```

---

### Task 3: Separate Control Completion From Replay Terminal Semantics

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:88-109,1198-1241`
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_imagination_replay_boundary.py`

**Interfaces:**
- Consumes: Task 2 `info["final_observation"]`, marker `bootstrap_on_truncation`, returned control observation `B`, and environment `terminated`/`truncated` tensors.
- Produces: `resolve_replay_transition(control_next_obs, terminated, truncated, info) -> tuple[dict, torch.Tensor]`.

- [ ] **Step 1: Write failing replay-boundary unit tests**

Create `test_imagination_replay_boundary.py`:

```python
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    add_chunk_transition,
    resolve_replay_transition,
)
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform


def _obs(value):
    return {
        "observation.state": torch.full((1, 3), float(value)),
        "observation.base_action": torch.full((1, 4), float(value)),
    }


def test_marked_truncation_ends_control_episode_but_not_replay_bootstrap():
    control_next_obs = _obs(20)
    final_obs = _obs(10)
    terminated = torch.tensor([False])
    truncated = torch.tensor([True])
    info = {
        "bootstrap_on_truncation": True,
        "final_observation": final_obs,
    }

    replay_next_obs, replay_done = resolve_replay_transition(
        control_next_obs, terminated, truncated, info)
    episode_done = terminated | truncated

    assert episode_done.item() is True
    assert replay_done.item() is False
    assert replay_next_obs is final_obs
    assert replay_next_obs["observation.state"][0, 0].item() == 10.0
    assert control_next_obs["observation.state"][0, 0].item() == 20.0


@pytest.mark.parametrize(
    "terminated,truncated,expected_done",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_unmarked_transitions_keep_existing_done_semantics(
        terminated, truncated, expected_done):
    control_next_obs = _obs(20)
    replay_next_obs, replay_done = resolve_replay_transition(
        control_next_obs,
        torch.tensor([terminated]),
        torch.tensor([truncated]),
        {},
    )
    assert replay_next_obs is control_next_obs
    assert replay_done.item() is expected_done


def test_marked_truncation_requires_final_observation():
    with pytest.raises(RuntimeError, match="final_observation"):
        resolve_replay_transition(
            _obs(20), torch.tensor([False]), torch.tensor([True]),
            {"bootstrap_on_truncation": True})


def test_marked_truncation_cannot_be_terminal():
    with pytest.raises(RuntimeError, match="terminated"):
        resolve_replay_transition(
            _obs(20), torch.tensor([True]), torch.tensor([True]), {
                "bootstrap_on_truncation": True,
                "final_observation": _obs(10),
            })
```

- [ ] **Step 2: Write a failing replay-transform/bootstrap test**

Append this capture buffer and test:

```python
class _TransformCaptureBuffer:
    def __init__(self, gamma):
        self.transform = MultiStepTransform(n_steps=1, gamma=gamma)
        self.stored = None

    def add(self, td):
        self.stored = self.transform._inv_call(td)


def test_second_segment_replay_item_bootstraps_from_s2():
    gamma = 0.995
    terminated = torch.tensor([False])
    truncated = torch.tensor([True])
    info = {
        "bootstrap_on_truncation": True,
        "final_observation": _obs(10),
        "scaled_action": torch.zeros(1, 4),
    }
    replay_next_obs, replay_done = resolve_replay_transition(
        _obs(20), terminated, truncated, info)
    rb = _TransformCaptureBuffer(gamma)

    add_chunk_transition(
        obs=_obs(1),
        next_obs=replay_next_obs,
        combined_action=info["scaled_action"],
        reward=torch.tensor([2.0]),
        done=replay_done,
        info=info,
        image_keys=[],
        lowdim_keys=["observation.state", "observation.base_action"],
        online_rb=rb,
    )

    assert rb.stored["next", "obs", "observation.state"][0, 0].item() == 10.0
    assert rb.stored["nonterminal"].item() is True
    assert rb.stored["gamma"].item() == pytest.approx(gamma)

    target_q_at_s2 = torch.tensor(4.0)
    target = (
        rb.stored["next", "reward"]
        + rb.stored["gamma"]
        * rb.stored["nonterminal"]
        * target_q_at_s2
    )
    assert target.item() == pytest.approx(2.0 + gamma * 4.0)
```

This test verifies the exact tensors consumed by the existing `QAgent.update`; do not mock or edit `QAgent`.

- [ ] **Step 3: Run the new tests and confirm RED**

Run:

```bash
pytest -q resfit/rl_finetuning/chunk_residual/tests/test_imagination_replay_boundary.py
```

Expected: collection fails because `resolve_replay_transition` does not exist.

- [ ] **Step 4: Implement the pure replay selection helper**

Add this function immediately before `add_chunk_transition`:

```python
def resolve_replay_transition(
        control_next_obs: dict, terminated: torch.Tensor,
        truncated: torch.Tensor, info: dict) -> tuple[dict, torch.Tensor]:
    """Choose replay successor/terminal independently from control autoreset."""
    if not bool(info.get("bootstrap_on_truncation", False)):
        return control_next_obs, terminated | truncated
    if bool(terminated.any()):
        raise RuntimeError(
            "bootstrap_on_truncation cannot be combined with terminated=True")
    if not bool(truncated.any()):
        raise RuntimeError(
            "bootstrap_on_truncation requires truncated=True")
    if "final_observation" not in info:
        raise RuntimeError(
            "bootstrap_on_truncation requires final_observation")
    return info["final_observation"], terminated
```

Update `add_chunk_transition`'s docstring:

```python
"""Store the already-resolved replay transition for a single environment.

The caller must pass replay next_obs/replay done, which may differ from the
control observation and episode boundary returned by a SAME_STEP autoreset.
"""
```

- [ ] **Step 5: Wire separate control and replay paths into the main loop**

Replace the boundary-handling portion of the training loop with:

```python
control_next_obs, reward, terminated, truncated, info = env.step(action)
if args.subgoal_conditioned:
    next_rel = info.get("rel_piece")
    control_next_obs["observation.subgoal"] = compute_online_subgoal(
        subgoal, control_next_obs, base_policy, next_rel).to(
            control_next_obs["observation.state"].device)

episode_done = terminated | truncated
replay_next_obs, replay_done = resolve_replay_transition(
    control_next_obs, terminated, truncated, info)

if gc_potential is not None:
    _pf = base_policy.last_prefix_feat() if subgoal.state_mode == "pi0_feat" else None
    _s_start = subgoal.encode_state(obs, rel_raw=cur_rel, prefix_feat=_pf)
    _s_end = subgoal.encode_state(
        control_next_obs, rel_raw=next_rel, prefix_feat=_pf)
    _z_start = obs["observation.subgoal"]
    from resfit.rl_finetuning.chunk_residual.hiql_potential import gc_subgoal_shaping
    _shape = gc_subgoal_shaping(
        gc_potential, _s_start, _s_end, _z_start,
        bonus=args.stage_reward_bonus, gamma=args.gamma,
        done=bool(episode_done.any()))
    reward = reward + _shape

add_chunk_transition(
    obs=obs,
    next_obs=replay_next_obs,
    combined_action=info["scaled_action"],
    reward=reward,
    done=replay_done,
    info=info,
    image_keys=image_keys,
    lowdim_keys=lowdim_keys,
    online_rb=online_rb,
)
```

Use `episode_done` for lifecycle operations:

```python
if harvester is not None:
    harvester.add(
        make_bc_entry(obs, info["scaled_action"], image_keys, lowdim_keys),
        info.get("max_stage_in_chunk", 0),
    )
    if bool(episode_done.any()):
        for e in harvester.flush():
            relabel_rb.extend(e)

if finetuner is not None and bool(episode_done.any()):
    finetuner.on_episode_end()

obs = control_next_obs
if args.subgoal_conditioned:
    cur_rel = next_rel
```

Do not use `replay_done` for harvester/finetuner lifecycle, and do not assign `replay_next_obs` to the next control-loop `obs`.

- [ ] **Step 6: Run replay-boundary tests and focused existing tests**

Run:

```bash
pytest -q \
  resfit/rl_finetuning/chunk_residual/tests/test_imagination_replay_boundary.py \
  resfit/rl_finetuning/chunk_residual/tests/test_stage_replay.py \
  resfit/rl_finetuning/chunk_residual/tests/test_relabel.py
```

Expected: all tests pass; the transformed imagined boundary item has `nonterminal=True`, while normal transitions retain prior done behavior.

- [ ] **Step 7: Commit only Task 3 files**

```bash
git add resfit/rl_finetuning/chunk_residual/tests/test_imagination_replay_boundary.py
git add -p resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: bootstrap TD3 across imagination truncation"
```

When answering `git add -p`, stage only the new
`resolve_replay_transition`, its docstring adjustment, and the
control/replay split in the main loop. Leave all pre-existing hunks unstaged.

---

### Task 4: Reject Unsupported Subgoal-Conditioned Imagination

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/contract.py:78-139`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_contract.py:10-76`

**Interfaces:**
- Consumes: trainer argument `subgoal_conditioned: bool`, both from an argparse namespace and passthrough CLI.
- Produces: `ContractError` before training starts when imagination and subgoal conditioning are combined.

- [ ] **Step 1: Write failing direct and passthrough contract tests**

Add `subgoal_conditioned=False` to `_args` defaults:

```python
d = {
    "reward_shaping": "none",
    "potential_source": "stage",
    "chunk_length": 50,
    "base_action_mode": "replan",
    "n_step": 1,
    "gamma": 0.995,
    "subgoal_conditioned": False,
}
```

Add:

```python
def test_subgoal_conditioned_imagination_is_rejected():
    with pytest.raises(ContractError, match="subgoal_conditioned"):
        contract.check_runtime_args(_args(subgoal_conditioned=True))


def test_passthrough_subgoal_conditioned_imagination_is_rejected():
    argv = [
        "--reward_shaping", "none",
        "--potential_source", "stage",
        "--chunk_length", "50",
        "--base_action_mode", "replan",
        "--n_step", "1",
        "--gamma", "0.995",
        "--subgoal_conditioned",
    ]
    with pytest.raises(ContractError, match="subgoal_conditioned"):
        contract.check_passthrough_runtime_args(
            argv, imagination_gamma=0.995)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest -q \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py::test_subgoal_conditioned_imagination_is_rejected \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py::test_passthrough_subgoal_conditioned_imagination_is_rejected
```

Expected: the direct check does not raise and passthrough parsing ignores the flag.

- [ ] **Step 3: Implement the launch contract**

In `check_runtime_args`, after the gamma check, add:

```python
if bool(getattr(args, "subgoal_conditioned", False)):
    raise ContractError(
        "想象路暂不支持 --subgoal_conditioned：truncation bootstrap 的 "
        "final_observation 尚无独立终点 subgoal 协议")
```

In `check_passthrough_runtime_args`, parse the flag:

```python
p.add_argument("--subgoal_conditioned", action="store_true")
```

- [ ] **Step 4: Run all contract and launcher tests**

Run:

```bash
pytest -q \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_launcher.py
```

Expected: all tests pass; valid non-subgoal imagination arguments remain accepted.

- [ ] **Step 5: Commit only Task 4 files**

```bash
git add -p \
  resfit/rl_finetuning/wm_bridge/contract.py \
  resfit/rl_finetuning/wm_bridge/tests/test_contract.py
git diff --cached --name-only
git diff --cached --check
git commit -m "fix: reject unsupported imagination subgoals"
```

When answering `git add -p`, stage only the `subgoal_conditioned` validation,
passthrough parser flag, and their tests. Leave all pre-existing hunks
unstaged.

---

## Final Verification

- [ ] **Step 1: Run the complete focused regression suite**

```bash
pytest -q \
  resfit/rl_finetuning/wm_bridge/tests \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/tests/test_imagination_replay_boundary.py \
  resfit/rl_finetuning/chunk_residual/tests/test_stage_replay.py \
  resfit/rl_finetuning/chunk_residual/tests/test_relabel.py
```

Expected: all tests pass.

- [ ] **Step 2: Verify no critic implementation changed**

```bash
git diff bcbb042 --name-only
```

Expected: output does not include `resfit/rl_finetuning/off_policy/rl/q_agent.py`.

- [ ] **Step 3: Verify endpoint/reset dataflow and query contract in tests**

```bash
pytest -q \
  resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py::test_truncation_exports_endpoint_base_chunk_before_reset_query \
  resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_imagination_truncation_augments_s2_but_returns_augmented_reset_seed \
  resfit/rl_finetuning/chunk_residual/tests/test_imagination_replay_boundary.py::test_second_segment_replay_item_bootstraps_from_s2
```

Expected: three tests pass, jointly proving:

```text
control:  second WM segment -> s2 -> SAME_STEP reset -> B -> next actor step
replay:   (s1, combined_chunk2, reward2, s2, done=False)
target:   reward2 + gamma * min Q_target(s2, base(s2) + residual_target(s2))
```

- [ ] **Step 4: Audit only intended changes**

```bash
git status --short
git log --oneline -5
```

Expected: the four task commits are present. Any pre-existing unrelated dirty files remain untouched and unstaged.
