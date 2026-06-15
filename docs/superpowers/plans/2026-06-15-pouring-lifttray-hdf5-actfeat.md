# pouring / lifttray 接 act_feat 残差 RL（纯 hdf5 路线）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 pouring / lifttray 走与 three_piece 完全相同的 act_feat hdf5 残差 RL 路线（离线 cache 图像+state 全读 hdf5），核心是把硬编码 18 维的 proprio 拼装换成 env-aware（按 task 选 key 集、完整维度 concat），实现真同源在线 + ~100x 提速 + 对 three_piece/threading 零回归。

**Architecture:** 在 `offline_hdf5_buffer.py` 复刻在线 dexmg 的 proprio key 选择（`expected_low_dim_keys` + `assemble_state_by_env`，用一致性测试守护不漂移），把 act_feat 三处硬编码 `assemble_state18` 的 hdf5 拼装点改成 env-aware 并把 `env_hint`（dataset_id / args.task）贯穿进去；现有 lerobot 路线与 eef/eef_piece/libero 路线完全不动。

**Tech Stack:** Python, h5py, numpy, torch, pytest, conda env `residual`；仓库 git 分支 `chunk-residual-validation`。

**Spec:** `docs/superpowers/specs/2026-06-15-pouring-lifttray-hdf5-actfeat-design.md`

**约定（所有 task 通用）：**
- 测试运行：`cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest <path> -v`
- 提交：每个 task 末尾 `git add` 相关文件 + `git commit`（分支 chunk-residual-validation，非默认分支，直接提交）。
- **Task 1、2 的实现代码已预落地**（brainstorming 前探索时写入）。执行时：先写/确认测试 → 跑测试；若已 PASS 说明预落地版与本 plan 一致，跳过该 task 的实现 step；若 FAIL 或代码缺失，按下方实现 step 写。

---

### Task 1: env-aware proprio 拼装核心函数（offline_hdf5_buffer.py）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_offline_hdf5_buffer.py`

- [ ] **Step 1: 写失败测试**（追加到 test_offline_hdf5_buffer.py 末尾）

```python
def test_expected_low_dim_keys_per_task():
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
        expected_low_dim_keys, LOW_DIM_KEYS_MULTI, LOW_DIM_KEYS_HUMANOID)
    for hint in ("ankile/dexmg-two-arm-pouring", "TwoArmPouring"):
        assert expected_low_dim_keys(hint) == LOW_DIM_KEYS_HUMANOID
    for hint in ("ankile/dexmg-two-arm-three-piece-assembly", "TwoArmThreePieceAssembly",
                 "ankile/dexmg-two-arm-threading", "TwoArmThreading",
                 "ankile/dexmg-two-arm-lift-tray", "TwoArmLiftTray"):
        assert expected_low_dim_keys(hint) == LOW_DIM_KEYS_MULTI


def test_assemble_state_by_env_dims():
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import assemble_state_by_env
    T = 2
    pouring = {"robot0_right_eef_pos": np.zeros((T, 3)), "robot0_right_eef_quat": np.zeros((T, 4)),
               "robot0_right_gripper_qpos": np.zeros((T, 11)),
               "robot0_left_eef_pos": np.zeros((T, 3)), "robot0_left_eef_quat": np.zeros((T, 4)),
               "robot0_left_gripper_qpos": np.zeros((T, 11))}
    assert assemble_state_by_env(pouring, "TwoArmPouring").shape == (T, 36)
    lifttray = {"robot0_eef_pos": np.zeros((T, 3)), "robot0_eef_quat": np.zeros((T, 4)),
                "robot0_gripper_qpos": np.zeros((T, 12)),
                "robot1_eef_pos": np.zeros((T, 3)), "robot1_eef_quat": np.zeros((T, 4)),
                "robot1_gripper_qpos": np.zeros((T, 12))}
    assert assemble_state_by_env(lifttray, "TwoArmLiftTray").shape == (T, 38)


def test_assemble_state_by_env_equals_state18_for_two_arm_panda():
    # two-arm Panda(gripper_qpos==2)：完整 concat 必须与 assemble_state18 逐位相等(零回归)
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
        assemble_state_by_env, assemble_state18, STATE18_KEYS)
    rs = np.random.RandomState(0)
    obs = {k: rs.randn(3, d) for k, d in STATE18_KEYS}      # gripper d==2
    a = assemble_state_by_env(obs, "TwoArmThreePieceAssembly")
    b = assemble_state18(obs)
    assert a.shape == (3, 18)
    assert np.array_equal(a, b)


def test_expected_low_dim_keys_consistent_with_dexmg():
    # 守护离线复刻与在线 dexmg 不漂移；dexmg 顶层硬 import robosuite-1.5，import 失败则 skip
    import pytest
    dexmg = pytest.importorskip(
        "resfit.dexmg.environments.dexmg",
        reason="dexmg 需 robosuite-1.5，CI 缺失时跳过；硬编码期望值已由其余测试兜底")
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import expected_low_dim_keys
    cls = dexmg.DexMG if hasattr(dexmg, "DexMG") else None
    assert cls is not None and hasattr(cls, "_get_expected_low_dim_keys")
    for env_name in ("TwoArmPouring", "TwoArmThreePieceAssembly", "TwoArmThreading", "TwoArmLiftTray"):
        ref = cls._get_expected_low_dim_keys(None, env_name)   # 仅用 env_name，self 未使用
        assert expected_low_dim_keys(env_name) == list(ref)
```

- [ ] **Step 2: 运行测试确认现状**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_offline_hdf5_buffer.py -v`
Expected: 若实现已预落地 → 全 PASS（一致性测试可能 SKIP）；若未落地 → 前几个测试 FAIL（ImportError / 函数未定义）。FAIL 则做 Step 3。

- [ ] **Step 3: 实现（仅当 Step 2 未全 PASS）**

在 `offline_hdf5_buffer.py` 的 `STATE18_KEYS` 定义之后加入：

```python
# env-aware proprio 拼装：observation.state 的 key 集与拼接语义，唯一真相对齐在线 dexmg
# resfit/dexmg/environments/dexmg.py::_get_expected_low_dim_keys + _process_obs
# （按 key 顺序、完整维度 np.concatenate、不截断）。离线不能 import dexmg.py（顶层硬 import
# robosuite-1.5，offline build 是纯读 hdf5 的 residual env），故在此复刻；
# test_expected_low_dim_keys_consistent_with_dexmg 守护不漂移。
# 维度由 hdf5/env 字段自然决定（实测）：Panda 夹爪 gripper_qpos==2 → three_piece/threading
# 完整拼=18（与 STATE18_KEYS 逐位等价）；pouring=36；lifttray=38。
LOW_DIM_KEYS_SINGLE = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
LOW_DIM_KEYS_MULTI = LOW_DIM_KEYS_SINGLE + [
    "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]
LOW_DIM_KEYS_HUMANOID = [
    "robot0_right_eef_pos", "robot0_right_eef_quat", "robot0_right_gripper_qpos",
    "robot0_left_eef_pos", "robot0_left_eef_quat", "robot0_left_gripper_qpos"]

_SINGLE_ARM_ENVS = {"lift", "can", "pickplacecan", "square", "nutassemblysquare", "threading"}


def expected_low_dim_keys(hint: str) -> list:
    """按 task 提示选 observation.state 的 proprio key 集（复刻 dexmg._get_expected_low_dim_keys）。

    hint 可为 robosuite env_name（'TwoArmPouring'）或 LeRobot dataset_id
    （'ankile/dexmg-two-arm-pouring'）—小写后用同款包含/精确匹配，对现有 4 个 task 结果一致。
    """
    h = str(hint).lower()
    if any(t in h for t in ("pouring", "coffee", "cansort", "can_sort")):
        return list(LOW_DIM_KEYS_HUMANOID)
    if h in _SINGLE_ARM_ENVS:
        return list(LOW_DIM_KEYS_SINGLE)
    return list(LOW_DIM_KEYS_MULTI)


def assemble_state_by_env(obs, hint: str) -> np.ndarray:
    """按 env 的 key 集完整维度拼 (T,D) observation.state（对齐在线 dexmg._process_obs，不截断）。

    three_piece/threading 得 18（== assemble_state18），pouring 得 36，lifttray 得 38。
    缺字段 KeyError（同源命门：拼不出即数据/任务不匹配）。
    """
    keys = expected_low_dim_keys(hint)
    return np.concatenate([np.asarray(obs[k]) for k in keys], axis=1)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_offline_hdf5_buffer.py -v`
Expected: PASS（一致性测试 PASS 或 SKIP）。

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py \
        resfit/rl_finetuning/chunk_residual/tests/test_offline_hdf5_buffer.py
git commit -m "feat(offline): env-aware proprio 拼装(expected_low_dim_keys/assemble_state_by_env),对齐在线 dexmg"
```

---

### Task 2: act_feat hdf5 路线 proprio 走 env-aware（train_hiql_value.py）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`（import + `_build_raw_obs_seqs` + act_feat build 调用点）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py`

- [ ] **Step 1: 写失败测试**（追加到 test_read_per_demo_states_act_feat.py 末尾）

```python
def test_build_raw_obs_seqs_env_hint_pouring(tmp_path):
    """env_hint=pouring → proprio 按 humanoid key 集完整拼 36 维；图像处理不变。"""
    import h5py
    import numpy as np
    import torch
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import _build_raw_obs_seqs
    T = 3
    p = str(tmp_path / "pour.hdf5")
    fields = {"robot0_right_eef_pos": 3, "robot0_right_eef_quat": 4, "robot0_right_gripper_qpos": 11,
              "robot0_left_eef_pos": 3, "robot0_left_eef_quat": 4, "robot0_left_gripper_qpos": 11}
    with h5py.File(p, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("obs/agentview_image",
                         data=(np.arange(T * 84 * 84 * 3).reshape(T, 84, 84, 3) % 256).astype(np.uint8))
        for k, d in fields.items():
            g.create_dataset(f"obs/{k}", data=np.zeros((T, d), np.float32))
    seqs = _build_raw_obs_seqs(p, ["observation.images.agentview"], "observation.state",
                               num_demos=None, env_hint="ankile/dexmg-two-arm-pouring")
    assert seqs[0]["observation.state"].shape == (T, 36)
    assert seqs[0]["observation.images.agentview"].shape == (T, 3, 84, 84)


def test_build_raw_obs_seqs_env_hint_none_backcompat(tmp_path):
    """env_hint=None → 回退旧 18 维 STATE18 行为(零回归)。"""
    import h5py
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import _build_raw_obs_seqs
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import STATE18_KEYS
    T = 2
    p = str(tmp_path / "tp.hdf5")
    with h5py.File(p, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("obs/agentview_image",
                         data=np.zeros((T, 84, 84, 3), np.uint8))
        for k, d in STATE18_KEYS:
            g.create_dataset(f"obs/{k}", data=np.zeros((T, d), np.float32))
    seqs = _build_raw_obs_seqs(p, ["observation.images.agentview"], "observation.state",
                               num_demos=None)
    assert seqs[0]["observation.state"].shape == (T, 18)
```

- [ ] **Step 2: 运行测试确认现状**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py -v`
Expected: 预落地则全 PASS；否则 `test_build_raw_obs_seqs_env_hint_pouring` FAIL（_build_raw_obs_seqs 不接受 env_hint）。FAIL 则做 Step 3。

- [ ] **Step 3: 实现（仅当 Step 2 未全 PASS）**

3a. import（把 offline_hdf5_buffer import 块改为）：

```python
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    STATE18_KEYS, assemble_state18, assemble_state_by_env, expected_low_dim_keys,
    sorted_demo_keys,
)
```

3b. `_build_raw_obs_seqs` 改为（加 env_hint 参数 + env-aware 拼装，env_hint=None 回退 STATE18）：

```python
def _build_raw_obs_seqs(hdf5_path, image_keys, proprio_key, num_demos, env_hint=None):
    """每条 demo -> 一个 raw_obs dict(整段 T 帧):ACT image_features 键 + proprio_key。

    proprio 按 env_hint(dataset_id/env_name)的 key 集完整维度拼,源头对齐在线 dexmg
    (three_piece/threading=18、pouring=36、lifttray=38);env_hint=None 回退 18 维 STATE18(旧默认)。
    """
    out = []
    with h5py.File(hdf5_path, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
        if num_demos is not None:
            eps = eps[:num_demos]
        keys = expected_low_dim_keys(env_hint) if env_hint is not None \
            else [k for k, _ in STATE18_KEYS]
        for ep in eps:
            grp = f[f"data/{ep}"]
            ro = {}
            for k in image_keys:
                name = k.replace("observation.images.", "")
                img = torch.as_tensor(grp[f"obs/{name}_image"][()])
                ro[k] = img.float().div(255.0).permute(0, 3, 1, 2)
            obs_arrays = {kk: grp[f"obs/{kk}"][()] for kk in keys}
            if env_hint is not None:
                state = assemble_state_by_env(obs_arrays, env_hint)
            else:
                state = assemble_state18(obs_arrays)
            ro[proprio_key] = torch.as_tensor(state, dtype=torch.float32)
            out.append(ro)
    return out
```

3c. act_feat build 调用点（`read_per_demo_states` 内 hdf5 分支）改为传 dataset_id：

```python
            raw_seqs = _build_raw_obs_seqs(hdf5_path, act_image_keys, act_proprio_key,
                                           num_demos, env_hint=dataset_id)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py -v`
Expected: 全 PASS（含原有 4 个测试，验证零回归）。

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py \
        resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py
git commit -m "feat(act_feat): hdf5 路线 proprio 走 env-aware(env_hint=dataset_id),支持 pouring/lifttray"
```

---

### Task 3: 主训 offline buffer hdf5 拼装走 env-aware（offline_stage_replay.py）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`（import + `_demo_base_actions` + `build_offline_buffer`）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_offline_stage_replay_env_hint.py`（新建）

- [ ] **Step 1: 写失败测试**（新建文件）

```python
"""offline_stage_replay 的 hdf5 proprio 拼装走 env-aware(env_hint)。

_demo_base_actions 喂 base ACT 的 observation.state 必须按 task 拼对维度
(pouring=36/lifttray=38),否则 KeyError 或喂错维度给 ACT。这里 mock base_policy
捕获它收到的 observation.state 维度,隔离验证拼装(不需真 ACT/env)。
"""
import h5py
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.offline_stage_replay import _demo_base_actions


class _RecBasePolicy:
    """记录每帧 select_action 收到的 observation.state 维度;返回固定动作。"""
    def __init__(self):
        self.state_dims = []

    def reset(self):
        pass

    def select_action(self, raw_obs):
        self.state_dims.append(int(raw_obs["observation.state"].shape[-1]))
        return torch.zeros(1, 7)


class _IdentityScaler:
    def scale(self, x):
        return torch.as_tensor(x, dtype=torch.float32)


def _make_pouring_grp(f, T):
    g = f.create_group("data/demo_0")
    g.create_dataset("obs/agentview_image", data=np.zeros((T, 84, 84, 3), np.uint8))
    fields = {"robot0_right_eef_pos": 3, "robot0_right_eef_quat": 4, "robot0_right_gripper_qpos": 11,
              "robot0_left_eef_pos": 3, "robot0_left_eef_quat": 4, "robot0_left_gripper_qpos": 11}
    for k, d in fields.items():
        g.create_dataset(f"obs/{k}", data=np.zeros((T, d), np.float32))
    return g


def test_demo_base_actions_pouring_state_dim(tmp_path):
    T = 4
    p = str(tmp_path / "pour.hdf5")
    with h5py.File(p, "w") as f:
        _make_pouring_grp(f, T)
    bp = _RecBasePolicy()
    with h5py.File(p, "r") as f:
        grp = f["data/demo_0"]
        out = _demo_base_actions(bp, grp, ["observation.images.agentview"],
                                 _IdentityScaler(), "cpu", env_hint="TwoArmPouring")
    assert out.shape == (T, 7)
    assert bp.state_dims == [36] * T        # 每帧喂 ACT 的 state 都是 36 维(humanoid)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_offline_stage_replay_env_hint.py -v`
Expected: FAIL —— `_demo_base_actions` 不接受 `env_hint`（TypeError），或对 pouring 走 assemble_state18 报 KeyError('robot0_eef_pos')。

- [ ] **Step 3: 实现**

3a. import 块（把 offline_hdf5_buffer import 改为加入两符号）：

```python
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    STATE18_KEYS,
    assemble_state18,
    assemble_state_by_env,
    expected_low_dim_keys,
    load_stage_cache,
    save_stage_cache,
    sorted_demo_keys,
```

（保持该 import 块其余符号不变。）

3b. `_demo_base_actions` 签名加 `env_hint`，proprio 拼装走 env-aware（env_hint=None 回退 STATE18）：

```python
def _demo_base_actions(base_policy, grp, image_keys, action_scaler, device, env_hint=None):
```

并把函数体内原拼装行：

```python
    state_raw = assemble_state18({k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS})  # (T,18) 原始
```

改为：

```python
    if env_hint is not None:
        keys = expected_low_dim_keys(env_hint)
        state_raw = assemble_state_by_env({k: grp[f"obs/{k}"][()] for k in keys}, env_hint)
    else:
        state_raw = assemble_state18({k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS})
```

3c. `build_offline_buffer` 签名加 `env_hint=None`（加在 `base_device="cpu"` 之后）：

```python
                         base_policy=None, base_mode="gt", base_device="cpu", env_hint=None) -> int:
```

3d. `build_offline_buffer` 内行 260 附近的 state_n 拼装：

```python
                obs_arrays = {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS}
                state_n = state_standardizer.standardize(
                    torch.as_tensor(assemble_state18(obs_arrays), dtype=torch.float32)).cpu()
```

改为：

```python
                if env_hint is not None:
                    _keys = expected_low_dim_keys(env_hint)
                    _state_raw = assemble_state_by_env(
                        {k: grp[f"obs/{k}"][()] for k in _keys}, env_hint)
                else:
                    _state_raw = assemble_state18(
                        {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS})
                state_n = state_standardizer.standardize(
                    torch.as_tensor(_state_raw, dtype=torch.float32)).cpu()
```

3e. `build_offline_buffer` 内对 `_demo_base_actions` 的调用（base_policy 分支）传 env_hint：

```python
                if base_mode == "base_policy":
                    base_n = _demo_base_actions(base_policy, grp, image_keys,
                                                action_scaler, base_device, env_hint=env_hint)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_offline_stage_replay_env_hint.py resfit/rl_finetuning/chunk_residual/tests/test_offline_stage_replay_smoke.py resfit/rl_finetuning/chunk_residual/tests/test_build_offline_buffer_rel.py resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -v`
Expected: 新测试 PASS + 原有 smoke/rel/base_policy 测试 PASS（零回归；这些用 robot0/robot1 fixture，env_hint=None 走原 assemble_state18）。

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/offline_stage_replay.py \
        resfit/rl_finetuning/chunk_residual/tests/test_offline_stage_replay_env_hint.py
git commit -m "feat(offline): 主训 offline buffer hdf5 拼装走 env-aware(env_hint),base_policy 喂 ACT state 维度按 task"
```

---

### Task 4: 主训把 env_hint=args.task 贯穿到 build_offline_buffer（train_chunk_residual.py）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:825-836`（build_offline_buffer 调用）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_env_hint.py`（新建，AST 静态检查，避免起重训练）

- [ ] **Step 1: 写失败测试**（新建文件，静态断言调用点传了 env_hint=args.task）

```python
"""静态守护:主训 build_offline_buffer 调用必须把 env_hint=args.task 传下去
(否则 pouring/lifttray hdf5 路 offline buffer 会按默认 18 维拼装 → KeyError)。
用 AST 检查而非起真训练(后者需 GPU/serve,过重)。"""
import ast
from pathlib import Path


def _find_call(tree, func_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == func_name:
            return node
    return None


def test_build_offline_buffer_call_passes_env_hint():
    src = Path("resfit/rl_finetuning/chunk_residual/train_chunk_residual.py").read_text()
    tree = ast.parse(src)
    call = _find_call(tree, "build_offline_buffer")
    assert call is not None, "找不到 build_offline_buffer 调用"
    kw = {k.arg: k.value for k in call.keywords}
    assert "env_hint" in kw, "build_offline_buffer 调用未传 env_hint"
    v = kw["env_hint"]
    # 期望 env_hint=args.task
    assert isinstance(v, ast.Attribute) and v.attr == "task" \
        and isinstance(v.value, ast.Name) and v.value.id == "args", \
        "env_hint 应为 args.task"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_env_hint.py -v`
Expected: FAIL —— "build_offline_buffer 调用未传 env_hint"。

- [ ] **Step 3: 实现**

在 `train_chunk_residual.py` 的 `build_offline_buffer(...)` 调用（行 825-836）参数末尾加 `env_hint=args.task,`：

```python
                build_offline_buffer(
                    offline_rb, args.offline_dataset_path,
                    action_scaler=action_scaler, state_standardizer=state_standardizer,
                    image_keys=image_keys, bonus=args.stage_reward_bonus,
                    mode=shaping_mode, gamma=args.gamma, num_demos=args.offline_num_demos,
                    stage_cache=args.offline_stage_cache, potential=potential,
                    subgoal=subgoal, way_steps=args.subgoal_way_steps,
                    act_feat_seqs=_offline_act_feat_seqs,
                    data_source=args.data_source, lerobot_repo_id=args.dataset,
                    lerobot_root=args.lerobot_root,
                    base_policy=base_policy, base_mode=args.offline_base_mode,
                    base_device=args.device, env_hint=args.task,)
```

- [ ] **Step 4: 运行测试确认通过 + 全量 import 不破**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_env_hint.py -v && conda run -n residual python -c "import resfit.rl_finetuning.chunk_residual.train_chunk_residual"`
Expected: PASS + import 无报错。

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
        resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_env_hint.py
git commit -m "feat(main): build_offline_buffer 传 env_hint=args.task,hdf5 路 offline buffer 按 task 拼 proprio"
```

---

### Task 5: 启动脚本 run_act_feat_hdf5_full_bp.sh（pouring|lifttray，hdf5 路线）

**Files:**
- Create: `run_act_feat_hdf5_full_bp.sh`

- [ ] **Step 1: 写脚本**

```bash
#!/usr/bin/env bash
# 全集正式实验:act_feat × hdf5 路线 × base_policy,超参对齐 three_piece_actfeat_bp_bc01。
# 与现有 run_act_feat_lerobot_full_bp.sh 的区别:离线 cache/主训全走 hdf5(--hdf5/--offline_dataset_path),
# 去 --data_source lerobot;真同源在线 env + ~100x 提速(省 pyav 解码)。
# 用法: CUDA_VISIBLE_DEVICES=<卡> bash run_act_feat_hdf5_full_bp.sh <pouring|lifttray>
set -uo pipefail
KEY="${1:?用法: CUDA_VISIBLE_DEVICES=N bash $0 pouring|lifttray}"
case "$KEY" in
  pouring)  TASK=TwoArmPouring;  SUB=dexmg-two-arm-pouring;   HDF5=resfit/dataset/two_arm_pouring.hdf5;   BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2 ;;
  lifttray) TASK=TwoArmLiftTray; SUB=dexmg-two-arm-lift-tray; HDF5=resfit/dataset/two_arm_lift_tray.hdf5; BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 ;;
  *) echo "未知 task: $KEY (只支持 pouring|lifttray)"; exit 2 ;;
esac

export PYTHONPATH=/mnt/mnt/data/resfit
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /mnt/mnt/data/resfit

DS=ankile/$SUB
CACHE=outputs_chunk/${KEY}_act_feat_hdf5.npz
GC=outputs_chunk/${KEY}_gc_value_actfeat_hdf5.pt
HA=outputs_chunk/${KEY}_high_actor_actfeat_hdf5.pt
OFFCACHE=outputs_chunk/${KEY}_actfeat_hdf5_bp_offcache
OUT=outputs_chunk/${KEY}_actfeat_hdf5_bp_bc01
RUN="conda run -n residual --no-capture-output python -u"
gate(){ [ -s "$1" ] || { echo "[driver] FATAL 产物缺失: $1 (上一步失败,停)"; exit 4; }; }

echo "[driver] START $KEY ($TASK) hdf5 路 GPU=${CUDA_VISIBLE_DEVICES:-?} $(date '+%F %T')"

echo "[driver] 1) gc_value 全集(hdf5,首跑建 act_feat 缓存,~2min)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$HDF5" --state_mode act_feat \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --output "$GC"
gate "$CACHE"; gate "$GC"; echo "[driver] 1) gc_value DONE $(date '+%T')"

echo "[driver] 2) high_actor(命中缓存)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$HDF5" --state_mode act_feat \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GC" --output "$HA"
gate "$HA"; echo "[driver] 2) high_actor DONE $(date '+%T')"

echo "[driver] 3) 主训 500k(hdf5,base_policy,对齐 three_piece 超参,no-stage)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name "${KEY}_actfeat_hdf5_bp_bc01" \
  --output_dir "$OUT"
echo "[driver] ALL DONE $KEY $(date '+%F %T')  out=$OUT"
```

- [ ] **Step 2: 语法检查**

Run: `cd /mnt/mnt/data/resfit && bash -n run_act_feat_hdf5_full_bp.sh && echo OK`
Expected: `OK`（无语法错；不实际执行）。

- [ ] **Step 3: 提交**

```bash
cd /mnt/mnt/data/resfit
git add run_act_feat_hdf5_full_bp.sh
git commit -m "feat(script): pouring/lifttray act_feat hdf5 路线全集启动脚本(对齐 three_piece 超参)"
```

---

### Task 6: 把 online-consistency hard gate 推广到 pouring/lifttray（verify_act_feat_online_consistency.py）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/verify_act_feat_online_consistency.py`

目的：把 three_piece 专用的"离线 hdf5 act_feat ≈ 在线 env act_feat"验证脚本推广到任意 dexmg task —— proprio 拼装走 `assemble_state_by_env(_, task)`，robots/camera 从 HDF5 `env_args` 读（不硬编码 robots_map）。

注：`ActFeatureExtractor` 的 `proprio_dim` 仅用于 `feature_dim` 属性(reporting/校验)，不影响
`embed_batch` 实际拼装(用运行时 tensor 维度)，保持默认 18 不影响数值正确性，本 task 无需改它。

- [ ] **Step 1: 离线 proprio 拼装走 env-aware**

把脚本中所有用 `STATE18_KEYS` / `assemble_state18` 拼 offline proprio 的地方，换成按 `--task` 走 `assemble_state_by_env`。具体：

离线读取（`assemble_state18(obs_arrays)` 处）改为：

```python
        keys = expected_low_dim_keys(args.task)
        obs_arrays = {k: grp[f"obs/{k}"][()] for k in keys}
        state_raw = assemble_state_by_env(obs_arrays, args.task)   # (T,Dp) raw
```

（变量名沿用脚本原有 `state18_raw` → 改为 `state_raw`，并同步其后引用。）

- [ ] **Step 2: 在线 proprio 拼装走 env-aware**

在线 env replay 处，把按 `STATE18_KEYS` 逐 key 取 `raw_rs_obs` 拼 18 维的循环，改为按 `expected_low_dim_keys(args.task)` 取并 `np.concatenate`（完整维度，对齐在线 `_process_obs`）：

```python
        on_keys = expected_low_dim_keys(args.task)
        on_state_parts = []
        for key_name in on_keys:
            if key_name in raw_rs_obs:
                on_state_parts.append(np.asarray(raw_rs_obs[key_name]).astype(np.float32))
            else:
                print(f"  WARNING: proprio key {key_name} not in env obs")
                exact_state_path = False
                break
        if not exact_state_path:
            break
        on_state_raw = np.concatenate(on_state_parts)
```

- [ ] **Step 3: robots/camera 从 HDF5 env_args 读（不硬编码 robots_map）**

把 `robots_map.get(args.task, ["Panda","Panda"])` 改为优先从 HDF5 `env_args` 读 robots：

```python
    with h5py.File(hdf5_path, "r") as f:
        _env_args = f["data"].attrs.get("env_args", None)
    robots = ["Panda", "Panda"]
    if _env_args is not None:
        _ek = json.loads(_env_args).get("env_kwargs", {})
        if _ek.get("robots"):
            robots = _ek["robots"]
            if isinstance(robots, str):
                robots = [robots]
    print(f"  robots={robots}")
```

并把 `robosuite.make(..., robots=robots, ...)` 的 controller_configs 用 `load_composite_controller_config(robot=robots[0])`（脚本已是此写法，确认即可）。

- [ ] **Step 4: 烟雾运行（pouring，少量帧）**

Run（GPU 卡号按空闲选；MUJOCO_EGL_DEVICE_ID 用该卡物理号）：
```bash
cd /mnt/mnt/data/resfit
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=2 HF_HUB_OFFLINE=1 \
conda run -n residual --no-capture-output python -u -m \
  resfit.rl_finetuning.chunk_residual.verify_act_feat_online_consistency \
  --base /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2 \
  --hdf5 resfit/dataset/two_arm_pouring.hdf5 \
  --dataset ankile/dexmg-two-arm-pouring \
  --task TwoArmPouring --frames 0 10
```
Expected: 打印 `proprio raw allclose ... max_diff` ≈ 0（离线 hdf5 与在线 env 同帧 proprio 对齐）；`max_abs_diff` ≤ 0.05（frame 0 接近 0，frame 10 ~0.03 EGL 噪声）；末行 `BOTTOM LINE: PASS`。

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/verify_act_feat_online_consistency.py
git commit -m "feat(gate): online-consistency 验证推广到任意 dexmg task(env-aware proprio + 从 env_args 读 robots)"
```

---

### Task 7: 全量回归 + hdf5 smoke + 双 task gate（启动全集前）

**Files:** 无（验证 task）。

- [ ] **Step 1: 全量 pytest 回归**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全 PASS（与改动前基线条数一致，新增测试增量；零回归）。记录通过条数。

- [ ] **Step 2: pouring hdf5 小规模 smoke（gc/high/主训 40 步）**

新建 `run_act_feat_hdf5_pouring_smoke.sh`（仿 Task 5 脚本，但 `--num_demos 4 --steps 80`、主训 `--offline_num_demos 4 --total_env_steps 40 --learning_starts 10 --eval_num_envs 2 --eval_num_episodes 2 --wandb_mode disabled --output_dir /tmp/pouring_hdf5_smoke`，gc/high 也带 `--num_demos 4`），跑：
```bash
cd /mnt/mnt/data/resfit && CUDA_VISIBLE_DEVICES=2 bash run_act_feat_hdf5_pouring_smoke.sh
```
Expected: 三阶段串行跑通，末行 `[smoke] ... DONE`；无 KeyError/维度错。

- [ ] **Step 3: lifttray online-consistency gate**

Run（同 Task6 Step5，换 lifttray）：
```bash
cd /mnt/mnt/data/resfit
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=2 HF_HUB_OFFLINE=1 \
conda run -n residual --no-capture-output python -u -m \
  resfit.rl_finetuning.chunk_residual.verify_act_feat_online_consistency \
  --base /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 \
  --hdf5 resfit/dataset/two_arm_lift_tray.hdf5 \
  --dataset ankile/dexmg-two-arm-lift-tray \
  --task TwoArmLiftTray --frames 0 10
```
Expected: `BOTTOM LINE: PASS`。

- [ ] **Step 4: 汇总并请用户确认启动全集**

汇总 pytest 条数、pouring/lifttray online-consistency PASS、smoke DONE，向用户报告并请确认 GPU 分配后再 `bash run_act_feat_hdf5_full_bp.sh pouring|lifttray`（500k，资源密集，启动前必须用户确认）。

---

## Self-Review

**Spec coverage：** spec §3.1→Task1；§3.2(train_hiql_value)→Task2、(offline_stage_replay)→Task3、(train_chunk_residual)→Task4；§3.3 向后兼容→Task1/2/3 的 None 回退测试 + two-arm 逐位等价测试；§4 测试→Task1-4 各测试；§5 脚本→Task5；§6 gate→Task6+Task7；§7 非目标(不动 lerobot/eef/libero)→各 task 仅改 act_feat hdf5 分支、env_hint=None 回退守护。全覆盖。

**Placeholder scan：** 无占位。所有步骤含完整 test/impl/命令代码；Task6 的 proprio_dim 在概述说明保持默认 18（仅影响 feature_dim 属性、不影响数值），无需改动步骤。

**Type consistency：** `expected_low_dim_keys`/`assemble_state_by_env` 签名在 Task1 定义，Task2/3/6 调用一致；`env_hint` 参数名在 Task2(`_build_raw_obs_seqs`)、Task3(`_demo_base_actions`/`build_offline_buffer`)、Task4(调用 `env_hint=args.task`)一致；`LOW_DIM_KEYS_MULTI/HUMANOID/SINGLE` 命名在 Task1 定义与测试引用一致。
