# pi0_feat 离线 waypoint 子目标 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 LIBERO+pi0_feat 残差 RL 补离线 waypoint 子目标，使 `--subgoal_conditioned + --offline_fraction>0 + --demo_bc_coef`(BC 锚) 能跑、与参照 run 对齐。

**Architecture:** 镜像 dexmg `build_offline_buffer` 已有的 act_feat 离线子目标模式：放开 `subgoal_waypoint` 对 pi0_feat 的禁用 → `build_libero_offline_buffer` 逐 demo 用 pi0_feat 缓存序列算 `z=vf.phi(s_t, s_{t+way})` → 逐 transition 存 `observation.subgoal`，与在线 `compute_online_subgoal` 同源。

**Tech Stack:** Python/PyTorch/torchrl/tensordict；测试 pytest。所有命令从 `/mnt/mnt/data/resfit` 用 `/mnt/mnt/data/envs/residual/bin/python` 跑。

**Spec:** `docs/superpowers/specs/2026-06-15-pi0-feat-offline-waypoint-subgoal-design.md`

## 文件结构

| 文件 | 职责 | 改动 |
|---|---|---|
| `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py` | 子目标计算 | `subgoal_waypoint` 删 pi0_feat assert + 更新 docstring |
| `resfit/rl_finetuning/chunk_residual/libero_offline.py` | libero 离线 buffer 构建 | `_demo_to_transitions` 加 `subgoal_z`；`build_libero_offline_buffer` 加 `subgoal/way_steps/feat_seqs` |
| `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` | 残差训练主流程 | `_pi0_seqs=None` 初始化 + libero 离线调用传 3 参 + `_libero_offline_signature` 纳入 subgoal 字段 |
| `tests/test_hiql_subgoal_pi0_feat.py` | 改写既有 reject 测试 | Task 1 |
| `tests/test_libero_offline.py` | 加 subgoal 测试 | Task 2/3 |
| `tests/test_train_chunk_residual_pi0_feat_subgoal.py` | 加签名测试 | Task 4 |

---

### Task 1: subgoal_waypoint 放开 pi0_feat

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py:130-141`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py`

**背景:** 现有测试 `test_subgoal_waypoint_rejects_pi0_feat` 断言 pi0_feat 抛 AssertionError。本任务删除该 assert，故须**改写**这个测试为"返回 z"。`subgoal_waypoint` 方法体本就通用(`vf.phi(b,t)` + `vf.state_dim` 维度校验)。

- [ ] **Step 1: 在测试文件加 `_StubVF` 并改写测试**

把 `tests/test_hiql_subgoal_pi0_feat.py` 末尾的 `test_subgoal_waypoint_rejects_pi0_feat` 整个函数替换为下面的 `_StubVF` + 新测试：

```python
class _StubVF(torch.nn.Module):
    rep_dim = 10
    state_dim = 2056

    def phi(self, b, t):
        # 返回 (B, rep_dim);用 base-target 前 rep_dim 维做可辨识输出便于断言
        return (b[..., :self.rep_dim] - t[..., :self.rep_dim])


def test_subgoal_waypoint_pi0_feat_returns_z():
    sg = HiqlSubgoal(gc_value=_StubVF(), high_actor=_StubHA(),
                     goal=np.zeros(2056, np.float32), device="cpu", state_mode="pi0_feat",
                     feat_stats=(np.zeros(2056, np.float32), np.ones(2056, np.float32)))
    z = sg.subgoal_waypoint(np.ones((4, 2056), np.float32), np.zeros((4, 2056), np.float32))
    assert z.shape == (4, 10) and torch.isfinite(z).all()
    # base=1,target=0 → phi=1-0=1
    assert torch.allclose(z, torch.ones(4, 10))


def test_subgoal_waypoint_pi0_feat_wrong_dim_raises():
    sg = HiqlSubgoal(gc_value=_StubVF(), high_actor=_StubHA(),
                     goal=np.zeros(2056, np.float32), device="cpu", state_mode="pi0_feat",
                     feat_stats=(np.zeros(2056, np.float32), np.ones(2056, np.float32)))
    with pytest.raises(AssertionError):
        sg.subgoal_waypoint(np.ones((4, 99), np.float32), np.zeros((4, 99), np.float32))
```

- [ ] **Step 2: 跑测试验证它失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py::test_subgoal_waypoint_pi0_feat_returns_z -v`
Expected: FAIL with `AssertionError: pi0_feat 暂不支持离线 waypoint...`(旧 assert 仍在)

- [ ] **Step 3: 删除 subgoal_waypoint 的 pi0_feat assert + 更新 docstring**

在 `hiql_subgoal.py` 的 `subgoal_waypoint`，删除这一行：

```python
        assert self.state_mode != "pi0_feat", "pi0_feat 暂不支持离线 waypoint(在线走 subgoal_online + prefix_feat)"
```

并把 docstring 改为(注明 pi0_feat 入 2056 维)：

```python
    def subgoal_waypoint(self, s30_base, s30_target):
        """离线:z = φ(base=s_t, target=s_{t+k})(真航点)。输入须是已拼好的、与 vf.state_dim
        同维的**已标准化** state:eef_piece=30(18 std + 12 标准化 rel)、act_feat=530、
        pi0_feat=2056(标准化 prefix_feat ⊕ proprio,来自 pi0_feat 缓存序列)。返回 [B, rep_dim]。"""
```

- [ ] **Step 4: 跑测试验证通过(含同文件其它 pi0_feat 测试不回归)**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py -v`
Expected: PASS(含 `test_subgoal_waypoint_pi0_feat_returns_z`、`..._wrong_dim_raises`，及原有 online 测试)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_subgoal.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py
git commit -m "feat(hiql_subgoal): subgoal_waypoint 放开 pi0_feat(2056维真航点φ)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: _demo_to_transitions 存 observation.subgoal

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_offline.py`(`_demo_to_transitions`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`

**背景:** `_demo_to_transitions` 用内部 `_obs(t)` 闭包同时造 curr 和 next.obs。加 `subgoal_z` 参数，`_obs` 在非 None 时加 `observation.subgoal` 键即覆盖两侧。复用该文件既有夹具 `_toy_demo / _IdScaler / _IdStd`。

- [ ] **Step 1: 写失败测试(加到 test_libero_offline.py 末尾)**

```python
def test_demo_to_transitions_stores_subgoal():
    import torch
    from resfit.rl_finetuning.chunk_residual.libero_offline import _demo_to_transitions
    demo = _toy_demo(T=3)
    sgz = torch.arange(3 * 10, dtype=torch.float32).reshape(3, 10)   # (T, rep_dim)
    tds = _demo_to_transitions(demo, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
                               base_actions=None, image_size=84, subgoal_z=sgz)
    assert len(tds) == 2
    assert torch.allclose(tds[0]["obs"]["observation.subgoal"], sgz[0])
    assert torch.allclose(tds[0]["next"]["obs"]["observation.subgoal"], sgz[1])
    assert torch.allclose(tds[1]["next"]["obs"]["observation.subgoal"], sgz[2])


def test_demo_to_transitions_no_subgoal_omits_key():
    from resfit.rl_finetuning.chunk_residual.libero_offline import _demo_to_transitions
    demo = _toy_demo(T=3)
    tds = _demo_to_transitions(demo, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
                               base_actions=None, image_size=84)   # subgoal_z 默认 None
    assert "observation.subgoal" not in tds[0]["obs"].keys()
    assert "observation.subgoal" not in tds[0]["next"]["obs"].keys()
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_demo_to_transitions_stores_subgoal -v`
Expected: FAIL with `TypeError: _demo_to_transitions() got an unexpected keyword argument 'subgoal_z'`

- [ ] **Step 3: 实现 — 给 `_demo_to_transitions` 加 subgoal_z 参数 + `_obs` 加键**

在 `libero_offline.py` 把 `_demo_to_transitions` 的签名与 `_obs` 改为：

```python
def _demo_to_transitions(demo, *, action_scaler, state_standardizer, base_actions,
                         image_size, subgoal_z=None):
```

并把内部 `_obs(t)`(原本一行 return dict)改为：

```python
    def _obs(t):
        d = {"observation.state": state_std[t], "observation.base_action": base[t],
             "observation.stage_id": torch.zeros(1, dtype=torch.float32),
             AGENTVIEW_KEY: img_av[t], WRIST_KEY: img_wr[t]}
        if subgoal_z is not None:
            d["observation.subgoal"] = subgoal_z[t]
        return d
```

(其余循环体不变;docstring 可补一句"subgoal_z 非 None 时每帧 obs 含 observation.subgoal=subgoal_z[t]"。)

- [ ] **Step 4: 跑测试验证通过(含既有 schema 测试不回归)**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py -k "demo_to_transitions" -v`
Expected: PASS(新 2 个 + 既有 `test_demo_to_transitions_schema_reward_noflip`、`..._single_frame_returns_empty`)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py
git commit -m "feat(libero_offline): _demo_to_transitions 可选存 observation.subgoal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: build_libero_offline_buffer 接 subgoal/way_steps/feat_seqs

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_offline.py`(`build_libero_offline_buffer`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`

**背景:** 逐 demo 用 pi0_feat 缓存序列 `feat_seqs[i]` 算 `z=subgoal.subgoal_waypoint(seq, seq[way])`(镜像 act_feat 模板)，传给 Task 2 的 `_demo_to_transitions`。复用既有 `test_build_libero_offline_buffer_gt` 的 monkeypatch 模式。

- [ ] **Step 1: 写失败测试(加到 test_libero_offline.py 末尾)**

```python
class _StubSubgoal:
    """假 subgoal:subgoal_waypoint 返回 base 前 10 维(便于断言对齐/维度)。"""
    def subgoal_waypoint(self, base, target):
        import torch
        b = torch.as_tensor(np.asarray(base), dtype=torch.float32)
        return b[:, :10]


def test_build_libero_offline_buffer_stores_subgoal(monkeypatch):
    import torch
    from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
    import resfit.rl_finetuning.chunk_residual.libero_offline as lo
    demos = [_toy_demo(T=3), _toy_demo(T=4)]                 # 2 + 3 = 5 transition
    monkeypatch.setattr(lo, "find_demo_episodes", lambda root, lang: ["p0", "p1"])
    monkeypatch.setattr(lo, "read_libero_demo", lambda p: demos[["p0", "p1"].index(p)])
    monkeypatch.setattr(lo, "libero_task_language", lambda s, t: "L")
    feat_seqs = [np.ones((3, 2056), np.float32), np.full((4, 2056), 2.0, np.float32)]
    rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=5, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority", batch_size=2)
    lo.build_libero_offline_buffer(rb, lerobot_root="root", suite="libero_10", task_id=8,
                                   action_scaler=_IdScaler(), state_standardizer=_IdStd(),
                                   base_policy=None, base_mode="gt", base_device="cpu", image_size=84,
                                   subgoal=_StubSubgoal(), way_steps=2, feat_seqs=feat_seqs)
    assert len(rb) == 5
    b = rb.sample(2)
    assert "observation.subgoal" in b["obs"].keys()
    assert b["obs"]["observation.subgoal"].shape[-1] == 10


def test_build_libero_offline_buffer_subgoal_needs_feat_seqs(monkeypatch):
    import pytest
    from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
    import resfit.rl_finetuning.chunk_residual.libero_offline as lo
    monkeypatch.setattr(lo, "find_demo_episodes", lambda root, lang: ["p0"])
    monkeypatch.setattr(lo, "read_libero_demo", lambda p: _toy_demo(T=3))
    monkeypatch.setattr(lo, "libero_task_language", lambda s, t: "L")
    rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=2, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority", batch_size=2)
    with pytest.raises(ValueError):
        lo.build_libero_offline_buffer(rb, lerobot_root="root", suite="libero_10", task_id=8,
            action_scaler=_IdScaler(), state_standardizer=_IdStd(), base_policy=None,
            base_mode="gt", base_device="cpu", image_size=84, subgoal=_StubSubgoal(), feat_seqs=None)


def test_build_libero_offline_buffer_subgoal_frame_mismatch(monkeypatch):
    import pytest
    from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
    import resfit.rl_finetuning.chunk_residual.libero_offline as lo
    monkeypatch.setattr(lo, "find_demo_episodes", lambda root, lang: ["p0"])
    monkeypatch.setattr(lo, "read_libero_demo", lambda p: _toy_demo(T=3))
    monkeypatch.setattr(lo, "libero_task_language", lambda s, t: "L")
    rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=2, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority", batch_size=2)
    with pytest.raises(AssertionError):
        lo.build_libero_offline_buffer(rb, lerobot_root="root", suite="libero_10", task_id=8,
            action_scaler=_IdScaler(), state_standardizer=_IdStd(), base_policy=None,
            base_mode="gt", base_device="cpu", image_size=84,
            subgoal=_StubSubgoal(), way_steps=2, feat_seqs=[np.zeros((99, 2056), np.float32)])
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_build_libero_offline_buffer_stores_subgoal -v`
Expected: FAIL with `TypeError: build_libero_offline_buffer() got an unexpected keyword argument 'subgoal'`

- [ ] **Step 3: 实现 — 给 build_libero_offline_buffer 加 3 参 + 逐 demo 算 z**

把 `build_libero_offline_buffer` 改为(签名加 3 参、循环改 enumerate、算 subgoal_z 并传入)：

```python
def build_libero_offline_buffer(offline_rb, *, lerobot_root, suite, task_id,
                                action_scaler, state_standardizer,
                                base_policy, base_mode, base_device="cpu",
                                image_size=84, num_demos=None,
                                subgoal=None, way_steps=25, feat_seqs=None) -> None:
    """灌当前任务的 demo 进 offline_rb。base_mode∈{gt,base_policy}。
    subgoal!=None 时逐 demo 用 feat_seqs[i](pi0_feat 缓存标准化序列)算真航点 z 存 observation.subgoal。"""
    if base_mode not in ("gt", "base_policy"):
        raise ValueError(f"base_mode 必须是 gt|base_policy,得到 {base_mode!r}")
    if base_mode == "base_policy" and base_policy is None:
        raise ValueError("base_mode=base_policy 需传 base_policy")
    if subgoal is not None and feat_seqs is None:
        raise ValueError("subgoal 离线子目标需 feat_seqs(pi0_feat 缓存 per-demo 标准化序列)")
    import torch
    language = libero_task_language(suite, task_id)
    paths = find_demo_episodes(lerobot_root, language)
    if num_demos is not None:
        paths = paths[:num_demos]
    for i, p in enumerate(paths):
        demo = read_libero_demo(p)
        T = demo["state"].shape[0]
        if T < 2:
            print(f"[libero-offline] 跳过 {p}(T<2)")
            continue
        base_actions = (_libero_demo_base_actions(demo, base_policy, action_scaler, image_size, base_device)
                        if base_mode == "base_policy" else None)
        subgoal_z = None
        if subgoal is not None:
            seq = torch.as_tensor(feat_seqs[i], dtype=torch.float32)        # (T, 2056) 已标准化
            assert seq.shape[0] == T, \
                f"pi0_feat 缓存帧数 {seq.shape[0]} != demo {T} (ep#{i}, {p})"
            way = np.minimum(np.arange(T) + way_steps, T - 1)
            subgoal_z = subgoal.subgoal_waypoint(seq, seq[way]).cpu()       # (T, rep_dim)
        for td in _demo_to_transitions(demo, action_scaler=action_scaler,
                                       state_standardizer=state_standardizer,
                                       base_actions=base_actions, image_size=image_size,
                                       subgoal_z=subgoal_z):
            offline_rb.add(td.unsqueeze(0))
```

(确认文件顶部已 `import numpy as np`;若无则补。)

- [ ] **Step 4: 跑测试验证通过(含既有 gt 测试不回归)**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py -v`
Expected: PASS(新 3 个 + 既有 `test_build_libero_offline_buffer_gt` 等全绿)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py
git commit -m "feat(libero_offline): build_libero_offline_buffer 接 subgoal 算离线 z

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: train_chunk_residual 传参 + _pi0_seqs 初始化 + 缓存签名

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(`_libero_offline_signature` + main() 两处)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_pi0_feat_subgoal.py`

**背景:** 三处改动。其中 `_libero_offline_signature` 可单元测；main() 内 `_pi0_seqs=None` 初始化 + build 调用传参无法独立单元测，由"全量测试不回归 + 实现后集成 smoke"覆盖。

- [ ] **Step 1: 写失败测试(签名纳入 subgoal,加到 test_train_chunk_residual_pi0_feat_subgoal.py 末尾)**

```python
def test_libero_offline_signature_includes_subgoal_when_conditioned():
    from types import SimpleNamespace
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _libero_offline_signature
    args = SimpleNamespace(
        libero_stats_json="/x/meta/stats.json", libero_suite="libero_10", libero_task_id=8,
        offline_base_mode="base_policy", base_policy_type="pi05", pi0_host="127.0.0.1",
        pi0_port=8000, pi0_action_dim=7, pi0_execute_horizon=10, action_scale=0.05,
        min_range_per_dim=0.1, gamma=0.99, n_step=3, offline_num_demos=None,
        subgoal_conditioned=True, gc_value_ckpt="gc.pt", high_actor_ckpt="ha.pt", subgoal_way_steps=25)
    sig = _libero_offline_signature(args, ["observation.images.agentview"], 100, 84)
    assert sig["subgoal"] is True
    assert sig["subgoal_way_steps"] == 25
    assert sig["gc_value_ckpt"].endswith("gc.pt") and sig["high_actor_ckpt"].endswith("ha.pt")


def test_libero_offline_signature_omits_subgoal_when_off():
    from types import SimpleNamespace
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _libero_offline_signature
    args = SimpleNamespace(
        libero_stats_json="/x/meta/stats.json", libero_suite="libero_10", libero_task_id=8,
        offline_base_mode="base_policy", base_policy_type="pi05", pi0_host="127.0.0.1",
        pi0_port=8000, pi0_action_dim=7, pi0_execute_horizon=10, action_scale=0.05,
        min_range_per_dim=0.1, gamma=0.99, n_step=3, offline_num_demos=None,
        subgoal_conditioned=False)
    sig = _libero_offline_signature(args, ["observation.images.agentview"], 100, 84)
    assert "subgoal" not in sig
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_pi0_feat_subgoal.py::test_libero_offline_signature_includes_subgoal_when_conditioned -v`
Expected: FAIL(`KeyError: 'subgoal'` —— 签名尚未含该字段)

- [ ] **Step 3a: 实现 — `_libero_offline_signature` 纳入 subgoal 字段**

把 `_libero_offline_signature` 的 body 改为(原 dict 赋给 `sig`,条件并入 subgoal 字段后返回)：

```python
def _libero_offline_signature(args, image_keys, offline_cap, image_size):
    """libero offline buffer 缓存签名(换数据源/base/缩放/图尺寸/subgoal即失效重建)。"""
    sig = {
        "env_family": "libero",
        "lerobot_root": os.path.abspath(os.path.join(os.path.dirname(args.libero_stats_json), "..")),
        "suite": args.libero_suite, "task_id": int(args.libero_task_id),
        "base_mode": args.offline_base_mode, "base_policy_type": args.base_policy_type,
        "pi0_host": args.pi0_host, "pi0_port": args.pi0_port, "pi0_action_dim": args.pi0_action_dim,
        "pi0_execute_horizon": args.pi0_execute_horizon,
        "action_scale": args.action_scale, "min_range_per_dim": args.min_range_per_dim,
        "offline_cap": offline_cap, "image_keys": sorted(image_keys), "image_size": int(image_size),
        "gamma": args.gamma, "n_step": args.n_step, "num_demos": args.offline_num_demos,
    }
    if getattr(args, "subgoal_conditioned", False):
        sig["subgoal"] = True
        sig["gc_value_ckpt"] = os.path.abspath(args.gc_value_ckpt)
        sig["high_actor_ckpt"] = os.path.abspath(args.high_actor_ckpt)
        sig["subgoal_way_steps"] = int(args.subgoal_way_steps)
    return sig
```

- [ ] **Step 3b: 实现 — main() 内 `_pi0_seqs=None` 初始化**

在 `train_chunk_residual.py` 找到 `_offline_act_feat_seqs = None`(约 line 712,在 `subgoal = None` 之后)，紧邻其下加一行：

```python
    _pi0_seqs = None                # pi0_feat:离线 buffer subgoal 复用的 2056 缓存序列
```

(pi0_feat 分支已有 `_pi0_seqs, _pi0_stats, _pi0_cache_sig = load_pi0_feat_cache(...)` 赋值,此处仅补默认值使非 pi0_feat 分支也有定义。)

- [ ] **Step 3c: 实现 — libero 离线构建调用传 3 参**

找到 `build_libero_offline_buffer(` 调用(约 line 850),把它改为(末尾加 3 个 kwarg)：

```python
                build_libero_offline_buffer(
                    offline_rb, lerobot_root=lerobot_root,
                    suite=args.libero_suite, task_id=args.libero_task_id,
                    action_scaler=action_scaler, state_standardizer=state_standardizer,
                    base_policy=base_policy, base_mode=args.offline_base_mode,
                    base_device=args.device, image_size=img_h, num_demos=args.offline_num_demos,
                    subgoal=subgoal, way_steps=args.subgoal_way_steps, feat_seqs=_pi0_seqs)
```

- [ ] **Step 4: 跑签名测试 + 全量回归**

Run(签名测试):
`cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_pi0_feat_subgoal.py -v`
Expected: PASS(两个签名测试)

Run(全量回归,确认无破坏):
`cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全绿(或仅既有与本改动无关的 skip);特别确认 test_libero_offline / test_hiql_subgoal_pi0_feat / test_train_chunk_residual_pi0_feat_subgoal 全 PASS

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_pi0_feat_subgoal.py
git commit -m "feat(train_chunk_residual): libero 离线 subgoal 传参 + 缓存签名纳入 subgoal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 实现后:集成 smoke(人工,不入 pytest)

全部 4 任务通过后，跑 `--smoke` rollout 验端到端(serve 已在 GPU2:8000、50k ckpts 已就绪)。从 `/mnt/mnt/data/resfit`：

```bash
CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=5 \
/mnt/mnt/data/envs/residual/bin/python -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --env_family libero --libero_suite libero_10 --libero_task_id 8 \
  --libero_stats_json /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json \
  --actor raw --chunk_length 1 --action_scale 0.05 \
  --offline_base_mode base_policy --demo_bc_coef 0.1 --offline_fraction 0.5 \
  --offline_buffer_cache outputs_chunk/libero_task8_pi0feat_offcache \
  --base_action_mode queue --base_policy_type pi05 \
  --pi0_host 127.0.0.1 --pi0_port 8000 --pi0_action_dim 7 --pi0_execute_horizon 10 \
  --pi0_prompt "put the black bowl in the bottom drawer of the cabinet and close it" \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/libero_task8_pi0_feat_gc_value.pt \
  --high_actor_ckpt outputs_chunk/libero_task8_pi0_feat_high_actor.pt \
  --pi0_feat_cache outputs_chunk/libero_task8_pi0_feat.npz \
  --subgoal_way_steps 25 --renorm_subgoal --device cuda --seed 0 \
  --output_dir outputs_chunk/libero_task8_pi0feat_residual \
  --wandb_mode disabled --smoke
```

Expected: 离线 buffer 带 observation.subgoal 构建并落盘 `outputs_chunk/libero_task8_pi0feat_offcache`(秒级复用)、2 步在线注入不报错。成功即解锁正式 rollout(去掉 `--smoke`、加 `--total_env_steps 500000` 等参,复用 offcache)。

---

## 自审清单(写完计划后)

**1. Spec 覆盖:** 改动 1(Task 1)、改动 2a(Task 2)、改动 2b(Task 3)、改动 3a/3b/3c(Task 4)、7 条同源不变量(Task 3 的帧数 assert + enumerate 同序 + Task 1 标准化态直喂 φ)、错误处理(Task 3 的 ValueError/AssertionError)、测试(Task 1-4 单元 + 集成 smoke)。全覆盖。

**2. 占位扫描:** 无 TBD/TODO;每步含完整测试与实现代码。

**3. 类型一致:** `subgoal_z`(Task 2)↔`build_libero_offline_buffer` 算的 `subgoal_z`(Task 3)同名同形 (T, rep_dim);`feat_seqs`(Task 3 参数)↔`_pi0_seqs`(Task 4 传入)同为 list[(T,2056)];`subgoal.subgoal_waypoint`(Task 3 调用)↔Task 1 放开的方法签名一致。
