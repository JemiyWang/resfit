# LiberoPi05Adapter per-env prefix_feat 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `LiberoPi05Adapter.last_prefix_feat()` 在向量化场景返回 per-env 堆叠 `[b,2048]`,使默认 `--eval_num_envs 8` 的 pi0_feat eval 不再撞 batch 不匹配;单 env / 直调 `_infer_chunk` 行为向后兼容。

**Architecture:** `select_action` 逐 env 重 infer 时把各 env 的 prefix_feat 记进 `self._prefix_feat_per_env[i]`(与 `_queues` 等长);`last_prefix_feat()` 全部就位时返回 `np.stack(...)=[b,2048]`,未就位(直调 `_infer_chunk` / 无 select_action / serve 不透特征含 None)回退单帧 `_last_prefix_feat`。`_infer_chunk` 仍照旧 set 单帧 `_last_prefix_feat`(直调测试不变)。

**Tech Stack:** Python/numpy;测试 pytest,用 `/mnt/mnt/data/envs/residual/bin/python`(纯 numpy/torch,不需 libero/serve)。

**Spec:** `docs/superpowers/specs/2026-06-16-libero-adapter-per-env-prefix-feat-design.md`

## 文件结构

| 文件 | 职责 | 改动 |
|---|---|---|
| `resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py` | pi0 base 适配器 | `__init__` 加 per-env 列表;`_ensure_queues` 同步增长;`select_action` 存 per-env;`last_prefix_feat` 新逻辑 |
| `resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py` | 适配器单测 | 加 1 个 stub + 3 个测试 |

---

### Task 1: per-env prefix_feat(TDD)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py`

**背景:** 现 `select_action`(libero_pi05_adapter.py:60)逐 env 循环重 infer,`_infer_chunk` 覆盖同一个 `self._last_prefix_feat`,跑完 b 个 env 只剩最后一个 → `last_prefix_feat()` batch=1,向量化 eval 注入时撞 proprio batch=b。本任务加 per-env 跟踪。复用测试文件已有 `_raw_obs(B)` 夹具。

- [ ] **Step 1: 写失败测试(加到 `test_libero_pi05_adapter.py` 末尾)**

```python
class _FeatPolicy:
    """每次 infer 返回可辨识 prefix_feat=full(2048, call_idx),验证 per-env 堆叠顺序/沿用。"""
    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        idx = self.calls
        self.calls += 1
        return {"actions": np.arange(10 * 8, dtype=np.float32).reshape(10, 8),
                "prefix_feat": np.full(2048, float(idx), dtype=np.float32)}


def test_last_prefix_feat_multi_env_stacks_per_env():
    ad = LiberoPi05Adapter(_FeatPolicy(), prompt="x", action_dim=7, execute_horizon=5)
    ad.select_action(_raw_obs(3))          # 3 env 队列全空 → 3 次 infer:call 0,1,2
    pf = np.asarray(ad.last_prefix_feat())
    assert pf.shape == (3, 2048)
    assert np.allclose(pf[0], 0.0) and np.allclose(pf[1], 1.0) and np.allclose(pf[2], 2.0)


def test_last_prefix_feat_single_env_shape():
    ad = LiberoPi05Adapter(_FeatPolicy(), prompt="x", action_dim=7, execute_horizon=5)
    ad.select_action(_raw_obs(1))
    assert np.asarray(ad.last_prefix_feat()).shape == (1, 2048)


def test_last_prefix_feat_carries_over_non_reinferred_env():
    # execute_horizon=5 → 首次 select_action 后每 env 队列剩 4;第二次不重 infer,per-env 沿用。
    ad = LiberoPi05Adapter(_FeatPolicy(), prompt="x", action_dim=7, execute_horizon=5)
    ad.select_action(_raw_obs(2))          # call 0,1 → per-env [0,1]
    ad.select_action(_raw_obs(2))          # 队列非空,不 infer
    pf = np.asarray(ad.last_prefix_feat())
    assert pf.shape == (2, 2048)
    assert np.allclose(pf[0], 0.0) and np.allclose(pf[1], 1.0)   # 仍是首次特征,未被清/覆盖
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py::test_last_prefix_feat_multi_env_stacks_per_env -v`
Expected: FAIL —— 现 `last_prefix_feat()` 返回最后一次单帧 (2048,),`np.asarray(pf).shape` 是 `(2048,)` 不是 `(3,2048)`(断言失败)。

- [ ] **Step 3: 实现 — 4 处改动**

**(3a)** `__init__` 里,在 `self._last_prefix_feat = None` 那行**之后**加:

```python
        self._prefix_feat_per_env = []       # per-env 最近一次 infer 的 prefix_feat(与 _queues 等长)
```

**(3b)** `_ensure_queues` 改为同步增长 per-env 列表:

```python
    def _ensure_queues(self, b):
        while len(self._queues) < b:
            self._queues.append(deque())
            self._prefix_feat_per_env.append(None)
```

**(3c)** `select_action` 循环里,env i 重 infer 后记下该 env 特征。把循环体改为:

```python
        for i in range(b):
            if not self._queues[i]:
                obs = build_libero_serve_obs(
                    raw_obs, base_key=self.BASE_KEY, wrist_key=self.WRIST_KEY,
                    state_key=self.STATE_KEY, prompt=self.prompt, env_index=i)
                self._queues[i].extend(self._infer_chunk(obs))
                self._prefix_feat_per_env[i] = self._last_prefix_feat   # 记下本 env 特征(_infer_chunk 刚 set)
            out.append(self._queues[i].popleft())
```

**(3d)** `last_prefix_feat` 改为:

```python
    def last_prefix_feat(self):
        """per-env 就位时返回堆叠 [b,2048](向量化 eval);未就位(直调 _infer_chunk/无 select_action/
        serve 不透特征含 None)回退单帧 _last_prefix_feat(向后兼容 + None)。"""
        if not self._prefix_feat_per_env or any(f is None for f in self._prefix_feat_per_env):
            return self._last_prefix_feat
        return np.stack([np.asarray(f, dtype=np.float32) for f in self._prefix_feat_per_env])
```

- [ ] **Step 4: 跑测试验证通过(含既有适配器测试零回归)**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py resfit/rl_finetuning/chunk_residual/tests/test_adapter_prefix_feat.py -v`
Expected: PASS —— 新 3 个 + `test_adapter_prefix_feat.py` 两个直调 `_infer_chunk` 测试仍绿(per-env 未就位→回退单帧 (2048,)/None)+ `test_libero_pi05_adapter.py` 既有多 env/队列测试全绿。

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py
git commit -m "feat(libero_adapter): last_prefix_feat per-env 堆叠 [b,2048](向量化 eval pi0_feat)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## 实现后:集成 smoke(人工,默认向量化)

跑 `run_libero_task3_pi0feat_smoke.sh` 但**去掉 `--eval_num_envs 1`**(回到默认 8):serve 在 8000、offcache 秒复用。从 `/mnt/mnt/data/resfit`,用卡 5。

**Expected:** 跨过 eval(无 `Sizes of tensors must match`)、打印 `[env_steps N] eval success_rate=...`、best.pt 落盘。成功即默认配置可正式向量化 eval。

---

## 自审清单

**1. Spec 覆盖:** per-env 列表(3a/3b)、select_action 存(3c)、last_prefix_feat 堆叠+回退(3d)、三测试(多 env 堆叠/单 env 形状/沿用)、回归(直调 _infer_chunk + 既有多 env)、集成 smoke。全覆盖。

**2. 占位扫描:** 无 TBD/TODO;每步含完整测试与实现代码。

**3. 类型一致:** `_prefix_feat_per_env`(list,元素 (2048,) ndarray 或 None)↔ `last_prefix_feat` stack 成 [b,2048];`_FeatPolicy.infer` 返回的 `prefix_feat` 形状 (2048,) ↔ `_infer_chunk` 的 `result.get("prefix_feat")` ↔ spec 声明的 serve 单帧 (2048,);新测试用的 `_raw_obs(B)` 与文件既有夹具同签名。
