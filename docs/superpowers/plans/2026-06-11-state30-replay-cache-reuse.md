# state30 replay 缓存复用(chokepoint)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `read_per_demo_states`(value/gc_value/state30 构建的共同汇聚点)在有完整缓存时直接读、跳过 MuJoCo 全 demo replay,消除重复 replay,且默认逐位等价。

**Architecture:** 缓存格式扩为 v2(30 维 seqs + rel_stats + dataset_id);把"能否复用"的等价性边界抽成纯函数 `state30_cache_reuse`(单测穷尽,不碰 MuJoCo);`read_per_demo_states` 加可选 `cache_path`,命中即返回、未命中走原 replay 并写 v2;value/gc_value 加 `--state30_cache` 透传;`load_or_build_state30` 改写 v2。不动 `build_offline_buffer`。

**Tech Stack:** Python / numpy(.npz)/ pytest。conda 环境 `residual`。

---

## 约定
- 工作目录仓库根 `/mnt/mnt/data/resfit`;单测 `conda run -n residual python -m pytest <path> -v`。
- 纯逻辑任务走 TDD;最后一个集成验证用真 threading 数据。
- 每个 Task 结束 commit(co-author trailer 见末尾)。

## 文件结构图
| 文件 | 责任 | 动作 |
|---|---|---|
| `resfit/rl_finetuning/chunk_residual/state30_cache.py` | 缓存 save/load + 复用决策 | 改:save v2、加 `load_state30_cache_v2`、加 `state30_cache_reuse` |
| `resfit/rl_finetuning/chunk_residual/train_hiql_value.py` | `read_per_demo_states` + CLI | 改:加 `cache_path` 复用、加 `--state30_cache` |
| `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py` | gc_value CLI | 改:加 `--state30_cache` 透传 |
| `.../tests/test_state30_cache.py` | 缓存层单测 | 改:加 v2 + 复用决策用例 |

---

## Task 1: 缓存格式 v2(save/load_v2)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/state30_cache.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_state30_cache.py` 末尾)

```python
from resfit.rl_finetuning.chunk_residual.state30_cache import load_state30_cache_v2


def test_state30_cache_v2_roundtrip(tmp_path):
    seqs = [np.ones((5, 30), np.float32), np.zeros((3, 30), np.float32)]
    mean = np.arange(12).astype(np.float32)
    std = (np.arange(12) + 1).astype(np.float32)
    p = str(tmp_path / "c.npz")
    save_state30_cache(p, seqs, rel_stats=(mean, std), dataset_id="ds/x")
    s, rel, ds = load_state30_cache_v2(p)
    assert len(s) == 2 and np.allclose(s[0], seqs[0])
    assert rel is not None and np.allclose(rel[0], mean) and np.allclose(rel[1], std)
    assert ds == "ds/x"


def test_state30_cache_v1_loaded_as_no_stats(tmp_path):
    # 旧格式(不传 rel_stats)→ load_v2 返回 rel_stats=None, ds=None
    p = str(tmp_path / "old.npz")
    save_state30_cache(p, [np.ones((2, 30), np.float32)])
    s, rel, ds = load_state30_cache_v2(p)
    assert len(s) == 1 and rel is None and ds is None


def test_old_loader_still_reads_v2_seqs(tmp_path):
    # 旧 reader load_state30_cache 仍能从 v2 读 seqs(忽略额外键)——保护 high_actor/goal30 路径
    p = str(tmp_path / "c.npz")
    save_state30_cache(p, [np.ones((2, 30), np.float32)],
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)),
                       dataset_id="d")
    assert len(load_state30_cache(p)) == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py -v -k "v2 or v1 or old_loader"`
Expected: FAIL（`ImportError: cannot import name 'load_state30_cache_v2'` / `save_state30_cache() got an unexpected keyword argument 'rel_stats'`）。

- [ ] **Step 3: 实现**。改 `state30_cache.py` 的 `save_state30_cache` 并加 `load_state30_cache_v2`:

```python
def save_state30_cache(path, seqs, rel_stats=None, dataset_id=None):
    """存 30 维 state 序列到 npz。

    rel_stats=(mean(12,),std(12,)) 给了则写 v2(额外 rel_mean/rel_std/dataset_id);
    不给则 v1(仅 n+s{i},向后兼容)。
    """
    payload = {"n": np.int64(len(seqs))}
    for i, s in enumerate(seqs):
        payload[f"s{i}"] = np.asarray(s, dtype=np.float32)
    if rel_stats is not None:
        mean, std = rel_stats
        payload["rel_mean"] = np.asarray(mean, dtype=np.float32)
        payload["rel_std"] = np.asarray(std, dtype=np.float32)
        if dataset_id is not None:
            payload["dataset_id"] = np.asarray(str(dataset_id))
    np.savez_compressed(path, **payload)


def load_state30_cache_v2(path):
    """读 npz → (seqs, rel_stats 或 None, dataset_id 或 None)。

    缺 rel_mean 键(旧 v1 格式)→ rel_stats=None, dataset_id=None。
    """
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        seqs = [z[f"s{i}"] for i in range(n)]
        if "rel_mean" in z.files:
            rel_stats = (z["rel_mean"], z["rel_std"])
            ds = str(z["dataset_id"]) if "dataset_id" in z.files else None
        else:
            rel_stats, ds = None, None
    return seqs, rel_stats, ds
```
（`load_state30_cache` 不动——它读 `n`+`s{i}`,v2 的额外键被忽略。）

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py -v`
Expected: 全 PASS（含旧的 `test_state30_cache_roundtrip`）。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/state30_cache.py \
        resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py
git commit -m "feat(state30-cache): v2 格式存 rel_stats/dataset_id + load_state30_cache_v2(v1 兼容)"
```

---

## Task 2: 复用决策纯函数 `state30_cache_reuse`(等价性边界)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/state30_cache.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_state30_cache.py` 末尾)

```python
from resfit.rl_finetuning.chunk_residual.state30_cache import state30_cache_reuse


def _write_v2(p, n=3, dataset_id="ds"):
    seqs = [np.ones((2, 30), np.float32) for _ in range(n)]
    save_state30_cache(p, seqs,
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)),
                       dataset_id=dataset_id)


def test_reuse_hit_full(tmp_path):
    p = str(tmp_path / "c.npz"); _write_v2(p, n=3, dataset_id="ds")
    out = state30_cache_reuse(p, dataset_id="ds", num_demos=None)
    assert out is not None
    seqs, rel = out
    assert len(seqs) == 3 and np.allclose(rel[1], 1.0)


def test_reuse_skip_partial_num_demos(tmp_path):
    # 部分 demo:rel_stats 会随 N 变 → 必须不复用(等价性边界,spec §2)
    p = str(tmp_path / "c.npz"); _write_v2(p)
    assert state30_cache_reuse(p, dataset_id="ds", num_demos=20) is None


def test_reuse_skip_missing_file(tmp_path):
    assert state30_cache_reuse(str(tmp_path / "nope.npz"), dataset_id="ds", num_demos=None) is None


def test_reuse_skip_old_format(tmp_path):
    p = str(tmp_path / "old.npz")
    save_state30_cache(p, [np.ones((2, 30), np.float32)])   # v1,无 stats
    assert state30_cache_reuse(p, dataset_id="ds", num_demos=None) is None


def test_reuse_skip_dataset_mismatch(tmp_path):
    p = str(tmp_path / "c.npz"); _write_v2(p, dataset_id="ds_a")
    assert state30_cache_reuse(p, dataset_id="ds_b", num_demos=None) is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py -v -k reuse`
Expected: FAIL（`ImportError: cannot import name 'state30_cache_reuse'`）。

- [ ] **Step 3: 实现**。在 `state30_cache.py` 加(`load_state30_cache_v2` 之后):

```python
def state30_cache_reuse(cache_path, *, dataset_id, num_demos):
    """决定能否复用 state30 缓存(等价性边界)。可复用→返回 (seqs, rel_stats);否则 None。

    规则(spec §2):缓存存在 且 num_demos is None(只有全量与新鲜 replay 严格等价,
    因 rel_stats 是对全量 raw rel 算的)且 v2(有 rel_stats)且 dataset_id 一致。
    """
    if not (cache_path and os.path.exists(cache_path)):
        return None
    if num_demos is not None:           # 部分 demo → rel_stats 不同,不复用
        return None
    seqs, rel_stats, ds = load_state30_cache_v2(cache_path)
    if rel_stats is None:               # 旧 v1 格式
        return None
    if ds != dataset_id:                # 张冠李戴防护
        return None
    return seqs, rel_stats
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py -v`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/state30_cache.py \
        resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py
git commit -m "feat(state30-cache): state30_cache_reuse 复用决策(num_demos=None 才复用,等价性边界)"
```

---

## Task 3: read_per_demo_states 接 cache + CLI flag + load_or_build_state30 写 v2

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`
- Modify: `resfit/rl_finetuning/chunk_residual/state30_cache.py`

- [ ] **Step 1: 改 `read_per_demo_states` 签名加 `cache_path`,命中走缓存、未命中写 v2**

`train_hiql_value.py`:把 `def read_per_demo_states(hdf5_path, dataset_id, state_mode="eef", num_demos=None, device="cpu"):` 改为加参 `cache_path=None`,并在建好 standardizer 后、eef_piece replay 前插命中检查,replay 后写缓存。改动后函数:

```python
def read_per_demo_states(hdf5_path, dataset_id, state_mode="eef", num_demos=None,
                         device="cpu", cache_path=None):
    """读每条 demo 的标准化 state 序列(与 RL 训练同源 mean/std)。

    state_mode=eef: (T,18) 纯 eef。eef_piece: (T,30)=[eef18 | 标准化 rel_piece12]。
    cache_path(仅 eef_piece+num_demos=None 时):命中完整 v2 缓存则跳过 MuJoCo replay。
    返回 (list[np.ndarray], standardizer, rel_piece_stats 或 None)。
    """
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.state30_cache import (
        state30_cache_reuse, save_state30_cache)
    meta = LeRobotDatasetMetadata(dataset_id)
    standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=device)
    if state_mode == "eef_piece" and cache_path is not None:
        hit = state30_cache_reuse(cache_path, dataset_id=dataset_id, num_demos=num_demos)
        if hit is not None:
            seqs30, rel_stats = hit
            print(f"[read_per_demo_states] 缓存命中 {cache_path} → 跳过 replay({len(seqs30)} demo)")
            return seqs30, standardizer, rel_stats
    seqs, replay_meta = [], []
    with h5py.File(hdf5_path, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
        if num_demos is not None:
            eps = eps[:num_demos]
        for ep in eps:
            grp = f[f"data/{ep}"]
            obs_arrays = {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS}
            state_raw = assemble_state18(obs_arrays)  # (T,18) np
            state_n = standardizer.standardize(
                torch.as_tensor(state_raw, dtype=torch.float32)).cpu().numpy()
            seqs.append(state_n)
            if state_mode == "eef_piece":
                replay_meta.append((grp["states"][()], grp.attrs["model_file"],
                                    grp.attrs.get("ep_meta")))
    if state_mode == "eef":
        return seqs, standardizer, None

    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
        make_replay_env, replay_eef_rel_piece)
    from resfit.rl_finetuning.chunk_residual.object_state import rel_piece_stats
    env, _ = make_replay_env(hdf5_path)
    rel_raws = []
    try:
        for states, model_file, ep_meta in replay_meta:
            rel_raws.append(replay_eef_rel_piece(
                env, states, model_file=model_file, ep_meta=ep_meta))
    finally:
        env.close()
    mean, std = rel_piece_stats(np.concatenate(rel_raws, axis=0))
    seqs30 = []
    for s18, rp in zip(seqs, rel_raws):
        t = min(len(s18), len(rp))
        rp_std = ((rp[:t] - mean) / std).astype(np.float32)
        seqs30.append(np.concatenate([s18[:t], rp_std], axis=1))
    if cache_path is not None and num_demos is None:
        save_state30_cache(cache_path, seqs30, rel_stats=(mean, std), dataset_id=dataset_id)
        print(f"[read_per_demo_states] 已写 v2 缓存 {cache_path}")
    return seqs30, standardizer, (mean, std)
```

- [ ] **Step 2: `train_hiql_value` 加 `--state30_cache` 并透传**

`train_hiql_value.py` argparse(在 `--num_demos` 之后)加:
```python
    p.add_argument("--state30_cache", default=None,
                   help="state30 v2 缓存路径(eef_piece+全量时命中跳过 replay;不传=每次 replay)")
```
`main()` 里把 `read_per_demo_states(args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos)` 改为加 `cache_path=args.state30_cache`:
```python
    seqs, standardizer, rel_stats = read_per_demo_states(
        args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos,
        cache_path=args.state30_cache)
```

- [ ] **Step 3: `train_hiql_gc_value` 加 `--state30_cache` 并透传**

`train_hiql_gc_value.py` argparse(`--num_demos` 之后)加同样一行:
```python
    p.add_argument("--state30_cache", default=None,
                   help="state30 v2 缓存路径(命中跳过 replay)")
```
`main()` 里把 `read_per_demo_states(args.hdf5, args.dataset, "eef_piece", num_demos=args.num_demos)` 改为:
```python
    seqs, standardizer, rel_stats = read_per_demo_states(
        args.hdf5, args.dataset, "eef_piece", num_demos=args.num_demos,
        cache_path=args.state30_cache)
```

- [ ] **Step 4: `load_or_build_state30` 构建分支写 v2(捕获 rel_stats)**

`state30_cache.py` 的 `load_or_build_state30`,把构建分支改为捕获并存 v2:
```python
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
    seqs, _, rel_stats = read_per_demo_states(hdf5_path, dataset_id, "eef_piece", num_demos=num_demos)
    if cache_path and num_demos is None:
        save_state30_cache(cache_path, seqs, rel_stats=rel_stats, dataset_id=dataset_id)
    elif cache_path:
        save_state30_cache(cache_path, seqs)   # 部分 demo 仍按 v1 存(不含全量 stats)
    return seqs
```
（读分支 `load_state30_cache` 不变;v2 缓存的 seqs 照样读。）

- [ ] **Step 5: 回归 + flag 存在性检查**

Run:
```bash
conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_state30_cache.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -v
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_value --help 2>/dev/null | grep state30_cache
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value --help 2>/dev/null | grep state30_cache
```
Expected: 测试全 PASS（默认 cache_path=None → 行为不变);两个 `--help` 都打印 `--state30_cache`。

- [ ] **Step 6: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py \
        resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py \
        resfit/rl_finetuning/chunk_residual/state30_cache.py
git commit -m "feat(state30-cache): read_per_demo_states 接 cache_path 复用 + value/gc_value --state30_cache(默认等价)"
```

---

## Task 4: 真实数据集成验证(等价 + 提速,执行期跑一次)

**Files:** 无代码(验证)。**前提**:`outputs_chunk/two_arm_threading_state30.npz` 已是 v2(若是旧 v1,本验证第一次跑会 replay 重写成 v2)。**不与正在跑的实验抢卡**:用空闲卡。

- [ ] **Step 1: 用现有 threading 缓存验证「命中跳过 replay」+ 数值等价**

Run(空闲 GPU,如 6;限线程):
```bash
cd /mnt/mnt/data/resfit
# (a) 第一次:显式传 cache。若缓存已是 v2 → 秒级命中;若旧 v1 → replay 一次并重写 v2
OMP_NUM_THREADS=8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
conda run -n residual python -u -c "
import time
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
t=time.time(); s,_,rel=read_per_demo_states('resfit/dataset/two_arm_threading.hdf5','ankile/dexmg-two-arm-threading','eef_piece',cache_path='outputs_chunk/two_arm_threading_state30.npz'); print('run1 %.1fs demos=%d rel_std0=%.4f'%(time.time()-t,len(s),rel[1][0]))
t=time.time(); s2,_,rel2=read_per_demo_states('resfit/dataset/two_arm_threading.hdf5','ankile/dexmg-two-arm-threading','eef_piece',cache_path='outputs_chunk/two_arm_threading_state30.npz'); print('run2 %.1fs'%(time.time()-t))
import numpy as np
print('EQUAL seqs0=%s rel=%s'%(np.array_equal(s[0],s2[0]), np.allclose(rel[0],rel2[0]) and np.allclose(rel[1],rel2[1])))
" 2>&1 | grep -vE "robosuite|Could not|macro|Warning"
```
Expected:`run2` 远快于（或等同秒级,若 run1 已命中）`run1`;打印 `EQUAL seqs0=True rel=True`(命中返回与 replay 逐位一致)。

- [ ] **Step 2: 验证「部分 demo 不误用缓存」**

Run:
```bash
OMP_NUM_THREADS=8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
conda run -n residual python -u -c "
from resfit.rl_finetuning.chunk_residual.state30_cache import state30_cache_reuse
print('partial reuse=', state30_cache_reuse('outputs_chunk/two_arm_threading_state30.npz', dataset_id='ankile/dexmg-two-arm-threading', num_demos=20))  # 应 None
" 2>&1 | tail -1
```
Expected:`partial reuse= None`。

- [ ] **Step 3: 无 commit(验证)。** 记录 run2 提速倍数。

---

## 自查覆盖(spec → task)
- spec §3.1 缓存格式 v2 → Task 1 ✓
- spec §3.2 read_per_demo_states 复用 + num_demos 边界 → Task 2(决策)+ Task 3 Step1(接线)✓
- spec §3.3 消费方接线(value/gc_value/load_or_build_state30)→ Task 3 Step2-4 ✓
- spec §3.4 默认等价 / 不动 build_offline_buffer/high_actor → Task 3 默认 cache_path=None + Task 1 旧 reader 兼容 ✓
- spec §4 测试(往返等价/默认等价/旧格式回退/部分不复用/dataset 不符/save-load v2)→ Task 1+2 单测 + Task 4 集成 ✓
- spec §6 风险(standardizer 一致/旧缓存误判/不影响在跑)→ dataset_id 校验(Task2)+ 旧格式→None(Task2)+ 默认 None(Task3)✓

---

提交用 co-author trailer:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
