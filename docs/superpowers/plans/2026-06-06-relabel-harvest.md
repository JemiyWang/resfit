# 在线 relay relabeling 生产端(模块②b)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在训练循环里 harvest 在线"产出前缀"段(失败但推进过 stage 的轨迹,起点→最后一次推进,丢尾巴)进 relabel buffer,bc_batch 改为 relabel+demo 50/50 混采喂 ②a 残差 BC 消费端;`--relabel` 默认关=逐位等价 ②a。

**Architecture:** 纯逻辑(`productive_prefix_len` + `RelabelHarvester` + `sample_bc_batch`)放新模块 `relabel.py`(可单测);train 循环用 `make_bc_entry` 把每 chunk 的 (obs, combined_action) 喂 harvester,done 时 flush 产出前缀进 relabel_rb;update 处 `--relabel` 开则 bc_batch 走 `sample_bc_batch`(relabel+demo 混采),否则走 ②a 原逻辑。只进 actor 侧 bc_batch,不碰 critic。

**Tech Stack:** PyTorch；torchrl/tensordict（TensorDictReplayBuffer）；pytest（conda env `residual`，从仓库根目录跑）。

设计依据：`docs/superpowers/specs/2026-06-06-relabel-harvest-design.md`。前置:②a 残差 demo-BC 已实现。

---

## 文件结构

- `resfit/rl_finetuning/chunk_residual/relabel.py`（新）—— `productive_prefix_len`(纯)+ `RelabelHarvester`(纯,存不透明 entry)+ `sample_bc_batch`(relabel/demo 混采)。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` —— 3 个 flag;`make_bc_entry` 辅助;建 relabel_rb + harvester(--relabel 时);循环 harvest 接线;update 处用 sample_bc_batch;依赖断言。
- `resfit/rl_finetuning/chunk_residual/tests/test_relabel.py`（新）—— 纯函数 + harvester + sample_bc_batch + CLI。

测试运行：`conda run -n residual python -m pytest <path> -v`（仓库根目录，默认排除 `-m manual`）。
**GPU 全占:本计划不启动任何 CUDA 训练;测试 CPU 跑。**

---

## Task 1: relabel.py 纯逻辑(productive_prefix_len + RelabelHarvester)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/relabel.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_relabel.py`（新建）

- [ ] **Step 1: 写失败测试**

新建 `resfit/rl_finetuning/chunk_residual/tests/test_relabel.py`:

```python
from resfit.rl_finetuning.chunk_residual.relabel import (
    productive_prefix_len, RelabelHarvester,
)


def test_prefix_drops_floundering_tail():
    assert productive_prefix_len([0, 1, 1, 2, 2], min_stage=1) == 4


def test_prefix_zero_when_no_advance():
    assert productive_prefix_len([0, 0, 0], min_stage=1) == 0


def test_prefix_full_when_monotone_climb():
    assert productive_prefix_len([1, 2, 3, 4], min_stage=1) == 4


def test_prefix_empty_seq():
    assert productive_prefix_len([], min_stage=1) == 0


def test_prefix_min_stage_filter():
    assert productive_prefix_len([0, 1, 1], min_stage=2) == 0     # max=1 < 2 -> 0


def test_harvester_returns_prefix_and_resets():
    h = RelabelHarvester(min_stage=1)
    for e, s in [("a", 0), ("b", 1), ("c", 1), ("d", 2), ("e", 2)]:
        h.add(e, s)
    assert h.flush() == ["a", "b", "c", "d"]    # 到最后推进(stage2 首次=idx3)含
    assert h.flush() == []                        # 已清空


def test_harvester_no_advance_returns_empty():
    h = RelabelHarvester(min_stage=1)
    h.add("a", 0)
    h.add("b", 0)
    assert h.flush() == []
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_relabel.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named '...relabel'`）

- [ ] **Step 3: 实现 relabel.py(本 Task 只放这两个)**

新建 `resfit/rl_finetuning/chunk_residual/relabel.py`:

```python
"""在线 relay relabeling 生产端(模块②b):产出前缀选段 + episode harvester。

只进 actor 侧 bc_batch(喂 ②a 残差 BC 消费端),不碰 critic 的 reward/done。
"""
from __future__ import annotations


def productive_prefix_len(stage_seq, min_stage: int = 1) -> int:
    """产出前缀长度:起点→最后一次 stage 推进(含),丢掉之后无推进的尾巴。

    stage_seq 是一条 episode 逐 chunk 的 latch(非降,由 chunk_env_wrapper 的 running-max 保证)。
    max < min_stage(没推进够)返回 0(该 episode 不 harvest)。latch 非降 -> max 首次出现 = 最后一次推进。
    """
    if not stage_seq:
        return 0
    top = max(stage_seq)
    if top < min_stage:
        return 0
    return stage_seq.index(top) + 1


class RelabelHarvester:
    """攒当前 episode 的 (entry, max_stage);flush() 按 productive_prefix_len 返回产出前缀 entry 并清空。

    entry 是调用方给的不透明对象(本项目=每 chunk 的 bc td)。min_stage 过滤走得不够远的 episode。
    """

    def __init__(self, min_stage: int = 1):
        self._min_stage = int(min_stage)
        self._entries: list = []
        self._stages: list = []

    def add(self, entry, max_stage) -> None:
        self._entries.append(entry)
        self._stages.append(int(max_stage))

    def flush(self) -> list:
        n = productive_prefix_len(self._stages, self._min_stage)
        out = self._entries[:n]
        self._entries, self._stages = [], []
        return out
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_relabel.py -v`
Expected: PASS（7 个）

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/relabel.py \
        resfit/rl_finetuning/chunk_residual/tests/test_relabel.py
git commit -m "feat(relabel): productive_prefix_len + RelabelHarvester 纯逻辑 + 单测"
```

---

## Task 2: sample_bc_batch(relabel+demo 50/50 混采)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/relabel.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_relabel.py`

- [ ] **Step 1: 追加失败测试**

在 `test_relabel.py` 末尾追加:

```python
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from resfit.rl_finetuning.chunk_residual.relabel import sample_bc_batch


def _tiny_rb(n):
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=max(n, 1), device="cpu"), batch_size=4)
    for _ in range(n):
        td = TensorDict({
            "obs": TensorDict({"observation.state": torch.randn(3)}, batch_size=[]),
            "action": torch.randn(5),
        }, batch_size=[]).unsqueeze(0)
        rb.add(td)
    return rb


def test_sample_bc_batch_mixes_when_relabel_full():
    bc = sample_bc_batch(_tiny_rb(20), _tiny_rb(20), batch_size=8, device="cpu")
    assert bc.shape[0] == 8                       # half relabel + half demo


def test_sample_bc_batch_falls_back_to_demo_when_relabel_short():
    bc = sample_bc_batch(_tiny_rb(1), _tiny_rb(20), batch_size=8, device="cpu")   # relabel<half(4)
    assert bc.shape[0] == 8                       # 整批来自 demo


def test_sample_bc_batch_none_relabel_uses_demo():
    bc = sample_bc_batch(None, _tiny_rb(20), batch_size=8, device="cpu")
    assert bc.shape[0] == 8
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_relabel.py -k sample_bc -v`
Expected: FAIL（`ImportError: cannot import name 'sample_bc_batch'`）

- [ ] **Step 3: 实现 sample_bc_batch(追加到 relabel.py)**

在 `relabel.py` 末尾追加:

```python
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import concat_mixed_batch


def sample_bc_batch(relabel_rb, offline_rb, batch_size: int, device):
    """bc_batch = relabel + demo 50/50 混采;relabel 不足半批 -> 整批回退 demo。搬到 device。

    relabel 条目只有 {obs, action};offline 还含 next/_priority/_weight。concat_mixed_batch 取两者
    公共 key 再 cat(bc 消费端只读 obs+action,公共 key 足够)。
    """
    half = batch_size // 2
    if relabel_rb is not None and len(relabel_rb) >= half:
        bc = concat_mixed_batch(relabel_rb.sample(half), offline_rb.sample(batch_size - half))
    else:
        bc = offline_rb.sample(batch_size)
    return bc.to(device, non_blocking=True)
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_relabel.py -v`
Expected: PASS（Task1 的 7 个 + 本 Task 3 个 = 10）

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/relabel.py \
        resfit/rl_finetuning/chunk_residual/tests/test_relabel.py
git commit -m "feat(relabel): sample_bc_batch(relabel+demo 50/50 混采,冷启动回退 demo)+ 单测"
```

---

## Task 3: train 接线(3 flag + harvest + 混采 bc_batch)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`（torchrl import；`make_bc_entry`；build_parser；offline_rb 之后建 relabel_rb/harvester；主循环 467 后 harvest；bc_batch 486-488）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_relabel.py`

- [ ] **Step 1: 追加失败测试(CLI light)**

在 `test_relabel.py` 末尾追加:

```python
def test_cli_relabel_defaults():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    a = build_parser().parse_args([])
    assert a.relabel is False
    assert a.relabel_buffer_size == 50_000
    assert a.relabel_min_stage == 1


def test_cli_relabel_parses():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    a = build_parser().parse_args(["--relabel", "--relabel_buffer_size", "1000",
                                   "--relabel_min_stage", "2"])
    assert a.relabel is True
    assert a.relabel_buffer_size == 1000
    assert a.relabel_min_stage == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_relabel.py -k cli_relabel -v`
Expected: FAIL（argparse `unrecognized arguments: --relabel` 或 AttributeError）

- [ ] **Step 3a: torchrl import 加 TensorDictReplayBuffer**

`train_chunk_residual.py` 顶部:
```python
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
```
改为:
```python
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
```

- [ ] **Step 3b: 加 make_bc_entry 辅助**

`train_chunk_residual.py` 里 `add_chunk_transition` 函数定义(约第 55 行)之后,加:
```python
def make_bc_entry(obs, action, image_keys, lowdim_keys):
    """构造 relabel 用的 bc 条目 td:{obs:{图+lowdim}, action}(与 offline_rb 同构,供混采 bc_batch)。"""
    keys = set(image_keys) | set(lowdim_keys)
    curr = {k: obs[k][0].detach().cpu() for k in keys}
    to_uint8(curr, image_keys)
    return TensorDict({"obs": TensorDict(curr, batch_size=[]),
                       "action": action[0].detach().cpu()}, batch_size=[]).unsqueeze(0)
```

- [ ] **Step 3c: build_parser 加 3 个 flag**

`build_parser` 里(放在 `--demo_bc_coef` 附近),加:
```python
    p.add_argument("--relabel", action="store_true",
                   help="在线 relay relabeling(模块②b):harvest 产出前缀段进 relabel buffer,"
                        "bc_batch 改 relabel+demo 50/50 混采。默认关=逐位等价 ②a。"
                        "需 --demo_bc_coef>0 + --offline_fraction>0 + --actor raw")
    p.add_argument("--relabel_buffer_size", type=int, default=50_000,
                   help="relabel buffer 容量(FIFO,CPU storage)")
    p.add_argument("--relabel_min_stage", type=int, default=1,
                   help="只 harvest 走到 stage>=该值的 episode(默认 1=只要推进过)")
```

- [ ] **Step 3d: offline_rb 块之后建 relabel_rb + harvester**

`main` 里 offline_rb 那一段(以 `print(f"[offline] 灌装 ...")` 收尾,约第 410 行附近)**之后**,加:
```python
    relabel_rb = None
    harvester = None
    if args.relabel:
        assert args.demo_bc_coef > 0, "--relabel 需 --demo_bc_coef>0(relabel 走 BC 路径)"
        assert args.offline_fraction > 0, "--relabel 需 --offline_fraction>0(demo 半边)"
        assert args.actor == "raw", "--relabel 第一版只支持 --actor raw"
        from resfit.rl_finetuning.chunk_residual.relabel import RelabelHarvester, sample_bc_batch
        relabel_rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=args.relabel_buffer_size, device="cpu"),
            batch_size=max(args.batch_size // 2, 1))
        harvester = RelabelHarvester(min_stage=args.relabel_min_stage)
        print(f"[relabel] on; buffer_size={args.relabel_buffer_size} min_stage={args.relabel_min_stage}")
```

- [ ] **Step 3e: 主循环 harvest 接线**

`add_chunk_transition(...)` 调用(约第 467-469 行)与 `obs = next_obs`(约第 470 行)之间,插入:
```python
        if harvester is not None:
            harvester.add(make_bc_entry(obs, info["scaled_action"], image_keys, lowdim_keys),
                          info.get("max_stage_in_chunk", 0))
            if bool(done.any()):
                for e in harvester.flush():
                    relabel_rb.add(e)
```

- [ ] **Step 3f: bc_batch 处用混采**

`main` 训练循环里 ②a 的(约第 486-488 行):
```python
                bc_batch = None
                if args.demo_bc_coef > 0 and update_actor and offline_rb is not None:
                    bc_batch = offline_rb.sample(args.batch_size).to(args.device, non_blocking=True)
```
替换为:
```python
                bc_batch = None
                if args.demo_bc_coef > 0 and update_actor and offline_rb is not None:
                    if args.relabel:
                        bc_batch = sample_bc_batch(relabel_rb, offline_rb, args.batch_size, args.device)
                    else:
                        bc_batch = offline_rb.sample(args.batch_size).to(args.device, non_blocking=True)
```
(`sample_bc_batch` 已在 Step 3d 的 `if args.relabel:` 块里 import;仅 --relabel 开时这条分支才执行,import 一定已发生。)

- [ ] **Step 4: 运行测试确认通过(全 CPU)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_relabel.py -v`
Expected: PASS（纯函数 7 + sample_bc 3 + cli 2 = 12）

回归(确保没踩坏 ②a / ①):
Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py -m "not manual" -q`
Expected: PASS

- [ ] **Step 5: 轻量接线验证(严禁起 CUDA 训练)**

```bash
conda run -n residual python -c "from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser; a=build_parser().parse_args(['--relabel']); assert a.relabel and a.relabel_buffer_size==50000; print('OK parser')"
grep -n "make_bc_entry\|harvester.add\|harvester.flush\|relabel_rb = TensorDictReplayBuffer\|sample_bc_batch(relabel_rb" resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
```
Expected: 打印 `OK parser`;grep 看到 make_bc_entry 定义+调用、harvester.add/flush、relabel_rb 构造、update 处 sample_bc_batch。
（完整 --relabel --smoke 训练 smoke 留待有空闲 GPU 时做。）

- [ ] **Step 6: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
        resfit/rl_finetuning/chunk_residual/tests/test_relabel.py
git commit -m "feat(relabel): train --relabel 接线(harvest 产出前缀 + 混采 bc_batch)+ CLI 测试"
```

---

## 完成后(不属本实现,A/B 在 plan/run 阶段)

见 spec §6:nas10 强 base + best 基座,`--demo_bc_coef 0.1 --offline_fraction 0.5`,
A=`--relabel` 关(=②a demo-only)vs B=`--relabel` 开。看 success/stage-reach、stage3 回退率,
旁证 relabel_rb 增长 + wandb `rft/bc_loss`。这才是 RPL relabeling 治稀疏的真正验证。
