# chunk_residual 训练默认值对齐金标准 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `train_chunk_residual.py` 的 6 个 argparse 默认改成近期实验的"金标准"值,让不带 flag 跑即金标准配置,旧值全部可显式回退。

**Architecture:** 只改 `build_parser()` 里的 `default=`(CLI 层),不动底层函数/训练逻辑。`stage_balanced` 用同-dest 互斥写法(store_true default=True + `--no_stage_balanced`)。`offline_base_mode` 默认 base_policy 与新默认 `chunk_length=1`/`queue` 配套满足其 assert;回退 chunk_length 须同时回退 offline_base_mode(耦合,有测试覆盖)。

**Tech Stack:** Python / argparse / pytest。env python: `/mnt/mnt/data/envs/residual/bin/python`。仓库根 `/mnt/mnt/data/resfit`。分支 `chunk-residual-validation`。

**改动总览(6 个默认值, `train_chunk_residual.py`)**:

| flag | 行 | 旧默认 | 新默认 |
|---|---|---|---|
| `--chunk_length` | 275 | 20 | 1 |
| `--actor_lr` | 286 | 5e-6 | 1e-6 |
| `--base_action_mode` | 339 | replan | queue |
| `--base_n_action_steps` | 356 | None | 10 |
| `--stage_balanced` | 293 | False(store_true) | True(+`--no_stage_balanced`) |
| `--offline_base_mode` | 310 | gt | base_policy |

`--wandb_project` 默认已是 `dexmg-chunk-residual`,不动。spec: `docs/superpowers/specs/2026-06-10-chunk-residual-golden-defaults-design.md`。

---

### Task 1: 4 个带值参数默认(chunk_length/actor_lr/base_action_mode/base_n_action_steps)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py`
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:275,286,339,356`

- [ ] **Step 1: 写失败测试** — 新建 `tests/test_golden_defaults.py`:

```python
"""train_chunk_residual 金标准默认值断言(2026-06-10)。"""
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def test_value_defaults_aligned_to_golden():
    """4 个带值参数 CLI 默认 = 金标准;旧值可显式回退。"""
    a = build_parser().parse_args([])
    assert a.chunk_length == 1
    assert a.actor_lr == 1e-6
    assert a.base_action_mode == "queue"
    assert a.base_n_action_steps == 10
    b = build_parser().parse_args(["--chunk_length", "20", "--actor_lr", "5e-6",
                                   "--base_action_mode", "replan", "--base_n_action_steps", "5"])
    assert b.chunk_length == 20
    assert b.actor_lr == 5e-6
    assert b.base_action_mode == "replan"
    assert b.base_n_action_steps == 5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py::test_value_defaults_aligned_to_golden -v`
Expected: FAIL(`assert 20 == 1` 等,当前默认仍旧值)

- [ ] **Step 3: 改 4 个 argparse 默认** — `train_chunk_residual.py` 逐行改:

第 275 行 `p.add_argument("--chunk_length", type=int, default=20)` 改为:
```python
    p.add_argument("--chunk_length", type=int, default=1)
```
第 286 行 `p.add_argument("--actor_lr", type=float, default=5e-6)` 改为:
```python
    p.add_argument("--actor_lr", type=float, default=1e-6)
```
第 339 行 `p.add_argument("--base_action_mode", choices=["replan", "queue"], default="replan",` 改为(只改 default,保留其后 help 行不动):
```python
    p.add_argument("--base_action_mode", choices=["replan", "queue"], default="queue",
```
第 356 行 `p.add_argument("--base_n_action_steps", type=int, default=None,` 改为(只改 default,保留其后 help 行不动):
```python
    p.add_argument("--base_n_action_steps", type=int, default=10,
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py
git -C /mnt/mnt/data/resfit commit -m "feat: chunk_residual 4 个带值默认对齐金标准(chunk_length1/actor_lr1e-6/queue/n10)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `--stage_balanced` 默认开 + 关闭开关

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:293`
- Modify: `resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py`(追加)

- [ ] **Step 1: 写失败测试** — 追加到 `tests/test_golden_defaults.py` 末尾:

```python
def test_stage_balanced_default_on():
    """stage_balanced 默认 True;--no_stage_balanced 可关;--stage_balanced 仍可显式开。"""
    assert build_parser().parse_args([]).stage_balanced is True
    assert build_parser().parse_args(["--no_stage_balanced"]).stage_balanced is False
    assert build_parser().parse_args(["--stage_balanced"]).stage_balanced is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py::test_stage_balanced_default_on -v`
Expected: FAIL(`assert False is True`,当前 store_true 默认 False;且 `--no_stage_balanced` 未定义会 SystemExit)

- [ ] **Step 3: 改 argparse(默认开 + 关闭开关)** — `train_chunk_residual.py` 第 293 行整行替换为两行:

```python
    p.add_argument("--stage_balanced", dest="stage_balanced", action="store_true", default=True,
                   help="按 stage 配额采样(stage-balanced replay;默认开)")
    p.add_argument("--no_stage_balanced", dest="stage_balanced", action="store_false",
                   help="关闭 stage-balanced 采样(回退旧行为)")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py
git -C /mnt/mnt/data/resfit commit -m "feat: chunk_residual stage_balanced 默认开(+--no_stage_balanced 关)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `--offline_base_mode` 默认 base_policy + 改现有断言 + 耦合测试

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:310`
- Modify: `resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py:139-151`(改 2 测试 + 追加耦合测试)

注意:`_validate_offline_base_mode`(定义 224、main 调用 410)要求 base_policy 时 `chunk_length==1 && base_action_mode=="queue"`。Task 1 后默认已满足。耦合测试须直接调 `_validate_offline_base_mode(args)`(parse_args 本身不触发该校验)。

- [ ] **Step 1: 改现有 2 个会被破坏的断言** — `tests/test_base_policy_as_base.py`:

第 139-140 行的 `test_cli_offline_base_mode_default_gt` 整个函数替换为:
```python
def test_cli_offline_base_mode_default_base_policy():
    assert build_parser().parse_args([]).offline_base_mode == "base_policy"
    # 旧值可显式回退(回退 chunk_length 时须一并回退,见 test_validate_* 耦合测试)
    assert build_parser().parse_args(["--offline_base_mode", "gt"]).offline_base_mode == "gt"
```

第 148-151 行的 `test_signature_gt_has_no_base_mode_key` 整个函数替换为(改成显式传 gt):
```python
def test_signature_gt_has_no_base_mode_key():
    img = ["observation.images.agentview"]
    s = _offline_buffer_signature(_sig_args(["--offline_base_mode", "gt"]), img, 100, "staged")
    assert "offline_base_mode" not in s          # gt 时 → 不加键 → 旧缓存向后兼容
```

- [ ] **Step 2: 追加耦合测试** — 追加到 `tests/test_base_policy_as_base.py` 末尾:

```python
import pytest
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _validate_offline_base_mode


def test_validate_default_combo_passes():
    """默认(base_policy + chunk_length 1 + queue)满足 base_policy 约束,不抛。"""
    _validate_offline_base_mode(build_parser().parse_args([]))


def test_validate_chunk_length_rollback_alone_fails():
    """单独回退 --chunk_length 20(base_policy 默认仍在)→ assert fail-fast。"""
    args = build_parser().parse_args(["--chunk_length", "20"])
    with pytest.raises(AssertionError):
        _validate_offline_base_mode(args)


def test_validate_chunk_length_with_gt_rollback_passes():
    """同时回退 --chunk_length 20 --offline_base_mode gt → gt 跳过校验,不抛。"""
    _validate_offline_base_mode(
        build_parser().parse_args(["--chunk_length", "20", "--offline_base_mode", "gt"]))
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -v`
Expected: FAIL,具体是依赖 base_policy 默认尚未生效的 2 个:`test_cli_offline_base_mode_default_base_policy`(默认仍 gt → 断言 base_policy 失败)、`test_validate_chunk_length_rollback_alone_fails`(默认仍 gt → `_validate` 跳过不抛 → `pytest.raises(AssertionError)` 落空)。另两个 `_passes` 耦合测试此时已通过(gt 跳过 / 默认组合不抛),`test_signature_gt_has_no_base_mode_key`(改成显式传 gt)也通过。

- [ ] **Step 4: 改 argparse 默认** — `train_chunk_residual.py` 第 310 行 `default="gt"` 改为 `default="base_policy"`(其余 choices/help 不动):
```python
    p.add_argument("--offline_base_mode", choices=["gt", "base_policy"], default="base_policy",
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -v`
Expected: PASS(全部,含改后 2 测试 + 3 耦合测试)

- [ ] **Step 6: 提交**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py
git -C /mnt/mnt/data/resfit commit -m "feat: chunk_residual offline_base_mode 默认 base_policy(修 gt_as_base bug)

改现有默认 gt 断言为 base_policy;新增 chunk_length-offline_base_mode 耦合测试。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 全量回归(验证 gate)

**Files:** 无代码改动(纯验证)。

- [ ] **Step 1: 跑全 chunk_residual 测试套件**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests -q`
Expected: 全绿,0 failed(确认 6 个默认改动 + 新增/改动测试全过,且未破坏其它既有测试,尤其 `test_stage_budget`/`test_relabel`/`test_demo_bc`/`test_hiql_potential`/`test_wandb_logging` 里的 `parse_args([])` 断言)

- [ ] **Step 2: 若 Step 1 有失败,定位并修**

若某既有测试因新默认失败(例如它隐式依赖旧默认),按"显式传旧值"原则修该测试断言(不改训练逻辑),重跑直至全绿。无失败则跳过。

- [ ] **Step 3: 无新增代码改动则无需提交**(Step 2 若改了测试,单独 commit 并说明)

---

## 备注:不影响进行中的 A/B

正在跑的 aligned run1/run2 **显式传**全部金标准 flag,不依赖默认;改默认不影响它们。本 plan 纯改 CLI 默认,与实验解耦。
