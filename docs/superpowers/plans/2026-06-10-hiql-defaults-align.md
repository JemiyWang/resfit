# HIQL 默认值对齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 resfit 三个训练脚本的 argparse 默认值翻向 HIQL 口径(折中口径 + LN 也开),让"不带 flag 跑训练"即对齐 HIQL,旧选项全部保留可显式回退。

**Architecture:** 只改 argparse `default=`(CLI 层),**不动底层函数 `train_gc_value`/`train_high_actor` 的签名默认**(它们被现有单测显式传参锚定,改了会破坏回归)。经核查,现有测试均直接调底层函数显式传参,改 argparse 默认不破坏任何现有单测;故本 plan 只**新增** parser 默认断言,不改现有断言。renorm 用同-dest 互斥 flag(`--renorm_subgoal`/`--no_renorm_subgoal`)实现 default=True + 可关。

**Tech Stack:** Python / argparse / pytest。env python: `/mnt/mnt/data/envs/residual/bin/python`。仓库根: `/mnt/mnt/data/resfit`。分支: `chunk-residual-validation`。

**对齐目标(6 处默认值)**:

| flag | 脚本 | 旧默认 | 新默认 |
|---|---|---|---|
| `--goal_future_mode` | `train_hiql_gc_value.py` | `stage_entry` | `geometric` |
| `--use_layer_norm` | `train_hiql_gc_value.py` | `0` | `1` |
| `--value_loss_mode` | `train_hiql_gc_value.py` | `shared_min` | `hiql` |
| `--target_mode` | `train_hiql_high_actor.py` | `fixed_waypoint` | `clamp_to_goal` |
| `--high_p_randomgoal` | `train_hiql_high_actor.py` | `0.0` | `0.3` |
| `--renorm_subgoal` | `train_chunk_residual.py` | `False`(store_true) | `True`(+`--no_renorm_subgoal` 关) |

**前置事实(已核查)**:三脚本 `build_parser` 均可 import(env 跑通);`run_hiql_align_ab.sh` 里 A/B 调用已显式传 `--target_mode clamp_to_goal`/`--goal_future_mode geometric --use_layer_norm 1`,不受默认改动影响;`test_hiql_gc_value.py:229 test_train_gc_value_loss_mode_default_equiv_and_hiql_differs` 测的是**底层函数** `train_gc_value` 的默认(显式传 `shared_min`),不受 argparse 改动影响。

参考 spec: `docs/superpowers/specs/2026-06-10-hiql-defaults-align-design.md`。

---

### Task 1: `train_hiql_gc_value.py` 默认对齐(geometric / LN=1 / hiql)

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`(新增 1 个测试函数,追加到文件末尾)
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py:59-66`(三个 `default=` + help 文字)

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_hiql_gc_value.py` 末尾:

```python
def test_gc_value_parser_defaults_aligned_to_hiql():
    """对齐 HIQL:不带 flag 时 CLI 默认 = geometric / LN 开 / hiql 口径;旧选项仍可显式回退。"""
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser
    req = ["--hdf5", "x.hdf5", "--dataset", "some/ds", "--stage_cache", "s.npz"]
    a = build_parser().parse_args(req)
    assert a.goal_future_mode == "geometric"
    assert a.use_layer_norm == 1
    assert a.value_loss_mode == "hiql"
    # 旧口径仍可显式回退
    b = build_parser().parse_args(req + ["--goal_future_mode", "stage_entry",
                                         "--use_layer_norm", "0",
                                         "--value_loss_mode", "shared_min"])
    assert b.goal_future_mode == "stage_entry"
    assert b.use_layer_norm == 0
    assert b.value_loss_mode == "shared_min"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_gc_value_parser_defaults_aligned_to_hiql -v`
Expected: FAIL(`assert 'stage_entry' == 'geometric'` 等,当前默认仍是旧值)

- [ ] **Step 3: 改 argparse 默认**

`train_hiql_gc_value.py` 把第 59-66 行三个 argument 整体替换为:

```python
    p.add_argument("--goal_future_mode", choices=["stage_entry", "geometric"], default="geometric",
                   help="未来目标采样:geometric(默认,HIQL 口径,几何分布取任意未来帧,覆盖中间态、填洞)| "
                        "stage_entry(旧口径,只锚 stage 入口)")
    p.add_argument("--use_layer_norm", type=int, choices=[0, 1], default=1,
                   help="value/rep 用 LN+GELU(1,默认,对齐 HIQL LayerNormMLP)| 裸 ReLU(0,旧口径)")
    p.add_argument("--value_loss_mode", choices=["shared_min", "hiql"], default="hiql",
                   help="value 损失:hiql(默认,对齐参考,per-critic 目标不取 min + adv 门控两参 expectile)| "
                        "shared_min(旧口径,两 critic 钉 min 共享目标 + 残差 expectile)")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -v`
Expected: PASS(新测试 + 该文件全部既有测试均绿)

- [ ] **Step 5: 提交**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git -C /mnt/mnt/data/resfit commit -m "feat: train_hiql_gc_value 默认对齐 HIQL(geom/LN/hiql)

argparse default: goal_future_mode→geometric, use_layer_norm→1, value_loss_mode→hiql。
底层 train_gc_value 签名默认不动(回归测试锚定)。旧选项可显式回退。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `train_hiql_high_actor.py` 默认对齐(clamp_to_goal / p_randomgoal=0.3)

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`(新增 1 个测试函数,追加到文件末尾)
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py:39-43`(两个 `default=` + help 文字)

注意:`target_mode` 与 `high_p_randomgoal` 必须**一起**改。`hiql_high_actor.py` 有 fail-fast 断言"fixed_waypoint 下 high_p_randomgoal 必须为 0"(见 `tests/test_hiql_high_actor.py:114`),只改其一会触发它;clamp_to_goal + 0.3 是合法组合。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_hiql_high_actor.py` 末尾:

```python
def test_high_actor_parser_defaults_aligned_to_hiql():
    """对齐 HIQL:不带 flag 时 CLI 默认 = clamp_to_goal / high_p_randomgoal=0.3;旧选项仍可显式回退。"""
    from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser
    req = ["--hdf5", "x.hdf5", "--dataset", "some/ds", "--stage_cache", "s.npz",
           "--gc_value_ckpt", "v.pt"]
    a = build_parser().parse_args(req)
    assert a.target_mode == "clamp_to_goal"
    assert a.high_p_randomgoal == 0.3
    # 旧口径仍可显式回退(fixed_waypoint 须配 0.0)
    b = build_parser().parse_args(req + ["--target_mode", "fixed_waypoint",
                                         "--high_p_randomgoal", "0.0"])
    assert b.target_mode == "fixed_waypoint"
    assert b.high_p_randomgoal == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_high_actor_parser_defaults_aligned_to_hiql -v`
Expected: FAIL(`assert 'fixed_waypoint' == 'clamp_to_goal'`)

- [ ] **Step 3: 改 argparse 默认**

`train_hiql_high_actor.py` 把第 39-43 行两个 argument 整体替换为:

```python
    p.add_argument("--target_mode", choices=["fixed_waypoint", "clamp_to_goal"],
                   default="clamp_to_goal",
                   help="高层 AWR 航点:clamp_to_goal(默认,HIQL,近 goal 塌到 goal)| fixed_waypoint(旧口径,恒 +way)")
    p.add_argument("--high_p_randomgoal", type=float, default=0.3,
                   help="clamp_to_goal 下高层 goal 取 random 的概率(HIQL 默认 0.3;fixed_waypoint 下须为 0)")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py -v`
Expected: PASS(新测试 + 该文件全部既有测试均绿)

- [ ] **Step 5: 提交**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git -C /mnt/mnt/data/resfit commit -m "feat: train_hiql_high_actor 默认对齐 HIQL(clamp_to_goal/p_randomgoal=0.3)

argparse default: target_mode→clamp_to_goal, high_p_randomgoal→0.3(配套,clamp 下合法)。
底层 train_high_actor 签名默认不动。旧选项可显式回退。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `train_chunk_residual.py` renorm 默认开 + 关闭开关

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py`(新增 1 个测试函数,追加到文件末尾)
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:303-305`(改 renorm argument + 新增关闭开关)

互斥写法已实测:`store_true,default=True` + 同-dest `store_false` → `[]`=True、`--no_renorm_subgoal`=False、`--renorm_subgoal`=True。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_hiql_subgoal_wiring.py` 末尾:

```python
def test_cli_renorm_subgoal_default_on():
    """对齐 HIQL eval:不带 flag 时 online z 默认投球面(renorm 默认 True);--no_renorm_subgoal 可关。"""
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    assert build_parser().parse_args([]).renorm_subgoal is True
    assert build_parser().parse_args(["--no_renorm_subgoal"]).renorm_subgoal is False
    assert build_parser().parse_args(["--renorm_subgoal"]).renorm_subgoal is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_cli_renorm_subgoal_default_on -v`
Expected: FAIL(`assert False is True`,当前默认 False)

- [ ] **Step 3: 改 argparse(默认开 + 关闭开关)**

`train_chunk_residual.py` 把第 303-305 行的 `--renorm_subgoal` argument 整体替换为:

```python
    p.add_argument("--renorm_subgoal", dest="renorm_subgoal", action="store_true", default=True,
                   help="online z 投回半径 sqrt(rep_dim) 的球面,对齐 offline φ 与 HIQL eval(evaluation.py:113-114);"
                        "只动模长不动方向。默认开=对齐 HIQL(实测偏差仅 1-2%,clamp 版尾部 ±10-20%)")
    p.add_argument("--no_renorm_subgoal", dest="renorm_subgoal", action="store_false",
                   help="关闭 online z 投球面(回到旧版逐位行为)")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py -v`
Expected: PASS(新测试 + 该文件全部既有测试均绿)

- [ ] **Step 5: 提交**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py
git -C /mnt/mnt/data/resfit commit -m "feat: train_chunk_residual renorm_subgoal 默认开(对齐 HIQL eval)

renorm_subgoal default False→True(同-dest 互斥 flag);新增 --no_renorm_subgoal 关闭。
行为变更:不写 flag 从'关'变'开'。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 全量回归 + 调用点核查(验证 gate)

**Files:** 无代码改动(纯验证)。

- [ ] **Step 1: 跑全 chunk_residual 测试套件**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests -q`
Expected: 全绿,失败数=0(确认 3 个新测试通过、且改 argparse 默认未破坏任何既有测试,尤其 `test_hiql_gc_value.py`/`test_hiql_high_actor.py`/`test_hiql_subgoal_wiring.py` 的底层函数测试)

- [ ] **Step 2: 核查 provenance 回退仍工作(旧 ckpt 兼容)**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_gc_value_save_load_value_loss_mode -v`
Expected: PASS(旧档无 `value_loss_mode` 键时 load 回退 `shared_min`,不受默认改动影响)

- [ ] **Step 3: 核查显式调用点不受默认漂移影响**

Run: `cd /mnt/mnt/data/resfit && grep -nE 'goal_future_mode|use_layer_norm|value_loss_mode|target_mode|high_p_randomgoal|renorm_subgoal' run_hiql_align_ab.sh`
Expected: 看到 `--target_mode clamp_to_goal`、`--goal_future_mode geometric --use_layer_norm 1` 均为显式传值。确认:A/B 脚本显式传 flag,行为不随默认改动而变(若发现有依赖旧默认的隐式调用,在此记录并报告,不擅自改 A/B 脚本)。

- [ ] **Step 4: 无新代码改动,无需提交**

本 task 仅验证。若 Step 3 发现需更新的调用点/文档,另起 commit 并在报告中说明。

---

## 交付后:让对齐默认真正生效(用户本机跑,非本 plan 任务)

改 argparse 默认只影响"以后不带 flag 重新训练"的产物。现有 canonical 产物(`three_piece_gc_value_geom.pt` / `three_piece_high_actor_geom_clamp.pt`)是 geom+clamp 但 shared_min + p_randomgoal=0 + 不 LN,**不含**本次 LN/hiql/p_randomgoal 折中项。要用上对齐默认需级联重训(EGL 回放重活,新产物用新文件名,旧的保留对照),前置 `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8`:

1. 重训 value(不带 flag = geom+LN+hiql):
```bash
cd /mnt/mnt/data/resfit
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /mnt/mnt/data/envs/residual/bin/python \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --output outputs_chunk/three_piece_gc_value_aligned.pt
```

2. 级联重训 high_actor(挂 aligned value,不带 flag = clamp+p_randomgoal0.3):
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /mnt/mnt/data/envs/residual/bin/python \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --state30_cache outputs_chunk/three_piece_state30.npz \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_aligned.pt \
  --output outputs_chunk/three_piece_high_actor_aligned.pt
```

3. 主训练挂两 aligned `.pt`(renorm 默认已开;主训练用 `--gc_value_ckpt outputs_chunk/three_piece_gc_value_aligned.pt --high_actor_ckpt outputs_chunk/three_piece_high_actor_aligned.pt`)。

4. 重训后 gate(可选,验对齐项是否退步;verify 脚本 `--hdf5`/`--dataset` 默认已指向 three_piece):
```bash
# value gate:重点看 LN 是否让 V(s,g=s) 自指负尾退步(spec §8 风险)、hiql 模式双 critic 分化度
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /mnt/mnt/data/envs/residual/bin/python \
  -m resfit.rl_finetuning.chunk_residual.verify_gc_value \
  --pt outputs_chunk/three_piece_gc_value_aligned.pt

# high_actor gate:近 goal 塌缩诊断(对齐 clamp_to_goal)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /mnt/mnt/data/envs/residual/bin/python \
  -m resfit.rl_finetuning.chunk_residual.verify_high_actor \
  --pt outputs_chunk/three_piece_high_actor_aligned.pt --goal_mode near
```
若 V(s,g=s) 明显退步,旧选项可一键回退(重训时显式传 `--use_layer_norm 0` 等)。
