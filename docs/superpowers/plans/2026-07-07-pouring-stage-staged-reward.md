# Pouring stage 检测器 + staged 稠密奖励实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `TwoArmPouring`(GR1)加一个 5 段 stage 检测器,并起一个 staged 稠密奖励实验(逐字对齐 pothiql 参照实验 rerun0703,仅 pothiql→staged)。

**Architecture:** 检测器极薄,复用 env 自带谓词(`_check_success`/`check_contact`/`_check_grasp`),只判瞬时阶段;闩锁由 wrapper(在线)/ offline_hdf5_buffer(离线)的 running-max 负责。注册到 `STAGE_DETECTORS`/`NUM_STAGES` 后,在线 worker(dexmg.py:353 通用调 `get_stage_detector`)与离线 replay(precompute_stage_cache)两条路自动生效,无需改 dexmg。staged 奖励复用现成 `shaping_reward(mode="staged")`。

**Tech Stack:** Python / pytest / robosuite+dexmimicgen(GR1 env)/ conda env `residual`。

## Global Constraints
- 顺序(1009 条 demo 100% 同序):`cup_grasped(0.26) → ball_in_bowl(0.45) → bowl_grasped(0.78) → bowl_on_pad(0.94)`。
- 5 段:`0 起步 / 1 抓cup / 2 球倒入碗 / 3 抓碗搬向pad / 4 成功`。`NUM_STAGES["TwoArmPouring"] = 5`。
- 顶段用 `env._check_success()`(不用 datagen `bowl_on_pad` 信号,后者仅 84% 成功 demo 触发)。
- **纯追加**:只动 `stage_detectors.py` / `tests/test_stage_detector.py` / `tests/test_run_scripts.py` 的追加,新建 `run_pouring_staged_joint.sh` / `verify_pouring_stages.py`。不动 dexmg.py 及其它原仓库文件。
- 参照实验 run 脚本:`run_pouring_pothiql_joint_rerun0703.sh`;base artifact `run_anw5pphu_best:v2`。
- 不改超参:bonus=1.0 / action_scale 0.05 / joint / sg15 全沿用参照。
- **上 500k 前必须先** smoke 掉 GR1 grasp 可用性 + stages npz 首次生成(Task 3)与 staged 训练路径(Task 4)。

---

### Task 1: `pouring_stage` 检测器 + 注册 + 契约测试

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/stage_detectors.py`(在 `threading_stage` 后追加函数;`STAGE_DETECTORS`/`NUM_STAGES` 各加一条)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py`(文件末尾追加 pouring 契约测试)

**Interfaces:**
- Consumes: 现成 `_grasped(env, obj)`(stage_detectors.py:14,对 dict-gripper 取 `.values()` 兜底)、测试文件已有的 `_DictGripperRobot`/`_GripperObj`(line 99/103)。
- Produces: `pouring_stage(env) -> int`(值域 0..4);`STAGE_DETECTORS["TwoArmPouring"]`;`NUM_STAGES["TwoArmPouring"] == 5`。

- [ ] **Step 1: 写失败的契约测试**

在 `tests/test_stage_detector.py` **文件末尾**追加(import 处补上 `pouring_stage`):
```python
# 在文件顶部 import 块把 stage_detectors 的导入补成:
# from resfit.rl_finetuning.chunk_residual.stage_detectors import (
#     NUM_STAGES, get_stage_detector, threading_stage, threepiece_stage, pouring_stage,
# )

# ---------- pouring 5 段检测器契约 ----------
class _FakeCup:
    contact_geoms = ["cup0", "cup1"]

class _FakeBowl:
    contact_geoms = ["bowl0", "bowl1"]

class _FakeBall:
    contact_geoms = ["ball0"]

def _is_bowl(object_geoms):
    return "bowl0" in object_geoms


class _FakePouringEnv:
    """GR1 dict-gripper fake;grasp_cup/grasp_bowl 独立控制;check_contact 返回 ball_in_bowl。"""
    def __init__(self, success=False, ball_in_bowl=False, grasp_cup=False, grasp_bowl=False):
        self._success = success
        self._bib = ball_in_bowl
        self._gc, self._gb = grasp_cup, grasp_bowl
        self.robots = [_DictGripperRobot()]     # GR1 单机器人,gripper=dict(迭代得 str 键)
        self.cup = _FakeCup()
        self.bowl = _FakeBowl()
        self.ball = _FakeBall()

    def _check_success(self):
        return self._success

    def check_contact(self, a, b):               # 检测器调 check_contact(env.bowl, env.ball)
        return self._bib

    def _check_grasp(self, gripper, object_geoms):
        if isinstance(gripper, str):
            return False
        return self._gb if _is_bowl(object_geoms) else self._gc


def test_pouring_stage_start_is_0():
    assert pouring_stage(_FakePouringEnv()) == 0

def test_pouring_stage_cup_grasped_is_1():
    assert pouring_stage(_FakePouringEnv(grasp_cup=True)) == 1

def test_pouring_stage_ball_in_bowl_is_2():
    assert pouring_stage(_FakePouringEnv(ball_in_bowl=True)) == 2

def test_pouring_stage_ball_in_bowl_and_bowl_grasped_is_3():
    assert pouring_stage(_FakePouringEnv(ball_in_bowl=True, grasp_bowl=True)) == 3

def test_pouring_stage_success_is_4():
    assert pouring_stage(_FakePouringEnv(success=True)) == 4

def test_pouring_stage_priority_success_over_lower():
    # 同时满足 → 高阶段优先 = 成功(4)
    assert pouring_stage(_FakePouringEnv(
        success=True, ball_in_bowl=True, grasp_cup=True, grasp_bowl=True)) == 4

def test_pouring_stage_bowl_grasped_before_pour_is_0():
    # 倒球前就抓碗(球未入碗、cup 未抓)→ 仍 0(stage 3 要求球已入碗;不误触发)
    assert pouring_stage(_FakePouringEnv(grasp_bowl=True)) == 0

def test_pouring_stage_ball_in_bowl_not_success_is_3_when_carrying():
    # 球入碗 + 抓碗但未 success → 3(搬运中)
    assert pouring_stage(_FakePouringEnv(ball_in_bowl=True, grasp_bowl=True, success=False)) == 3

def test_pouring_stage_monotonic_separation():
    # 三相位依次:抓cup(1)→倒入+松cup(2)→抓碗搬运(3),证明 5 段能切开
    seq = [
        _FakePouringEnv(grasp_cup=True),                              # 1
        _FakePouringEnv(ball_in_bowl=True, grasp_cup=False),          # 2
        _FakePouringEnv(ball_in_bowl=True, grasp_bowl=True),          # 3
    ]
    assert [pouring_stage(e) for e in seq] == [1, 2, 3]

def test_pouring_registered():
    assert NUM_STAGES["TwoArmPouring"] == 5
    assert get_stage_detector("TwoArmPouring") is pouring_stage
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -k pouring -q`
Expected: FAIL / ImportError `cannot import name 'pouring_stage'`。

- [ ] **Step 3: 实现 `pouring_stage` + 注册**

在 `stage_detectors.py` 的 `threading_stage` 函数**之后**、`STAGE_DETECTORS` 之前追加:
```python
def pouring_stage(env) -> int:
    """TwoArmPouring 5 段:
    0 起步 / 1 抓起 cup / 2 球倒入碗 / 3 抓起碗搬向 pad / 4 碗放上 pad(成功)。

    顺序由 demo 数据核实(100% 同序):cup_grasped→ball_in_bowl→bowl_grasped→bowl_on_pad。
    顶段用 _check_success(比 datagen bowl_on_pad 信号更可靠,后者仅 84% 成功 demo 触发);
    高段短路优先,闩锁(max-so-far)由 wrapper 负责,这里只判瞬时阶段。
    """
    if env._check_success():                            # 4 成功
        return 4
    ball_in_bowl = env.check_contact(env.bowl, env.ball)
    if ball_in_bowl and _grasped(env, env.bowl):        # 3 球已入碗 + 碗被抓起搬运
        return 3
    if ball_in_bowl:                                    # 2 球已倒入碗
        return 2
    if _grasped(env, env.cup):                          # 1 抓起 cup
        return 1
    return 0
```
在 `STAGE_DETECTORS` 字典加一条:`"TwoArmPouring": pouring_stage,`
在 `NUM_STAGES` 字典加一条:`"TwoArmPouring": 5,`

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -k pouring -q`
Expected: PASS(10 passed)。

- [ ] **Step 5: 全量回归**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全绿(含 piece 4 段现状 + threading + wrapper + `test_shaping_reward.py` 的 none-mode→0.0 已守 §6a)。若 collect 报无关模块错(如 diagnose_subgoal 等未提交脚本),只看本目录 stage/shaping/staged 相关测试全绿即可。

- [ ] **Step 6: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/stage_detectors.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py
git commit -m "feat: pouring 5-stage detector (cup→pour→bowl→pad) + contract tests"
```
> 提交边界见 spec §10:工作树已有未提交的 piece 5→4 段重构。**提交前先与用户确认**是否把 piece 改动单独先提、再叠本 commit。默认征得同意后分提。

---

### Task 2: `run_pouring_staged_joint.sh` + 脚本内容测试

**Files:**
- Create: `run_pouring_staged_joint.sh`(仓库根)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_run_scripts.py`(追加一个 pouring staged 断言)

**Interfaces:**
- Consumes: 参照脚本 `run_pouring_pothiql_joint_rerun0703.sh` 的 env/venv/artifact;Task 1 注册的 `TwoArmPouring` 检测器(训练时按 `--offline_stage_cache` 触发 5 段生成)。
- Produces: 一条 ready 的 500k 命令 `bash run_pouring_staged_joint.sh <gpu>`。

- [ ] **Step 1: 写失败的脚本内容测试**

在 `tests/test_run_scripts.py` 末尾追加:
```python
def test_pouring_staged_joint_script_aligned_with_rerun0703():
    text = Path("run_pouring_staged_joint.sh").read_text()
    # —— staged 差异 ——
    assert "--reward_shaping staged" in text
    assert "--stage_reward_bonus 1.0" in text
    assert "--stage_balanced" in text
    assert "two_arm_pouring_stages.npz" in text
    assert "--potential_source" not in text        # 删掉(staged 无势函数)
    assert "--hiql_value_ckpt" not in text          # staged 不需 value.pt
    assert "_staged_joint_offcache" in text         # 新 offcache(重建)
    assert "pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_joint" in text
    # —— 对齐参照的关键不变量 ——
    for flag in (
        "--task TwoArmPouring",
        "--dataset ankile/dexmg-two-arm-pouring",
        "--offline_dataset_path resfit/dataset/two_arm_pouring.hdf5",
        "--chunk_length 1 --base_action_mode queue --base_n_action_steps 10",
        "--action_scale 0.05 --actor_lr 1e-6 --actor raw",
        "--offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1",
        "--subgoal_conditioned",
        "--gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt",
        "--high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt",
        "--act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz",
        "--subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor",
        "--total_env_steps 500000",
        "run_anw5pphu_best:v2",
    ):
        assert flag in text, flag
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_run_scripts.py -k pouring_staged -q`
Expected: FAIL(FileNotFoundError:脚本不存在)。

- [ ] **Step 3: 创建 `run_pouring_staged_joint.sh`**

写入仓库根 `run_pouring_staged_joint.sh`(逐字对齐 rerun0703,仅 staged 差异):
```bash
#!/usr/bin/env bash
# pouring staged + 联合训练(staged_joint)。
#   逐字对齐 pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0703,仅改:
#     ① reward: potential(单状态 hiql V) → staged(stage 整数净加 bonus;不需要 value.pt)
#     ② 删 --potential_source hiql / --hiql_value_ckpt;加 --stage_reward_bonus 1.0 --stage_balanced
#        --offline_stage_cache outputs_chunk/two_arm_pouring_stages.npz(不存在→首跑 sim-replay 生成 5 段)
#     ③ 新 OFFCACHE(staged 签名→重建)/ 新 wandb_name / output_dir
#   其余 100% 一致:chunk1 / queue / base_n10 / action_scale0.05 / actor_lr1e-6 / actor raw /
#     offline_fraction0.5 / base_policy锚 + bc0.1 / subgoal sg15(gc_value+high_actor+act_feat)/
#     joint online_finetune / 500k。
#   ⚠ 上 500k 前先 smoke 掉 stages npz 在 GR1+hdf5 上的首次生成(见 verify_pouring_stages.py)。
# 用法: setsid bash run_pouring_staged_joint.sh <gpu> > pouring_staged_joint.log 2>&1 < /dev/null &
set -u
GPU=${1:-4}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_staged_joint_offcache   # 新(staged 签名→重建)
STAGES=outputs_chunk/two_arm_pouring_stages.npz
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
gate(){ [ -s "$1" ] || { echo "[pouring_staged_joint] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
[ -e "$STAGES" ] && echo "[pouring_staged_joint] 注意:$STAGES 已存在,将直接读(确认是5段);若想重生成请先删它" \
                  || echo "[pouring_staged_joint] $STAGES 不存在 → 首次将 sim-replay 生成 5 段(GR1,较慢,一次性)"
echo "[pouring_staged_joint] START pouring staged+joint  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 首建)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced \
  --reward_shaping staged --stage_reward_bonus 1.0 \
  --offline_stage_cache "$STAGES" \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_joint \
  --output_dir outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_joint
echo "[pouring_staged_joint] ALL DONE $(date '+%F %T')"
```

- [ ] **Step 4: 跑测试 + bash 语法检查**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_run_scripts.py -k pouring_staged -q && bash -n run_pouring_staged_joint.sh && echo SYNTAX_OK`
Expected: PASS + `SYNTAX_OK`。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit
git add run_pouring_staged_joint.sh resfit/rl_finetuning/chunk_residual/tests/test_run_scripts.py
git commit -m "feat: run_pouring_staged_joint.sh (rerun0703 aligned, pothiql->staged) + script test"
```

---

### Task 3: 离线 replay 验证(GR1 grasp 可用性 + stages npz 生成)—— 交付物验证

> 这是 spec §7.2 的关键验证:**先过这一步才允许上 500k**。跑纯物理 sim-replay(不需 GPU/EGL),
> 用训练同款 `precompute_stage_cache` 在前 N 条 demo 上生成瞬时 stage,看 GR1 双灵巧手上 `_grasped`
> 是否触发(stage 1、3 是否出现)、分桶是否均衡、有没有干净的 0→1→2→3→4。

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/verify_pouring_stages.py`

- [ ] **Step 1: 创建验证脚本**

```python
"""离线 sim-replay 验证 pouring 5 段检测器 + GR1 grasp 可用性 + stages npz 生成。

跑 precompute_stage_cache(与训练 --offline_stage_cache 缺失时同款路径)在前 N 条 demo 上生成瞬时
stage,统计闩锁后分布。回答单测答不了的问题:GR1 双灵巧手上 _grasped 是否触发(stage 1/3 是否出现)。

用法(仓库根, conda env residual, 不需 GPU/EGL,纯物理):
  python -m resfit.rl_finetuning.chunk_residual.verify_pouring_stages --num_demos 20
退出码: 0=五段(0..4)齐现且非退化; 1=有阶段从未出现(grasp 可能没触发→走几何代理退路 Task 3b)。
"""
from __future__ import annotations
import argparse
import os
import tempfile
from collections import Counter

import numpy as np

from resfit.rl_finetuning.chunk_residual.offline_stage_replay import precompute_stage_cache
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import load_stage_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="resfit/dataset/two_arm_pouring.hdf5")
    p.add_argument("--num_demos", type=int, default=20)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "pour_stages_smoke.npz")
        n = precompute_stage_cache(args.dataset, out, num_demos=args.num_demos)
        cache = load_stage_cache(out)

    seen = Counter()
    latched = Counter()
    reached = Counter()
    for _ep, inst in cache.items():
        inst = np.asarray(inst)
        latch = np.maximum.accumulate(inst)
        for v in inst:
            seen[int(v)] += 1
        for v in latch:
            latched[int(v)] += 1
        reached[int(latch[-1])] += 1

    total = sum(latched.values())
    print(f"\n{'#'*60}\n[pouring stages] {n} demos")
    print("  瞬时出现次数:", dict(sorted(seen.items())))
    print("  闩锁后桶占比:", {k: round(latched[k] / total, 3) for k in sorted(latched)})
    print("  每 demo 终到阶段:", dict(sorted(reached.items())))
    all_present = all(seen.get(s, 0) > 0 for s in range(5))
    print(f"  五段(0..4)是否齐现: {all_present}")
    if not all_present:
        missing = [s for s in range(5) if seen.get(s, 0) == 0]
        print(f"  ✗ 缺失阶段 {missing};若缺 1/3(抓取段)→ 大概率 GR1 _grasped 没触发,走 Task 3b 几何代理退路")
    raise SystemExit(0 if all_present else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑验证(交付物;需 residual env + hdf5,CPU 即可)**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual --no-capture-output python -m resfit.rl_finetuning.chunk_residual.verify_pouring_stages --num_demos 20`
Expected(理想):退出码 0;`五段(0..4)是否齐现: True`;闩锁后桶占比接近 `{0:0.26, 1:0.19, 2:0.34, 3:0.17, 4:0.05}`(允许偏差,4 段本就小)。
> **决策门**:退出码 0 且五段齐现 → 检测器在 GR1 上工作,进 Task 4。退出码 1 且缺 stage 1/3 → GR1 `_check_grasp` 没触发,转 **Task 3b**(几何代理退路)。stage 4 若偏小/个别 demo 未达属正常(datagen 里也有 16% demo 的 bowl_on_pad 信号不触发,但 `_check_success` 更宽,通常能达)。

- [ ] **Step 3: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/verify_pouring_stages.py
git commit -m "test: offline replay verifier for pouring 5-stage (GR1 grasp + stages npz)"
```

---

### Task 3b(仅当 Task 3 缺 stage 1/3 时执行):几何 grasp 代理退路

> **触发条件**:Task 3 Step 2 退出码 1 且缺失阶段含 1 或 3 → GR1 灵巧手上 `_grasped`/`_check_grasp`
> 不触发。改用「物体抬离桌面」几何持久代理(与夹爪无关,GR1/Panda 通用)。若 Task 3 已过,**跳过本任务**。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/stage_detectors.py`(加 `_lifted` 辅助 + 改 `pouring_stage` 的抓取分支)
- Test: `tests/test_stage_detector.py`(pouring fake env 增加 z 位置字段 + 用例)

- [ ] **Step 1: 加 `_lifted` 辅助 + 用它替换 grasp 判据**

在 `stage_detectors.py` 加(表高 `env.table_offset[2]`≈0.9,阈值 3cm):
```python
def _lifted(env, body_key: str, margin: float = 0.03) -> bool:
    """物体质心 z 抬离桌面 margin 以上(持久几何里程碑,不依赖夹爪;GR1/Panda 通用)。"""
    try:
        z = float(env.sim.data.body_xpos[env.obj_body_id[body_key]][2])
        return z > float(env.table_offset[2]) + margin
    except Exception:
        return False
```
把 `pouring_stage` 里 `_grasped(env, env.bowl)` 换成 `_lifted(env, "bowl")`、`_grasped(env, env.cup)` 换成 `_lifted(env, "cup")`(其余不变)。

- [ ] **Step 2: 更新契约测试的 fake env**

给 `_FakePouringEnv` 加最小 sim 桩,使 `_lifted` 可测(不改已有语义:`grasp_cup`→cup 抬起、`grasp_bowl`→bowl 抬起):
```python
class _FakeSim:
    def __init__(self, cup_z, bowl_z):
        class _D: pass
        self.data = _D()
        self.data.body_xpos = {"cup_bid": [0, 0, cup_z], "bowl_bid": [0, 0, bowl_z]}
# 在 _FakePouringEnv.__init__ 末尾加:
#   self.table_offset = [0, 0, 0.9]
#   self.obj_body_id = {"cup": "cup_bid", "bowl": "bowl_bid"}
#   self.sim = _FakeSim(cup_z=0.95 if grasp_cup else 0.9, bowl_z=0.95 if grasp_bowl else 0.9)
```
既有 pouring 用例语义不变(grasp_cup/grasp_bowl 仍驱动 stage 1/3)。

- [ ] **Step 3: 跑测试 + 重跑 Task 3 验证**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -k pouring -q && conda run -n residual --no-capture-output python -m resfit.rl_finetuning.chunk_residual.verify_pouring_stages --num_demos 20`
Expected: 单测全绿;验证退出码 0,五段齐现(阈值 margin 若不准,按 Task 3 打印的 `瞬时出现次数` 微调 0.02–0.05)。

- [ ] **Step 4: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/stage_detectors.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py
git commit -m "fix: pouring stage grasp milestones via geometric lifted proxy (GR1-agnostic)"
```

---

### Task 4: smoke staged 训练路径 + ready 500k 命令 —— 交付物验证(GPU,需用户授权)

> spec §7.3:上 500k 前 smoke 掉 staged 训练路径不崩 + stages npz 首次生成跑通。**需 GPU + EGL +
> base artifact,按本机惯例先取得用户授权/空闲卡再跑。**

- [ ] **Step 1: 确认复用产物在位**

Run: `cd /mnt/mnt/data/resfit && for f in outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt outputs_chunk/pouring_act_feat_hdf5.npz /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2/policy/model.safetensors; do [ -s "$f" ] && echo "OK $f" || echo "MISSING $f"; done`
Expected: 全部 `OK`。有 MISSING → 先补产物(gc_value/high_actor/act_feat 与参照实验同源,base artifact 同 rerun0703),不改本 plan。

- [ ] **Step 2: smoke 跑 staged 路径(短步数,验不崩 + stages npz 生成)**

在空闲卡 `<gpu>` 上(向用户确认):
Run: `cd /mnt/mnt/data/resfit && CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 PYTHONPATH=/mnt/mnt/data/resfit conda run -n residual --no-capture-output python -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual --task TwoArmPouring --base_wandb_id "/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2" --dataset ankile/dexmg-two-arm-pouring --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping staged --stage_reward_bonus 1.0 --offline_stage_cache outputs_chunk/two_arm_pouring_stages.npz --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 --offline_buffer_cache /tmp/pour_staged_smoke_offcache --subgoal_conditioned --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor --smoke 2>&1 | tail -40`
Expected: 不崩;日志出现 stages npz 生成(首次 sim-replay)或读取;staged reward 生效(critic 更新有信号)。
> 注:`--smoke` 用短步数;`two_arm_pouring_stages.npz` 首次会 sim-replay 全量 1009 条(GR1,较慢,一次性,之后复用)。smoke 若只想快验路径可先用 verify(Task 3)产出的小样本确认逻辑,再让正式 run 全量生成。`--offline_buffer_cache` 用 `/tmp` 临时路径避免污染正式 offcache。

- [ ] **Step 3: 交付 ready 500k 命令(不自动起长跑)**

向用户交付:`bash run_pouring_staged_joint.sh <free_gpu>`(500k;offcache 首建 ~80min/~41G;A/B 对标 `..._pothiql_joint_rerun0703`)。长跑由用户在空闲卡启动。

---

## Self-Review

**1. Spec coverage:**
- spec §4.1 检测器 + 注册 → Task 1 ✓
- spec §4.2 GR1 grasp 风险 + 退路 → Task 3(验证)+ Task 3b(退路)✓
- spec §4.3 契约测试 → Task 1 Step 1 ✓
- spec §4.4 run 脚本 → Task 2 ✓
- spec §5 数据流 → Task 1 注册后在线/离线自动生效(dexmg.py:353 通用)✓
- spec §6 回归安全(none-mode→0)→ Task 1 Step 5 全量回归(`test_shaping_reward.py:17-18` 已守)✓
- spec §7 交付边界(代码+单测+真env验证+smoke+ready 命令)→ Task 1–4 ✓
- spec §10 提交边界 → Task 1 Step 6 注记 ✓

**2. Placeholder scan:** 无 TBD/TODO;所有 step 有具体代码/命令。Task 3b 是条件任务但代码完整(触发条件明确)。

**3. Type consistency:** `pouring_stage(env)->int` 值域 0..4 在 Task 1 定义,Task 2/3/4 一致引用;`precompute_stage_cache(dataset_path,out_path,num_demos)` / `load_stage_cache(path)` 签名与实现一致;fake env 字段(`_check_success`/`check_contact`/`_check_grasp`/`cup`/`bowl`/`ball`/`robots`)与 `_grasped` 消费一致。
