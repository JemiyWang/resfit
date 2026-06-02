# ThreePiece stage 4→5 拆细 + staged 稠密奖励 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 chunk-RL 的 critic 一个按 stage 推进的稠密奖励(收集时 additive 加分,flag 门控),并把瓶颈段 stage2→3 拆成两段,验证能否解除"critic 没信号"导致的负结果。

**Architecture:** detector 由 4 段扩到 5 段(插入"piece2 抓起"子阶段);chunk wrapper 在 step 内按本 chunk 的 stage 推进量(max_in_chunk − 入 chunk 时的 stage)给 additive bonus,折进 chunk reward 后入 buffer(MultiStepTransform 自动烤进 n-step);训练 env bonus 由 CLI 控,eval env 恒 0 不污染指标。全部改动在 chunk_residual/ 自有文件,零新增原仓库改动。

**Tech Stack:** Python / PyTorch / torchrl / pytest;conda env `residual`;从仓库根目录 `/data2/RL/residual-offpolicy-rl` 跑;分支 `chunk-residual-validation`。

> **关于 commit**:用户自己管 commit。下方每个 Task 末尾的 commit step 是建议节奏(均在 feature 分支 `chunk-residual-validation` 上);执行时若用户未明确要求提交,可跳过 commit step,改为人工检查点。

---

## 约定(所有命令)
- 工作目录:仓库根 `/data2/RL/residual-offpolicy-rl`(editable 安装,import 路径依赖此根)。
- 单测:`conda run -n residual python -m pytest <path> -v`
- 真环境/训练:前缀 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp`(根盘 99% 满,TMPDIR 指 /data2)。

## File Structure
| 文件 | 责任 | 动作 |
|------|------|------|
| `resfit/rl_finetuning/chunk_residual/stage_detectors.py` | 5 段 threepiece detector + NUM_STAGES | Modify |
| `resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py` | detector 契约(更新到 5 段)+ wrapper staged 集成 | Modify |
| `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py` | `staged_bonus` 纯函数 + 构造参数 + step 加分 | Modify |
| `resfit/rl_finetuning/chunk_residual/tests/test_staged_reward.py` | staged_bonus 纯函数单测 | Create |
| `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` | CLI flag + wrapper 构造接线 | Modify |
| `resfit/rl_finetuning/chunk_residual/verify_stage5_instant.py` | 真环境探针:确认 env.piece_2 & 5 段分布 | Create |

---

### Task 1: stage_detectors 4 段 → 5 段

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/stage_detectors.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py`

- [ ] **Step 1: 改测试到新契约 + 加新用例(失败优先)**

在 `tests/test_stage_detector.py` 顶部 import 补 `NUM_STAGES`:
```python
from resfit.rl_finetuning.chunk_residual.stage_detectors import (
    NUM_STAGES,
    get_stage_detector,
    threepiece_stage,
)
```

把 piece fake 与两个 fake env 扩成能区分 piece_1 / piece_2 抓取。替换现有
`_FakePiece` / `_FakeTPEnv` / `_RealisticTPEnv` 三个类(及其 `_check_grasp`)为:
```python
class _FakePiece:                 # piece_1
    contact_geoms = ["p1g0", "p1g1"]


class _FakePiece2:                # piece_2
    contact_geoms = ["p2g0", "p2g1"]


def _is_piece2(object_geoms):
    return "p2g0" in object_geoms


class _FakeTPEnv:
    """list-gripper 变体(非 dict);grasp=piece1, grasp2=piece2。"""
    def __init__(self, second=False, first=False, grasp=False, grasp2=False):
        self._second, self._first = second, first
        self._grasp, self._grasp2 = grasp, grasp2
        self.robots = [_FakeRobot()]
        self.piece_1 = _FakePiece()
        self.piece_2 = _FakePiece2()

    def _check_second_piece_is_assembled(self):
        return self._second

    def _check_first_piece_is_assembled(self):
        return self._first

    def _check_grasp(self, gripper, object_geoms):
        return self._grasp2 if _is_piece2(object_geoms) else self._grasp


class _RealisticTPEnv:
    """真环境:gripper=dict(迭代得 str 键);_check_grasp 收到 str 键恒 False。"""
    def __init__(self, second=False, first=False, grasp=False, grasp2=False):
        self._second, self._first = second, first
        self._grasp, self._grasp2 = grasp, grasp2
        self.robots = [_DictGripperRobot(), _DictGripperRobot()]
        self.piece_1 = _FakePiece()
        self.piece_2 = _FakePiece2()

    def _check_second_piece_is_assembled(self):
        return self._second

    def _check_first_piece_is_assembled(self):
        return self._first

    def _check_grasp(self, gripper, object_geoms):
        if isinstance(gripper, str):
            return False
        return self._grasp2 if _is_piece2(object_geoms) else self._grasp
```

更新既有断言到新编号 + 加新用例(success 现在是 4,piece2 抓起是 3):
```python
def test_threepiece_stage_success_is_4():
    assert threepiece_stage(_FakeTPEnv(second=True)) == 4


def test_threepiece_stage_priority_success_over_lower():
    assert threepiece_stage(_FakeTPEnv(second=True, first=True, grasp=True)) == 4


def test_threepiece_stage_piece2_grasped_is_3():
    # piece1 已装好 + 正握 piece2 → 3
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=False, grasp2=True)) == 3


def test_threepiece_stage_assembled_and_released_is_2():
    # 靠近 base 且都没握(piece2 也没抓)→ 2
    assert threepiece_stage(_RealisticTPEnv(first=True, grasp=False, grasp2=False)) == 2


def test_num_stages_threepiece_is_5():
    assert NUM_STAGES["TwoArmThreePieceAssembly"] == 5
```
（删除旧的 `test_threepiece_stage_success_is_3`;旧
`test_threepiece_stage_assembled_and_released_is_2` 用 `_RealisticTPEnv(first=True, grasp=False)`,
grasp2 默认 False,语义不变,可保留或并入上面那条。其余
`first_assembled_is_2 / grasped_is_1 / start_is_0 / grasped_with_dict_gripper_is_1 /
assembled_but_still_grasped_is_1` 不动。)

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -v`
Expected: FAIL —— `test_threepiece_stage_success_is_4` / `_piece2_grasped_is_3` / `_num_stages_threepiece_is_5` 失败(当前 detector 仍返回 3 段、NUM_STAGES=4)。

- [ ] **Step 3: 实现 5 段 detector**

在 `stage_detectors.py` 把 `threepiece_stage` 与 `NUM_STAGES` 改为:
```python
def threepiece_stage(env) -> int:
    """TwoArmThreePieceAssembly 5 段:
    0 起步 / 1 piece1抓 / 2 piece1放好释放 / 3 piece2抓起 / 4 成功。

    瓶颈段 stage2→3 拆细:在"piece1 装好释放"与"成功"之间插入"已抓起 piece2"子阶段,
    让 staged 稠密奖励在该大段内有梯度化信用。闩锁(max-so-far)由 wrapper 负责。
    """
    if env._check_second_piece_is_assembled():    # == _check_success
        return 4
    if env._check_first_piece_is_assembled() and _grasped(env, env.piece_2):
        return 3
    grasped_piece1 = _grasped(env, env.piece_1)
    if env._check_first_piece_is_assembled() and not grasped_piece1:
        return 2
    if grasped_piece1:
        return 1
    return 0


STAGE_DETECTORS = {
    "TwoArmThreePieceAssembly": threepiece_stage,
}
NUM_STAGES = {
    "TwoArmThreePieceAssembly": 5,
}
```
(`_grasped` 已对 dict-gripper 取 `.values()`,piece_2 复用同实现。)

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -v`
Expected: PASS（全部,含改后既有用例与 3 个新用例）。

- [ ] **Step 5: 建议 commit**

```bash
git -C /data2/RL/residual-offpolicy-rl add \
  resfit/rl_finetuning/chunk_residual/stage_detectors.py \
  resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py
git -C /data2/RL/residual-offpolicy-rl commit -m "feat(stage): ThreePiece detector 4→5 段(插入 piece2 抓起子阶段)"
```

---

### Task 2: chunk_env_wrapper staged 稠密奖励

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_staged_reward.py`
- Test(集成,复用 fake env): `resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py`

- [ ] **Step 1: 写失败的纯函数单测**

新建 `tests/test_staged_reward.py`:
```python
"""staged_bonus 纯函数:按本 chunk 的 stage 推进量给 additive 加分。"""
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import staged_bonus


def test_staged_bonus_zero_delta():
    assert staged_bonus(2, 2, 1.0) == 0.0


def test_staged_bonus_positive_delta():
    assert staged_bonus(0, 2, 1.0) == 2.0


def test_staged_bonus_uses_magnitude():
    assert staged_bonus(1, 3, 0.5) == 1.0


def test_staged_bonus_negative_delta_clamped():
    assert staged_bonus(3, 1, 1.0) == 0.0


def test_staged_bonus_zero_bonus_disabled():
    assert staged_bonus(0, 4, 0.0) == 0.0
```

- [ ] **Step 2: 写失败的 wrapper 集成测试**

在 `tests/test_stage_detector.py` 末尾追加(复用同文件已有的 `_StageVecEnv` / `_mk_wrapper`)。
先把 `_mk_wrapper` 加一个可选 bonus 参数:
```python
def _mk_wrapper(env, bonus=0.0):
    return ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                   chunk_length=L, stage_reward_bonus=bonus)
```
再追加 3 条:
```python
def test_wrapper_staged_reward_adds_bonus():
    # stage 0→2(峰值 2),bonus=1.0 → 奖励含 +2.0(env 基础奖励为 0)
    env = _StageVecEnv([0, 1, 2, 1, 0])
    w = _mk_wrapper(env, bonus=1.0)
    w.reset()
    _, reward, _, _, info = w.step(torch.zeros(1, L * D))
    assert info["max_stage_in_chunk"] == 2
    assert float(reward[0]) == 2.0


def test_wrapper_no_bonus_is_regression():
    # bonus=0.0 → 与现状逐位相同(奖励仍为 env 的 0.0)
    env = _StageVecEnv([0, 1, 2, 1, 0])
    w = _mk_wrapper(env, bonus=0.0)
    w.reset()
    _, reward, _, _, _ = w.step(torch.zeros(1, L * D))
    assert float(reward[0]) == 0.0


def test_wrapper_staged_reward_done_midchunk_uses_peak():
    # 第 3 步 terminate,期间到过 stage 2 → 用归零前峰值算 Δ=2 → +2.0
    env = _StageVecEnv([1, 2, 2, 0, 0], term_at=3)
    w = _mk_wrapper(env, bonus=1.0)
    w.reset()
    _, reward, term, _, info = w.step(torch.zeros(1, L * D))
    assert term.item() is True
    assert info["max_stage_in_chunk"] == 2
    assert float(reward[0]) == 2.0
```

- [ ] **Step 3: 跑测试确认失败**

Run:
```
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_staged_reward.py \
  resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -v
```
Expected: FAIL —— `ImportError: cannot import name 'staged_bonus'`;且 `_mk_wrapper` 传
`stage_reward_bonus=` 时 `TypeError`(构造器还没这个参数)。

- [ ] **Step 4: 实现 staged_bonus + 构造参数 + step 加分**

在 `chunk_env_wrapper.py` 顶部(class 之前)加纯函数:
```python
def staged_bonus(start_stage: int, end_stage: int, bonus: float) -> float:
    """本 chunk 内 stage 推进量的 additive 加分(非负;bonus=0 即关闭)。"""
    return bonus * max(0, int(end_stage) - int(start_stage))
```
构造器签名与字段(在 `__init__` 末尾加):
```python
    def __init__(self, vec_env, base_policy, action_scaler, state_standardizer,
                 chunk_length: int, stage_reward_bonus: float = 0.0):
        ...
        self.stage_reward_bonus = stage_reward_bonus
        self._stage = 0
```
`step()` 内:循环开始处已有 `max_in_chunk = self._stage`,在其同处加捕获入 chunk 起点;
循环结束后、done-reset 之前,把 bonus 折进 `total_reward`:
```python
        start_stage = self._stage                        # 入 chunk 时的闩锁值(= max_in_chunk 初值)
        max_in_chunk = self._stage
        for t in range(self.chunk_length):
            ...
        total_reward = total_reward + staged_bonus(start_stage, max_in_chunk,
                                                    self.stage_reward_bonus)
        if bool((terminated | truncated).any()):
            self.base_policy.reset()
            self._stage = 0
        ...
```
(bonus 用 `max_in_chunk` 峰值,与 `self._stage` 是否被 reset 无关,故放 reset 前后都对;放 reset 前更醒目。)

- [ ] **Step 5: 跑测试确认通过**

Run:
```
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_staged_reward.py \
  resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -v
```
Expected: PASS（含 5 条纯函数 + 3 条 wrapper 集成,且既有 wrapper 闩锁/归零用例仍绿）。

- [ ] **Step 6: 全量回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -v`
Expected: PASS（全部;原 35 个 stage 相关 + 本轮新增,无回归）。

- [ ] **Step 7: 建议 commit**

```bash
git -C /data2/RL/residual-offpolicy-rl add \
  resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py \
  resfit/rl_finetuning/chunk_residual/tests/test_staged_reward.py \
  resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py
git -C /data2/RL/residual-offpolicy-rl commit -m "feat(reward): chunk wrapper 收集时 staged 稠密奖励(bonus*Δstage,flag 门控,eval 不污染)"
```

---

### Task 3: train_chunk_residual CLI 接线

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`

- [ ] **Step 1: 加 CLI 参数**

在 `main()` 的 argparse 段(`--stage_balanced` 附近)加:
```python
    p.add_argument("--staged_reward", action="store_true",
                   help="开启 collection-time staged 稠密奖励(按 stage 推进加分)")
    p.add_argument("--stage_reward_bonus", type=float, default=1.0,
                   help="每推进一个 stage 的 additive 加分量(仅 --staged_reward 时生效)")
```

- [ ] **Step 2: 训练 env 接 bonus,eval env 恒 0**

把训练 env 与 eval env 的构造改为:
```python
    train_bonus = args.stage_reward_bonus if args.staged_reward else 0.0
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length,
                                  stage_reward_bonus=train_bonus)
    eval_vec = create_vectorized_env(env_name=args.task, num_envs=args.eval_num_envs,
                                     device=args.device)
    eval_env = ChunkResidualEnvWrapper(eval_vec, base_policy, action_scaler, state_standardizer,
                                       chunk_length=args.chunk_length,
                                       stage_reward_bonus=0.0)   # eval 不加 shaping,指标纯净
```

- [ ] **Step 3: smoke 验证(staged 路径不崩)**

Run（`<gpu>` 换空闲卡;`<base_dir>` = 本地 ACT 基座目录,见下方"交付与长跑"）:
```
CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --base_wandb_id bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999 \
  --actor raw --staged_reward --stage_reward_bonus 1.0 \
  --smoke --eval_num_envs 2 --eval_num_episodes 2
```
Expected: 跑完不崩(打印 `done. best success_rate=...`),证明 staged 奖励路径 + 5 段 stage 端到端通。

- [ ] **Step 4: 建议 commit**

```bash
git -C /data2/RL/residual-offpolicy-rl add \
  resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git -C /data2/RL/residual-offpolicy-rl commit -m "feat(train): --staged_reward/--stage_reward_bonus 接线;eval env bonus=0"
```

---

### Task 4: 真环境探针 verify_stage5_instant.py(确认 env.piece_2 & 5 段分布)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/verify_stage5_instant.py`

> 唯一真·未知:`env.piece_2` 属性名(detector 此前只用过 `env.piece_1`)。`_grasped` 对任何异常
> 返回 False,故属性名错时 **不崩**,而是 stage 3 永不出现。探针用一个反证逻辑兜住:
> 若观测到 stage 4(成功)却从未见 stage 3,则 piece_2 属性名几乎必错(不抓 piece2 不可能成功)。

- [ ] **Step 1: 写探针脚本**

新建 `verify_stage5_instant.py`:
```python
"""真环境探针:base(零残差)在 ThreePiece 上的 5 段瞬时分布,确认 stage 拆细生效。

回答两件事:
  1. env.piece_2 属性名是否正确 —— stage 3(piece2 抓起)是否真出现。
  2. 5 段 stage 的瞬时出现频次,目测 stage2→3 拆细给瓶颈段带来分辨率。
反证:若见过 stage 4(成功)却从未见 stage 3 → piece_2 属性名几乎必错,需修 detector。

用法:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
  python -m resfit.rl_finetuning.chunk_residual.verify_stage5_instant \
    --base_dir bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_base_policy
from resfit.rl_finetuning.chunk_residual.stage_detectors import NUM_STAGES
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="TwoArmThreePieceAssembly")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-three-piece-assembly")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--max_chunks", type=int, default=700)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    meta = LeRobotDataset(args.dataset).meta
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device=args.device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=args.device)

    base_policy = build_base_policy(args.base_dir, args.device)
    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device)
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length)
    env.reset()
    zero = torch.zeros(1, env.action_dim * args.chunk_length, device=args.device)

    seen = Counter()        # 每个 chunk 收尾的 max_in_chunk 计数
    for _ in range(args.max_chunks):
        _, reward, term, trunc, info = env.step(zero)
        seen[int(info.get("max_stage_in_chunk", 0))] += 1
        if bool((term | trunc).any()):
            env.reset()

    print(f"\n{'#'*60}")
    print(f"[NUM_STAGES] {NUM_STAGES[args.task]}(期望 5)")
    print(f"[chunk 级 max_stage 频次] {dict(sorted(seen.items()))}")
    saw3, saw4 = seen.get(3, 0), seen.get(4, 0)
    print(f"[stage 3(piece2 抓起)出现] {saw3} 次")
    print(f"[stage 4(成功)出现] {saw4} 次")
    if saw3 == 0 and saw4 > 0:
        print("[!! 反证] 见过成功却从未见 stage3 → env.piece_2 属性名几乎必错,需修 detector")
    elif saw3 > 0:
        print("[OK] stage3 真触发 → piece_2 属性名正确、stage2→3 拆细生效")
    else:
        print("[?] stage3/4 都没出现(可能 base 太弱/chunk 数太少)→ 加 --max_chunks 再看")
    vec_env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑探针**

Run（`<gpu>` 换空闲卡）:
```
CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.verify_stage5_instant \
  --base_dir bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999
```
Expected: 打印 `[OK] stage3 真触发 ...`,且 max_stage 频次里 3 出现可观。
- 若打印 `[!! 反证]`:`env.piece_2` 属性名错。回到 `stage_detectors.py`,确认真实属性名
  (候选:`env.piece_2` / `env.pieces[1]` / robosuite obj 注册名),改 detector 后重跑本探针,
  并在 spec/handoff 记一笔。

- [ ] **Step 3: 建议 commit**

```bash
git -C /data2/RL/residual-offpolicy-rl add \
  resfit/rl_finetuning/chunk_residual/verify_stage5_instant.py
git -C /data2/RL/residual-offpolicy-rl commit -m "test: 真环境探针确认 ThreePiece 5 段 stage(env.piece_2 反证)"
```

---

## 交付与长跑(交命令给用户)

代码 + 全绿单测 + 探针(stage3 真触发)+ smoke 通过后,交付下面这条 300k 命令给用户启动长跑:
```
CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --base_wandb_id bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999 \
  --actor raw --stage_balanced --staged_reward --stage_reward_bonus 1.0 \
  --total_env_steps 300000 --seed 0 \
  2>&1 | tee outputs_chunk/threepiece_staged_balanced.log
```
（建议 `tmux` 后台跑防断线。）

**验证判据(用户跑完后看):**
1. critic_loss 不再恒 0(对照负结果里原版单步 `critic_loss≡0.0000`)。
2. `[stage-diag]` 的 `target_q/stageK` 出现 stage 结构(不再全 ≈0)。
3. eval success_rate 能否超过基座开环 ~20-30%。

任一不成立都是有价值结果;若 success 上不去且诊断显示策略被 shaping 带偏,下一步换
potential-based shaping 或上 advantage-gating 止损(handoff item 3)。

---

## Self-Review(已对 spec 核对)
- **Spec 覆盖**:4.1 detector→Task1;4.2 wrapper staged→Task2;4.3 CLI→Task3;§8 探针→Task4;
  §8 ready 命令→"交付与长跑";§7 测试三类(detector / 纯函数 / wrapper 集成 + 回归)分布在
  Task1 Step1、Task2 Step1-2、Task2 Step6。
- **占位扫描**:无 TBD/TODO;每个 code step 给了完整代码与确切命令/期望。
- **类型/命名一致**:`staged_bonus(start_stage, end_stage, bonus)` 在 Task2 定义、wrapper step
  与纯函数测试同名同签名;构造参数 `stage_reward_bonus` 在 wrapper 构造、`_mk_wrapper`、
  train 接线三处一致;CLI `--staged_reward` / `--stage_reward_bonus` 与 train 内引用一致;
  `NUM_STAGES` 在 detector 与探针/测试 import 一致。
- **真·未知**:env.piece_2 属性名,Task4 探针用反证兜住并给修正路径。
