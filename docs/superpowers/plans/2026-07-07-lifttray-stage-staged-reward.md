# LiftTray stage 检测器 + staged 奖励实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `TwoArmLiftTray` 加 4 段 stage 检测器(按搬上盘方块数计数),并起一个 staged 稠密奖励实验(配置对齐 pothiql_joint_rerun0629,仅 pothiql→staged)。

**Architecture:** 检测器复用 env 的 `check_contact("pot_base", obj)` 持久接触谓词(非瞬时离地),经现有无状态检测器 + wrapper max 闩锁把两次搬运切成 stage 1/2;staged 奖励复用现有 `shaping_reward(mode="staged")`,零新奖励逻辑;run 脚本对参照实验做 pothiql→staged 的 flag 差异。

**Tech Stack:** Python / pytest / robosuite(dexmimicgen)/ torchrl;residual conda env(`/mnt/mnt/data/envs/residual/bin/python`)。

## Global Constraints
- 参照实验 `lifttray_actfeat_hdf5_bp_bcfixed01_hiqlv512_sg15_pothiql_joint_rerun0629`;超参逐字沿用:`--action_scale 0.05 --actor_lr 1e-6 --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 --subgoal_conditioned --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor --total_env_steps 500000`。
- staged 差异:`--reward_shaping staged --stage_reward_bonus 1.0 --stage_balanced --offline_stage_cache <lifttray_stages.npz>`;**删** `--potential_source hiql` 与 `--hiql_value_ckpt`。
- 对 `stage_detectors.py` / `test_stage_detector.py` 只**追加**(工作树含未提交的 piece 5→4 段重构,不得丢弃/改写);提交边界见收尾。
- 检测器返回瞬时 stage ∈ [0,3];闩锁(max-so-far)由 wrapper 负责,检测器无状态。
- **上 500k 前必须先跑通 Task 3/4 的真 env smoke。**
- Python:`/mnt/mnt/data/envs/residual/bin/python`;真 env 步骤需 `MUJOCO_GL=egl PYTHONPATH=/mnt/mnt/data/resfit`,cwd=`/mnt/mnt/data/resfit`。

---

### Task 1: `lifttray_stage` 检测器 + 注册(纯逻辑,单测)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/stage_detectors.py`(加函数 + registry 两条目)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py`(加 fake env + 用例,追加到文件末)

**Interfaces:**
- Produces: `lifttray_stage(env) -> int`;`STAGE_DETECTORS["TwoArmLiftTray"] = lifttray_stage`;`NUM_STAGES["TwoArmLiftTray"] == 4`
- Consumes(env 契约):`env._check_success() -> bool`;`env.check_contact("pot_base", env.obj0/obj1) -> bool`;属性 `env.obj0` / `env.obj1`

- [ ] **Step 1: 写失败测试**(追加到 `test_stage_detector.py` 末尾;并在文件顶部 import 块把 `lifttray_stage` 加入 `from ...stage_detectors import (...)`)

```python
# ---------- lifttray 4 段检测器契约 ----------
class _FakeBox:
    def __init__(self, name):
        self.name = name


class _FakeLiftTrayEnv:
    """success + obj0/obj1 是否与 pot_base 接触 独立控制。
    check_contact(geoms_1, geoms_2) 仅 geoms_1=='pot_base' 时按 obj.name 返回其 on_tray。"""
    def __init__(self, success=False, obj0_on=False, obj1_on=False):
        self._success = success
        self._on = {"obj0": obj0_on, "obj1": obj1_on}
        self.obj0 = _FakeBox("obj0")
        self.obj1 = _FakeBox("obj1")

    def _check_success(self):
        return self._success

    def check_contact(self, geoms_1, geoms_2=None):
        if geoms_1 != "pot_base":
            return False
        return self._on.get(getattr(geoms_2, "name", None), False)


def test_lifttray_stage_start_is_0():
    assert lifttray_stage(_FakeLiftTrayEnv()) == 0


def test_lifttray_stage_one_block_on_tray_is_1():
    assert lifttray_stage(_FakeLiftTrayEnv(obj0_on=True)) == 1
    assert lifttray_stage(_FakeLiftTrayEnv(obj1_on=True)) == 1   # 顺序无关,哪块都算 1


def test_lifttray_stage_both_on_tray_is_2():
    assert lifttray_stage(_FakeLiftTrayEnv(obj0_on=True, obj1_on=True)) == 2


def test_lifttray_stage_success_is_3():
    assert lifttray_stage(_FakeLiftTrayEnv(success=True)) == 3


def test_lifttray_stage_priority_success_over_lower():
    assert lifttray_stage(_FakeLiftTrayEnv(success=True, obj0_on=True, obj1_on=True)) == 3


def test_lifttray_stage_sequential_pick_separates_two_stages():
    # 核心意图:obj1 先上盘→1;obj0 后上盘(两块都在)→2 → 两次搬运落在不同段(持久里程碑)
    assert lifttray_stage(_FakeLiftTrayEnv(obj1_on=True)) == 1
    assert lifttray_stage(_FakeLiftTrayEnv(obj0_on=True, obj1_on=True)) == 2


def test_lifttray_registered():
    assert NUM_STAGES["TwoArmLiftTray"] == 4
    assert get_stage_detector("TwoArmLiftTray") is lifttray_stage
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -k lifttray -q`
Expected: FAIL(`ImportError: cannot import name 'lifttray_stage'` 或 `NameError`)

- [ ] **Step 3: 实现检测器 + 注册**(在 `stage_detectors.py` 的 `threading_stage` 之后加函数;更新两个 dict)

```python
def lifttray_stage(env) -> int:
    """TwoArmLiftTray 4 段:0 起步 / 1 一个方块搬上盘 / 2 两个方块都上盘 / 3 抬盘成功。

    里程碑用"方块与盘底 pot_base 接触"(持久态,复用 _check_success 同款谓词),按已上盘
    方块数计数(顺序无关);高段短路优先,闩锁(max-so-far)由 wrapper 负责,这里只判瞬时。
    注:不用"离地/抓起"——那是瞬时信号,两次搬运不重叠,经 wrapper max 闩锁会被压成同一段。
    """
    if env._check_success():                              # 3 抬盘成功
        return 3
    n = int(env.check_contact("pot_base", env.obj0)) + \
        int(env.check_contact("pot_base", env.obj1))
    if n >= 2:                                            # 两块都在盘上
        return 2
    if n >= 1:                                            # 一块在盘上
        return 1
    return 0
```

`STAGE_DETECTORS` 加 `"TwoArmLiftTray": lifttray_stage,`;`NUM_STAGES` 加 `"TwoArmLiftTray": 4,`。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -q`
Expected: PASS(含既有 piece 4 段 / threading / wrapper / potential 全绿)

- [ ] **Step 5: 提交**(只加这两个文件;不 `git add -A`,避开工作树里 piece/其他未提交改动)

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/stage_detectors.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py
git commit -m "feat: lifttray_stage 4-stage detector (on-tray block count)"
```

---

### Task 2: `run_lifttray_staged_joint.sh` + 脚本 flag 测试

**Files:**
- Create: `run_lifttray_staged_joint.sh`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_run_scripts.py`(追加一个测试函数)

**Interfaces:**
- Consumes: 复用产物路径 `outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt` / `..._high_actor_...pt` / `lifttray_act_feat_hdf5.npz`;base artifact `run_e0o0sckj_best:v4`(均来自参照脚本)
- Produces: 一条 ready 500k 命令 `bash run_lifttray_staged_joint.sh <gpu>`

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_run_scripts.py`)

```python
def test_lifttray_staged_joint_script_pothiql_to_staged():
    text = Path("run_lifttray_staged_joint.sh").read_text()
    # staged 特征齐全
    assert "--reward_shaping staged" in text
    assert "--stage_reward_bonus 1.0" in text
    assert "--stage_balanced" in text
    assert "--offline_stage_cache outputs_chunk/two_arm_lift_tray_stages.npz" in text
    # pothiql 残留必须清除
    assert "--reward_shaping potential" not in text
    assert "--potential_source" not in text
    assert "--hiql_value_ckpt" not in text
    # 对齐参照实验的关键不变量
    for flag in ("--task TwoArmLiftTray", "--action_scale 0.05", "--actor_lr 1e-6",
                 "--offline_fraction 0.5", "--offline_base_mode base_policy",
                 "--demo_bc_coef 0.1", "--subgoal_conditioned", "--subgoal_way_steps 15",
                 "--online_finetune_value", "--online_finetune_high_actor",
                 "--total_env_steps 500000",
                 "--base_action_mode queue", "--base_n_action_steps 10"):
        assert flag in text
    # 新 offcache / 输出名(staged 签名,勿复用 pothiql)
    assert "lifttray_actfeat_hdf5_bp_hiqlv512_sg15_staged_offcache" in text
    assert "lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_staged_joint" in text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_run_scripts.py -k lifttray_staged -q`
Expected: FAIL(`FileNotFoundError: run_lifttray_staged_joint.sh`)

- [ ] **Step 3: 建脚本**(`run_lifttray_staged_joint.sh`,内容如下)

```bash
#!/bin/bash
# lifttray staged + 联合训练。参照 run_lifttray_pothiql_joint_rerun0629.sh,仅 pothiql→staged:
#   ① --reward_shaping potential→staged;删 --potential_source hiql / --hiql_value_ckpt
#   ② 加 --stage_reward_bonus 1.0 --stage_balanced --offline_stage_cache(不存在→首跑 sim-replay 自动生成)
#   ③ 新 offcache(staged 签名重建)+ 新 wandb_name/output_dir
# 其余逐字照搬 pothiql。用法: setsid bash run_lifttray_staged_joint.sh <gpu> > lifttray_staged_joint.log 2>&1 < /dev/null &
set -euo pipefail
GPU=${1:?用法: run_lifttray_staged_joint.sh <gpu>}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd /mnt/mnt/data/resfit
PY=/mnt/mnt/data/envs/residual/bin/python

GCV=outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt
HIA=outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt
CACHE=outputs_chunk/lifttray_act_feat_hdf5.npz
gate(){ [ -s "$1" ] || { echo "[lifttray_staged] FATAL 复用产物缺失: $1"; exit 4; }; }
echo "[lifttray_staged] gate 复用产物(gc_value/high_actor/act_feat)..."
gate "$GCV"; gate "$HIA"; gate "$CACHE"
echo "[lifttray_staged] START GPU=$GPU  $(date '+%F %T')"

exec $PY -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmLiftTray \
  --base_wandb_id /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 \
  --dataset ankile/dexmg-two-arm-lift-tray \
  --offline_dataset_path resfit/dataset/two_arm_lift_tray.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping staged --stage_reward_bonus 1.0 --stage_balanced \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/two_arm_lift_tray_stages.npz \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/lifttray_actfeat_hdf5_bp_hiqlv512_sg15_staged_offcache \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_staged_joint \
  --output_dir outputs_chunk/lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_staged_joint
```

- [ ] **Step 4: 测试通过 + bash 语法检查 + 复用产物存在性**

Run:
```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_run_scripts.py -k lifttray_staged -q
bash -n run_lifttray_staged_joint.sh && echo "bash 语法 OK"
for f in outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt \
         outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt \
         outputs_chunk/lifttray_act_feat_hdf5.npz; do
  [ -s "$f" ] && echo "OK  $f" || echo "缺失 $f (需先备好/复用参照实验产物)"
done
```
Expected: pytest PASS;`bash 语法 OK`;三个复用产物 `OK`(若缺失,记录并在 Task 4 前解决)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add run_lifttray_staged_joint.sh resfit/rl_finetuning/chunk_residual/tests/test_run_scripts.py
git commit -m "feat: lifttray staged_joint run script (pothiql->staged) + script test"
```

---

### Task 3: 真 env stage 验证(controller 跑,非 CI)—— 检测器 + stages 生成路径 smoke

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/verify_stage_lifttray.py`

**Interfaces:**
- Consumes: `offline_stage_replay.make_replay_env(path) -> (env, env_name)`;`replay_instant_stages(env, states, *, model_file, detector, ep_meta=None) -> np.ndarray`;`get_stage_detector`、`NUM_STAGES`;数据 `resfit/dataset/two_arm_lift_tray.hdf5`

- [ ] **Step 1: 写验证脚本**

```python
"""真 env 验证 lifttray_stage:从 hdf5 起不渲染 env,逐帧 set_state 跑检测器,打印每条 demo 的
瞬时 stage 分布与终段闩锁。确认 set_state 后 check_contact("pot_base",obj) 正确、分布呈 0→1→2→3。
用法: MUJOCO_GL=egl PYTHONPATH=/mnt/mnt/data/resfit \
      /mnt/mnt/data/envs/residual/bin/python -m resfit.rl_finetuning.chunk_residual.verify_stage_lifttray [n]
"""
import sys

import h5py
import numpy as np

from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
    make_replay_env,
    replay_instant_stages,
)
from resfit.rl_finetuning.chunk_residual.stage_detectors import (
    NUM_STAGES,
    get_stage_detector,
)

DATA = "resfit/dataset/two_arm_lift_tray.hdf5"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    env, env_name = make_replay_env(DATA)
    det = get_stage_detector(env_name)
    assert det is not None, f"无 {env_name} 检测器"
    goal = NUM_STAGES[env_name] - 1
    ok = 0
    with h5py.File(DATA, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:n]
        for k in keys:
            d = f["data"][k]
            stages = replay_instant_stages(
                env, d["states"][:], model_file=d.attrs["model_file"], detector=det)
            latched = int(np.maximum.accumulate(stages)[-1])
            uniq, cnt = np.unique(stages, return_counts=True)
            passed = latched == goal and set(range(goal + 1)) <= set(uniq.tolist())
            ok += passed
            print(f"{k}: instant={dict(zip(uniq.tolist(), cnt.tolist()))} "
                  f"final_latched={latched}/{goal} {'OK' if passed else 'CHECK'}")
    print(f"\n{ok}/{n} demo 达终段且历经 0..{goal}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑验证(5 条 demo)**

Run:
```bash
cd /mnt/mnt/data/resfit && MUJOCO_GL=egl PYTHONPATH=/mnt/mnt/data/resfit \
  /mnt/mnt/data/envs/residual/bin/python -m resfit.rl_finetuning.chunk_residual.verify_stage_lifttray 5
```
Expected: env 成功构建;每条 demo `final_latched=3/3` 且 instant 分布含 0,1,2,3;末行 `5/5`。
若 env 构建/set_state/check_contact 报错或分布异常(如恒 0、跳段)→ 停,按 systematic-debugging 排查(常见:geom 名非 "pot_base"、obj 属性名、PandaDex env kwargs)。

- [ ] **Step 3: 提交验证脚本**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/verify_stage_lifttray.py
git commit -m "test: real-env lifttray_stage distribution probe (offline replay)"
```

---

### Task 4: staged 全链 `--smoke`(controller 跑)+ ready 500k 命令

**Files:** 无新文件(用 Task 2 的脚本 + 临时路径做小规模冒烟)

**Interfaces:**
- Consumes: Task 1 检测器、Task 2 脚本、Task 3 已验证的 stages 生成路径

- [ ] **Step 1: 小规模端到端 smoke**(4 demo,临时 stages/offcache/输出;`--smoke` 跑 2 步)

Run:
```bash
cd /mnt/mnt/data/resfit && MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONPATH=/mnt/mnt/data/resfit \
CUDA_VISIBLE_DEVICES=<free_gpu> LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
/mnt/mnt/data/envs/residual/bin/python -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmLiftTray \
  --base_wandb_id /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 \
  --dataset ankile/dexmg-two-arm-lift-tray \
  --offline_dataset_path resfit/dataset/two_arm_lift_tray.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping staged --stage_reward_bonus 1.0 --stage_balanced \
  --offline_fraction 0.5 --offline_stage_cache /tmp/lifttray_stages_smoke.npz \
  --offline_base_mode base_policy --demo_bc_coef 0.1 --offline_num_demos 4 \
  --offline_buffer_cache /tmp/lifttray_staged_offcache_smoke \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/lifttray_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --smoke --learning_starts 1 --wandb_mode disabled --output_dir /tmp/lifttray_staged_smoke_out
```
Expected:全链跑通不崩;日志出现 stages sim-replay 生成(4 demo)、offcache 构建、staged 奖励与 subgoal/joint 装载;正常退出。
(注:`--offline_num_demos 4` 让 stages/offcache 只在 4 条 demo 上建,秒级~分钟级;真跑不加此 flag。)

- [ ] **Step 2: 清理临时产物**

Run: `rm -rf /tmp/lifttray_stages_smoke.npz /tmp/lifttray_staged_offcache_smoke /tmp/lifttray_staged_smoke_out`

- [ ] **Step 3: 定稿 ready 命令 + 起 500k**(GPU 由用户/controller 选空闲卡)

Run:
```bash
cd /mnt/mnt/data/resfit
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # 选空闲卡
setsid bash run_lifttray_staged_joint.sh <free_gpu> > lifttray_staged_joint.log 2>&1 < /dev/null &
```
Expected:后台起跑;`tail -f lifttray_staged_joint.log` 见 stages npz 首建(全量,慢一次性)、offcache 重建、训练推进。首建阶段盯磁盘。

---

## 收尾:两段审查 + 提交边界

- 每个 code 任务(1、2)后按 subagent-driven 两段审查(实现 review + 独立 verify)。
- **提交边界**:落码前先与用户确认工作树里未提交的 piece 5→4 段改动怎么处理(默认:先单独提一个 "refactor: threepiece 4-stage" commit,再叠本计划的 lifttray commit),避免混提。
- Task 3/4 为真 env 验证,**controller 亲自跑**(勿交长时 subagent,防超时);500k 长跑由 controller 后台起、用户盯。

## Self-Review 记录
- Spec 覆盖:§4.1→Task1;§4.2→Task1 测试;§4.3→Task2;§7.2 真 env 验证→Task3;§7.3 smoke→Task4;§7.4 ready 命令→Task4 Step3;§6 回归→Task1 全量 pytest(既有 wrapper 回归用例守 mode="none");§10 提交边界→收尾。
- 无占位;类型/路径一致(stages/offcache/输出名在 Task2 脚本与 Task4 smoke 一致,仅 smoke 用 /tmp 临时)。
- 已知留验:Task3 真 env `check_contact` 正确性、stages 生成在 PandaDex 上跑通(这正是 Task3 目的)。
