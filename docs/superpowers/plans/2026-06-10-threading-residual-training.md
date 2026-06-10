# TwoArmThreading 残差 RL 训练接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 TwoArmThreading 走通已为 three_piece 实现的残差 RL 端到端管线(stage 整形 + object-aware 30 维 + HIQL value potential),只改最少的 task-specific 代码。

**Architecture:** 镜像 three_piece 管线,产物按 `two_arm_threading_*` 命名隔离。代码改动收敛到 **2 个文件**:`object_state.py`(让 object-aware 物体名从 env 实际属性解析,不写死字符串,所有调用点零改动)、`stage_detectors.py`(新增 3 段 `threading_stage` 并注册)。`dexmg.py` 无需改——online `_rel_piece_info` 与所有 offline 路径都汇到同一个 `compute_eef_rel_piece_from_env(env)`。其余为"换参数跑一遍"。

**Tech Stack:** Python / PyTorch / robosuite + dexmimicgen(MuJoCo EGL)/ LeRobot / torchrl / wandb / pytest。conda 环境 `residual`。

---

## 约定(每个命令都遵守)

- 工作目录:仓库根 `/mnt/mnt/data/resfit`。
- 跑 Python:`conda run -n residual --no-capture-output python ...`。
- 任何要起 MuJoCo 渲染/env 的命令前置:`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`,并按本机单卡掩码约定设 `CUDA_VISIBLE_DEVICES`。
- 单测:`conda run -n residual python -m pytest <path> -v`。
- 纯逻辑代码任务走 TDD(先写失败测试);数据/跑批任务以"命令 + 期望输出 + gate"卡。
- 每个任务结束 commit(co-author trailer 见末尾)。

## 文件结构图

| 文件 | 责任 | 动作 |
|---|---|---|
| `resfit/rl_finetuning/chunk_residual/object_state.py` | object-aware rel_piece 计算 | 改:加 `get_object_root_bodies(env)`;`compute_eef_rel_piece_from_env` 默认从 env 解析 bodies |
| `resfit/rl_finetuning/chunk_residual/stage_detectors.py` | 特权 stage 检测器 | 改:加 `threading_stage` + 注册 `STAGE_DETECTORS`/`NUM_STAGES` |
| `.../tests/test_object_state.py` | object_state 单测 | 改:加 threading 解析/维度用例 |
| `.../tests/test_stage_detector.py` | 检测器契约单测 | 改:加 `threading_stage` 用例 |
| `resfit/dataset/two_arm_threading.hdf5` | 原始 demo | 新:下载 |
| `outputs_chunk/two_arm_threading_{stages,state30}.npz`, `two_arm_threading_value.pt` | 离线产物 | 新:跑批生成 |
| `run_threading_pipeline.sh` | 复现脚本 | 新:记录离线+主训命令 |

---

## Task 0: 前置——下载数据 + 验证 BC base

**Files:** 无代码改动(IO/验证)。

- [ ] **Step 1: 下载 threading raw hdf5**

Run:
```bash
cd /mnt/mnt/data/resfit
conda run -n residual python deps/dexmimicgen/scripts/download_hf_dataset.py \
  --path resfit/dataset --tasks TwoArmThreading
```
Expected: 打印 `Downloading dataset for two_arm_threading.hdf5` … `Download complete`。

- [ ] **Step 2: 定位并归一化 hdf5 路径**

Run:
```bash
find resfit/dataset -iname "*two_arm_threading*.hdf5"
```
Expected: 找到文件(可能在子目录)。把它落到规范路径(已在则跳过):
```bash
f=$(find resfit/dataset -iname "*two_arm_threading*.hdf5" | head -1)
[ "$f" = "resfit/dataset/two_arm_threading.hdf5" ] || ln -sf "$(realpath "$f")" resfit/dataset/two_arm_threading.hdf5
ls -la resfit/dataset/two_arm_threading.hdf5
```
Expected: `resfit/dataset/two_arm_threading.hdf5` 存在(GB 级)。

- [ ] **Step 3: 验证 hdf5 可读、含 demo**

Run:
```bash
conda run -n residual python -c "
import h5py
with h5py.File('resfit/dataset/two_arm_threading.hdf5','r') as f:
    ks=list(f['data'].keys()); print('demos=',len(ks),'sample=',ks[:2])
    g=f['data/'+ks[0]]; print('obs keys=',list(g['obs'].keys())[:6]); print('has states=', 'states' in g)
"
```
Expected: `demos= <N>`(几百~1000),obs 含 `robot0_eef_pos` 等,`has states= True`。

- [ ] **Step 4: 验证 BC base 可下载可加载**

Run:
```bash
conda run -n residual python -c "
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
d,_ = download_policy_from_wandb('dexmg-twoarmthreading-bc/cbv7mqw3', step='best', artifact_version='latest')
p = load_policy(d); print('BASE OK', d)
"
```
Expected: `BASE OK <path>`。
**若失败(wandb 死链/artifact 不存在)→ 走风险分支**:用 `train_bc_dexmg.py --dataset ankile/dexmg-two-arm-threading --eval_env TwoArmThreading ...`(镜像 three_piece 的 BC 命令)重训一个 base,记下新 wandb_id 供后续 `--base_wandb_id`。**先停下与用户确认**再开重训。

- [ ] **Step 5: 验证 lerobot 数据集 stats 可取**

Run:
```bash
conda run -n residual python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
m=LeRobotDatasetMetadata('ankile/dexmg-two-arm-threading')
print('state stats keys=', list(m.stats['observation.state'].keys()))
"
```
Expected: 打印含 `mean`/`std`(首次会下载数据集元信息)。

- [ ] **Step 6: 无代码改动,不 commit。** 在 PR 描述/笔记记录:base 是否复用 `cbv7mqw3` 还是重训的新 id。

---

## Task 1: object_state —— object-aware 物体名从 env 解析(TDD)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/object_state.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_object_state.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_object_state.py` 末尾)

```python
# ---- object-aware 物体名从 env 解析(threading/three_piece/fallback)----
from resfit.rl_finetuning.chunk_residual.object_state import get_object_root_bodies


class _Obj:
    def __init__(self, root): self.root_body = root

class _ThreadingEnv:
    def __init__(self):
        self.needle = _Obj("needle_obj_root")
        self.tripod = _Obj("tripod_obj_root")

class _ThreePieceEnv:
    def __init__(self):
        self.piece_1 = _Obj("piece_1_root")
        self.piece_2 = _Obj("piece_2_root")

class _UnknownEnv:
    pass


def test_get_object_root_bodies_threading():
    assert get_object_root_bodies(_ThreadingEnv()) == ("needle_obj_root", "tripod_obj_root")

def test_get_object_root_bodies_threepiece():
    assert get_object_root_bodies(_ThreePieceEnv()) == ("piece_1_root", "piece_2_root")

def test_get_object_root_bodies_fallback_to_constant():
    assert get_object_root_bodies(_UnknownEnv()) == PIECE_ROOT_BODIES


class _ThreadingFullEnv:
    """mock threading env:_eefN_xpos(sim 实时)+ sim.data.get_body_xpos + needle/tripod 属性。"""
    def __init__(self):
        import numpy as np
        self.needle = _Obj("needle_obj_root")
        self.tripod = _Obj("tripod_obj_root")
        self._eef0_xpos = np.array([1., 0., 0.])
        self._eef1_xpos = np.array([0., 2., 0.])
        body = {"needle_obj_root": [0., 0., 0.], "tripod_obj_root": [0., 0., 1.]}
        class _Data:
            def get_body_xpos(_s, name): return np.asarray(body[name], dtype=np.float64)
        class _Sim:
            data = _Data()
        self.sim = _Sim()


def test_compute_from_env_resolves_threading_bodies_no_arg():
    # 不传 bodies → 应自动用 needle/tripod root,而非三件套默认
    out = compute_eef_rel_piece_from_env(_ThreadingFullEnv())
    assert out.shape == (12,)
    np.testing.assert_allclose(out[0:3], [1, 0, 0])    # eef0 - needle
    np.testing.assert_allclose(out[3:6], [1, 0, -1])   # eef0 - tripod
    np.testing.assert_allclose(out[6:9], [0, 2, 0])    # eef1 - needle
    np.testing.assert_allclose(out[9:12], [0, 2, -1])  # eef1 - tripod
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_object_state.py -v -k "get_object_root_bodies or resolves_threading"`
Expected: FAIL（`ImportError: cannot import name 'get_object_root_bodies'`）。

- [ ] **Step 3: 实现**

在 `object_state.py` 加 `get_object_root_bodies`(放在 `PIECE_ROOT_BODIES` 之后):
```python
def get_object_root_bodies(env):
    """按 env 实际物体属性解析 object-aware 用的 root body 名(无写死字符串)。

    threading: (needle, tripod);three_piece: (piece_1, piece_2);都没有则回退 PIECE_ROOT_BODIES。
    online(_rel_piece_info)与所有 offline 路径都不传 bodies → 统一在此解析,保证同源。
    """
    if hasattr(env, "needle") and hasattr(env, "tripod"):
        return (env.needle.root_body, env.tripod.root_body)
    if hasattr(env, "piece_1") and hasattr(env, "piece_2"):
        return (env.piece_1.root_body, env.piece_2.root_body)
    return PIECE_ROOT_BODIES
```
把 `compute_eef_rel_piece_from_env` 改为默认从 env 解析:
```python
def compute_eef_rel_piece_from_env(env, piece_root_bodies=None):
    """从 env 读 eef(双臂,sim 实时)+ object(sim) → rel_piece(12,)。

    piece_root_bodies=None(默认,所有现有调用点都这么调)→ get_object_root_bodies(env) 自动解析。
    online/offline **共用此函数** → 严格同源。
    """
    if piece_root_bodies is None:
        piece_root_bodies = get_object_root_bodies(env)
    eefs = read_eef_positions(env)
    pieces = read_piece_positions(env.sim, piece_root_bodies)
    return eef_rel_piece(eefs, pieces)
```

- [ ] **Step 4: 跑新测试确认通过 + 全量回归(不破坏 three_piece)**

Run:
```bash
conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_object_state.py -v
```
Expected: 全 PASS（含旧的 `test_piece_root_bodies_constant`、`_FakeEnv` 相关用例——`PIECE_ROOT_BODIES` 常量未变、无 piece_1/2 属性的 fake env 回退到常量,与旧行为逐位一致）。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/object_state.py \
        resfit/rl_finetuning/chunk_residual/tests/test_object_state.py
git commit -m "feat(object-aware): rel_piece 物体名从 env 解析(支持 threading,three_piece 等价)"
```

---

## Task 2: stage_detectors —— threading 3 段检测器(TDD)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/stage_detectors.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_stage_detector.py` 检测器契约区)

```python
# ---------- threading 3 段检测器契约 ----------
from resfit.rl_finetuning.chunk_residual.stage_detectors import threading_stage


class _FakeNeedle:
    contact_geoms = ["ndl0", "ndl1"]

class _FakeTripod:
    contact_geoms = ["trp0", "trp1"]

def _is_tripod(object_geoms):
    return "trp0" in object_geoms


class _FakeThreadEnv:
    """list-gripper fake;grasp_needle / grasp_tripod 独立控制。"""
    def __init__(self, success=False, grasp_needle=False, grasp_tripod=False):
        self._success = success
        self._gn, self._gt = grasp_needle, grasp_tripod
        self.robots = [_FakeRobot()]          # 复用本文件已有的 _FakeRobot(gripper=[_FakeGripper()])
        self.needle = _FakeNeedle()
        self.tripod = _FakeTripod()

    def _check_success(self):
        return self._success

    def _check_grasp(self, gripper, object_geoms):
        return self._gt if _is_tripod(object_geoms) else self._gn


def test_threading_stage_success_is_2():
    assert threading_stage(_FakeThreadEnv(success=True)) == 2

def test_threading_stage_both_grasped_is_1():
    assert threading_stage(_FakeThreadEnv(grasp_needle=True, grasp_tripod=True)) == 1

def test_threading_stage_only_needle_is_0():
    # 仅抓针、未抓脚架 → 仍 0(stage 1 要求两物都抓)
    assert threading_stage(_FakeThreadEnv(grasp_needle=True)) == 0

def test_threading_stage_only_tripod_is_0():
    assert threading_stage(_FakeThreadEnv(grasp_tripod=True)) == 0

def test_threading_stage_start_is_0():
    assert threading_stage(_FakeThreadEnv()) == 0

def test_threading_stage_priority_success_over_grasp():
    assert threading_stage(_FakeThreadEnv(success=True, grasp_needle=True, grasp_tripod=True)) == 2

def test_threading_registered():
    assert NUM_STAGES["TwoArmThreading"] == 3
    assert get_stage_detector("TwoArmThreading") is threading_stage
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -v -k threading`
Expected: FAIL（`ImportError: cannot import name 'threading_stage'`）。

- [ ] **Step 3: 实现**

在 `stage_detectors.py` 的 `threepiece_stage` 之后加:
```python
def threading_stage(env) -> int:
    """TwoArmThreading 3 段:0 起步 / 1 脚架与针都被抓起 / 2 成功(针穿入环)。

    threading 无 env 内置子阶段谓词(只有 _check_success),故中间段用双物体 grasp 里程碑。
    闩锁(max-so-far)由 wrapper 负责,这里只判瞬时阶段。
    """
    if env._check_success():
        return 2
    if _grasped(env, env.needle) and _grasped(env, env.tripod):
        return 1
    return 0
```
并扩注册表:
```python
STAGE_DETECTORS = {
    "TwoArmThreePieceAssembly": threepiece_stage,
    "TwoArmThreading": threading_stage,
}
NUM_STAGES = {
    "TwoArmThreePieceAssembly": 5,
    "TwoArmThreading": 3,
}
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py -v`
Expected: 全 PASS(threading 新用例 + 旧 threepiece 用例都绿)。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/stage_detectors.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_detector.py
git commit -m "feat(stage): TwoArmThreading 3 段检测器(起步/双物抓起/成功)"
```

---

## Task 3: sim 探针 —— 确认物体名 + 脚架可抓(裁决 stage 1 回退)

**Files:** 无代码(一次性集成验证)。**依赖 Task 0 的数据 + GPU/EGL。**

- [ ] **Step 1: 探针打印真实 root body 名,核对 Task 1 假设**

Run:
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<gpu> \
conda run -n residual python -c "
from resfit.rl_finetuning.chunk_residual.offline_stage_replay import make_replay_env
env,name = make_replay_env('resfit/dataset/two_arm_threading.hdf5')
print('env_name=',name)
print('needle.root_body=',env.needle.root_body,'| tripod.root_body=',env.tripod.root_body)
from resfit.rl_finetuning.chunk_residual.object_state import get_object_root_bodies
print('resolved=',get_object_root_bodies(env))
"
```
Expected: `needle.root_body= needle_obj_root`、`tripod.root_body= tripod_obj_root`(若不同,Task 1 的解析仍正确,因为是从 env 实读;只需更新 `test_get_object_root_bodies_threading` 的期望字符串并补跑 Task 1 测试)。

- [ ] **Step 2: 无 commit**(探针是验证,不留文件)。记录实际 body 名。

---

## Task 4: 离线 stage 缓存 + 直方图 gate(裁决 stage 1)

**Files:** 产出 `outputs_chunk/two_arm_threading_stages.npz`。

- [ ] **Step 1: 预计算 stage 缓存**

Run:
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<gpu> \
conda run -n residual python -c "
from resfit.rl_finetuning.chunk_residual.offline_stage_replay import precompute_stage_cache
n=precompute_stage_cache('resfit/dataset/two_arm_threading.hdf5','outputs_chunk/two_arm_threading_stages.npz',num_demos=None)
print('cached demos=',n)
" 2>&1 | tee two_arm_threading_stages.log
```
Expected: `cached demos= <N>`(数小时;耐心)。

- [ ] **Step 2: 直方图 gate —— 三段都非空、单调,裁决 stage 1**

Run:
```bash
conda run -n residual python -c "
import numpy as np
d=np.load('outputs_chunk/two_arm_threading_stages.npz')
import collections
cnt=collections.Counter()
reached1=reached2=0
for k in d.files:
    a=d[k].astype(int); cnt.update(a.tolist())
    reached1+= int((a>=1).any()); reached2+= int((a>=2).any())
print('per-step stage counts=',dict(cnt))
print('demos reaching stage1=',reached1,'/',len(d.files),'| stage2=',reached2,'/',len(d.files))
"
```
Expected(PASS 判据):stage 0/1/2 三段在 per-step 计数里都非零,且大多数 demo `reaching stage1` 接近总数(成功 demo 应都经过"两物抓起")。
**FAIL 处理**:若 `reaching stage1` 远低于总数(脚架很少被 grasp 检测到)→ 执行 §6 回退:把 `threading_stage` 的 stage 1 改为只判 `_grasped(env, env.needle)`,更新 Task 2 对应单测(`test_threading_stage_only_needle_is_0` → `..._is_1`、`test_threading_stage_only_tripod_is_0` 删去),补跑 Task 2 测试并 commit,然后**重跑本 Task 的 Step 1-2**。

- [ ] **Step 3: 缓存是数据产物,不 commit**(`outputs_chunk/` 见 .gitignore)。

---

## Task 5: 离线 state30(object-aware)缓存

**Files:** 产出 `outputs_chunk/two_arm_threading_state30.npz`。

- [ ] **Step 1: 建 state30 缓存(eef_piece,30 维)**

Run:
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<gpu> \
conda run -n residual python -c "
from resfit.rl_finetuning.chunk_residual.state30_cache import load_or_build_state30
seqs=load_or_build_state30('resfit/dataset/two_arm_threading.hdf5','ankile/dexmg-two-arm-threading',None,'outputs_chunk/two_arm_threading_state30.npz')
import numpy as np
print('demos=',len(seqs),'dim=',seqs[0].shape[1])
" 2>&1 | tee two_arm_threading_state30.log
```
Expected: `dim= 30`(18 eef + 12 rel_piece;rel_piece 经 Task 1 解析自动用 needle/tripod)。数小时。

- [ ] **Step 2: 同源抽检 —— offline rel_piece 数值合理(非全零、非 NaN)**

Run:
```bash
conda run -n residual python -c "
import numpy as np
d=np.load('outputs_chunk/two_arm_threading_state30.npz')
s=d['s0']; rp=s[:,18:30]
print('rel_piece shape=',rp.shape,'finite=',np.isfinite(rp).all(),'nonzero_frac=',float((rp!=0).mean()))
"
```
Expected: `shape=(T,12)`、`finite= True`、`nonzero_frac` 明显 >0。

- [ ] **Step 3: 不 commit(数据产物)。**

---

## Task 6: HIQL value 训练(potential 源)

**Files:** 产出 `outputs_chunk/two_arm_threading_value.pt`。

- [ ] **Step 1: 先小冒烟(num_demos 少、steps 少)验通路**

Run:
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<gpu> \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --hdf5 resfit/dataset/two_arm_threading.hdf5 \
  --dataset ankile/dexmg-two-arm-threading --state_mode eef_piece \
  --num_demos 20 --steps 500 --output /tmp/threading_value_smoke.pt 2>&1 | tail -5
```
Expected: 打印 `[hiql_value] state_mode=eef_piece demos=20 ... state_dim=30` 与 `saved`。

- [ ] **Step 2: 全量训练**

Run:
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<gpu> \
conda run -n residual python -u -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --hdf5 resfit/dataset/two_arm_threading.hdf5 \
  --dataset ankile/dexmg-two-arm-threading --state_mode eef_piece \
  --output outputs_chunk/two_arm_threading_value.pt 2>&1 | tee two_arm_threading_value.log
```
Expected: `state_dim=30`,末尾 `saved outputs_chunk/two_arm_threading_value.pt; state_mode=eef_piece v_stats={...}`。

- [ ] **Step 3: 校验 ckpt 元信息**

Run:
```bash
conda run -n residual python -c "
import torch; c=torch.load('outputs_chunk/two_arm_threading_value.pt',map_location='cpu')
print('state_mode=',c.get('state_mode'),'dataset_id=',c.get('dataset_id'),'v_stats=',c.get('v_stats'))
"
```
Expected: `state_mode= eef_piece`、`dataset_id= ankile/dexmg-two-arm-threading`、`v_stats` 有 min/max/mean。

- [ ] **Step 4: 不 commit(产物)。**

---

## Task 7: 残差 RL 主训冒烟

**Files:** 写复现脚本 `run_threading_pipeline.sh`(可 commit);产出 `outputs_chunk/threading_smoke/`。

- [ ] **Step 1: 写复现脚本**(`run_threading_pipeline.sh`,记录全量主训命令)

```bash
#!/usr/bin/env bash
set -euo pipefail
GPU=${1:?usage: run_threading_pipeline.sh <gpu>}
BASE_ID=${2:-dexmg-twoarmthreading-bc/cbv7mqw3}   # Task0 若重训 base,传新 id
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$GPU"
conda run -n residual --no-capture-output python -u \
  -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading \
  --base_wandb_id "$BASE_ID" \
  --dataset ankile/dexmg-two-arm-threading \
  --offline_dataset_path resfit/dataset/two_arm_threading.hdf5 \
  --offline_stage_cache outputs_chunk/two_arm_threading_stages.npz \
  --reward_shaping potential --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/two_arm_threading_value.pt \
  --state_mode eef_piece --stage_balanced --actor raw \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 \
  --offline_fraction 0.5 \
  --wandb_project dexmg-chunk-residual --wandb_name threading_best_pothiql \
  --output_dir outputs_chunk/threading_best_pothiql \
  2>&1 | tee threading_best_pothiql.log
```

- [ ] **Step 2: 冒烟跑(`--smoke`,确认接线/shaping 数值合理)**

用内置 `--smoke`(少量步数)+ `--wandb_mode disabled` 跑一遍接线:
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<gpu> \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading --base_wandb_id dexmg-twoarmthreading-bc/cbv7mqw3 \
  --dataset ankile/dexmg-two-arm-threading \
  --offline_dataset_path resfit/dataset/two_arm_threading.hdf5 \
  --offline_stage_cache outputs_chunk/two_arm_threading_stages.npz \
  --reward_shaping potential --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/two_arm_threading_value.pt \
  --state_mode eef_piece --stage_balanced --actor raw \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --offline_fraction 0.5 \
  --smoke --wandb_mode disabled --output_dir outputs_chunk/threading_smoke \
  2>&1 | tee outputs_chunk/threading_smoke.log
```
Expected(确认 3 件事):
- `[hiql-phi] potential on; ckpt=...two_arm_threading_value.pt state_mode=eef_piece scale=...` 打印出现;
- 无维度/同源报错;offline buffer 命中 `two_arm_threading_stages.npz`(秒读,非数小时 replay);
- reward shaping 数值有限、非全零、非 NaN。

- [ ] **Step 3: Commit 复现脚本**

```bash
git add run_threading_pipeline.sh
git commit -m "chore(threading): 残差 RL 复现脚本(potential=hiql, state_mode=eef_piece)"
```

---

## Task 8: 全量长跑 + eval

**Files:** 产出 `outputs_chunk/threading_best_pothiql/`。

- [ ] **Step 1: 长跑(后台)**

Run: `bash run_threading_pipeline.sh <gpu> [<base_id>]`(后台,`tee` 已落日志)。

- [ ] **Step 2: eval gate**

监控 wandb `dexmg-chunk-residual/threading_best_pothiql` 的 `eval/success_rate`。
Expected(PASS 判据):随训练上升且**超过 BC base 的 success_rate**(残差有增益)。若长期不升 → 先退 `--potential_source stage`(Φ=整数 stage,更稳)排查是 value 还是整形的问题(§9 fallback)。

- [ ] **Step 3: 收尾**——记录最终 success_rate、与 three_piece 对照;按 finishing-a-development-branch 决定合并/PR。

---

## 自查覆盖(spec → task)

- spec §4 数据/base → Task 0 ✓(含 base 死链回退)
- spec §5 object_state 物体名按任务 → Task 1 ✓(精化为从 env 解析,所有调用点零改动;§7 dexmg 因此无需改)
- spec §6 threading 3 段检测器 + 回退裁决 → Task 2(实现)+ Task 4 Step 2(直方图裁决回退)✓
- spec §8 离线管线 → Task 4/5/6 ✓
- spec §9 主训 + potential=hiql → Task 7/8 ✓
- spec §10 验证 gate(env smoke/直方图/同源/value/主训/长跑)→ Task 3/4/5/6/7/8 ✓
- spec §12 风险(base 死链 / stage1 少触发 / body 字符串)→ Task 0 Step4 / Task 4 Step2 / Task 3 Step1 ✓

> **与 spec 的一处实现精化**:§5 原写"物体名字符串注册表 + dexmg 接线(§7)"。规划时确认所有 rel_piece 调用点都汇到 `compute_eef_rel_piece_from_env(env)` 不传 bodies,故改为"从 env 实读 `needle/tripod.root_body`"——更省(无写死字符串要核、所有调用点零改动)、`dexmg.py` 无需改。功能与 spec 意图一致(物体名 task-correct、维度不变、同源保持)。

---

提交用 co-author trailer:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
