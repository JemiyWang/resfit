# LIBERO 单臂 env 接进 chunk_residual(base=pi0_libero)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 resfit chunk_residual 加 `--env_family libero` 分支(默认 dexmg 零回归),让现成"pi0 动作当 base + 残差修正"机制在 LIBERO 单臂上跑,base=pi0_libero。

**Architecture:** 纯逻辑(`libero_obs.py`:翻转/resize/8维state/4→5tuple/扁平serve-schema)与集成层(`libero_env.py` LiberoGymWrapper + 工厂、`LiberoPi05Adapter`)分离。纯逻辑在 residual 环境单测(不 import libero/openpi_client);env 真集成只在 opt-in live smoke。运行在新建 LIBERO-residual conda env(robosuite 1.4.1)。

**Tech Stack:** Python/PyTorch/gymnasium/numpy/PIL/pytest;LIBERO(libero/bddl1.0.1/robomimic0.2.0/robosuite1.4.1);openpi-client(pi05 websocket base)。

参照 spec:`docs/superpowers/specs/2026-06-12-libero-residual-pi0-onboarding-design.md`。

单测命令(residual 环境,纯逻辑不需新 env):
```
conda run -n residual python -m pytest <path>::<test> -v
```

---

## File Structure
- Create `resfit/rl_finetuning/chunk_residual/libero_obs.py` — 纯函数(无 libero/openpi_client 依赖)。
- Create `resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py` — `LiberoPi05Adapter`(子类化 kai0 `Pi05PolicyAdapter`,override `_to_openpi_obs` 委托纯函数)。
- Create `resfit/rl_finetuning/chunk_residual/libero_env.py` — `LiberoGymWrapper` + `make_libero_env` + `create_libero_vectorized_env`(惰性 import libero)。
- Modify `resfit/lerobot/policies/pi05/load_pi05.py` — `schema="libero"` 分发 `LiberoPi05Adapter`。
- Modify `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` — `--env_family/--libero_suite/--libero_task_id` + `validate_libero_cfg` + libero env 分支。
- Create `scripts/setup_libero_residual_env.sh` — 新 conda env 构建 + 依赖共存核验。
- Tests:`tests/test_libero_obs.py`、`tests/test_libero_pi05_adapter.py`、`tests/test_env_family_wiring.py`、`tests/test_libero_env_wrapper.py`、`tests/test_libero_smoke.py`(opt-in)。

---

## Task 1: 新 conda env 构建 + 依赖共存核验(前置命门,先行)

**目的(spec §2/§7):** 验证"resfit RL 依赖 + LIBERO robosuite 1.4.1 栈能否在一个 conda env 共存"。不成立 → 停下回退 env-RPC,不写后续代码。

**Files:**
- Create: `resfit/scripts/setup_libero_residual_env.sh`

- [ ] **Step 1: 写构建脚本**

```bash
#!/usr/bin/env bash
# 新建 LIBERO-residual conda env:resfit RL 依赖 + LIBERO 1.4.1 栈(不含 dexmg)。
# 用法:bash resfit/scripts/setup_libero_residual_env.sh [ENV_NAME]
set -euo pipefail
ENV_NAME="${1:-libero-residual}"
LIBERO_SRC=/mnt/mnt/data/chj/openpi/third_party/libero
KAI0_PATH=/mnt/mnt/data/wjm/kai0_new4090

conda create -y -n "$ENV_NAME" python=3.8
# LIBERO 栈(钉版本,见 third_party/libero/requirements.txt)
conda run -n "$ENV_NAME" pip install robosuite==1.4.1 bddl==1.0.1 robomimic==0.2.0 mujoco==3.2.3 egl_probe
conda run -n "$ENV_NAME" pip install -e "$LIBERO_SRC"
# resfit RL 依赖(在线训练 + pi05 websocket base)
conda run -n "$ENV_NAME" pip install torch torchrl gymnasium numpy pillow lerobot
conda run -n "$ENV_NAME" pip install -e "$KAI0_PATH"   # openpi_client + resfit_pi05 adapter
echo "[setup] env $ENV_NAME built"
```

- [ ] **Step 2: 跑核验(手动,需 conda)**

Run:
```bash
bash resfit/scripts/setup_libero_residual_env.sh libero-residual
conda run -n libero-residual python -c "
import torch, torchrl, gymnasium, numpy, PIL, lerobot, openpi_client
import robosuite; assert robosuite.__version__.startswith('1.4'), robosuite.__version__
import bddl, robomimic
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
print('[verify] all imports OK; robosuite', robosuite.__version__)
"
```
Expected: 打印 `[verify] all imports OK; robosuite 1.4.x`,无 import/版本冲突。
- 若任一 import 失败或 robosuite 版本被别的包顶成 1.5 → **共存不成立**:记录冲突包,**停止本计划**,回到 spec §7 的 env-RPC 回退(另起 spec)。后续 Task 2-6 的纯逻辑单测仍可在现有 `residual` 环境跑,但 env 真集成(Task 4 集成路 / Task 6 smoke)依赖本 env。

- [ ] **Step 3: Commit**

```bash
git add resfit/scripts/setup_libero_residual_env.sh
git commit -m "build: setup script for libero-residual conda env (robosuite 1.4.1 + resfit RL deps)"
```

---

## Task 2: `libero_obs.py` 纯函数 + 单测

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/libero_obs.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_obs.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest
from resfit.rl_finetuning.chunk_residual.libero_obs import (
    quat2axisangle, resize_with_pad, flip_resize_image,
    assemble_libero_state, adapt_4tuple, build_libero_serve_obs,
)


def test_quat2axisangle_identity_quat_is_zero():
    # 单位四元数 (x,y,z,w)=(0,0,0,1) → 零轴角
    assert np.allclose(quat2axisangle(np.array([0., 0, 0, 1.])), np.zeros(3))


def test_quat2axisangle_clamps_w_over_one():
    out = quat2axisangle(np.array([0., 0, 0, 1.5]))  # w>1 被 clamp,den=0 → 0
    assert np.allclose(out, np.zeros(3))


def test_resize_with_pad_shape_and_dtype():
    img = np.zeros((128, 256, 3), np.uint8)
    out = resize_with_pad(img, 224, 224)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_flip_resize_image_flips_then_resizes():
    img = np.zeros((100, 120, 3), np.uint8); img[0, 0] = 255      # 左上角白点
    out = flip_resize_image(img)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8   # [::-1,::-1] 后白点到右下区


def test_assemble_libero_state_is_8d_axisangle():
    obs = {"robot0_eef_pos": np.ones(3), "robot0_eef_quat": np.array([0., 0, 0, 1.]),
           "robot0_gripper_qpos": np.array([0.04, -0.04])}
    s = assemble_libero_state(obs)
    assert s.shape == (8,) and s.dtype == np.float32
    assert np.allclose(s[:3], 1.0) and np.allclose(s[3:6], 0.0)   # axisangle(单位quat)=0


def test_adapt_4tuple_success_terminates():
    o, r, term, trunc, info = adapt_4tuple({"x": 1}, 1.0, True, {})
    assert term is True and trunc is False and r == 1.0


def test_adapt_4tuple_timeout_truncates():
    o, r, term, trunc, info = adapt_4tuple({"x": 1}, 0.0, True, {})
    assert term is False and trunc is True       # done 但非成功 → 超时


def test_build_libero_serve_obs_flat_schema():
    raw = {
        "observation.images.agentview": np.zeros((3, 4, 4), np.float32),       # CHW float
        "observation.images.robot0_eye_in_hand": np.zeros((3, 4, 4), np.float32),
        "observation.state": np.arange(8, dtype=np.float32)[None, :],          # [B,8]
    }
    out = build_libero_serve_obs(raw, base_key="observation.images.agentview",
                                 wrist_key="observation.images.robot0_eye_in_hand",
                                 state_key="observation.state", prompt="do it", env_index=0)
    assert set(out) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    assert out["observation/image"].shape == (224, 224, 3) and out["observation/image"].dtype == np.uint8
    assert out["observation/state"].shape == (8,) and out["prompt"] == "do it"
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_obs.py -v`
Expected: FAIL(`ModuleNotFoundError: libero_obs`)。

- [ ] **Step 3: 实现 `libero_obs.py`**

```python
"""LIBERO obs/动作纯逻辑(无 libero/openpi_client 依赖,可在 residual 环境单测)。
语义对齐 chj/openpi/scripts/rollout_libero.py(可用 client)。"""
import math
import numpy as np
from PIL import Image


def quat2axisangle(quat) -> np.ndarray:
    """robosuite (x,y,z,w) 四元数 → 3 维轴角。对齐 rollout_libero._quat2axisangle。"""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def resize_with_pad(img, h, w) -> np.ndarray:
    """保纵横比缩放 + 居中 pad 到 (h,w);uint8 HWC。对齐 openpi image_tools.resize_with_pad。"""
    img = np.asarray(img)
    ih, iw = img.shape[:2]
    ratio = min(h / ih, w / iw)
    nh, nw = int(round(ih * ratio)), int(round(iw * ratio))
    resized = np.asarray(Image.fromarray(img.astype(np.uint8)).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def _to_hwc_uint8(img) -> np.ndarray:
    """CHW float[0,1] 或 HWC → HWC uint8。"""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3:           # CHW → HWC
        a = np.transpose(a, (1, 2, 0))
    if np.issubdtype(a.dtype, np.floating):
        a = (255.0 * a).clip(0, 255).astype(np.uint8)
    return a.astype(np.uint8)


def flip_resize_image(img) -> np.ndarray:
    """LIBERO 原始图上下左右翻转 [::-1,::-1] + resize_with_pad 224。对齐 rollout_libero:425-430。"""
    a = _to_hwc_uint8(img)
    a = np.ascontiguousarray(a[::-1, ::-1])
    return resize_with_pad(a, 224, 224)


def assemble_libero_state(obs) -> np.ndarray:
    """eef_pos(3)+quat2axisangle(eef_quat)(3)+gripper_qpos(2) = 8 维。对齐 rollout_libero:436-440。"""
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], np.float32).reshape(-1)[:3],
        quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"], np.float32).reshape(-1)[:2],
    ]).astype(np.float32)


def adapt_4tuple(obs, reward, done, info):
    """LIBERO 4-tuple → gym 5-tuple。成功(reward==1.0)→terminated;done 非成功→truncated(超时)。"""
    success = bool(np.asarray(reward).reshape(-1)[0] == 1.0)
    terminated = success
    truncated = bool(done) and not success
    info = dict(info or {}); info["success"] = success
    return obs, reward, terminated, truncated, info


def build_libero_serve_obs(raw_obs, *, base_key, wrist_key, state_key, prompt, env_index=0):
    """从(批后)raw_obs 取 env_index,出 pi0_libero serve 的扁平 schema。
    图像走 _to_hwc_uint8+resize_with_pad(env 侧已翻转,故这里不再翻);state 8 维。"""
    def _img(key):
        arr = np.asarray(raw_obs[key])
        arr = arr[env_index] if arr.ndim == 4 else arr
        return resize_with_pad(_to_hwc_uint8(arr), 224, 224)
    st = np.asarray(raw_obs[state_key])
    st = st[env_index] if st.ndim == 2 else st
    return {
        "observation/image": _img(base_key),
        "observation/wrist_image": _img(wrist_key),
        "observation/state": st.astype(np.float32).reshape(-1)[:8],
        "prompt": str(prompt),
    }
```

- [ ] **Step 4: Run → 通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_obs.py -v`
Expected: PASS(8 项)。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/libero_obs.py resfit/rl_finetuning/chunk_residual/tests/test_libero_obs.py
git commit -m "feat: libero_obs pure helpers (flip/resize/8d-state/4to5tuple/flat-serve-schema)"
```

---

## Task 3: `LiberoPi05Adapter`(扁平 schema base shim)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py`

- [ ] **Step 1: 写失败测试(纯函数已覆盖核心;此处只锁 override 委托正确)**

```python
import numpy as np
import pytest
from resfit.rl_finetuning.chunk_residual.libero_obs import build_libero_serve_obs


def test_libero_serve_obs_used_by_adapter_contract():
    """LiberoPi05Adapter._to_openpi_obs 必须产出与 build_libero_serve_obs 一致的扁平 schema。
    adapter 基类在 kai0(运行时 sys.path 注入),单测不依赖它,只锁委托的纯函数契约。"""
    raw = {"observation.images.agentview": np.zeros((3, 8, 8), np.float32),
           "observation.images.robot0_eye_in_hand": np.zeros((3, 8, 8), np.float32),
           "observation.state": np.zeros((1, 8), np.float32)}
    out = build_libero_serve_obs(raw, base_key="observation.images.agentview",
                                 wrist_key="observation.images.robot0_eye_in_hand",
                                 state_key="observation.state", prompt="p", env_index=0)
    assert set(out) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}


def test_adapter_module_imports_with_kai0_on_path(monkeypatch):
    """有 kai0 路径时 LiberoPi05Adapter 可构造且是 Pi05PolicyAdapter 子类;无则 skip。"""
    import sys, importlib
    sys.path.insert(0, "/mnt/mnt/data/wjm/kai0_new4090")
    try:
        from resfit_pi05.pi05_policy_adapter import Pi05PolicyAdapter
    except Exception:
        pytest.skip("kai0 adapter 不可用,跳过(纯函数契约已由上一测试覆盖)")
    from resfit.rl_finetuning.chunk_residual.libero_pi05_adapter import LiberoPi05Adapter
    assert issubclass(LiberoPi05Adapter, Pi05PolicyAdapter)
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py -v`
Expected: 第一测试 FAIL→PASS(build_libero_serve_obs 已存在则 PASS);第二测试 import LiberoPi05Adapter FAIL(`ModuleNotFoundError`)。

- [ ] **Step 3: 实现 `libero_pi05_adapter.py`**

```python
"""LiberoPi05Adapter:让 pi05 base 发 pi0_libero serve 吃的扁平 schema。
子类化 kai0 Pi05PolicyAdapter,只覆写 _to_openpi_obs 委托 libero_obs.build_libero_serve_obs。
kai0 在运行时由 load_pi05 经 sys.path 注入,故本模块惰性 import 基类。"""
from resfit.rl_finetuning.chunk_residual.libero_obs import build_libero_serve_obs


def _base_cls():
    from resfit_pi05.pi05_policy_adapter import Pi05PolicyAdapter
    return Pi05PolicyAdapter


def make_libero_pi05_adapter_cls():
    Base = _base_cls()

    class LiberoPi05Adapter(Base):
        # base/wrist/state 在 env obs 里的键(LIBERO 单臂固定)
        BASE_KEY = "observation.images.agentview"
        WRIST_KEY = "observation.images.robot0_eye_in_hand"
        STATE_KEY = "observation.state"

        def _to_openpi_obs(self, raw_obs, env_index):
            return build_libero_serve_obs(
                raw_obs, base_key=self.BASE_KEY, wrist_key=self.WRIST_KEY,
                state_key=self.STATE_KEY, prompt=self.prompt, env_index=env_index)

    return LiberoPi05Adapter


# 模块级惰性导出:kai0 在 path 上才可解析
try:
    LiberoPi05Adapter = make_libero_pi05_adapter_cls()
except Exception:                                     # kai0 不在 path:延迟到 load_pi05 注入后再建
    LiberoPi05Adapter = None
```

(注:`load_pi05`(Task 5)在注入 kai0 path 后,若模块级 `LiberoPi05Adapter is None` 则调 `make_libero_pi05_adapter_cls()` 重建。)

- [ ] **Step 4: Run → 通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py -v`
Expected: PASS(契约测试)+ 第二测试 PASS 或 SKIP(取决于 kai0 可用性)。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py resfit/rl_finetuning/chunk_residual/tests/test_libero_pi05_adapter.py
git commit -m "feat: LiberoPi05Adapter emits flat openpi LIBERO serve schema"
```

---

## Task 4: `libero_env.py`(LiberoGymWrapper + 工厂)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/libero_env.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py`

- [ ] **Step 1: 写失败测试(用 stub libero env,不 import 真 libero)**

```python
import numpy as np
from resfit.rl_finetuning.chunk_residual.libero_env import LiberoGymWrapper


class _StubLiberoEnv:
    """假 LIBERO OffScreenRenderEnv:返回 4-tuple + 原始 obs 键。"""
    def __init__(self): self.t = 0
    def _obs(self):
        return {"agentview_image": np.zeros((128, 128, 3), np.uint8),
                "robot0_eye_in_hand_image": np.zeros((128, 128, 3), np.uint8),
                "robot0_eef_pos": np.ones(3), "robot0_eef_quat": np.array([0., 0, 0, 1.]),
                "robot0_gripper_qpos": np.array([0.04, -0.04])}
    def reset(self): self.t = 0; return self._obs()
    def set_init_state(self, s): return self._obs()
    def step(self, a):
        self.t += 1
        return self._obs(), 0.0, False, {}        # 4-tuple


def test_wrapper_reset_returns_obs_info_with_contract_keys():
    w = LiberoGymWrapper(_StubLiberoEnv(), init_states=np.zeros((1, 10)), num_steps_wait=2)
    obs, info = w.reset()
    assert "observation.state" in obs and obs["observation.state"].shape[-1] == 8
    assert "observation.images.agentview" in obs
    assert "observation.images.robot0_eye_in_hand" in obs
    assert w.action_space.shape == (7,)


def test_wrapper_step_returns_5tuple():
    w = LiberoGymWrapper(_StubLiberoEnv(), init_states=np.zeros((1, 10)), num_steps_wait=0)
    w.reset()
    obs, r, term, trunc, info = w.step(np.zeros(7, np.float32))
    assert isinstance(term, (bool, np.bool_)) and isinstance(trunc, (bool, np.bool_))
    assert "observation.state" in obs


def test_wrapper_reset_runs_dummy_steps():
    env = _StubLiberoEnv()
    LiberoGymWrapper(env, init_states=np.zeros((1, 10)), num_steps_wait=5).reset()
    assert env.t == 5                                # reset 内置 dummy step
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py -v`
Expected: FAIL(`ModuleNotFoundError: libero_env`)。

- [ ] **Step 3: 实现 `libero_env.py`**

```python
"""LIBERO 单臂 env,适配成 dexmg RobosuiteGymWrapper 同款 gym 契约。
纯转换走 libero_obs;OffScreenRenderEnv 惰性 import(单测用 stub,真集成在 live smoke)。"""
import numpy as np
import gymnasium as gym

from resfit.rl_finetuning.chunk_residual.libero_obs import (
    flip_resize_image, assemble_libero_state, adapt_4tuple)

_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)   # 张开夹爪、不动


class LiberoGymWrapper(gym.Env):
    """单 (suite,task) LIBERO env → resfit 契约:obs{observation.state(8), images.{agentview,
    robot0_eye_in_hand}}、action Box(-1,1,(7,))、reset->(obs,info)、step->5tuple。"""

    def __init__(self, libero_env, *, init_states, num_steps_wait=10, init_idx=0):
        self.env = libero_env
        self.init_states = np.asarray(init_states)
        self.num_steps_wait = int(num_steps_wait)
        self.init_idx = int(init_idx)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = gym.spaces.Dict({})    # 占位;消费方按键取,不校验

    def _process(self, obs):
        return {
            "observation.state": assemble_libero_state(obs).reshape(1, 8),
            "observation.images.agentview": _chw01(flip_resize_image(obs["agentview_image"])),
            "observation.images.robot0_eye_in_hand": _chw01(flip_resize_image(obs["robot0_eye_in_hand_image"])),
        }

    def reset(self, *, seed=None, options=None):
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[self.init_idx])
        for _ in range(self.num_steps_wait):          # LIBERO 物体落稳
            obs, _, _, _ = self.env.step(_DUMMY_ACTION)
        return self._process(obs), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action, np.float32))
        obs2, r, term, trunc, info = adapt_4tuple(obs, reward, done, info)
        return self._process(obs2), float(r), bool(term), bool(trunc), info

    def render(self, *a, **k): return None
    def close(self): self.env.close() if hasattr(self.env, "close") else None
    def seed(self, s=None): return [s]


def _chw01(hwc_uint8):
    return (np.transpose(hwc_uint8, (2, 0, 1)).astype(np.float32) / 255.0)


def make_libero_env(suite, task_id, *, camera_size=256, render_gpu_device_id=0):
    """惰性构造真 LIBERO env(live smoke / 训练用)。"""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_size,
                             camera_widths=camera_size, render_gpu_device_id=render_gpu_device_id)
    init_states = task_suite.get_task_init_states(task_id)
    return LiberoGymWrapper(env, init_states=init_states), task.language


def create_libero_vectorized_env(suite, task_id, num_envs, device, **kw):
    """镜像 dexmg create_vectorized_env:复用其 VectorizedEnvWrapper + AsyncVectorEnv + EGL 映射。"""
    from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper, cuda_to_egl_device_id
    import gymnasium
    def _factory(env_id):
        egl = cuda_to_egl_device_id(device, env_id, num_envs)
        env, _ = make_libero_env(suite, task_id, render_gpu_device_id=egl)
        return env
    venv = gymnasium.vector.AsyncVectorEnv(
        [lambda i=i: _factory(i) for i in range(num_envs)],
        autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP, context="spawn")
    return VectorizedEnvWrapper(venv, device=device)
```

(注:`create_libero_vectorized_env` 用真 libero,只在新 env / live smoke 跑;`VectorizedEnvWrapper`/`cuda_to_egl_device_id` 的确切签名按 dexmg.py 现状对齐,执行时核对。单测只覆盖 `LiberoGymWrapper`(stub env)。)

- [ ] **Step 4: Run → 通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py -v`
Expected: PASS(3 项;均走 stub,不 import 真 libero)。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/libero_env.py resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py
git commit -m "feat: LiberoGymWrapper (dexmg-contract) + libero vectorized env factory"
```

---

## Task 5: 接线(load_pi05 schema 分发 + train --env_family + validate_libero_cfg)

**Files:**
- Modify: `resfit/lerobot/policies/pi05/load_pi05.py`
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_env_family_wiring.py`

- [ ] **Step 1: 写失败测试**

```python
import pytest
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    build_parser, validate_libero_cfg,
)


def test_env_family_default_dexmg():
    a = build_parser().parse_args(["--task", "TwoArmBoxCleanup"])
    assert a.env_family == "dexmg"


def test_libero_args_present():
    a = build_parser().parse_args(["--env_family", "libero", "--libero_suite", "libero_spatial",
                                   "--libero_task_id", "0", "--base_policy_type", "pi05"])
    assert a.libero_suite == "libero_spatial" and a.libero_task_id == 0


def test_validate_libero_cfg_rejects_stage_on():
    a = build_parser().parse_args(["--env_family", "libero", "--base_policy_type", "pi05",
                                   "--stage_conditioned"])
    with pytest.raises(ValueError):
        validate_libero_cfg(a)


def test_validate_libero_cfg_rejects_non_pi05_base():
    a = build_parser().parse_args(["--env_family", "libero", "--base_policy_type", "act"])
    with pytest.raises(ValueError):
        validate_libero_cfg(a)


def test_validate_libero_cfg_noop_for_dexmg():
    a = build_parser().parse_args(["--task", "TwoArmBoxCleanup"])
    validate_libero_cfg(a)        # dexmg 不校验
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_env_family_wiring.py -v`
Expected: FAIL(`cannot import name 'validate_libero_cfg'` / `--env_family` 非法)。

- [ ] **Step 3: 实现接线**

`train_chunk_residual.py` parser 加:
```python
    p.add_argument("--env_family", choices=["dexmg", "libero"], default="dexmg")
    p.add_argument("--libero_suite", default="libero_spatial")
    p.add_argument("--libero_task_id", type=int, default=0)
```
加守卫函数:
```python
def validate_libero_cfg(args):
    """--env_family libero 的硬约束(spec §4)。dexmg 不校验。"""
    if args.env_family != "libero":
        return
    if args.base_policy_type != "pi05":
        raise ValueError("--env_family libero 需 --base_policy_type pi05")
    if getattr(args, "reward_shaping_mode", "none") not in ("none", None):
        raise ValueError("--env_family libero 第一版需 --reward_shaping_mode none")
    if getattr(args, "offline_fraction", 0.0):
        raise ValueError("--env_family libero 第一版需 offline_fraction=0")
    if args.base_action_mode != "queue" or args.chunk_length != 1:
        raise ValueError("--env_family libero 需 --base_action_mode queue --chunk_length 1")
    for flag in ("stage_conditioned", "stage_budget", "subgoal_conditioned"):
        if getattr(args, flag, False):
            raise ValueError(f"--env_family libero 不支持 --{flag}(需双臂 stage/rel_piece)")
    if getattr(args, "potential_source", None) == "hiql":
        raise ValueError("--env_family libero 不支持 --potential_source hiql")
```
`main()` 在 `create_vectorized_env` 处分发(`:444`/`:453`):
```python
    validate_libero_cfg(args)
    if args.env_family == "libero":
        from resfit.rl_finetuning.chunk_residual.libero_env import create_libero_vectorized_env
        vec_env = create_libero_vectorized_env(args.libero_suite, args.libero_task_id, 1, args.device)
        eval_vec = create_libero_vectorized_env(args.libero_suite, args.libero_task_id,
                                                args.eval_num_envs, args.device)
    else:
        vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device, state_mode=env_state_mode)
        eval_vec = create_vectorized_env(env_name=args.task, num_envs=args.eval_num_envs, device=args.device, state_mode=eval_state_mode)
```
`load_pi05.py`:`load_pi05_base_policy(cfg, device, schema="dexmg")`,`schema=="libero"` 时用 `LiberoPi05Adapter`:
```python
    if schema == "libero":
        from resfit.rl_finetuning.chunk_residual import libero_pi05_adapter as _m
        AdapterCls = _m.LiberoPi05Adapter or _m.make_libero_pi05_adapter_cls()  # kai0 path 已注入
        return AdapterCls.from_policy(client, prompt=cfg.prompt, action_dim=cfg.action_dim,
                                      device=str(device), execute_horizon=cfg.execute_horizon,
                                      image_key_map=dict(cfg.image_key_map))
```
`build_base_policy`(pi05 分支)按 `args.env_family` 传 `schema`。

- [ ] **Step 4: Run → 通过 + dexmg 回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_env_family_wiring.py resfit/rl_finetuning/chunk_residual/tests/ -k "wiring or base or chunk" -v`
Expected: 新 5 项 PASS;既有 dexmg/base/chunk 用例全绿。

- [ ] **Step 5: Commit**

```bash
git add resfit/lerobot/policies/pi05/load_pi05.py resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_env_family_wiring.py
git commit -m "feat: --env_family libero wiring + validate_libero_cfg guard + libero pi05 schema dispatch"
```

---

## Task 6: 全量回归 + opt-in live smoke(新 env,真 pi0_libero serve)

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_smoke.py`(默认 skip)

- [ ] **Step 1: 写 live smoke(env-gate)**

```python
import os
import pytest

RUN = os.environ.get("LIBERO_RESIDUAL_SMOKE") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="opt-in:需 libero-residual env + pi0_libero serve + GPU")


def test_libero_env_constructs_and_steps():
    """真起单个 LIBERO env,reset+step 几步,验证契约 obs/action 维度。"""
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.libero_env import make_libero_env
    env, prompt = make_libero_env("libero_spatial", 0)
    obs, info = env.reset()
    assert obs["observation.state"].shape[-1] == 8 and env.action_space.shape == (7,)
    for _ in range(3):
        obs, r, term, trunc, info = env.step(np.zeros(7, np.float32))
    env.close()
```

- [ ] **Step 2: Run(默认 skip)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_smoke.py -v`
Expected: 1 SKIPPED。
手动 opt-in(在 **libero-residual** env):`LIBERO_RESIDUAL_SMOKE=1 MUJOCO_GL=egl conda run -n libero-residual python -m pytest .../test_libero_smoke.py -v`。

- [ ] **Step 3: 全量回归(确认 dexmg 默认路零影响)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全绿。**任何既有用例变红 → 停下修到 bit 等价**。

- [ ] **Step 4: 端到端 live smoke(手动,需新 env + GPU + pi0_libero serve,见 spec §5/§7)**

文档化命令(不在自动测试内):起 `pi0_libero` serve(见 MEMORY `project_pi0_libero_run_recipe`)→ 在 libero-residual env 跑 `train_chunk_residual.py --env_family libero --libero_suite libero_spatial --libero_task_id 0 --base_policy_type pi05 --base_action_mode queue --chunk_length 1 --pi0_action_dim 7 --pi0_prompt "<task instruction>" --reward_shaping_mode none --offline_fraction 0 --total_steps 50`,观察 RL loop 起得来、env step/base+残差/reward 流通、不崩。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/tests/test_libero_smoke.py
git commit -m "test: opt-in libero-residual live smoke + full dexmg regression green"
```

---

## Self-Review

**Spec 覆盖**:§2 env 命门→Task1;§3.1 libero_obs→Task2;§3.3 adapter→Task3;§3.2 env→Task4;§3.4 接线/守卫+§3.5 归一化(--dataset 复用现有逻辑)→Task5;§4 错误处理→Task5 validate_libero_cfg;§5 测试(纯逻辑单测+stub+opt-in smoke+回归)→各 Task Step1 + Task6;§6 文件→全覆盖;§7 范围/前置→Task1 + Task6 Step4。

**占位符扫描**:无 TBD/TODO。`create_libero_vectorized_env` 里 `VectorizedEnvWrapper`/`cuda_to_egl_device_id` 标注"执行时按 dexmg.py 现状核对签名"——这是真集成层、只在新 env 跑,非占位逃避(单测走 stub 不触达);Task1 未通过则整条 env 集成不启用。

**类型/命名一致**:`build_libero_serve_obs(raw_obs,*,base_key,wrist_key,state_key,prompt,env_index)`、`assemble_libero_state`、`flip_resize_image`、`adapt_4tuple`、`LiberoGymWrapper(libero_env,*,init_states,num_steps_wait,init_idx)`、`make_libero_env`/`create_libero_vectorized_env`、`validate_libero_cfg`、`load_pi05_base_policy(...,schema=)` 跨 Task 一致。
