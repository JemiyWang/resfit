# HiRes-RL ↔ RISE WM 接入模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成 `resfit/rl_finetuning/wm_bridge/` 包，让 HiRes-RL 残差 TD3 能在 RISE 微调好的 dynamics model 想象空间里训练，且 WM / RL 两侧源码 0 行改动。

**Architecture:** 一个新包 + 一个 launcher。launcher 在 `sys.modules` 预置 3 个假模块拦截 env 工厂、evaluator、pi05 基座加载器，再用 `runpy` 以 `__main__` 跑原 trainer。`ImaginationVecEnv` 鸭子类型 `VectorizedEnvWrapper`，内部攒 50 次单步调用后点火一次 WM，reward 用 PBRS 端点差。

**Tech Stack:** PyTorch, numpy, pytest, LeRobot dataset, openpi-client websocket (kai0 serve), RISE `dynamics_model.infer`

**Spec:** `docs/superpowers/specs/2026-07-19-hires-rl-wm-bridge-design.md`

## Global Constraints

- **WM 侧（`RISE_Hi/`）与 RL 侧（`resfit/rl_finetuning/` 既有文件）严格 0 行改动。** 本计划只新增 `resfit/rl_finetuning/wm_bridge/` 下的文件。任何"顺手改一下上游"都是计划失败。
- Python 解释器：`/mnt/mnt/data/chj/conda_envs/residual/bin/python`
- 测试命令前缀：`cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest`
- 动作维度恒 `16`；chunk 长度恒 `50`；WM token `(1, 25, 30)`；WM 输出帧 `192×256`；残差输入图 `84×84`（`min_vit.py:38` `num_patch=81` 写死，别的尺寸必崩）
- WM batch 维恒 `1`（`train_chunk_residual.py:841` 写死 `num_envs=1`，`:96-105` 入 buffer 只取 `[0]`）
- caption 恒为常量 `"build block"`，**禁止从数据集读**
- 想象段最多 2 个 chunk（`seg_step >= 2` 时 `truncated=True`）
- **`truncated` 时禁止把 Φ 置零**（与仓库既有 `potential_shaping` 约定相反，见 Task 6）
- 运行时必须 `--reward_shaping none` 且不传 `--potential_source`（防 double-shaping）

---

### Task 1: `state_tracker.py` —— proprio 物理空间外推

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/__init__.py`
- Create: `resfit/rl_finetuning/wm_bridge/state_tracker.py`
- Create: `resfit/rl_finetuning/wm_bridge/tests/__init__.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_state_tracker.py`

**Interfaces:**
- Consumes: 无
- Produces: `ProprioTracker(init_proprio: np.ndarray, action_dim: int = 16)`，属性 `proprio -> np.ndarray (16,)`，方法 `advance(action_chunk_physical: np.ndarray (L,16)) -> np.ndarray (16,)`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_state_tracker.py`：

```python
import numpy as np
import pytest

from resfit.rl_finetuning.wm_bridge.state_tracker import ProprioTracker


def test_initial_proprio_is_returned_as_copy():
    init = np.arange(16, dtype=np.float32)
    t = ProprioTracker(init)
    got = t.proprio
    got[0] = 999.0                      # 改返回值不得污染内部状态
    assert t.proprio[0] == 0.0


def test_advance_takes_last_action_of_chunk():
    t = ProprioTracker(np.zeros(16, dtype=np.float32))
    chunk = np.arange(50 * 16, dtype=np.float32).reshape(50, 16)
    out = t.advance(chunk)
    np.testing.assert_allclose(out, chunk[-1])
    np.testing.assert_allclose(t.proprio, chunk[-1])


def test_advance_rejects_wrong_action_dim():
    t = ProprioTracker(np.zeros(16, dtype=np.float32))
    with pytest.raises(AssertionError):
        t.advance(np.zeros((50, 14), dtype=np.float32))


def test_init_rejects_wrong_dim():
    with pytest.raises(AssertionError):
        ProprioTracker(np.zeros(14, dtype=np.float32))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_state_tracker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge'`

- [ ] **Step 3: 写实现**

创建空文件 `resfit/rl_finetuning/wm_bridge/__init__.py` 和 `resfit/rl_finetuning/wm_bridge/tests/__init__.py`。

创建 `resfit/rl_finetuning/wm_bridge/state_tracker.py`：

```python
"""想象空间的 proprio 外推。

WM 两头都没有机器人状态(输入无 state 键、输出无 state head),所以 proprio 必须由本模块
自己维护。做法沿用 RISE:下一段的关节位置 = 本段动作 chunk 的最后一个动作——成立前提是
action_type=absolute + action_space=joint。

★ 全程在物理空间。RISE 原实现的 next_state 取自归一化空间而喂 WM 的 token 取自物理空间,
混用会静默错位;本模块只在打包 act_tokens 时才归一化。
"""
from __future__ import annotations

import numpy as np

ACTION_DIM = 16


class ProprioTracker:
    def __init__(self, init_proprio, action_dim: int = ACTION_DIM):
        arr = np.asarray(init_proprio, dtype=np.float32).reshape(-1)
        assert arr.shape == (action_dim,), \
            f"init_proprio 须 ({action_dim},),got {arr.shape}"
        self.action_dim = action_dim
        self._p = arr.copy()

    @property
    def proprio(self) -> np.ndarray:
        return self._p.copy()

    def advance(self, action_chunk_physical) -> np.ndarray:
        """action_chunk_physical: (L, action_dim) 物理空间绝对关节动作。"""
        arr = np.asarray(action_chunk_physical, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == self.action_dim, \
            f"chunk 须 (L,{self.action_dim}),got {arr.shape}"
        assert arr.shape[0] > 0, "chunk 不能为空"
        self._p = arr[-1].copy()
        return self.proprio
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_state_tracker.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/__init__.py \
        resfit/rl_finetuning/wm_bridge/state_tracker.py \
        resfit/rl_finetuning/wm_bridge/tests/__init__.py \
        resfit/rl_finetuning/wm_bridge/tests/test_state_tracker.py
git commit -m "feat(wm_bridge): proprio tracker (physical space, absolute joint extrapolation)"
```

---

### Task 2: `wm_driver.py` —— 动作打包/归一化 + 帧解包

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/wm_driver.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_wm_driver.py`

**Interfaces:**
- Consumes: Task 1 的 `ACTION_DIM`
- Produces:
  - `ActionNormalizer(min_val: np.ndarray (16,), max_val: np.ndarray (16,))`，方法 `normalize(a) -> np.ndarray`、`denormalize(a) -> np.ndarray`
  - `pack_act_tokens(actions_physical: np.ndarray (50,16), normalizer) -> torch.Tensor (1,25,30) bfloat16`
  - `split_predicted_frames(video: torch.Tensor (3,3,29,192,256)) -> torch.Tensor (3,3,25,192,256)`
  - `frames_to_obs_images(frames: torch.Tensor (3,3,T,192,256), t_index: int) -> dict[str, torch.Tensor]`（值域 [0,1]，尺寸 84×84，形状 `[1,3,84,84]`）
  - 常量 `CHUNK_LENGTH=50`、`WM_TOKEN_STEPS=25`、`WM_TOKEN_SLOTS=30`、`WM_ACTION_INTERVAL=2`、`N_PREVIOUS=4`、`CAMERA_KEYS`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_wm_driver.py`：

```python
import numpy as np
import torch

from resfit.rl_finetuning.wm_bridge.wm_driver import (
    CAMERA_KEYS, CHUNK_LENGTH, N_PREVIOUS, WM_TOKEN_SLOTS, WM_TOKEN_STEPS,
    ActionNormalizer, frames_to_obs_images, pack_act_tokens,
    split_predicted_frames,
)


def _norm():
    return ActionNormalizer(min_val=-np.ones(16, np.float32),
                            max_val=np.ones(16, np.float32))


def test_normalize_roundtrip():
    n = ActionNormalizer(min_val=np.full(16, -2.0, np.float32),
                         max_val=np.full(16, 6.0, np.float32))
    a = np.linspace(-2.0, 6.0, 16, dtype=np.float32)
    np.testing.assert_allclose(n.denormalize(n.normalize(a)), a, atol=1e-5)


def test_normalize_maps_endpoints_to_pm1():
    n = ActionNormalizer(min_val=np.full(16, -2.0, np.float32),
                         max_val=np.full(16, 6.0, np.float32))
    np.testing.assert_allclose(n.normalize(np.full(16, -2.0, np.float32)),
                               -np.ones(16), atol=1e-6)
    np.testing.assert_allclose(n.normalize(np.full(16, 6.0, np.float32)),
                               np.ones(16), atol=1e-6)


def test_pack_act_tokens_shape_and_dtype():
    acts = np.zeros((CHUNK_LENGTH, 16), dtype=np.float32)
    tok = pack_act_tokens(acts, _norm())
    assert tok.shape == (1, WM_TOKEN_STEPS, WM_TOKEN_SLOTS)
    assert tok.dtype == torch.bfloat16


def test_pack_act_tokens_subsamples_every_other_action():
    acts = np.zeros((CHUNK_LENGTH, 16), dtype=np.float32)
    acts[:, 0] = np.arange(CHUNK_LENGTH)          # 0..49
    tok = pack_act_tokens(acts, _norm()).float().numpy()
    # interval=2 → 取 index 0,2,4,...,48;归一化恒等(min=-1,max=1)
    np.testing.assert_allclose(tok[0, :, 0], np.arange(0, CHUNK_LENGTH, 2), atol=1e-2)


def test_pack_act_tokens_zero_pads_slots_16_to_30():
    acts = np.ones((CHUNK_LENGTH, 16), dtype=np.float32)
    tok = pack_act_tokens(acts, _norm()).float().numpy()
    assert np.all(tok[0, :, 16:] == 0.0)


def test_split_predicted_frames_drops_history():
    video = torch.zeros(3, 3, N_PREVIOUS + WM_TOKEN_STEPS, 192, 256)
    video[:, :, :N_PREVIOUS] = 1.0                # 历史段标记
    pred = split_predicted_frames(video)
    assert pred.shape == (3, 3, WM_TOKEN_STEPS, 192, 256)
    assert torch.all(pred == 0.0)                 # 历史段已被切掉


def test_frames_to_obs_images_keys_shape_and_range():
    # WM 输出值域 [-1,1];-1 应映射到 0.0,+1 应映射到 1.0
    frames = torch.full((3, 3, WM_TOKEN_STEPS, 192, 256), -1.0)
    obs = frames_to_obs_images(frames, t_index=0)
    assert set(obs.keys()) == set(CAMERA_KEYS)
    for k in CAMERA_KEYS:
        assert obs[k].shape == (1, 3, 84, 84)
        assert torch.all(obs[k] >= 0.0) and torch.all(obs[k] <= 1.0)
        assert torch.allclose(obs[k], torch.zeros_like(obs[k]), atol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_wm_driver.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.wm_driver'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/wm_driver.py`：

```python
"""WM 张量契约:动作打包 / 帧解包 / 归一化。

绕开的上游坑(均不改上游):
- 坑1 rl_release.yaml 指向的 infer.yaml 缺 min_val/max_val → 本模块自带归一化统计量
- 坑2 wm_utils.py:193 用 CPU FloatTensor 减可能在 CUDA 上的 actions → 本模块自己打包
- 坑4 dynamics_model.py:227 把 device 当尺寸传 → 永远显式传 act_tokens,不走兜底
- 坑6 custom_pipeline.py:997 退出时 .train() → 调用方每次 infer 后显式 .eval()
- 坑7 动作维度在 14/16/30/32 之间漂移 → 显式钉死 16 并断言
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

ACTION_DIM = 16
CHUNK_LENGTH = 50
WM_TOKEN_STEPS = 25
WM_TOKEN_SLOTS = 30
WM_ACTION_INTERVAL = 2          # 50 动作 @30Hz → 25 token;WM 出 25 帧 @15Hz,同覆盖 1.667s
N_PREVIOUS = 4                  # 历史帧数
FRAME_H, FRAME_W = 192, 256
AGENT_IMG = 84                  # min_vit PatchEmbed2.num_patch=81 写死,只能 84

# 三视角。顺序即 WM video 张量第 0 维的顺序,不可改。
CAMERA_KEYS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)


class ActionNormalizer:
    """min-max 到 [-1,1],逐维。统计量由本模块持有,不依赖上游 config。"""

    def __init__(self, min_val, max_val):
        self.min_val = np.asarray(min_val, dtype=np.float32).reshape(-1)
        self.max_val = np.asarray(max_val, dtype=np.float32).reshape(-1)
        assert self.min_val.shape == (ACTION_DIM,), \
            f"min_val 须 ({ACTION_DIM},),got {self.min_val.shape}"
        assert self.max_val.shape == (ACTION_DIM,), \
            f"max_val 须 ({ACTION_DIM},),got {self.max_val.shape}"
        span = self.max_val - self.min_val
        assert np.all(span > 0), "max_val 须逐维严格大于 min_val"
        self._span = span

    def normalize(self, a):
        arr = np.asarray(a, dtype=np.float32)
        return (2.0 * (arr - self.min_val) / self._span - 1.0).astype(np.float32)

    def denormalize(self, a):
        arr = np.asarray(a, dtype=np.float32)
        return ((arr + 1.0) * 0.5 * self._span + self.min_val).astype(np.float32)


def pack_act_tokens(actions_physical, normalizer: ActionNormalizer) -> torch.Tensor:
    """(50,16) 物理空间绝对关节动作 → (1,25,30) bf16 act_tokens。"""
    arr = np.asarray(actions_physical, dtype=np.float32)
    assert arr.shape == (CHUNK_LENGTH, ACTION_DIM), \
        f"actions 须 ({CHUNK_LENGTH},{ACTION_DIM}),got {arr.shape}"
    sampled = arr[::WM_ACTION_INTERVAL]
    assert sampled.shape[0] == WM_TOKEN_STEPS, \
        f"抽样后须 {WM_TOKEN_STEPS} 步,got {sampled.shape[0]}"
    normed = normalizer.normalize(sampled)
    tokens = np.zeros((WM_TOKEN_STEPS, WM_TOKEN_SLOTS), dtype=np.float32)
    tokens[:, :ACTION_DIM] = normed
    return torch.from_numpy(tokens).unsqueeze(0).to(torch.bfloat16)


def split_predicted_frames(video: torch.Tensor) -> torch.Tensor:
    """(V,C,29,H,W) → (V,C,25,H,W),切掉前 4 帧历史。"""
    assert video.ndim == 5, f"video 须 5 维 (V,C,T,H,W),got {tuple(video.shape)}"
    total = N_PREVIOUS + WM_TOKEN_STEPS
    assert video.shape[2] == total, f"时间维须 {total},got {video.shape[2]}"
    return video[:, :, N_PREVIOUS:]


def frames_to_obs_images(frames: torch.Tensor, t_index: int) -> dict:
    """(V,C,T,H,W) 值域[-1,1] 的第 t_index 帧 → {camera_key: [1,3,84,84] float[0,1]}。"""
    assert frames.ndim == 5, f"frames 须 5 维,got {tuple(frames.shape)}"
    assert frames.shape[0] == len(CAMERA_KEYS), \
        f"视角数须 {len(CAMERA_KEYS)},got {frames.shape[0]}"
    out = {}
    for v, key in enumerate(CAMERA_KEYS):
        img = frames[v, :, t_index].float()            # (3,H,W) in [-1,1]
        img = (img + 1.0) * 0.5                        # → [0,1]
        img = img.clamp(0.0, 1.0).unsqueeze(0)         # (1,3,H,W)
        img = F.interpolate(img, size=(AGENT_IMG, AGENT_IMG),
                            mode="bilinear", align_corners=False)
        out[key] = img.contiguous()
    return out


def build_obs_window(frames: torch.Tensor, t_indices) -> torch.Tensor:
    """从预测帧取 4 帧作为下一段的 obs 窗口 → (V,C,4,H,W)。"""
    idx = list(t_indices)
    assert len(idx) == N_PREVIOUS, f"须 {N_PREVIOUS} 个索引,got {len(idx)}"
    return frames[:, :, idx].contiguous()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_wm_driver.py -q`
Expected: `7 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/wm_driver.py \
        resfit/rl_finetuning/wm_bridge/tests/test_wm_driver.py
git commit -m "feat(wm_bridge): action token packing + frame unpacking (self-owned norm stats)"
```

---

### Task 3: `scorers.py` —— Φ 打分器

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/scorers.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_scorers.py`

**Interfaces:**
- Consumes: `resfit.rl_finetuning.chunk_residual.hiql_value.load_value(path, map_location) -> (ValueMLP, info)`；`info` 含 `mean` / `std` / `state_dim` / `pi0_feat_signature`（同源锚，含 `serve_ckpt_id`）
- Produces:
  - `Scorer` Protocol：`phi(psi: np.ndarray, proprio: np.ndarray) -> float`
  - `DummyScorer()` —— 恒返回 `0.0`，仅测试用
  - `Kai0HiqlScorer(value_model, mean, std, expected_psi_anchor=None)`，类方法 `from_value_ckpt(path, device="cpu") -> Kai0HiqlScorer`，属性 `expected_psi_anchor`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_scorers.py`：

```python
import numpy as np
import pytest
import torch
from torch import nn

from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer, Kai0HiqlScorer


class _ConstValue(nn.Module):
    """把标准化后的 state 求和当 V,便于精确断言。"""

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim

    def forward(self, s):
        return s.sum(dim=-1, keepdim=True)


def test_dummy_scorer_returns_zero():
    s = DummyScorer()
    assert s.phi(np.ones(8, np.float32), np.ones(16, np.float32)) == 0.0


def test_kai0_scorer_concats_psi_and_proprio_then_standardizes():
    psi = np.array([1.0, 2.0], np.float32)
    proprio = np.array([3.0], np.float32)
    mean = np.array([1.0, 1.0, 1.0], np.float32)
    std = np.array([1.0, 2.0, 3.0], np.float32)
    sc = Kai0HiqlScorer(_ConstValue(3), mean=mean, std=std)
    # 标准化后 = [(1-1)/1, (2-1)/2, (3-1)/3] = [0, 0.5, 0.6667];求和 ≈ 1.1667
    assert sc.phi(psi, proprio) == pytest.approx(1.0 / 2 + 2.0 / 3, abs=1e-5)


def test_kai0_scorer_rejects_dim_mismatch():
    sc = Kai0HiqlScorer(_ConstValue(3),
                        mean=np.zeros(3, np.float32), std=np.ones(3, np.float32))
    with pytest.raises(AssertionError):
        sc.phi(np.ones(5, np.float32), np.ones(1, np.float32))


def test_kai0_scorer_returns_python_float():
    sc = Kai0HiqlScorer(_ConstValue(2),
                        mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    out = sc.phi(np.ones(1, np.float32), np.ones(1, np.float32))
    assert isinstance(out, float)


def test_kai0_scorer_does_not_track_grad():
    model = _ConstValue(2)
    sc = Kai0HiqlScorer(model, mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    sc.phi(np.ones(1, np.float32), np.ones(1, np.float32))
    for p in model.parameters():
        assert p.grad is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_scorers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.scorers'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/scorers.py`：

```python
"""Φ 打分器。

PBRS 端点差只需要"给一个 ψ 打一个 Φ",所以接口是单点的,不是逐帧的。
ψ 由 base_bridge 从 kai0 serve 的 prefix_feat 取得,本模块不自己编码。

安全线:shaping 只用单状态 V;gc value 的 V(s,z) 绝不进 reward。
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class Scorer(Protocol):
    def phi(self, psi: np.ndarray, proprio: np.ndarray) -> float:
        ...


class DummyScorer:
    """恒 0。仅供单元测试;真实训练须显式 --allow_dummy_scorer 才可用(见 launcher)。"""

    expected_psi_anchor = None

    def phi(self, psi, proprio) -> float:
        return 0.0


class Kai0HiqlScorer:
    """ψ ⊕ proprio → 标准化 → ValueMLP → Φ。"""

    def __init__(self, value_model, *, mean, std, expected_psi_anchor=None):
        self.model = value_model
        self.model.eval()
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        assert self.mean.shape == self.std.shape, "mean/std 维度须一致"
        assert np.all(self.std > 0), "std 须逐维为正"
        self.state_dim = int(self.mean.shape[0])
        self.expected_psi_anchor = expected_psi_anchor

    @classmethod
    def from_value_ckpt(cls, path, device="cpu"):
        from resfit.rl_finetuning.chunk_residual.hiql_value import load_value
        model, info = load_value(path, map_location=device)
        # 同源锚 = value.pt 的 pi0_feat_signature.serve_ckpt_id(Task 14 落地机制)。
        # base 统一用 kai0/pi05,不涉及 ACT;不再用 act_weight_sha。
        sig = info.get("pi0_feat_signature") or {}
        return cls(model, mean=info["mean"], std=info["std"],
                   expected_psi_anchor=sig.get("serve_ckpt_id"))

    def phi(self, psi, proprio) -> float:
        p = np.asarray(psi, dtype=np.float32).reshape(-1)
        q = np.asarray(proprio, dtype=np.float32).reshape(-1)
        state = np.concatenate([p, q])
        assert state.shape[0] == self.state_dim, \
            f"ψ⊕proprio 维度 {state.shape[0]} != value.pt 的 state_dim {self.state_dim}"
        z = (state - self.mean) / self.std
        with torch.no_grad():
            v = self.model(torch.from_numpy(z.astype(np.float32)))
        return float(v.reshape(-1)[0])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_scorers.py -q`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/scorers.py \
        resfit/rl_finetuning/wm_bridge/tests/test_scorers.py
git commit -m "feat(wm_bridge): Phi scorers (DummyScorer + Kai0HiqlScorer)"
```

---

### Task 4: `base_bridge.py` —— kai0 shim + 窗口缓存

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/base_bridge.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_base_bridge.py`

**Interfaces:**
- Consumes: Task 2 的 `CAMERA_KEYS` / `ACTION_DIM` / `CHUNK_LENGTH`
- Produces: `Kai0ImaginationBase(client, prompt: str, action_dim: int = 16)`，方法：
  - `query(raw_obs: dict) -> tuple[np.ndarray (50,16), np.ndarray]` —— 返回 `(actions_physical, psi)`，按 `raw_obs["_wm_window_token"]` 缓存
  - `get_action_chunk(raw_obs: dict, chunk_length: int) -> torch.Tensor [1,50,16]`
  - `reset() -> None`
  - `config` —— 带 `.image_features` 属性（dict，键为 `CAMERA_KEYS`）
  - 属性 `call_count: int`（测试用）

`client` 须提供 `infer(obs: dict) -> dict`，返回含 `"actions"`（`(50,16)` array-like）与 `"prefix_feat"`。

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_base_bridge.py`：

```python
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.wm_bridge.base_bridge import Kai0ImaginationBase
from resfit.rl_finetuning.wm_bridge.wm_driver import CAMERA_KEYS


class _StubClient:
    def __init__(self, psi=None):
        self.n = 0
        self._psi = np.ones(8, np.float32) if psi is None else psi

    def infer(self, obs):
        self.n += 1
        acts = np.full((50, 16), float(self.n), dtype=np.float32)
        return {"actions": acts, "prefix_feat": self._psi}


def _obs(token):
    o = {"_wm_window_token": token,
         "_wm_native_frames": np.zeros((3, 3, 192, 256), np.float32),
         "observation.state": np.zeros((1, 16), np.float32)}
    for k in CAMERA_KEYS:
        o[k] = torch.zeros(1, 3, 84, 84)
    return o


def test_same_window_token_hits_cache_only_one_serve_call():
    c = _StubClient()
    b = Kai0ImaginationBase(c, prompt="build block")
    a1, p1 = b.query(_obs(7))
    a2, p2 = b.query(_obs(7))
    assert c.n == 1                        # ★ 一个 chunk 内只调一次 serve
    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(p1, p2)


def test_new_window_token_triggers_new_serve_call():
    c = _StubClient()
    b = Kai0ImaginationBase(c, prompt="build block")
    b.query(_obs(1))
    b.query(_obs(2))
    assert c.n == 2


def test_get_action_chunk_returns_batched_tensor():
    b = Kai0ImaginationBase(_StubClient(), prompt="build block")
    out = b.get_action_chunk(_obs(0), 50)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 50, 16)


def test_get_action_chunk_shares_cache_with_query():
    c = _StubClient()
    b = Kai0ImaginationBase(c, prompt="build block")
    b.query(_obs(3))
    b.get_action_chunk(_obs(3), 50)
    assert c.n == 1                        # ★ query 与 wrapper 的取动作共用同一次调用


def test_missing_prefix_feat_raises():
    class _NoFeat(_StubClient):
        def infer(self, obs):
            return {"actions": np.zeros((50, 16), np.float32), "prefix_feat": None}

    b = Kai0ImaginationBase(_NoFeat(), prompt="build block")
    with pytest.raises(RuntimeError, match="prefix_feat"):
        b.query(_obs(0))


def test_config_exposes_camera_image_features():
    b = Kai0ImaginationBase(_StubClient(), prompt="build block")
    assert set(b.config.image_features.keys()) == set(CAMERA_KEYS)


def test_wrong_chunk_length_raises():
    b = Kai0ImaginationBase(_StubClient(), prompt="build block")
    with pytest.raises(AssertionError):
        b.get_action_chunk(_obs(0), 25)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_base_bridge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.base_bridge'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/base_bridge.py`：

```python
"""kai0 serve 的想象空间 shim。

替换 resfit.lerobot.policies.pi05.load_pi05_base_policy 的返回物(不是替换
build_base_policy —— 后者在 trainer 自身,runpy 以 __main__ 跑会造新模块对象,patch 不到)。

★ 按窗口 token 缓存。同一个观测窗口会被请求两次:
   ① ImaginationVecEnv 自己要 ψ 算 reward
   ② wrapper 在 chunk_env_wrapper.py:219 调 _base_chunk_flat 要基座动作
   两次针对同一窗口,缓存后每 chunk 只需 1 次 serve 调用 —— ψ 完全是基座调用的副产物。

★ 用原生分辨率帧,不用 84×84。实机部署时 kai0 吃原生帧,想象空间须与部署一致;
   喂降采样帧会人为削弱基座,反而抬高残差的相对增益,是论文的效度威胁。
"""
from __future__ import annotations

import numpy as np
import torch

from resfit.rl_finetuning.wm_bridge.wm_driver import (
    ACTION_DIM, CAMERA_KEYS, CHUNK_LENGTH,
)


class _BaseConfig:
    """鸭子类型 lerobot policy config:trainer 只用 .image_features 的 keys()。"""

    def __init__(self, image_keys):
        self.image_features = {k: None for k in image_keys}


class Kai0ImaginationBase:
    def __init__(self, client, *, prompt: str, action_dim: int = ACTION_DIM):
        self.client = client
        self.prompt = prompt
        self.action_dim = action_dim
        self.config = _BaseConfig(CAMERA_KEYS)
        self.call_count = 0
        self._cache_token = None
        self._cache = None

    def reset(self):
        """想象段之间不需要清缓存(token 单调递增,天然失效)。"""
        return None

    def _serve_obs(self, raw_obs) -> dict:
        native = np.asarray(raw_obs["_wm_native_frames"], dtype=np.float32)
        assert native.shape[0] == len(CAMERA_KEYS), \
            f"原生帧视角数须 {len(CAMERA_KEYS)},got {native.shape[0]}"
        state = np.asarray(raw_obs["observation.state"],
                           dtype=np.float32).reshape(-1)[:self.action_dim]
        obs = {"prompt": self.prompt, "observation/state": state}
        for i, key in enumerate(CAMERA_KEYS):
            obs[f"observation/{key.split('.')[-1]}"] = native[i]
        return obs

    def query(self, raw_obs):
        """→ (actions_physical (50,16), psi)。同一窗口 token 只打一次 serve。"""
        token = raw_obs.get("_wm_window_token")
        assert token is not None, "raw_obs 缺 _wm_window_token(ImaginationVecEnv 须填)"
        if token == self._cache_token and self._cache is not None:
            return self._cache

        result = self.client.infer(self._serve_obs(raw_obs))
        self.call_count += 1

        psi = result.get("prefix_feat")
        if psi is None:
            raise RuntimeError(
                "kai0 serve 未透出 prefix_feat —— Kai0HiqlScorer 无法工作。"
                "serve 须以透出 prefix_feat 的方式启动。")

        actions = np.asarray(result["actions"], dtype=np.float32)
        assert actions.shape == (CHUNK_LENGTH, self.action_dim), \
            f"serve 返回动作须 ({CHUNK_LENGTH},{self.action_dim}),got {actions.shape}"

        self._cache_token = token
        self._cache = (actions, np.asarray(psi, dtype=np.float32).reshape(-1))
        return self._cache

    def get_action_chunk(self, raw_obs, chunk_length: int) -> torch.Tensor:
        assert chunk_length == CHUNK_LENGTH, \
            f"想象路只支持 chunk_length={CHUNK_LENGTH},got {chunk_length}"
        actions, _ = self.query(raw_obs)
        return torch.from_numpy(actions).unsqueeze(0)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_base_bridge.py -q`
Expected: `7 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/base_bridge.py \
        resfit/rl_finetuning/wm_bridge/tests/test_base_bridge.py
git commit -m "feat(wm_bridge): kai0 serve shim with per-window cache (1 serve call per chunk)"
```

---

### Task 5: `init_states.py` —— 想象起点采样

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/init_states.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_init_states.py`

**Interfaces:**
- Consumes: Task 2 的 `N_PREVIOUS` / `CAMERA_KEYS` / `ACTION_DIM`
- Produces:
  - 常量 `BLOCK_CAPTION = "build block"`、`BLOCK_DATASETS`（两个数据集路径的 tuple）
  - `InitState`（dataclass）：字段 `obs_window: np.ndarray (3,3,4,192,256)`、`proprio: np.ndarray (16,)`、`caption: str`
  - `InitStateSampler(episode_sources, rng)`，方法 `sample() -> InitState`
  - `episode_sources` 是 `list[EpisodeSource]`，`EpisodeSource` 须有 `n_frames: int` 与 `read(frame_idx) -> tuple[np.ndarray (3,3,192,256), np.ndarray (16,)]`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_init_states.py`：

```python
import numpy as np
import pytest

from resfit.rl_finetuning.wm_bridge.init_states import (
    BLOCK_CAPTION, InitStateSampler,
)


class _StubEpisode:
    """每帧的像素值 = 帧号,便于断言取到的是哪 4 帧。"""

    def __init__(self, n_frames, tag=0.0):
        self.n_frames = n_frames
        self.tag = tag

    def read(self, frame_idx):
        frame = np.full((3, 3, 192, 256), float(frame_idx), np.float32)
        proprio = np.full(16, self.tag, np.float32)
        return frame, proprio


def test_caption_is_pinned_constant_not_from_data():
    s = InitStateSampler([_StubEpisode(100)], rng=np.random.default_rng(0))
    assert s.sample().caption == BLOCK_CAPTION == "build block"


def test_obs_window_has_four_consecutive_frames():
    s = InitStateSampler([_StubEpisode(100)], rng=np.random.default_rng(0))
    st = s.sample()
    assert st.obs_window.shape == (3, 3, 4, 192, 256)
    # 时间维上 4 帧应是连续递增的帧号
    vals = [st.obs_window[0, 0, t, 0, 0] for t in range(4)]
    assert vals == [vals[0] + i for i in range(4)]


def test_never_samples_before_frame_index_three():
    """前 3 帧凑不满 4 帧历史窗口,必须被排除。"""
    s = InitStateSampler([_StubEpisode(4)], rng=np.random.default_rng(0))
    for _ in range(20):
        st = s.sample()
        assert st.obs_window[0, 0, 0, 0, 0] >= 0.0   # 起始帧号 >= 0
        assert st.obs_window[0, 0, 3, 0, 0] <= 3.0   # 末帧号 <= n_frames-1


def test_samples_from_all_sources():
    """expert 与 rollout 混采:两个源都必须被抽到。"""
    eps = [_StubEpisode(50, tag=1.0), _StubEpisode(50, tag=2.0)]
    s = InitStateSampler(eps, rng=np.random.default_rng(0))
    tags = {float(s.sample().proprio[0]) for _ in range(60)}
    assert tags == {1.0, 2.0}


def test_proprio_shape():
    s = InitStateSampler([_StubEpisode(50)], rng=np.random.default_rng(0))
    assert s.sample().proprio.shape == (16,)


def test_rejects_episode_too_short():
    with pytest.raises(AssertionError):
        InitStateSampler([_StubEpisode(3)], rng=np.random.default_rng(0))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_init_states.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.init_states'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/init_states.py`：

```python
"""想象段起点采样。

★ caption 钉死为常量,禁止从数据集读。WM 微调时 block 域 caption 已统一为 "build block"
(RISE_Hi/temp/三域数据构建方案.md §3.4),而本机数据集里是 "build blocks"(复数)。
WM 靠 T5 编 caption 做条件,喂错一个字符即偏离训练分布。

★ expert ∪ rollout 混采。残差的主战场是"基座跑偏"的状态,只用专家集会让它从没见过
需要救场的局面;rollout 集是基座实跑轨迹,与部署时的状态分布天然对齐。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resfit.rl_finetuning.wm_bridge.wm_driver import ACTION_DIM, N_PREVIOUS

BLOCK_CAPTION = "build block"

# ★ 不写死路径。用户实际使用的 block 数据不在本机,且成功集与失败集完全分开。
#   由 --init_state_dataset(可重复)在运行时传入;缺失即硬失败,不用任何默认路径猜测。
BLOCK_DATASETS = ()


@dataclass
class InitState:
    obs_window: np.ndarray      # (3, 3, 4, 192, 256)
    proprio: np.ndarray         # (16,)
    caption: str


class InitStateSampler:
    def __init__(self, episode_sources, rng=None):
        assert len(episode_sources) > 0, "episode_sources 不能为空"
        for src in episode_sources:
            assert src.n_frames >= N_PREVIOUS, \
                f"每集至少需 {N_PREVIOUS} 帧才能凑满历史窗口,got {src.n_frames}"
        self.sources = list(episode_sources)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self) -> InitState:
        src = self.sources[self.rng.integers(len(self.sources))]
        # 末帧索引须 >= N_PREVIOUS-1,否则历史窗口越界
        end = int(self.rng.integers(N_PREVIOUS - 1, src.n_frames))
        frames = []
        proprio = None
        for t in range(end - N_PREVIOUS + 1, end + 1):
            f, p = src.read(t)
            frames.append(np.asarray(f, dtype=np.float32))
            proprio = p
        window = np.stack(frames, axis=2)          # (V,C,4,H,W)
        proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)[:ACTION_DIM]
        assert proprio.shape == (ACTION_DIM,), \
            f"proprio 须 ({ACTION_DIM},),got {proprio.shape}"
        return InitState(obs_window=window, proprio=proprio, caption=BLOCK_CAPTION)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_init_states.py -q`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/init_states.py \
        resfit/rl_finetuning/wm_bridge/tests/test_init_states.py
git commit -m "feat(wm_bridge): imagination start-state sampler (expert+rollout, pinned caption)"
```

---

### Task 6: `imagination_env.py` —— 核心想象 env

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/imagination_env.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py`

**Interfaces:**
- Consumes: Task 1 `ProprioTracker`；Task 2 全部；Task 3 `Scorer`；Task 4 `Kai0ImaginationBase`；Task 5 `InitStateSampler`
- Produces: `ImaginationVecEnv(wm, base, scorer, sampler, normalizer, gamma=0.995, max_segments=2)`
  - 鸭子类型：`reset()`、`step(action)`、`action_space`（`.shape == (16,)`）、`num_envs == 1`、`render()`、`close()`
  - `wm` 须有 `infer(obs=..., act_tokens=..., prompt=...) -> dict`，返回 `{"video": Tensor (3,3,29,192,256)}`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py`：

```python
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.wm_bridge.imagination_env import ImaginationVecEnv
from resfit.rl_finetuning.wm_bridge.init_states import InitStateSampler
from resfit.rl_finetuning.wm_bridge.wm_driver import (
    ACTION_DIM, CAMERA_KEYS, CHUNK_LENGTH, ActionNormalizer,
)


class _StubEpisode:
    n_frames = 50

    def read(self, frame_idx):
        return (np.zeros((3, 3, 192, 256), np.float32),
                np.zeros(16, np.float32))


class _StubWM:
    def __init__(self):
        self.n = 0

    def infer(self, obs=None, act_tokens=None, prompt="", **kw):
        self.n += 1
        assert act_tokens is not None, "永远不可走 act_tokens=None 兜底路径(坑4)"
        assert tuple(act_tokens.shape) == (1, 25, 30)
        return {"video": torch.zeros(3, 3, 29, 192, 256)}


class _StubBase:
    def __init__(self):
        self.n = 0

    def reset(self):
        pass

    def query(self, raw_obs):
        self.n += 1
        return (np.zeros((CHUNK_LENGTH, ACTION_DIM), np.float32),
                np.full(8, float(self.n), np.float32))

    def get_action_chunk(self, raw_obs, chunk_length):
        acts, _ = self.query(raw_obs)
        return torch.from_numpy(acts).unsqueeze(0)


class _CountingScorer:
    """Φ = ψ 的第 0 维,便于精确验算 PBRS。"""

    expected_psi_anchor = None

    def phi(self, psi, proprio):
        return float(np.asarray(psi).reshape(-1)[0])


def _env(**kw):
    return ImaginationVecEnv(
        wm=kw.pop("wm", _StubWM()),
        base=kw.pop("base", _StubBase()),
        scorer=kw.pop("scorer", _CountingScorer()),
        sampler=InitStateSampler([_StubEpisode()], rng=np.random.default_rng(0)),
        normalizer=ActionNormalizer(-np.ones(16, np.float32), np.ones(16, np.float32)),
        **kw)


def _act():
    return torch.zeros(1, ACTION_DIM)


def test_reset_returns_required_obs_keys():
    env = _env()
    obs, info = env.reset()
    for k in CAMERA_KEYS:
        assert obs[k].shape == (1, 3, 84, 84)
    assert obs["observation.state"].shape == (1, 16)
    assert "_wm_native_frames" in obs and "_wm_window_token" in obs


def test_first_49_steps_are_noops():
    env = _env()
    wm = env.wm
    env.reset()
    for i in range(CHUNK_LENGTH - 1):
        obs, r, term, trunc, info = env.step(_act())
        assert float(r.reshape(-1)[0]) == 0.0
        assert not bool(term.reshape(-1)[0]) and not bool(trunc.reshape(-1)[0])
    assert wm.n == 0                       # ★ 前 49 步一次 WM 都没调


def test_wm_fires_exactly_once_on_fiftieth_step():
    env = _env()
    wm = env.wm
    env.reset()
    for _ in range(CHUNK_LENGTH):
        env.step(_act())
    assert wm.n == 1


def test_truncated_only_after_two_chunks():
    env = _env()
    env.reset()
    for _ in range(CHUNK_LENGTH):
        _, _, _, trunc, _ = env.step(_act())
    assert not bool(trunc.reshape(-1)[0])          # 第 1 段末不截断
    for _ in range(CHUNK_LENGTH):
        _, _, term, trunc, _ = env.step(_act())
    assert bool(trunc.reshape(-1)[0])              # 第 2 段末截断
    assert not bool(term.reshape(-1)[0])           # terminated 恒 False


def test_pbrs_reward_is_gamma_phi_next_minus_phi_prev():
    """_StubBase 的 ψ[0] 依次是 1,2,3... → Φ 依次 1,2,3。"""
    env = _env(gamma=0.9)
    env.reset()                                    # 基座第 1 次调用 → Φ_prev = 1
    r = None
    for _ in range(CHUNK_LENGTH):
        _, r, _, _, _ = env.step(_act())
    # 第 2 次基座调用 → Φ_next = 2;reward = 0.9*2 - 1 = 0.8
    assert float(r.reshape(-1)[0]) == pytest.approx(0.8, abs=1e-6)


def test_phi_is_not_zeroed_at_truncation():
    """★ 与仓库既有 potential_shaping 约定相反:truncated 时 Φ 不置零。

    若置零,第二段 reward 会变成 -Φ_prev(负数);不置零则是 gamma*Φ_next - Φ_prev。
    """
    env = _env(gamma=0.9)
    env.reset()
    for _ in range(CHUNK_LENGTH):
        env.step(_act())                           # 第 1 段:Φ 1→2
    r = None
    for _ in range(CHUNK_LENGTH):
        _, r, _, trunc, _ = env.step(_act())       # 第 2 段:Φ 2→3,且 truncated
    assert bool(trunc.reshape(-1)[0])
    assert float(r.reshape(-1)[0]) == pytest.approx(0.9 * 3 - 2, abs=1e-6)
    assert float(r.reshape(-1)[0]) > 0             # 置零的话会是 -2


def test_two_segment_return_telescopes():
    """r_0 + gamma*r_1 == gamma^2*Phi_2 - Phi_0。"""
    g = 0.9
    env = _env(gamma=g)
    env.reset()                                    # Phi_0 = 1
    r0 = r1 = None
    for _ in range(CHUNK_LENGTH):
        _, r0, _, _, _ = env.step(_act())          # Phi_1 = 2
    for _ in range(CHUNK_LENGTH):
        _, r1, _, _, _ = env.step(_act())          # Phi_2 = 3
    total = float(r0.reshape(-1)[0]) + g * float(r1.reshape(-1)[0])
    assert total == pytest.approx(g * g * 3 - 1, abs=1e-6)


def test_serve_called_once_per_chunk():
    base = _StubBase()
    env = _env(base=base)
    env.reset()                                    # 1 次(起点)
    for _ in range(CHUNK_LENGTH):
        env.step(_act())
    assert base.n == 2                             # 起点 1 次 + 本段末 1 次


def test_num_envs_is_one_and_action_space_dim():
    env = _env()
    assert env.num_envs == 1
    assert env.action_space.shape[-1] == ACTION_DIM


def test_window_token_advances_after_each_chunk():
    env = _env()
    obs, _ = env.reset()
    t0 = obs["_wm_window_token"]
    for _ in range(CHUNK_LENGTH):
        obs, _, _, _, _ = env.step(_act())
    assert obs["_wm_window_token"] != t0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.imagination_env'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/imagination_env.py`：

```python
"""想象空间的向量化 env,鸭子类型 VectorizedEnvWrapper。

★ 攒批机制:ChunkResidualEnvWrapper(chunk_env_wrapper.py:181-182)是逐时间步循环,
   chunk_length=50 时会调 step() 50 次、每次 1 个动作。而 WM 要 50 个动作一次性打包成
   25 个 token。故本 env 内部攒动作,第 50 次调用才点火 WM。

   前 49 次返回缓存 obs / reward=0 / term=trunc=False 是安全的,因为 wrapper 对中间步的
   返回值本就不消费:raw_obs 与 last_info 每轮被覆盖(只有最后一轮进 :219/:222),
   reward 是累加(加 0 无影响),term/trunc 是 OR 累积(False 无影响)。

★ reward = PBRS 端点差 gamma*Phi(psi_next) - Phi(psi_prev)。不用 RISE Eq.(2) 的
   25 帧均值:均值形式跨 chunk 不 telescoping,正是 potsubgoal 塌方的根因(farming 伪奖励);
   且 prefix_feat 无纯特征端点,每个 psi 要跑一次完整 pi05 扩散推理,均值形式要贵 25 倍。

★ truncated 时绝不把 Phi 置零。仓库既有 potential_shaping(chunk_env_wrapper.py:32)约定
   done 时 Phi(s')=0,那是为真终止态设计的;想象段的 truncated 是人为视界切断。照抄会让
   两段回报退化成常数 -Phi_0,学习信号全丢。见 test_phi_is_not_zeroed_at_truncation。
"""
from __future__ import annotations

import numpy as np
import torch

from resfit.rl_finetuning.wm_bridge.state_tracker import ProprioTracker
from resfit.rl_finetuning.wm_bridge.wm_driver import (
    ACTION_DIM, CHUNK_LENGTH, N_PREVIOUS, WM_TOKEN_STEPS,
    build_obs_window, frames_to_obs_images, pack_act_tokens,
    split_predicted_frames,
)


class _ActionSpace:
    def __init__(self, dim):
        self.shape = (dim,)


class ImaginationVecEnv:
    def __init__(self, *, wm, base, scorer, sampler, normalizer,
                 gamma: float = 0.995, max_segments: int = 2,
                 num_denois_steps: int = 10):
        self.wm = wm
        self.base = base
        self.scorer = scorer
        self.sampler = sampler
        self.normalizer = normalizer
        self.gamma = float(gamma)
        self.max_segments = int(max_segments)
        self.num_denois_steps = int(num_denois_steps)

        self.num_envs = 1
        self.action_space = _ActionSpace(ACTION_DIM)

        self._window = None          # (V,C,4,H,W) [-1,1]
        self._caption = None
        self._tracker = None
        self._phi_prev = None
        self._act_buf = []
        self._seg_step = 0
        self._token = 0
        self._obs_cache = None

    # ---------- obs 构造 ----------

    def _build_obs(self, images: dict) -> dict:
        obs = dict(images)
        obs["observation.state"] = torch.from_numpy(
            self._tracker.proprio).unsqueeze(0)
        obs["_wm_native_frames"] = self._window[:, :, -1].copy()   # (V,C,H,W) 末帧
        obs["_wm_window_token"] = self._token
        return obs

    def _obs_from_window(self) -> dict:
        # 窗口末帧作为当前观测
        images = frames_to_obs_images(torch.from_numpy(self._window),
                                      t_index=N_PREVIOUS - 1)
        return self._build_obs(images)

    # ---------- 生命周期 ----------

    def reset(self, **kwargs):
        st = self.sampler.sample()
        self._window = np.asarray(st.obs_window, dtype=np.float32)
        self._caption = st.caption
        self._tracker = ProprioTracker(st.proprio)
        self._act_buf = []
        self._seg_step = 0
        self._token += 1

        self.base.reset()
        obs = self._obs_from_window()
        _, psi = self.base.query(obs)
        self._phi_prev = self.scorer.phi(psi, self._tracker.proprio)
        self._obs_cache = obs
        return obs, {}

    def step(self, action):
        a = np.asarray(
            action.detach().cpu().numpy() if isinstance(action, torch.Tensor)
            else action, dtype=np.float32).reshape(-1)[:ACTION_DIM]
        assert a.shape == (ACTION_DIM,), \
            f"动作须 ({ACTION_DIM},),got {a.shape}"
        self._act_buf.append(a)

        if len(self._act_buf) < CHUNK_LENGTH:
            return (self._obs_cache,
                    torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                    torch.zeros(1, dtype=torch.bool), {})

        # ---- 第 50 次:点火 ----
        actions = np.stack(self._act_buf, axis=0)      # (50,16) 物理空间
        self._act_buf = []

        act_tokens = pack_act_tokens(actions, self.normalizer)
        obs_in = torch.from_numpy(self._window).to(torch.bfloat16)
        out = self.wm.infer(obs=obs_in, act_tokens=act_tokens,
                            prompt=self._caption,
                            num_denois_steps=self.num_denois_steps)
        # 坑6:custom_pipeline.py:997 退出时会把模块留在 train 模式
        inner = getattr(self.wm, "transformer", None)
        if inner is not None:
            inner.eval()

        video = out["video"]
        if video.ndim == 6:                            # (b,v,c,t,h,w) → 取 b=0
            video = video[0]
        pred = split_predicted_frames(video.float())   # (V,C,25,H,W)

        self._tracker.advance(actions)
        self._window = build_obs_window(
            pred, range(WM_TOKEN_STEPS - N_PREVIOUS, WM_TOKEN_STEPS)).numpy()
        self._token += 1

        obs = self._obs_from_window()
        _, psi = self.base.query(obs)
        phi_next = self.scorer.phi(psi, self._tracker.proprio)

        # ★ PBRS 端点差。truncated 时也不置零 phi_next(见 module docstring)
        reward = self.gamma * phi_next - self._phi_prev
        self._phi_prev = phi_next

        self._seg_step += 1
        terminated = False
        truncated = self._seg_step >= self.max_segments

        self._obs_cache = obs
        return (obs,
                torch.tensor([reward], dtype=torch.float32),
                torch.tensor([terminated], dtype=torch.bool),
                torch.tensor([truncated], dtype=torch.bool),
                {})

    def render(self):
        raise NotImplementedError("想象空间不支持 render")

    def close(self):
        return None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py -q`
Expected: `10 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/imagination_env.py \
        resfit/rl_finetuning/wm_bridge/tests/test_imagination_env.py
git commit -m "feat(wm_bridge): ImaginationVecEnv (50-step buffering, PBRS endpoint reward)"
```

---

### Task 7: `fake_eval.py` —— 接管 checkpoint 存盘

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/fake_eval.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_fake_eval.py`

**Interfaces:**
- Consumes: `resfit.rl_finetuning.utils.checkpoint.save_checkpoint`
- Produces: `make_imagination_evaluator(output_dir: str, config) -> callable`，返回的函数签名兼容 `run_dexmg_evaluation(*, env, agent, num_episodes, device, global_step, save_video, save_q_plots, run_name, output_dir, subgoal, base_policy, **kw) -> dict`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_fake_eval.py`：

```python
import os

import pytest

from resfit.rl_finetuning.wm_bridge import fake_eval


def test_returns_constant_zero_success_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(str(tmp_path), config=None)
    m = ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=0)
    assert m["eval/success_rate"] == 0.0


def test_saves_checkpoint_every_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fake_eval, "save_checkpoint",
                        lambda agent, path, **k: calls.append((path, k)))
    ev = fake_eval.make_imagination_evaluator(str(tmp_path), config=None)
    ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=10)
    ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=20)
    assert len(calls) == 2                       # ★ 每次 eval 都存,不靠 best 比较
    assert calls[0][0].endswith("imagination_last.pt")
    assert calls[0][1]["global_step"] == 10
    assert calls[1][1]["global_step"] == 20


def test_tolerates_extra_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(str(tmp_path), config=None)
    m = ev(env=None, agent=object(), num_episodes=1, device="cpu",
           global_step=0, save_video=False, save_q_plots=False,
           run_name="x", output_dir="y", subgoal=None, base_policy=None)
    assert m["eval/success_rate"] == 0.0


def test_creates_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    target = tmp_path / "nested" / "run"
    ev = fake_eval.make_imagination_evaluator(str(target), config=None)
    ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=0)
    assert os.path.isdir(str(target))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_fake_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.fake_eval'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/fake_eval.py`：

```python
"""接管 train_chunk_residual.py:1315 那个全脚本唯一的 save_checkpoint。

想象空间没有真环境可 eval。但 :1315 的存盘在 eval 块里,简单关掉 eval(把
--eval_every_env_steps 设大)会让整个 run 一个 checkpoint 都不存,训练白跑。

★ 返回恒定 0.0,不返回递增哨兵去骗过 :1312 的 `sr > best_sr` 比较 —— 那是把存盘寄托在
   一个假指标上。存盘由本模块显式负责,trainer 的 best 逻辑自然失效。

想象空间不产生任何被报告的指标。真实成功率只在实机 eval 测。
"""
from __future__ import annotations

import os

from resfit.rl_finetuning.utils.checkpoint import save_checkpoint


def make_imagination_evaluator(output_dir: str, config):
    def run_dexmg_evaluation(*, env=None, agent=None, num_episodes=None,
                             device=None, global_step=0, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        save_checkpoint(agent, os.path.join(output_dir, "imagination_last.pt"),
                        global_step=global_step, config=config,
                        success_rate=0.0)
        print(f"[imagination-eval] saved imagination_last.pt "
              f"@ env_steps={global_step} (想象空间不产出成功率指标)", flush=True)
        return {"eval/success_rate": 0.0}

    return run_dexmg_evaluation
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_fake_eval.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/fake_eval.py \
        resfit/rl_finetuning/wm_bridge/tests/test_fake_eval.py
git commit -m "feat(wm_bridge): imagination evaluator that owns checkpoint saving"
```

---

### Task 7b: 优势估计器 proxy 值 logging（选项 2 监控曲线）

> 用户 2026-07-19 选选项 2：每 eval 点 rollout 10 集想象轨迹，用优势估计器逐帧打分，
> **只存原始值不判成败**（阈值 θ 留给用户离线用 block_success/block_fail 标定）。
> 设计与效度命门见 spec §6.7。**这是 imagined proxy,不是实机成功率。**

**Files:**
- Modify: `resfit/rl_finetuning/wm_bridge/fake_eval.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_adv_eval_logging.py`

**Interfaces:**
- `make_imagination_evaluator(output_dir, config, *, adv_scorer=None, eval_env=None, n_eval_episodes=10)`
  —— `adv_scorer` 为 None 时退化为纯存 checkpoint（Task 7 行为逐位不变）
- `adv_scorer.score_frames(frames) -> np.ndarray (T,)` —— 优势估计器逐帧标量
- 追加写 `<output_dir>/imagined_adv_eval.jsonl`，每集一行:
  `{env_step, episode_idx, adv_final, adv_max, adv_mean, adv_traj:[...]}`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_adv_eval_logging.py`：

```python
import json
import os

import numpy as np

from resfit.rl_finetuning.wm_bridge import fake_eval


class _StubAdvScorer:
    """逐帧打分:第 k 集第 t 帧 → 0.1*episode + 0.01*t,便于断言写出的值。"""
    def __init__(self):
        self.ep = 0
    def score_frames(self, frames):
        T = frames.shape[0] if hasattr(frames, "shape") else len(frames)
        vals = np.array([0.1 * self.ep + 0.01 * t for t in range(T)], np.float32)
        self.ep += 1
        return vals


class _StubEvalEnv:
    """reset→step×(2*chunk) 产出想象轨迹的帧;这里只需给 evaluator 帧序列。"""
    def rollout_frames(self, agent):        # evaluator 用它拿一集的逐帧
        return np.zeros((50, 3, 8, 8), np.float32)


def test_no_adv_scorer_is_bitwise_task7(tmp_path, monkeypatch):
    """adv_scorer=None 时行为与 Task 7 一致:只存 checkpoint,不写 jsonl。"""
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(str(tmp_path), config=None)
    m = ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=0)
    assert m["eval/success_rate"] == 0.0
    assert not os.path.exists(tmp_path / "imagined_adv_eval.jsonl")


def test_logs_raw_values_per_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(
        str(tmp_path), config=None,
        adv_scorer=_StubAdvScorer(), eval_env=_StubEvalEnv(), n_eval_episodes=3)
    ev(env=None, agent=object(), num_episodes=None, device="cpu", global_step=50000)
    rows = [json.loads(l) for l in open(tmp_path / "imagined_adv_eval.jsonl")]
    assert len(rows) == 3
    assert all(r["env_step"] == 50000 for r in rows)
    # 第 1 集 traj = [0.0, 0.01, ...]; final = 0.49, max = 0.49
    assert rows[0]["episode_idx"] == 0
    assert abs(rows[0]["adv_final"] - 0.49) < 1e-4
    assert len(rows[0]["adv_traj"]) == 50


def test_no_success_failure_classification_in_log(tmp_path, monkeypatch):
    """★ 只存原始值,绝不写 success/failure 字段(阈值留给用户离线定)。"""
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(
        str(tmp_path), config=None,
        adv_scorer=_StubAdvScorer(), eval_env=_StubEvalEnv(), n_eval_episodes=1)
    ev(env=None, agent=object(), num_episodes=None, device="cpu", global_step=0)
    row = json.loads(open(tmp_path / "imagined_adv_eval.jsonl").readline())
    assert "success" not in row and "is_success" not in row and "threshold" not in row


def test_appends_across_eval_points(tmp_path, monkeypatch):
    """训练中断不丢已存:每个 eval 点追加,不覆盖。"""
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(
        str(tmp_path), config=None,
        adv_scorer=_StubAdvScorer(), eval_env=_StubEvalEnv(), n_eval_episodes=2)
    ev(env=None, agent=object(), num_episodes=None, device="cpu", global_step=50000)
    ev(env=None, agent=object(), num_episodes=None, device="cpu", global_step=100000)
    rows = [json.loads(l) for l in open(tmp_path / "imagined_adv_eval.jsonl")]
    assert len(rows) == 4
    assert {r["env_step"] for r in rows} == {50000, 100000}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_adv_eval_logging.py -q`
Expected: FAIL（`make_imagination_evaluator` 尚不收 `adv_scorer` 等参数）

- [ ] **Step 3: 扩展 `make_imagination_evaluator`**

在 Task 7 的 `make_imagination_evaluator` 上加可选参数与 logging（`adv_scorer=None` 时逐位等价 Task 7）：

```python
def make_imagination_evaluator(output_dir, config, *, adv_scorer=None,
                               eval_env=None, n_eval_episodes=10):
    import json
    log_path = os.path.join(output_dir, "imagined_adv_eval.jsonl")

    def run_dexmg_evaluation(*, env=None, agent=None, num_episodes=None,
                             device=None, global_step=0, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        save_checkpoint(agent, os.path.join(output_dir, "imagination_last.pt"),
                        global_step=global_step, config=config, success_rate=0.0)
        # 选项2:优势估计器 proxy 值 logging(只存原始值,不判成败)
        if adv_scorer is not None and eval_env is not None:
            with open(log_path, "a") as f:
                for ep in range(n_eval_episodes):
                    frames = eval_env.rollout_frames(agent)      # 冻结策略 rollout 一集
                    vals = np.asarray(adv_scorer.score_frames(frames), np.float32)
                    f.write(json.dumps({
                        "env_step": int(global_step), "episode_idx": ep,
                        "adv_final": float(vals[-1]), "adv_max": float(vals.max()),
                        "adv_mean": float(vals.mean()),
                        "adv_traj": [round(float(v), 5) for v in vals],
                    }) + "\n")
        return {"eval/success_rate": 0.0}

    return run_dexmg_evaluation
```

（`np` 须在 `fake_eval.py` 顶部 `import numpy as np`。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_adv_eval_logging.py resfit/rl_finetuning/wm_bridge/tests/test_fake_eval.py -q`
Expected: 全绿（Task 7 的 4 个测试仍过 → `adv_scorer=None` 逐位等价）

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/fake_eval.py \
        resfit/rl_finetuning/wm_bridge/tests/test_adv_eval_logging.py
git commit -m "feat(wm_bridge): log raw advantage-estimator values per imagined eval rollout"
```

> **eval_env.rollout_frames 与 adv_scorer 的接线**是子任务：`eval_env` 复用 `ImaginationVecEnv`
> 做冻结-策略 rollout；`adv_scorer` 是优势估计器（另一个 kai0 组件/serve，独立加载）。
> `--eval_every_env_steps 50000`、`n_eval_episodes=10` 由 launcher 透传。

---

### Task 8: `contract.py` —— 结构断言兜底

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/contract.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_contract.py`

**Interfaces:**
- Produces:
  - `ContractError(RuntimeError)`
  - `check_upstream_symbols() -> None`
  - `check_agent_image_size() -> None`
  - `check_runtime_args(args) -> None`
  - `check_scorer(scorer, allow_dummy: bool) -> None`
  - `check_psi_samesource(scorer, serve_ckpt_id) -> None`
  - `check_all(args, scorer, serve_ckpt_id, allow_dummy) -> None`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_contract.py`：

```python
import types

import pytest

from resfit.rl_finetuning.wm_bridge import contract
from resfit.rl_finetuning.wm_bridge.contract import ContractError
from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer


def _args(**kw):
    d = {"reward_shaping": "none", "potential_source": None,
         "chunk_length": 50, "base_action_mode": "replan"}
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_agent_image_size_contract_holds_on_current_repo():
    contract.check_agent_image_size()          # 当前仓库应通过


def test_upstream_symbols_present_on_current_repo():
    contract.check_upstream_symbols()          # 当前仓库应通过


def test_wrapper_still_loops_per_timestep():
    """攒批机制的前提:wrapper 必须逐时间步调 vec_env.step()。"""
    contract.check_wrapper_step_loop()         # 当前仓库应通过


def test_reward_shaping_must_be_none():
    with pytest.raises(ContractError, match="reward_shaping"):
        contract.check_runtime_args(_args(reward_shaping="staged"))


def test_potential_source_must_be_unset():
    with pytest.raises(ContractError, match="potential_source"):
        contract.check_runtime_args(_args(potential_source="hiql"))


def test_chunk_length_must_be_fifty():
    with pytest.raises(ContractError, match="chunk_length"):
        contract.check_runtime_args(_args(chunk_length=1))


def test_valid_args_pass():
    contract.check_runtime_args(_args())


def test_dummy_scorer_rejected_without_flag():
    with pytest.raises(ContractError, match="allow_dummy_scorer"):
        contract.check_scorer(DummyScorer(), allow_dummy=False)


def test_dummy_scorer_allowed_with_flag():
    contract.check_scorer(DummyScorer(), allow_dummy=True)


def test_psi_sha_mismatch_raises():
    sc = types.SimpleNamespace(expected_psi_anchor="aaa")
    with pytest.raises(ContractError, match="同源"):
        contract.check_psi_samesource(sc, serve_ckpt_id="bbb")


def test_psi_sha_match_passes():
    sc = types.SimpleNamespace(expected_psi_anchor="aaa")
    contract.check_psi_samesource(sc, serve_ckpt_id="aaa")


def test_psi_sha_missing_warns_but_does_not_raise():
    """指纹缺失=无法验证,不等于已知异源。对齐 act_feature.py:149-153 的先例。"""
    sc = types.SimpleNamespace(expected_psi_anchor=None)
    with pytest.warns(UserWarning, match="无法验证同源"):
        contract.check_psi_samesource(sc, serve_ckpt_id="bbb")


def test_serve_sha_missing_warns_but_does_not_raise():
    sc = types.SimpleNamespace(expected_psi_anchor="aaa")
    with pytest.warns(UserWarning, match="无法验证同源"):
        contract.check_psi_samesource(sc, serve_ckpt_id=None)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_contract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.contract'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/contract.py`：

```python
"""结构断言 —— 零改动路线唯一的兜底。

monkeypatch 最大的风险是上游改了符号/签名/调用点而我们静默失配,跑出一堆垃圾数据。
本模块在启动时把所有前提检一遍,不符当场硬失败。
"""
from __future__ import annotations

import inspect


class ContractError(RuntimeError):
    pass


def check_upstream_symbols() -> None:
    """被 patch 的目标符号必须存在且签名未变。"""
    import importlib

    dexmg = importlib.import_module("resfit.dexmg.environments.dexmg")
    if not hasattr(dexmg, "create_vectorized_env"):
        raise ContractError(
            "resfit.dexmg.environments.dexmg.create_vectorized_env 不存在 —— "
            "注入点 1 失效")
    sig = inspect.signature(dexmg.create_vectorized_env)
    for p in ("num_envs", "device"):
        if p not in sig.parameters:
            raise ContractError(
                f"create_vectorized_env 签名缺参数 {p!r} —— 注入点 1 的假工厂需同签名")

    ev = importlib.import_module("resfit.rl_finetuning.utils.evaluate_dexmg")
    if not hasattr(ev, "run_dexmg_evaluation"):
        raise ContractError(
            "resfit.rl_finetuning.utils.evaluate_dexmg.run_dexmg_evaluation 不存在 —— "
            "注入点 2 失效,checkpoint 将无处存盘")

    pi05 = importlib.import_module("resfit.lerobot.policies.pi05")
    if not hasattr(pi05, "load_pi05_base_policy"):
        raise ContractError(
            "resfit.lerobot.policies.pi05.load_pi05_base_policy 不存在 —— 注入点 3 失效")


def check_agent_image_size() -> None:
    """残差 ViT 的 patch 数写死 81(=84×84),别的尺寸会在加位置编码时崩。"""
    from resfit.rl_finetuning.off_policy.networks.min_vit import PatchEmbed2

    embed = PatchEmbed2(128, use_norm=False)
    if embed.num_patch != 81:
        raise ContractError(
            f"PatchEmbed2.num_patch={embed.num_patch} != 81 —— "
            "84×84 前提已变,wm_driver.AGENT_IMG 需同步调整")


def check_wrapper_step_loop() -> None:
    """攒批机制的前提:ChunkResidualEnvWrapper.step 必须逐时间步调 vec_env.step()。

    若上游改成"一次把整个 chunk 交给 vec_env",我们攒 50 步再点火的时序就全错了,
    而且不会报错——只会把动作错位地喂给 WM。故在此硬检源码结构。
    """
    import inspect

    from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import (
        ChunkResidualEnvWrapper,
    )

    src = inspect.getsource(ChunkResidualEnvWrapper.step)
    if "for t in range(self.chunk_length)" not in src:
        raise ContractError(
            "ChunkResidualEnvWrapper.step 不再逐时间步循环 —— "
            "ImaginationVecEnv 攒 50 步再点火 WM 的时序前提已失效")
    if "self.vec_env.step(env_chunk[:, t])" not in src:
        raise ContractError(
            "ChunkResidualEnvWrapper.step 不再以单步动作调 vec_env.step —— "
            "ImaginationVecEnv.step 的入参约定已失效")


def check_runtime_args(args) -> None:
    """防 double-shaping:wrapper 会在我们的 PBRS reward 之上再叠一层自己的 shaping。"""
    shaping = getattr(args, "reward_shaping", None)
    if shaping != "none":
        raise ContractError(
            f"想象路必须 --reward_shaping none(当前 {shaping!r})—— "
            "否则 chunk_env_wrapper:202-213 会在 PBRS reward 上再叠一层 = double-shaping")

    if getattr(args, "potential_source", None) is not None:
        raise ContractError(
            "想象路不可传 --potential_source —— Φ 由 wm_bridge 自己算,"
            "wrapper 的 potential 分支必须走 shaping_reward(mode=none) 的零路径")

    cl = getattr(args, "chunk_length", None)
    if cl != 50:
        raise ContractError(
            f"想象路必须 --chunk_length 50(当前 {cl!r})—— WM 一次吃 25 token = 50 个动作")


def check_scorer(scorer, allow_dummy: bool) -> None:
    from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer

    if isinstance(scorer, DummyScorer) and not allow_dummy:
        raise ContractError(
            "DummyScorer 的 reward 恒 0,TD3 会安静跑完全程并产出看起来正常的曲线。"
            "确要如此请显式传 --allow_dummy_scorer")


def check_psi_samesource(scorer, serve_ckpt_id) -> None:
    """训 V 时编 ψ 的 kai0(pi05)权重必须与在线 serve 同源,否则静默给出垃圾势。

    同源锚 = value.pt 的 pi0_feat_signature.serve_ckpt_id(base 统一 kai0,不涉及 ACT)。
    在线 serve_ckpt_id 由 launcher 的 --pi0_serve_ckpt_id 提供,与之比对。

    ★ 锚缺失 = 无法验证,不等于已知异源。对齐仓库既有 warn-not-raise 惯例
      (pi0_feat 的 assert_pi0_caches_samesource / hiql 既有 same-source 检查):
      warn 并要求先过 S1.5 一致性检查,不 raise。只有"两边都在且不等"才 raise。
    """
    import warnings

    expected = getattr(scorer, "expected_psi_anchor", None)
    if expected is None or serve_ckpt_id is None:
        warnings.warn(
            "[wm_bridge] ψ 同源锚缺失(value.pt 未记 pi0_feat_signature.serve_ckpt_id "
            "或未传在线 --pi0_serve_ckpt_id),无法验证同源。异源不会报错,只会静默给出"
            "垃圾势 —— 务必先跑 S1.5 一致性检查再开训。", stacklevel=2)
        return
    if str(expected) != str(serve_ckpt_id):
        raise ContractError(
            f"ψ 不同源:value.pt 记的 serve_ckpt_id={expected},在线是 {serve_ckpt_id}。"
            "Φ 会被喂进它没见过的特征空间,且不会报错,只会静默给出垃圾势")


def check_all(args, scorer, serve_ckpt_id, allow_dummy: bool) -> None:
    check_upstream_symbols()
    check_wrapper_step_loop()
    check_agent_image_size()
    check_runtime_args(args)
    check_scorer(scorer, allow_dummy)
    if not allow_dummy:
        check_psi_samesource(scorer, serve_ckpt_id)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_contract.py -q`
Expected: `13 passed`

> 若 `test_upstream_symbols_present_on_current_repo` 因 robosuite 未装而 import 失败，
> 这本身就是有效信号：说明当前环境跑不了 dexmg 路。此时把该测试标记
> `@pytest.mark.skipif` 并在 skip 理由里写明依赖，**不要**放宽 `check_upstream_symbols`。

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/contract.py \
        resfit/rl_finetuning/wm_bridge/tests/test_contract.py
git commit -m "feat(wm_bridge): structural contract assertions (fail fast on upstream drift)"
```

---

### Task 9: `launch_imagination.py` —— 零改动 launcher

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/launch_imagination.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_launcher.py`

**Interfaces:**
- Consumes: Task 4/6/7/8
- Produces: `install_fakes(factories: dict) -> None`、`main(argv: list[str]) -> None`

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_launcher.py`：

```python
import sys
import types

from resfit.rl_finetuning.wm_bridge.launch_imagination import install_fakes


def test_install_fakes_registers_all_three_modules():
    saved = {k: sys.modules.get(k) for k in (
        "resfit.dexmg.environments.dexmg",
        "resfit.rl_finetuning.utils.evaluate_dexmg",
        "resfit.lerobot.policies.pi05")}
    try:
        install_fakes({
            "create_vectorized_env": lambda **kw: "ENV",
            "run_dexmg_evaluation": lambda **kw: {"eval/success_rate": 0.0},
            "load_pi05_base_policy": lambda *a, **k: "BASE",
        })
        m = sys.modules["resfit.dexmg.environments.dexmg"]
        assert m.create_vectorized_env(num_envs=1, device="cpu") == "ENV"
        assert sys.modules[
            "resfit.rl_finetuning.utils.evaluate_dexmg"].run_dexmg_evaluation()[
            "eval/success_rate"] == 0.0
        assert sys.modules[
            "resfit.lerobot.policies.pi05"].load_pi05_base_policy(None, "cpu") == "BASE"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_parent_package_gets_attribute_set():
    """`from a.b.c import d` 有的形式会走父包属性,不只查 sys.modules。"""
    saved = sys.modules.get("resfit.dexmg.environments.dexmg")
    try:
        install_fakes({"create_vectorized_env": lambda **kw: "ENV"})
        parent = sys.modules["resfit.dexmg.environments"]
        assert getattr(parent, "dexmg").create_vectorized_env(
            num_envs=1, device="cpu") == "ENV"
    finally:
        if saved is None:
            sys.modules.pop("resfit.dexmg.environments.dexmg", None)
        else:
            sys.modules["resfit.dexmg.environments.dexmg"] = saved


def test_fake_module_is_a_real_module_object():
    saved = sys.modules.get("resfit.lerobot.policies.pi05")
    try:
        install_fakes({"load_pi05_base_policy": lambda *a, **k: None})
        assert isinstance(sys.modules["resfit.lerobot.policies.pi05"],
                          types.ModuleType)
    finally:
        if saved is None:
            sys.modules.pop("resfit.lerobot.policies.pi05", None)
        else:
            sys.modules["resfit.lerobot.policies.pi05"] = saved
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_launcher.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.launch_imagination'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/launch_imagination.py`：

```python
"""零改动入口。

在 sys.modules 预置 3 个假模块拦截符号,再用 runpy 以 __main__ 方式跑原 trainer。
WM / RL 两侧源码 0 行改动。仓库先例:run_td3_meta_only_wrapper.py。

★ 不能 patch build_base_policy —— 它定义在 train_chunk_residual.py 自身,而 runpy 以
   run_name="__main__" 执行会创建新的模块对象,对已 import 版本的 patch 作用不到。
   故往下钻一层,patch 它在 :173 lazy import 的 load_pi05_base_policy。

用法:
  python -m resfit.rl_finetuning.wm_bridge.launch_imagination \
      --reward_shaping none --chunk_length 50 --base_action_mode replan \
      --base_policy_type pi05 --pi0_host <kai0> --pi0_port <port> \
      --value_ckpt <value.pt> --wm_ckpt <D> [--allow_dummy_scorer] <其余原有参数>
"""
from __future__ import annotations

import runpy
import sys
import types

_TARGETS = {
    "create_vectorized_env": "resfit.dexmg.environments.dexmg",
    "run_dexmg_evaluation": "resfit.rl_finetuning.utils.evaluate_dexmg",
    "load_pi05_base_policy": "resfit.lerobot.policies.pi05",
}

TRAINER = "resfit.rl_finetuning.chunk_residual.train_chunk_residual"


def install_fakes(factories: dict) -> None:
    """把 {符号名: 可调用} 装进对应的假模块。"""
    for symbol, fn in factories.items():
        mod_name = _TARGETS[symbol]
        mod = sys.modules.get(mod_name)
        if mod is None or not getattr(mod, "_wm_bridge_fake", False):
            mod = types.ModuleType(mod_name)
            mod._wm_bridge_fake = True
            sys.modules[mod_name] = mod
        setattr(mod, symbol, fn)

        # 部分 import 形式会走父包属性而非 sys.modules,两头都设上
        parent_name, _, leaf = mod_name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is None:
                parent = types.ModuleType(parent_name)
                parent._wm_bridge_fake = True
                sys.modules[parent_name] = parent
            setattr(parent, leaf, mod)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 延迟到此处 import:contract 会 import 上游真模块做断言,须在装假模块之前
    from resfit.rl_finetuning.wm_bridge import contract
    contract.check_upstream_symbols()
    contract.check_agent_image_size()

    from resfit.rl_finetuning.wm_bridge.builder import (
        build_imagination_factories, parse_bridge_args,
    )
    bridge_args, passthrough = parse_bridge_args(argv)
    factories = build_imagination_factories(bridge_args)

    install_fakes(factories)
    sys.argv = [TRAINER] + passthrough
    runpy.run_module(TRAINER, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_launcher.py -q`
Expected: `3 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/launch_imagination.py \
        resfit/rl_finetuning/wm_bridge/tests/test_launcher.py
git commit -m "feat(wm_bridge): zero-touch launcher via sys.modules pre-seeding + runpy"
```

---

### Task 10: `builder.py` —— 参数解析与组装

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/builder.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_builder.py`

**Interfaces:**
- Consumes: Task 3/4/6/7/8
- Produces:
  - `parse_bridge_args(argv) -> tuple[argparse.Namespace, list[str]]` —— 摘走 bridge 专属参数，其余原样透传给 trainer
  - `build_imagination_factories(bridge_args) -> dict` —— 返回 `install_fakes` 要的 3 个可调用

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_builder.py`：

```python
from resfit.rl_finetuning.wm_bridge.builder import parse_bridge_args


def test_bridge_args_are_consumed_and_rest_passed_through():
    argv = ["--wm_ckpt", "/x/D", "--value_ckpt", "/x/V.pt",
            "--pi0_host", "h", "--pi0_port", "9000",
            "--chunk_length", "50", "--reward_shaping", "none"]
    b, rest = parse_bridge_args(argv)
    assert b.wm_ckpt == "/x/D"
    assert b.value_ckpt == "/x/V.pt"
    assert "--wm_ckpt" not in rest and "--value_ckpt" not in rest
    # trainer 自己的参数必须原样保留
    assert "--chunk_length" in rest and "50" in rest
    assert "--reward_shaping" in rest and "none" in rest
    assert "--pi0_host" in rest          # pi0_* 是 trainer 的参数,不摘


def test_allow_dummy_scorer_defaults_false():
    b, _ = parse_bridge_args([])
    assert b.allow_dummy_scorer is False


def test_allow_dummy_scorer_flag():
    b, rest = parse_bridge_args(["--allow_dummy_scorer"])
    assert b.allow_dummy_scorer is True
    assert "--allow_dummy_scorer" not in rest


def test_gamma_default_matches_rise():
    b, _ = parse_bridge_args([])
    assert b.imagination_gamma == 0.995


def test_max_segments_default_is_two():
    b, _ = parse_bridge_args([])
    assert b.max_segments == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_builder.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.builder'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/builder.py`：

```python
"""bridge 专属参数解析 + 三个假符号的组装。

bridge 参数用 parse_known_args 摘走,其余原样透传给 trainer —— trainer 的 argparse
不认识 --wm_ckpt 之类,不摘会直接报错。
"""
from __future__ import annotations

import argparse

import numpy as np


def parse_bridge_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--wm_ckpt", type=str, default=None,
                   help="RISE 微调后的 dynamics model 权重目录")
    p.add_argument("--value_ckpt", type=str, default=None,
                   help="HIQL value.pt(Kai0HiqlScorer 用)")
    p.add_argument("--action_norm_json", type=str, default=None,
                   help="block 域动作 min/max 统计量 JSON;缺省用 ±1")
    p.add_argument("--allow_dummy_scorer", action="store_true",
                   help="允许 reward 恒 0 的 DummyScorer 进入真实训练(危险)")
    p.add_argument("--imagination_gamma", type=float, default=0.995,
                   help="PBRS 折扣(RISE Table IX 为 0.995)")
    p.add_argument("--max_segments", type=int, default=2,
                   help="每个想象段最多几个 chunk(RISE 递归上限为 2)")
    p.add_argument("--num_denois_steps", type=int, default=10)
    p.add_argument("--init_state_dataset", action="append", default=[],
                   help="想象起点采样的数据集路径,可重复。成功集与失败集都要传"
                        "(残差的主战场是基座跑偏的状态)。无默认值")
    p.add_argument("--pi0_serve_ckpt_id", type=str, default=None,
                   help="在线 kai0 serve 的 ckpt 标签(如 pi05_block_awbc_49999),与 value.pt 的"
                        "pi0_feat_signature.serve_ckpt_id 比对做同源校验。须与训 V/建 ψ 缓存时同一标签")
    p.add_argument("--kai0_ckpt", type=str, default=None,
                   help="(可选加固)kai0 权重目录;传了则额外用 psi_fingerprint 做内容哈希级校验。"
                        "默认同源校验走 serve_ckpt_id 标签即可,此项可不传")
    bridge_args, rest = p.parse_known_args(argv)
    return bridge_args, rest


def _load_normalizer(path):
    from resfit.rl_finetuning.wm_bridge.wm_driver import ACTION_DIM, ActionNormalizer

    if path is None:
        return ActionNormalizer(-np.ones(ACTION_DIM, np.float32),
                                np.ones(ACTION_DIM, np.float32))
    import json
    with open(path) as f:
        d = json.load(f)
    return ActionNormalizer(np.asarray(d["min"], np.float32),
                            np.asarray(d["max"], np.float32))


def _build_scorer(bridge_args):
    from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer, Kai0HiqlScorer

    if bridge_args.value_ckpt is None:
        return DummyScorer()
    return Kai0HiqlScorer.from_value_ckpt(bridge_args.value_ckpt)


def _build_episode_sources(bridge_args):
    """从 --init_state_dataset 传入的数据集建 EpisodeSource 列表。

    LeRobotEpisodeSource 读原生 192×256 三视角帧 + 16 维 proprio,值域 [-1,1]。
    """
    from resfit.rl_finetuning.wm_bridge.lerobot_source import build_block_sources
    return build_block_sources(bridge_args.init_state_dataset)


def build_imagination_factories(bridge_args) -> dict:
    from resfit.rl_finetuning.wm_bridge import contract
    from resfit.rl_finetuning.wm_bridge.base_bridge import Kai0ImaginationBase
    from resfit.rl_finetuning.wm_bridge.fake_eval import make_imagination_evaluator
    from resfit.rl_finetuning.wm_bridge.imagination_env import ImaginationVecEnv
    from resfit.rl_finetuning.wm_bridge.init_states import (
        BLOCK_CAPTION, InitStateSampler,
    )

    scorer = _build_scorer(bridge_args)
    contract.check_scorer(scorer, bridge_args.allow_dummy_scorer)
    normalizer = _load_normalizer(bridge_args.action_norm_json)

    state = {"base": None, "env": None, "output_dir": None, "config": None}

    def fake_load_pi05_base_policy(cfg, device, schema="dexmg"):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)
        base = Kai0ImaginationBase(client, prompt=BLOCK_CAPTION,
                                   action_dim=cfg.action_dim)
        if state["base"] is None:
            state["base"] = base
            if not bridge_args.allow_dummy_scorer:
                # 同源锚 = serve_ckpt_id 标签(Task 14):value.pt 记的 vs 在线传的比对。
                # 在线值来自 --pi0_serve_ckpt_id(与训 V / 建 ψ 缓存时同一标签)。
                contract.check_psi_samesource(scorer, bridge_args.pi0_serve_ckpt_id)
        return base

    def fake_create_vectorized_env(*, env_name=None, num_envs=1, device="cpu",
                                   state_mode=None, **kw):
        if state["env"] is not None:
            return state["env"]          # eval_vec 复用同一个,不另起想象流
        from resfit.rl_finetuning.wm_bridge.wm_loader import load_dynamics_model
        wm = load_dynamics_model(bridge_args.wm_ckpt, device=device)
        env = ImaginationVecEnv(
            wm=wm, base=state["base"], scorer=scorer,
            sampler=InitStateSampler(_build_episode_sources(bridge_args)),
            normalizer=normalizer,
            gamma=bridge_args.imagination_gamma,
            max_segments=bridge_args.max_segments,
            num_denois_steps=bridge_args.num_denois_steps)
        state["env"] = env
        return env

    def fake_run_dexmg_evaluation(**kw):
        out_dir = kw.get("output_dir") or state["output_dir"] or "outputs_imagination"
        return make_imagination_evaluator(out_dir, state["config"])(**kw)

    return {
        "create_vectorized_env": fake_create_vectorized_env,
        "run_dexmg_evaluation": fake_run_dexmg_evaluation,
        "load_pi05_base_policy": fake_load_pi05_base_policy,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_builder.py -q`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/builder.py \
        resfit/rl_finetuning/wm_bridge/tests/test_builder.py
git commit -m "feat(wm_bridge): bridge arg parsing + fake symbol assembly"
```

---

### Task 11: `lerobot_source.py` + `wm_loader.py` —— 真实数据与 D 的装载

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/lerobot_source.py`
- Create: `resfit/rl_finetuning/wm_bridge/psi_fingerprint.py`
- Create: `resfit/rl_finetuning/wm_bridge/wm_loader.py`
- Test: `resfit/rl_finetuning/wm_bridge/tests/test_lerobot_source.py`

**Interfaces:**
- Produces:
  - `LeRobotEpisodeSource(dataset, episode_index, camera_map)`，属性 `n_frames`，方法 `read(frame_idx) -> (np.ndarray (3,3,192,256) in [-1,1], np.ndarray (16,))`
  - `build_block_sources(dataset_paths=BLOCK_DATASETS) -> list[LeRobotEpisodeSource]`
  - `load_dynamics_model(ckpt_dir, device) -> object`（带 `.infer`）

- [ ] **Step 1: 写失败的测试**

创建 `resfit/rl_finetuning/wm_bridge/tests/test_lerobot_source.py`：

```python
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.wm_bridge.lerobot_source import LeRobotEpisodeSource

CAMERA_MAP = {
    "observation.images.top_head": 0,
    "observation.images.hand_left": 1,
    "observation.images.hand_right": 2,
}


class _StubDataset:
    """LeRobot 风格:__getitem__ 返回 CHW float[0,1] 图 + state。"""

    def __init__(self, n=10):
        self.n = n

    def __getitem__(self, i):
        d = {"observation.state": torch.arange(16, dtype=torch.float32) + i}
        for k in CAMERA_MAP:
            d[k] = torch.full((3, 192, 256), i / 100.0)
        return d


def _src(n=10):
    return LeRobotEpisodeSource(_StubDataset(n), frame_indices=list(range(n)),
                               camera_map=CAMERA_MAP)


def test_n_frames_matches_frame_indices():
    assert _src(7).n_frames == 7


def test_read_returns_three_views_at_native_resolution():
    frames, proprio = _src().read(3)
    assert frames.shape == (3, 3, 192, 256)
    assert proprio.shape == (16,)


def test_read_converts_zero_one_to_minus_one_one():
    """LeRobot 出 [0,1],WM 吃 [-1,1]。"""
    frames, _ = _src().read(0)                 # 像素值 0.0 → 应为 -1.0
    np.testing.assert_allclose(frames, -np.ones_like(frames), atol=1e-5)


def test_camera_order_follows_camera_map():
    src = _src()
    frames, _ = src.read(5)
    # 三个视角像素值相同(stub),但顺序必须稳定为 3 个视角
    assert frames.shape[0] == len(CAMERA_MAP)


def test_read_out_of_range_raises():
    with pytest.raises(IndexError):
        _src(5).read(9)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_lerobot_source.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'resfit.rl_finetuning.wm_bridge.lerobot_source'`

- [ ] **Step 3: 写实现**

创建 `resfit/rl_finetuning/wm_bridge/lerobot_source.py`：

```python
"""从 LeRobot 数据集读想象起点的原生帧。

值域约定:LeRobot 出 CHW float[0,1],WM 吃 [-1,1] —— 本模块负责转换。
分辨率保持原生 192×256(base_bridge 要原生帧;84×84 由 wm_driver 在 obs 构造时才降)。
"""
from __future__ import annotations

import numpy as np
import torch

from resfit.rl_finetuning.wm_bridge.init_states import BLOCK_DATASETS
from resfit.rl_finetuning.wm_bridge.wm_driver import ACTION_DIM, CAMERA_KEYS

# 数据集里的相机键 → CAMERA_KEYS 的顺序位。真实键名须在接数据当天核对。
DEFAULT_CAMERA_MAP = {k: i for i, k in enumerate(CAMERA_KEYS)}


class LeRobotEpisodeSource:
    def __init__(self, dataset, frame_indices, camera_map=None):
        self.dataset = dataset
        self.frame_indices = list(frame_indices)
        self.camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
        assert len(self.camera_map) == len(CAMERA_KEYS), \
            f"须 {len(CAMERA_KEYS)} 个相机,got {len(self.camera_map)}"

    @property
    def n_frames(self) -> int:
        return len(self.frame_indices)

    def read(self, frame_idx: int):
        if not 0 <= frame_idx < self.n_frames:
            raise IndexError(
                f"frame_idx {frame_idx} 越界(本集 {self.n_frames} 帧)")
        item = self.dataset[self.frame_indices[frame_idx]]

        views = [None] * len(self.camera_map)
        for key, slot in self.camera_map.items():
            img = item[key]
            arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) \
                else np.asarray(img)
            arr = arr.astype(np.float32)
            views[slot] = arr * 2.0 - 1.0           # [0,1] → [-1,1]
        frames = np.stack(views, axis=0)            # (V,C,H,W)

        st = item["observation.state"]
        proprio = (st.detach().cpu().numpy() if isinstance(st, torch.Tensor)
                   else np.asarray(st)).astype(np.float32).reshape(-1)[:ACTION_DIM]
        return frames, proprio


def build_block_sources(dataset_paths=BLOCK_DATASETS, camera_map=None):
    """expert ∪ rollout 混采池。每集一个 EpisodeSource。

    dataset_paths 必须由调用方显式传入(--init_state_dataset,可重复)。
    ★ 不提供默认路径:用户实际数据不在本机,任何硬编码默认都会静默读到错的数据集。
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if not dataset_paths:
        raise ValueError(
            "须显式传 --init_state_dataset(可重复,成功集与失败集都要传)—— "
            "本模块不提供默认数据路径")
    sources = []
    for path in dataset_paths:
        ds = LeRobotDataset(path, root=path)
        n_eps = ds.num_episodes
        for ep in range(n_eps):
            lo = int(ds.episode_data_index["from"][ep])
            hi = int(ds.episode_data_index["to"][ep])
            sources.append(LeRobotEpisodeSource(
                ds, frame_indices=range(lo, hi), camera_map=camera_map))
    assert sources, f"没从 {dataset_paths} 读到任何 episode"
    return sources
```

创建 `resfit/rl_finetuning/wm_bridge/psi_fingerprint.py`（**可选加固,非默认路径**）：

> 默认同源校验走 serve_ckpt_id 标签比对（value.pt 的 pi0_feat_signature.serve_ckpt_id vs
> 在线 --pi0_serve_ckpt_id,Task 14 落地机制,§6.4）。本文件提供更强的**内容哈希级**校验:
> 只在 launcher 传了 --kai0_ckpt 时启用,防"标签对但权重被换过"。base 用 kai0/pi05,不涉及 ACT。

```python
"""kai0 权重指纹 —— ψ 同源的可选内容哈希级校验(默认走 serve_ckpt_id 标签,见 §6.4)。

★ 对本地 checkpoint 目录算,不问运行中的 serve:kai0 在 websocket 后面,
  客户端拿不到 state_dict()。所以指纹的对象是"启动 serve 时用的那份权重文件"。

确定性策略:按路径排序、逐文件哈希内容(与仓库既有权重指纹惯例一致)。
"""
from __future__ import annotations

import hashlib
import os


def kai0_ckpt_fingerprint(ckpt_dir) -> str:
    """对 checkpoint 目录下所有权重文件的确定性 sha256。"""
    assert ckpt_dir and os.path.isdir(ckpt_dir), \
        f"--kai0_ckpt 须指向存在的目录,got {ckpt_dir!r}"
    exts = (".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".npz")
    paths = []
    for root, _, files in os.walk(ckpt_dir):
        for fn in files:
            if fn.endswith(exts):
                paths.append(os.path.join(root, fn))
    assert paths, f"{ckpt_dir} 下没找到任何权重文件({exts})"

    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.relpath(p, ckpt_dir).encode("utf-8") + b"\x00")
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()
```

创建 `resfit/rl_finetuning/wm_bridge/wm_loader.py`：

```python
"""装载 RISE 微调后的 dynamics model。只读,不改上游。"""
from __future__ import annotations

import sys


def load_dynamics_model(ckpt_dir, device="cuda"):
    assert ckpt_dir is not None, "须传 --wm_ckpt(RISE 微调后的 dynamics model 目录)"
    rise_root = "/mnt/mnt/data/resfit/RISE_Hi/policy_and_value/policy_online"
    if rise_root not in sys.path:
        sys.path.insert(0, rise_root)

    from rlinf.models.embodiment.modules.dynamics_model import DynamicsModel

    model = DynamicsModel(ckpt_dir, device=device)
    inner = getattr(model, "transformer", None)
    if inner is not None:
        inner.eval()          # 坑6:上游会把模块留在 train 模式
    return model
```

> `DynamicsModel` 的确切构造签名须在 D 权重到位当天核对（`dynamics_model.py` 的
> `__init__`）。若签名不同，只改本文件的这一行，**不要**改上游。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/wm_bridge/tests/test_lerobot_source.py -q`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/lerobot_source.py \
        resfit/rl_finetuning/wm_bridge/psi_fingerprint.py \
        resfit/rl_finetuning/wm_bridge/wm_loader.py \
        resfit/rl_finetuning/wm_bridge/tests/test_lerobot_source.py
git commit -m "feat(wm_bridge): LeRobot episode source + psi fingerprint + dynamics loader"
```

---

### Task 12: 全量回归 + S0 契约固化脚本

**Files:**
- Create: `resfit/rl_finetuning/wm_bridge/s0_contract_check.py`
- Test: 复用全部既有测试

**Interfaces:**
- Consumes: Task 1–11 全部
- Produces: 可执行脚本 `s0_contract_check.py`，产出 25 帧预测视频 + 动作可控性对比

- [ ] **Step 1: 跑全量回归，确认既有两条路径逐位未变**

Run:
```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest \
    resfit/rl_finetuning/wm_bridge/tests/ \
    resfit/rl_finetuning/chunk_residual/tests/ -q
```
Expected: 全绿。`chunk_residual/tests/` 里的 `test_env_family_wiring.py` 等必须全过 —— 它们证明 dexmg / libero 两条既有路径**逐位未变**。任何一条失败都说明违反了"0 行改动"约束。

- [ ] **Step 2: 确认两侧源码确实 0 行改动**

Run:
```bash
cd /mnt/mnt/data/resfit
git status --short RISE_Hi/ | grep -v '^??' || echo "RISE_Hi 干净"
git diff --stat HEAD -- resfit/rl_finetuning/chunk_residual/ resfit/lerobot/ resfit/dexmg/
```
Expected: `RISE_Hi 干净`，且 `git diff --stat` 无输出（既有文件零改动）。

- [ ] **Step 3: 写 S0 契约固化脚本**

创建 `resfit/rl_finetuning/wm_bridge/s0_contract_check.py`：

```python
"""S0:WM 契约固化 + 动作可控性检查。

★ 动作可控性检查不能跳过。shipped 配置下 WM 确实是动作条件的(train_mode='video_only'
   只冻结名字含 'action_' 的参数,而动作条件模块叫 act_vit_in/act_in),但条件方式是
   "动作 token 加到 T5 文本嵌入上",是相对弱的耦合。动作敏感度是经验问题,不能假设——
   若 WM 对动作不敏感,整条想象 RL 路线的前提就不成立。

用法:
  python -m resfit.rl_finetuning.wm_bridge.s0_contract_check \
      --wm_ckpt <D 目录> --out_dir /tmp/s0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from resfit.rl_finetuning.wm_bridge.init_states import BLOCK_CAPTION, InitStateSampler
from resfit.rl_finetuning.wm_bridge.lerobot_source import build_block_sources
from resfit.rl_finetuning.wm_bridge.wm_driver import (
    ACTION_DIM, CHUNK_LENGTH, ActionNormalizer, pack_act_tokens,
    split_predicted_frames,
)
from resfit.rl_finetuning.wm_bridge.wm_loader import load_dynamics_model


def _save_frames(frames, out_dir, tag):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for v in range(frames.shape[0]):
        for t in range(frames.shape[2]):
            img = ((frames[v, :, t].clamp(-1, 1) + 1) * 127.5).byte()
            arr = img.permute(1, 2, 0).cpu().numpy()
            Image.fromarray(arr).save(
                os.path.join(out_dir, f"{tag}_v{v}_t{t:02d}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_ckpt", required=True)
    ap.add_argument("--out_dir", default="/tmp/s0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    wm = load_dynamics_model(args.wm_ckpt, device=args.device)
    sampler = InitStateSampler(build_block_sources(), rng=np.random.default_rng(0))
    st = sampler.sample()
    norm = ActionNormalizer(-np.ones(ACTION_DIM, np.float32),
                            np.ones(ACTION_DIM, np.float32))

    obs_in = torch.from_numpy(st.obs_window).to(torch.bfloat16)
    print(f"[S0] obs_window {tuple(st.obs_window.shape)} caption={BLOCK_CAPTION!r}")

    results = {}
    for tag, acts in (
        ("zero", np.zeros((CHUNK_LENGTH, ACTION_DIM), np.float32)),
        ("posit", np.full((CHUNK_LENGTH, ACTION_DIM), 0.5, np.float32)),
        ("negat", np.full((CHUNK_LENGTH, ACTION_DIM), -0.5, np.float32)),
    ):
        tok = pack_act_tokens(acts, norm)
        assert tuple(tok.shape) == (1, 25, 30), tok.shape
        out = wm.infer(obs=obs_in, act_tokens=tok, prompt=BLOCK_CAPTION)
        video = out["video"]
        if video.ndim == 6:
            video = video[0]
        pred = split_predicted_frames(video.float())
        assert tuple(pred.shape)[2] == 25, pred.shape
        results[tag] = pred
        _save_frames(pred, os.path.join(args.out_dir, tag), tag)
        print(f"[S0] {tag}: pred {tuple(pred.shape)} 已存 {args.out_dir}/{tag}")

    # ★ 动作可控性:不同动作必须产生不同画面
    for a, b in (("zero", "posit"), ("zero", "negat"), ("posit", "negat")):
        d = (results[a] - results[b]).abs().mean().item()
        print(f"[S0] 动作可控性 |{a} - {b}| 平均像素差 = {d:.5f}")
        if d < 1e-3:
            print(f"[S0] ★★ 警告:{a} 与 {b} 的预测几乎相同 —— "
                  f"WM 可能对动作不敏感,想象 RL 的前提不成立")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: D 权重到位后跑 S0**

Run:
```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/conda_envs/residual/bin/python -m \
    resfit.rl_finetuning.wm_bridge.s0_contract_check \
    --wm_ckpt <D 目录> --out_dir /tmp/s0
```
Expected:
- 打印 `obs_window (3, 3, 4, 192, 256)`
- 三组 `pred (3, 3, 25, 192, 256)`
- **三个动作可控性平均像素差都显著大于 1e-3**
- 肉眼确认 `/tmp/s0/*/` 下的帧合理（不是噪声、不是静止）

若可控性差 < 1e-3，**停止后续所有阶段**并向用户报告：WM 对动作不敏感，想象 RL 前提不成立。

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/wm_bridge/s0_contract_check.py
git commit -m "feat(wm_bridge): S0 contract + action-controllability check script"
```

---

### Task 13: V 训练的 ±1 终端奖励（RL 侧改动，用户已授权）

> **本任务改的是 `hiql_value.py` / `train_hiql_value.py`，不在 Global Constraints 的
> "0 行改动"范围内** —— 那条约束针对接入模块；训 V 是独立的离线管线步骤。
> **所有既有仿真 run 的 V 必须逐位不变**，由 Step 1 的回归测试证明。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_value.py`
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value_terminal_reward.py`

**Interfaces:**
- Consumes: 既有 `build_transitions` / `train_value` / `discounted_target`
- Produces:
  - `build_transitions_with_rewards(state_seqs, success_flags) -> (s, s_next, done, reward)`
  - `train_value(..., reward=None)` —— `reward=None` 时逐位等价于现状
  - CLI `--terminal_reward_mode {legacy, success_signed}`（默认 `legacy`）、
    `--success_dataset` / `--failure_dataset`（可重复）

- [ ] **Step 1: 写失败的测试（含默认等价的回归锁）**

创建 `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value_terminal_reward.py`：

```python
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_value import (
    build_transitions, build_transitions_with_rewards, train_value,
)


def _seqs():
    return [np.arange(8, dtype=np.float32).reshape(4, 2),      # T=4
            np.arange(6, dtype=np.float32).reshape(3, 2),      # T=3
            np.zeros((1, 2), dtype=np.float32)]                # T=1 → 应被跳过


def test_done_matches_legacy_build_transitions():
    """★ 回归锁:新函数的 s/s_next/done 必须与既有 build_transitions 逐位相同。"""
    s0, sn0, d0 = build_transitions(_seqs())
    s1, sn1, d1, _ = build_transitions_with_rewards(_seqs(), [True, False, True])
    torch.testing.assert_close(s0, s1)
    torch.testing.assert_close(sn0, sn1)
    torch.testing.assert_close(d0, d1)


def test_short_sequences_skipped_consistently():
    """T<2 的 demo 被跳过时,success_flags 必须跟着跳,不能错位。"""
    # 第 3 条 T=1 被跳过 → 只剩 seq0(成功) 与 seq1(失败) 的 transition
    _, _, done, reward = build_transitions_with_rewards(
        _seqs(), [True, False, True])
    # seq0 产 3 个 transition,seq1 产 2 个 → 共 5
    assert done.shape[0] == 5
    assert float(reward[2]) == pytest.approx(1.0)    # seq0 末 → 成功 +1
    assert float(reward[4]) == pytest.approx(-1.0)   # seq1 末 → 失败 −1


def test_intermediate_rewards_are_zero():
    _, _, _, reward = build_transitions_with_rewards(_seqs(), [True, False, True])
    for i in (0, 1, 3):
        assert float(reward[i]) == 0.0


def test_flags_length_mismatch_raises():
    with pytest.raises(AssertionError):
        build_transitions_with_rewards(_seqs(), [True, False])


def test_train_value_default_is_bitwise_legacy():
    """★ 回归锁:reward=None 时必须与不传 reward 的现状逐位相同。"""
    s, sn, done = build_transitions(_seqs())
    m0, _ = train_value(s, sn, done, steps=50, seed=0)
    m1, _ = train_value(s, sn, done, reward=None, steps=50, seed=0)
    for p0, p1 in zip(m0.parameters(), m1.parameters()):
        torch.testing.assert_close(p0, p1)


def test_failure_terminal_gets_lower_value_than_success_terminal():
    """训完后,失败轨迹末态的 V 应显著低于成功轨迹末态。"""
    succ = [np.linspace(0, 1, 20, dtype=np.float32).reshape(10, 2)]
    fail = [np.linspace(0, -1, 20, dtype=np.float32).reshape(10, 2)]
    s, sn, done, reward = build_transitions_with_rewards(
        succ + fail, [True, False])
    model, _ = train_value(s, sn, done, reward=reward, steps=3000, seed=0)
    with torch.no_grad():
        v_succ = model(torch.from_numpy(succ[0][-1])).item()
        v_fail = model(torch.from_numpy(fail[0][-1])).item()
    assert v_succ > v_fail
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value_terminal_reward.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_transitions_with_rewards'`

- [ ] **Step 3: 改 `hiql_value.py`**

在 `build_transitions` 之后新增（**既有 `build_transitions` 一行不动**）：

```python
def build_transitions_with_rewards(state_seqs, success_flags):
    """带成功/失败标签的 transitions。

    ★ 跳过 T<2 的逻辑与 build_transitions 逐位一致,且 success_flags 在同一次遍历里
      消费 —— 若分两次构造,被跳过的短 demo 会让标签整体错位一格且不报错。

    reward: 成功轨迹末 +1,失败轨迹末 −1,中间 0。done 语义不变(轨迹末=1)。
    """
    assert len(success_flags) == len(state_seqs), \
        f"success_flags 长度 {len(success_flags)} != state_seqs {len(state_seqs)}"
    s_list, sn_list, done_list, rew_list = [], [], [], []
    for seq, ok in zip(state_seqs, success_flags):
        seq = np.asarray(seq, dtype=np.float32)
        T = seq.shape[0]
        if T < 2:
            continue
        s_list.append(seq[:-1])
        sn_list.append(seq[1:])
        d = np.zeros(T - 1, dtype=np.float32)
        d[-1] = 1.0
        done_list.append(d)
        r = np.zeros(T - 1, dtype=np.float32)
        r[-1] = 1.0 if ok else -1.0
        rew_list.append(r)
    s = torch.from_numpy(np.concatenate(s_list, axis=0))
    sn = torch.from_numpy(np.concatenate(sn_list, axis=0))
    done = torch.from_numpy(np.concatenate(done_list, axis=0)).unsqueeze(1)
    reward = torch.from_numpy(np.concatenate(rew_list, axis=0)).unsqueeze(1)
    return s, sn, done, reward
```

改 `train_value` 签名与那一行（**其余一行不动**）：

```python
def train_value(s, s_next, done, *, reward=None, gamma=0.99, expectile=0.7, ema=0.005,
                lr=3e-4, batch_size=256, steps=50000, hidden=256, seed=0):
```
```python
    reward = done if reward is None else reward   # 默认逐位等价:r = 1 if done else 0
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run:
```bash
cd /mnt/mnt/data/resfit
/mnt/mnt/data/chj/conda_envs/residual/bin/python -m pytest \
    resfit/rl_finetuning/chunk_residual/tests/ -q
```
Expected: 新测试 6 passed，且 `chunk_residual/tests/` **其余全部原样通过** —— 证明既有仿真
run 的 V 训练逐位未变。任何一条既有测试失败都说明破坏了向后兼容，必须修到全绿才能继续。

- [ ] **Step 5: 改 `train_hiql_value.py` 的 CLI**

新增参数（默认 `legacy`，既有命令行为逐位不变）：

```python
    p.add_argument("--terminal_reward_mode", choices=["legacy", "success_signed"],
                   default="legacy",
                   help="legacy: r=done(每条轨迹末端都 +1,不分成败,既有行为);"
                        "success_signed: 成功末端 +1、失败末端 −1(RISE 口径)")
    p.add_argument("--success_dataset", action="append", default=[],
                   help="成功 episode 数据集路径,可重复。success_signed 模式必填")
    p.add_argument("--failure_dataset", action="append", default=[],
                   help="失败 episode 数据集路径,可重复。success_signed 模式必填")
```

在读数据处分发：

```python
    if args.terminal_reward_mode == "success_signed":
        # ★ 禁止按目录名推断标签:RISE 的 'fail'/'infer' 约定在本项目数据上会静默失效
        #   (rollout_* 两个关键词都不含,会被全部误判为成功)
        assert args.success_dataset and args.failure_dataset, \
            "--terminal_reward_mode success_signed 需同时传 --success_dataset 与 " \
            "--failure_dataset(各至少一个);标签只能显式给,不从目录名推断"
        seqs, flags = [], []
        for path in args.success_dataset:
            got = read_per_demo_states(path, dataset_id=path, state_mode=args.state_mode)
            seqs.extend(got); flags.extend([True] * len(got))
        for path in args.failure_dataset:
            got = read_per_demo_states(path, dataset_id=path, state_mode=args.state_mode)
            seqs.extend(got); flags.extend([False] * len(got))
        n_succ, n_fail = flags.count(True), flags.count(False)
        print(f"[terminal-reward] success_signed: {n_succ} 成功 / {n_fail} 失败 demo")
        assert n_succ > 0 and n_fail > 0, "两类都必须非空,否则 V 退化"
        s, sn, done, reward = build_transitions_with_rewards(seqs, flags)
    else:
        s, sn, done = build_transitions(seqs)
        reward = None
```

并把 `reward` 透传进 `train_value(s, sn, done, reward=reward, ...)`。

> `read_per_demo_states` 的确切签名以 `train_hiql_value.py:27` 为准；上面按现有调用形式转写，
> 实现时对齐即可，**不要**改它的签名。

- [ ] **Step 6: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_value.py \
        resfit/rl_finetuning/chunk_residual/train_hiql_value.py \
        resfit/rl_finetuning/chunk_residual/tests/test_hiql_value_terminal_reward.py
git commit -m "feat(hiql_value): success-signed terminal reward (+1/-1), default legacy"
```

---

## 遗留给执行者的核对项

以下三处**必须在 D 权重与 kai0 serve 到位当天核对**，核对结果只改 `wm_bridge/` 内的文件，不改上游：

1. **`DynamicsModel` 构造签名**（`wm_loader.py`）—— 见 `RISE_Hi/.../dynamics_model.py` 的 `__init__`
2. **数据集真实相机键名**（`lerobot_source.py:DEFAULT_CAMERA_MAP`）—— `dynamics_model.py:44` 用 `front_color/left_color/right_color`，数据集用 `top_head/hand_left/hand_right`，两者不一致（spec §6 坑 5）
3. **kai0 serve 的 obs schema 与权重 sha 上报字段**（`base_bridge.py:_serve_obs`、`builder.py` 的 `weight_sha`）—— 实机是三相机 16 维，与既有 libero（单臂 8 维）/ dexmg schema 都不同

**动作归一化统计量**（`--action_norm_json`）须从 block 域数据实算，默认的 ±1 只是让单元测试跑通的占位，**真实训练前必须替换**，否则 act_tokens 会全部饱和在 ±1。

## 数据前提（2026-07-19 更新）

**用户实际使用的 block 数据不在本机。** 本机 `lingyu_datasets/` 下的
`expert_build_blocks`(201) / `rollout_build_blocks`(119) **不是**用户的数据，
不可作为默认路径。所有数据路径一律由 CLI 显式传入，无默认值、不猜测：

- 想象起点：`--init_state_dataset`（可重复，成功集与失败集都要传）
- V 训练：`--success_dataset` / `--failure_dataset`（Task 13）

数据格式前提（用户确认）：**成功 episode 与失败 episode 完全分开存放，已人工区分**。
故标签落在数据集层面，不需要逐集标注文件。

**今天下午需从另一台服务器传入三样**：D 权重、block 数据集、kai0 权重（专家版；
且须与训 V 时冻结的那份同源，见 `psi_fingerprint.py`）。
