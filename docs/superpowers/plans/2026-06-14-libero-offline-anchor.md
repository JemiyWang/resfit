# LIBERO + offline 锚点 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `--env_family libero` 支持 offline demo 锚(`offline_fraction>0` RLPD 混采 + `demo_bc_coef` BC 锚),demo 取自 LeRobot `physical-intelligence/libero`,base=pi0_libero。

**Architecture:** 新增 `libero_offline.py`(纯逻辑 + 一个调 base_policy 的 builder),从 LeRobot parquet 直读当前训练任务的 demo,产出与在线 `add_chunk_transition` 同构的 transition 灌进 `offline_rb`;`train_chunk_residual.py` 放开 libero 的 offline 守卫并在 offline 构建处按 env_family 分流(复用现成 memmap 缓存助手)。混采/BC 训练循环不改(靠 td 同构)。

**Tech Stack:** Python 3.11(conda env `resfit-libero`)、pandas/pyarrow(读 parquet)、PIL(解 JPEG)、torch/torchrl/tensordict、libero benchmark(取任务语言)。

**三个命门(贯穿所有 task)**:① LeRobot 数据集图是**预翻正**的 → 只 `resize_with_pad`、**绝不** `[::-1,::-1]`;② reward 用**末帧 1.0 / done**(不用 `rewards/*.npy` 事件 reward);③ state/action 用**和在线同一套** `ActionScaler`/`StateStandardizer`。

**测试运行环境**:所有单测在 `resfit-libero`:
`conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest <path> -q`

**数据集常量**:
- 根目录 `LEROBOT_LIBERO_ROOT = /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero`
- parquet 路径模板 `data/chunk-{episode_index//1000:03d}/episode_{episode_index:06d}.parquet`(chunks_size=1000,从 `meta/info.json` 读)
- parquet 列:`image`/`wrist_image`(dict{bytes,path},JPEG 256×256)、`state`(f32[8])、`actions`(f32[7])、`episode_index`
- `meta/episodes.jsonl` 行:`{"episode_index": int, "tasks": [language], "length": int}`
- buffer 图像键:`observation.images.agentview`←`image`,`observation.images.robot0_eye_in_hand`←`wrist_image`(与 `LiberoPi05Adapter.BASE_KEY/WRIST_KEY` 一致)

---

### Task 1: `libero_task_language` — 任务语言串

把 `(suite, task_id)` 映射到 benchmark 任务语言串(demo 匹配的 key)。

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/libero_offline.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_libero_offline.py
from resfit.rl_finetuning.chunk_residual.libero_offline import libero_task_language


def test_libero_task_language_moka():
    lang = libero_task_language("libero_10", 8)
    assert lang == "put both moka pots on the stove"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_libero_task_language_moka -q`
Expected: FAIL（ModuleNotFoundError: libero_offline）

- [ ] **Step 3: 写最小实现**

```python
# libero_offline.py
"""LIBERO offline 锚 buffer:从 LeRobot physical-intelligence/libero 直读当前任务的 demo,
产出与在线 add_chunk_transition 同构的 transition。命门:数据集图已预翻正(只 resize 不翻);
reward 末帧+1(不用事件 reward npy);归一化与在线同源。"""
from __future__ import annotations

import glob
import io
import json
import os

import numpy as np

AGENTVIEW_KEY = "observation.images.agentview"
WRIST_KEY = "observation.images.robot0_eye_in_hand"


def libero_task_language(suite: str, task_id: int) -> str:
    """benchmark 任务语言串(demo 按它匹配 LeRobot 数据集)。"""
    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()[suite]()
    return task_suite.get_task(int(task_id)).language.strip()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_libero_task_language_moka -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py
git commit -m "feat(libero-offline): libero_task_language (suite,task_id -> demo language)"
```

---

### Task 2: `find_demo_episodes` — 任务语言 → episode parquet 路径

按语言串在 `episodes.jsonl` 里筛 episodes,映射到 parquet 路径。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_offline.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`

- [ ] **Step 1: 写失败测试(用 tmp stub meta,不依赖大数据集)**

```python
import json as _json

def _make_stub_root(tmp_path, episodes):
    """episodes: list[(episode_index, language, length)] → 建 meta/episodes.jsonl + 空 parquet 占位。"""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for ei, lang, length in episodes:
            f.write(_json.dumps({"episode_index": ei, "tasks": [lang], "length": length}) + "\n")
    for ei, _, _ in episodes:
        d = root / "data" / f"chunk-{ei // 1000:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"episode_{ei:06d}.parquet").write_text("stub")
    return str(root)


def test_find_demo_episodes_filters_by_language(tmp_path):
    from resfit.rl_finetuning.chunk_residual.libero_offline import find_demo_episodes
    root = _make_stub_root(tmp_path, [(0, "task A", 5), (1, "task B", 7), (1001, "task A", 9)])
    paths = find_demo_episodes(root, "task A")
    assert len(paths) == 2
    assert paths[0].endswith("data/chunk-000/episode_000000.parquet")
    assert paths[1].endswith("data/chunk-001/episode_001001.parquet")


def test_find_demo_episodes_no_match_raises(tmp_path):
    from resfit.rl_finetuning.chunk_residual.libero_offline import find_demo_episodes
    import pytest
    root = _make_stub_root(tmp_path, [(0, "task A", 5)])
    with pytest.raises(ValueError):
        find_demo_episodes(root, "nonexistent task")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py -k find_demo_episodes -q`
Expected: FAIL（find_demo_episodes 未定义）

- [ ] **Step 3: 写最小实现**

```python
# libero_offline.py 追加
def find_demo_episodes(lerobot_root: str, language: str) -> list[str]:
    """读 meta/episodes.jsonl,返回 tasks[0]==language 的 episode parquet 路径(按 episode_index 升序)。"""
    ep_path = os.path.join(lerobot_root, "meta", "episodes.jsonl")
    matched = []
    with open(ep_path) as f:
        for line in f:
            rec = json.loads(line)
            tasks = rec.get("tasks", [])
            if tasks and tasks[0].strip() == language.strip():
                matched.append(int(rec["episode_index"]))
    if not matched:
        raise ValueError(f"find_demo_episodes: 数据集 {lerobot_root} 无任务语言 {language!r} 的 episode")
    matched.sort()
    return [os.path.join(lerobot_root, "data", f"chunk-{ei // 1000:03d}",
                         f"episode_{ei:06d}.parquet") for ei in matched]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py -k find_demo_episodes -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py
git commit -m "feat(libero-offline): find_demo_episodes (language -> parquet paths)"
```

---

### Task 3: `read_libero_demo` — 解码一条 parquet

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_offline.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`

- [ ] **Step 1: 写失败测试(合成迷你 parquet:JPEG bytes 图 + state/action)**

```python
def _make_demo_parquet(path, T=3):
    import pandas as pd
    from PIL import Image
    def jpg(rgb):
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG"); return {"bytes": buf.getvalue(), "path": None}
    rows = []
    for t in range(T):
        av = np.full((256, 256, 3), t * 10, np.uint8)
        wr = np.full((256, 256, 3), t * 5, np.uint8)
        rows.append({"image": jpg(av), "wrist_image": jpg(wr),
                     "state": np.full(8, float(t), np.float32), "actions": np.full(7, float(t) * 0.1, np.float32),
                     "episode_index": 0})
    pd.DataFrame(rows).to_parquet(path)


def test_read_libero_demo_shapes(tmp_path):
    import io as _io  # noqa
    from resfit.rl_finetuning.chunk_residual.libero_offline import read_libero_demo
    p = tmp_path / "episode_000000.parquet"
    _make_demo_parquet(str(p), T=4)
    d = read_libero_demo(str(p))
    assert d["state"].shape == (4, 8) and d["state"].dtype == np.float32
    assert d["action"].shape == (4, 7) and d["action"].dtype == np.float32
    assert d["agentview"].shape == (4, 256, 256, 3) and d["agentview"].dtype == np.uint8
    assert d["wrist"].shape == (4, 256, 256, 3)
```

注意：测试文件顶部已 `import io`（见 Task 2/3 用到）。`_make_demo_parquet` 内部已 `import pandas`/`PIL`。

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_read_libero_demo_shapes -q`
Expected: FAIL（read_libero_demo 未定义）

- [ ] **Step 3: 写最小实现**

```python
# libero_offline.py 追加
def _decode_img_col(col) -> np.ndarray:
    """LeRobot v2.0 image 列(每帧 dict{bytes,path} 的 JPEG)→ (T,H,W,3) uint8。"""
    from PIL import Image
    out = []
    for cell in col:
        b = cell["bytes"] if isinstance(cell, dict) else cell
        out.append(np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8))
    return np.stack(out, axis=0)


def read_libero_demo(parquet_path: str) -> dict:
    """一条 episode parquet → {state(T,8) f32, action(T,7) f32, agentview(T,256,256,3) u8, wrist(...) u8}。"""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    state = np.stack([np.asarray(x, np.float32) for x in df["state"]], axis=0)
    action = np.stack([np.asarray(x, np.float32) for x in df["actions"]], axis=0)
    return {"state": state, "action": action,
            "agentview": _decode_img_col(df["image"]), "wrist": _decode_img_col(df["wrist_image"])}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_read_libero_demo_shapes -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py
git commit -m "feat(libero-offline): read_libero_demo (parquet -> per-frame state/action/images)"
```

---

### Task 4: `_demo_to_transitions` — 一条 demo → 同构 transition(核心 + 命门①②③)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_offline.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`

- [ ] **Step 1: 写失败测试(stub 标量缩放 + 非对称图验没翻转 + reward/done/schema)**

```python
class _IdScaler:
    def scale(self, x):  # 恒等,便于断言值穿透
        return x

class _IdStd:
    def standardize(self, x):
        return x


def _toy_demo(T=3):
    # agentview 上半亮(255)下半暗(0):用来验"没被上下翻转"
    av = np.zeros((T, 256, 256, 3), np.uint8); av[:, :128] = 255
    wr = np.zeros((T, 256, 256, 3), np.uint8)
    return {"state": np.arange(T * 8, dtype=np.float32).reshape(T, 8),
            "action": np.arange(T * 7, dtype=np.float32).reshape(T, 7) * 0.1,
            "agentview": av, "wrist": wr}


def test_demo_to_transitions_schema_reward_noflip():
    import torch
    from resfit.rl_finetuning.chunk_residual.libero_offline import _demo_to_transitions, AGENTVIEW_KEY, WRIST_KEY
    demo = _toy_demo(T=3)
    tds = _demo_to_transitions(demo, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
                               base_actions=None, image_size=84)
    assert len(tds) == 2                         # T-1 个 transition
    td = tds[0]
    obs = td["obs"]
    # 同构键集(与在线 add_chunk_transition:obs={state,base_action,stage_id,images})
    assert set(obs.keys()) >= {"observation.state", "observation.base_action",
                               "observation.stage_id", AGENTVIEW_KEY, WRIST_KEY}
    assert obs["observation.state"].shape == (8,)
    assert obs["observation.base_action"].shape == (7,)
    assert obs[AGENTVIEW_KEY].shape == (3, 84, 84) and obs[AGENTVIEW_KEY].dtype == torch.uint8
    assert obs["observation.stage_id"].item() == 0.0
    # gt 模式(base_actions=None):base_action == action
    assert torch.allclose(obs["observation.base_action"], td["action"])
    # reward/done:仅末帧 transition 的 next 给 reward=1/done=True
    assert tds[-1]["next"]["reward"].item() == 1.0 and bool(tds[-1]["next"]["done"]) is True
    assert tds[0]["next"]["reward"].item() == 0.0 and bool(tds[0]["next"]["done"]) is False
    assert td["max_stage"].item() == 0.0 and td["_priority"].item() == 10.0
    # 命门①:没被上下翻转 → 上半(行<42)应远亮于下半
    img = obs[AGENTVIEW_KEY].float()
    assert img[:, :42, :].mean() > img[:, 42:, :].mean() + 50
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_demo_to_transitions_schema_reward_noflip -q`
Expected: FAIL（_demo_to_transitions 未定义）

- [ ] **Step 3: 写最小实现**

```python
# libero_offline.py 追加
def _img_chw_uint8(hwc_uint8, size):
    """命门①:只 resize_with_pad(不翻转)→ CHW uint8。"""
    from resfit.rl_finetuning.chunk_residual.libero_obs import resize_with_pad
    resized = resize_with_pad(hwc_uint8, size, size)        # HWC uint8,无翻转
    return np.transpose(resized, (2, 0, 1))                 # CHW uint8


def _demo_to_transitions(demo, *, action_scaler, state_standardizer, base_actions, image_size):
    """一条 demo → list[TensorDict],与在线 add_chunk_transition 同构。

    base_actions: (T,7) 已缩放的 base 动作(base_policy 模式),或 None(gt 模式 → base=action)。
    命门②:reward 仅末帧 transition=1.0、done 仅末帧 True(demo 是成功轨迹)。
    命门③:state 用 state_standardizer、action 用 action_scaler(与在线同源)。
    """
    import torch
    from tensordict import TensorDict
    state = torch.as_tensor(demo["state"], dtype=torch.float32)
    action_raw = torch.as_tensor(demo["action"], dtype=torch.float32)
    T = state.shape[0]
    if T < 2:
        return []
    state_std = state_standardizer.standardize(state)                  # (T,8)
    action = action_scaler.scale(action_raw)                          # (T,7) 缩放
    base = base_actions if base_actions is not None else action       # (T,7)
    img_av = torch.stack([torch.as_tensor(_img_chw_uint8(demo["agentview"][t], image_size)) for t in range(T)])
    img_wr = torch.stack([torch.as_tensor(_img_chw_uint8(demo["wrist"][t], image_size)) for t in range(T)])

    def _obs(t):
        return {"observation.state": state_std[t], "observation.base_action": base[t],
                "observation.stage_id": torch.zeros(1, dtype=torch.float32),
                AGENTVIEW_KEY: img_av[t], WRIST_KEY: img_wr[t]}

    out = []
    for t in range(T - 1):
        last = (t == T - 2)
        td = TensorDict({
            "obs": TensorDict(_obs(t), batch_size=[]),
            "next": TensorDict({"obs": TensorDict(_obs(t + 1), batch_size=[]),
                                "done": torch.tensor(bool(last)),
                                "reward": torch.tensor(1.0 if last else 0.0, dtype=torch.float32)},
                               batch_size=[]),
            "action": action[t],
            "max_stage": torch.tensor(0.0, dtype=torch.float32),
            "_priority": torch.tensor(10.0, dtype=torch.float32),
        }, batch_size=[])
        out.append(td)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_demo_to_transitions_schema_reward_noflip -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py
git commit -m "feat(libero-offline): _demo_to_transitions (isomorphic td, no-flip img, terminal reward)"
```

---

### Task 5: `_libero_demo_base_actions` — base_policy 模式逐帧出 base

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_offline.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`

- [ ] **Step 1: 写失败测试(stub base_policy 记录 raw_obs)**

```python
class _StubBase:
    """记录每次 select_action 的 raw_obs;返回固定原始动作。"""
    def __init__(self):
        self.seen = []
        self.reset_calls = 0
    def reset(self):
        self.reset_calls += 1
    def select_action(self, raw_obs):
        import torch
        self.seen.append({k: (v.clone() if hasattr(v, "clone") else v) for k, v in raw_obs.items()})
        return torch.full((1, 7), 0.5)


def test_libero_demo_base_actions_feeds_raw_state_and_noflip_84():
    import torch
    from resfit.rl_finetuning.chunk_residual.libero_offline import _libero_demo_base_actions, AGENTVIEW_KEY
    demo = _toy_demo(T=3)
    base = _StubBase()
    out = _libero_demo_base_actions(demo, base, _IdScaler(), image_size=84, device="cpu")
    assert out.shape == (3, 7)
    assert torch.allclose(out, torch.full((3, 7), 0.5))      # 恒等 scaler 穿透
    assert base.reset_calls == 1                              # 逐 demo reset 一次
    first = base.seen[0]
    # 命门③(base 侧):喂 raw(未标准化)state,即 demo["state"][0]
    assert torch.allclose(first["observation.state"].reshape(-1), torch.as_tensor(demo["state"][0]))
    # 喂 84 CHW float[0,1] 图、且没翻转(上半亮)
    img = first[AGENTVIEW_KEY].reshape(3, 84, 84)
    assert img.max() <= 1.0 + 1e-6
    assert img[:, :42, :].mean() > img[:, 42:, :].mean() + 0.2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_libero_demo_base_actions_feeds_raw_state_and_noflip_84 -q`
Expected: FAIL（_libero_demo_base_actions 未定义）

- [ ] **Step 3: 写最小实现**

```python
# libero_offline.py 追加
def _libero_demo_base_actions(demo, base_policy, action_scaler, image_size, device):
    """逐帧调 base_policy.select_action 出 base 动作(缩放后,(T,7))。

    对齐在线 select_action:喂 raw(未标准化)state(8)+ 84 CHW float[0,1] 图(adapter 内部
    build_libero_serve_obs 再 resize 224,无翻转)。逐 demo 先 reset 清队列、顺序不可批。
    """
    import torch
    state = torch.as_tensor(demo["state"], dtype=torch.float32)
    T = state.shape[0]
    img_av = torch.stack([torch.as_tensor(_img_chw_uint8(demo["agentview"][t], image_size), dtype=torch.float32) / 255.0
                          for t in range(T)])
    img_wr = torch.stack([torch.as_tensor(_img_chw_uint8(demo["wrist"][t], image_size), dtype=torch.float32) / 255.0
                          for t in range(T)])
    base_policy.reset()
    out = []
    for t in range(T):
        raw_obs = {"observation.state": state[t:t + 1].to(device),
                   AGENTVIEW_KEY: img_av[t:t + 1].to(device),
                   WRIST_KEY: img_wr[t:t + 1].to(device)}
        out.append(base_policy.select_action(raw_obs).to("cpu"))
    base_raw = torch.cat(out, dim=0)                                  # (T,7) 原始尺度
    return action_scaler.scale(base_raw)                             # (T,7) 缩放
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_libero_demo_base_actions_feeds_raw_state_and_noflip_84 -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py
git commit -m "feat(libero-offline): _libero_demo_base_actions (base_policy per-frame, raw state, 84 no-flip)"
```

---

### Task 6: `count_libero_offline_transitions` + `build_libero_offline_buffer` 编排

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_offline.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`

- [ ] **Step 1: 写失败测试(monkeypatch resolve/read,真 prioritized rb)**

```python
def test_build_libero_offline_buffer_gt(monkeypatch, tmp_path):
    import torch
    from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
    import resfit.rl_finetuning.chunk_residual.libero_offline as lo

    demos = [_toy_demo(T=3), _toy_demo(T=4)]            # 2 + 3 = 5 个 transition
    monkeypatch.setattr(lo, "libero_task_language", lambda s, t: "L")
    monkeypatch.setattr(lo, "find_demo_episodes", lambda root, lang: ["p0", "p1"])
    monkeypatch.setattr(lo, "read_libero_demo", lambda p: demos[["p0", "p1"].index(p)])

    cap = lo.count_libero_offline_transitions("root", "libero_10", 8)
    assert cap == 5

    rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=cap, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority", batch_size=2)
    lo.build_libero_offline_buffer(rb, lerobot_root="root", suite="libero_10", task_id=8,
                                   action_scaler=_IdScaler(), state_standardizer=_IdStd(),
                                   base_policy=None, base_mode="gt", base_device="cpu", image_size=84)
    assert len(rb) == 5
    b = rb.sample(2)
    assert "observation.state" in b["obs"] and b["action"].shape[-1] == 7
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_build_libero_offline_buffer_gt -q`
Expected: FAIL（count_libero_offline_transitions / build_libero_offline_buffer 未定义）

- [ ] **Step 3: 写最小实现**

```python
# libero_offline.py 追加
def count_libero_offline_transitions(lerobot_root, suite, task_id, num_demos=None) -> int:
    """从 episodes.jsonl 的 length 求 sum(T-1)(不解 parquet,秒级),给 LazyTensorStorage 精确定容。"""
    language = libero_task_language(suite, task_id)
    ep_path = os.path.join(lerobot_root, "meta", "episodes.jsonl")
    lengths = []
    with open(ep_path) as f:
        for line in f:
            rec = json.loads(line)
            tasks = rec.get("tasks", [])
            if tasks and tasks[0].strip() == language:
                lengths.append(int(rec["length"]))
    lengths.sort()
    if num_demos is not None:
        lengths = lengths[:num_demos]
    return int(sum(max(0, L - 1) for L in lengths))


def build_libero_offline_buffer(offline_rb, *, lerobot_root, suite, task_id,
                                action_scaler, state_standardizer,
                                base_policy, base_mode, base_device="cpu",
                                image_size=84, num_demos=None) -> None:
    """灌当前任务的 demo 进 offline_rb(prioritized + MultiStepTransform)。base_mode∈{gt,base_policy}。"""
    if base_mode not in ("gt", "base_policy"):
        raise ValueError(f"base_mode 必须是 gt|base_policy,得到 {base_mode!r}")
    if base_mode == "base_policy" and base_policy is None:
        raise ValueError("base_mode=base_policy 需传 base_policy")
    language = libero_task_language(suite, task_id)
    paths = find_demo_episodes(lerobot_root, language)
    if num_demos is not None:
        paths = paths[:num_demos]
    for p in paths:
        demo = read_libero_demo(p)
        if demo["state"].shape[0] < 2:
            print(f"[libero-offline] 跳过 {p}(T<2)")
            continue
        base_actions = (_libero_demo_base_actions(demo, base_policy, action_scaler, image_size, base_device)
                        if base_mode == "base_policy" else None)
        for td in _demo_to_transitions(demo, action_scaler=action_scaler,
                                       state_standardizer=state_standardizer,
                                       base_actions=base_actions, image_size=image_size):
            offline_rb.add(td)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py::test_build_libero_offline_buffer_gt -q`
Expected: PASS

- [ ] **Step 5: 跑整文件确认全绿 + 提交**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py -q`
Expected: PASS（全部）

```bash
git add resfit/rl_finetuning/chunk_residual/libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py
git commit -m "feat(libero-offline): build_libero_offline_buffer + count (orchestration)"
```

---

### Task 7: train 接线 — validate 放开 + offline 构建 libero 分支 + 缓存签名

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
  - `validate_libero_cfg`(~315 行:offline_fraction 硬禁)
  - offline 构建块(~717 行 `if args.offline_fraction > 0.0:`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline_validate.py`

- [ ] **Step 1: 写失败测试(validate 放开 + 仍禁 subgoal/stage)**

```python
# tests/test_libero_offline_validate.py
import argparse
import pytest
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import validate_libero_cfg


def _libero_args(**over):
    a = argparse.Namespace(
        env_family="libero", libero_stats_json="/x/stats.json", base_policy_type="pi05",
        base_action_mode="queue", chunk_length=1, reward_shaping=None, staged_reward=False,
        offline_fraction=0.0, pi0_action_dim=7, stage_conditioned=False, stage_budget=None,
        subgoal_conditioned=False, potential_source="stage", actor="raw", demo_bc_coef=0.0,
        offline_dataset_path=None)
    a.__dict__.update(over)
    return a


def test_validate_allows_libero_offline_fraction():
    validate_libero_cfg(_libero_args(offline_fraction=0.5))   # 不再抛

def test_validate_libero_offline_requires_actor_raw():
    with pytest.raises(ValueError):
        validate_libero_cfg(_libero_args(offline_fraction=0.5, actor="flow"))

def test_validate_libero_still_rejects_subgoal_with_offline():
    with pytest.raises(ValueError):
        validate_libero_cfg(_libero_args(offline_fraction=0.5, subgoal_conditioned=True))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline_validate.py -q`
Expected: FAIL（`test_validate_allows_libero_offline_fraction` 抛 ValueError，因当前仍硬禁）

- [ ] **Step 3: 改 validate_libero_cfg(放开 offline,新增 actor=raw 约束)**

把这段（约 313-317 行）：
```python
    if getattr(args, "offline_fraction", 0) not in (0, 0.0, None):
        raise ValueError(
            "--env_family libero 不支持 offline 锚 buffer,需 --offline_fraction 0;当前 "
            f"offline_fraction={getattr(args, 'offline_fraction', None)!r}")
```
替换为：
```python
    if getattr(args, "offline_fraction", 0) and args.offline_fraction > 0:
        if getattr(args, "actor", "raw") != "raw":
            raise ValueError(
                "--env_family libero + offline_fraction>0 需 --actor raw(BC 锚走 raw actor);"
                f"当前 actor={getattr(args, 'actor', None)!r}")
        # 注:subgoal/stage 仍由下方 stage_conditioned/subgoal_conditioned 检查拦截;
        # offline demo 源是 LeRobot 数据集(自动按任务匹配),无需 --offline_dataset_path。
```
（`subgoal_conditioned`/`stage_conditioned`/`stage_budget` 的拒绝在同函数后面,已覆盖 offline+subgoal 组合,无需重复。）

- [ ] **Step 4: 跑 validate 测试确认通过**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline_validate.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 改 offline 构建块(env_family 分流 + libero 签名)**

在 `_offline_buffer_signature`(176 行)附近**新增** libero 签名函数：
```python
def _libero_offline_signature(args, image_keys, offline_cap):
    """libero offline buffer 缓存签名(换数据源/base/缩放/图尺寸即失效重建)。"""
    return {
        "env_family": "libero", "lerobot_root": os.path.abspath(args.libero_stats_json + "/../.."),
        "suite": args.libero_suite, "task_id": int(args.libero_task_id),
        "base_mode": args.offline_base_mode, "base_policy_type": args.base_policy_type,
        "pi0_host": args.pi0_host, "pi0_port": args.pi0_port, "pi0_action_dim": args.pi0_action_dim,
        "action_scale": args.action_scale, "min_range_per_dim": args.min_range_per_dim,
        "offline_cap": offline_cap, "image_keys": sorted(image_keys),
        "gamma": args.gamma, "n_step": args.n_step, "num_demos": args.offline_num_demos,
    }
```
注：`lerobot_root` 由 `--libero_stats_json`(`<root>/meta/stats.json`)上溯两级得到,与 demo 源同根。

把 offline 构建块（约 717-765 行）的开头改成按 env_family 分流。原块：
```python
    if args.offline_fraction > 0.0:
        assert args.offline_dataset_path is not None, \
            "offline_fraction>0 需 --offline_dataset_path 指向源 dexmimicgen HDF5"
        from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import concat_mixed_batch
        from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
            build_offline_buffer, count_offline_transitions)
        offline_cap = count_offline_transitions(args.offline_dataset_path,
                                                num_demos=args.offline_num_demos)
        sig = _offline_buffer_signature(args, image_keys, offline_cap, shaping_mode,
                                        potential=potential)
```
改为：
```python
    if args.offline_fraction > 0.0:
        from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import concat_mixed_batch
        is_libero = getattr(args, "env_family", "dexmg") == "libero"
        if is_libero:
            from resfit.rl_finetuning.chunk_residual.libero_offline import (
                build_libero_offline_buffer, count_libero_offline_transitions)
            lerobot_root = os.path.abspath(os.path.join(os.path.dirname(args.libero_stats_json), ".."))
            offline_cap = count_libero_offline_transitions(
                lerobot_root, args.libero_suite, args.libero_task_id, num_demos=args.offline_num_demos)
            sig = _libero_offline_signature(args, image_keys, offline_cap)
        else:
            assert args.offline_dataset_path is not None, \
                "offline_fraction>0 需 --offline_dataset_path 指向源 dexmimicgen HDF5"
            from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
                build_offline_buffer, count_offline_transitions)
            offline_cap = count_offline_transitions(args.offline_dataset_path,
                                                    num_demos=args.offline_num_demos)
            sig = _offline_buffer_signature(args, image_keys, offline_cap, shaping_mode,
                                            potential=potential)
```
再把构建分支（cache miss 的 `else:` 里 `build_offline_buffer(...)`）改成按 `is_libero` 分流。原：
```python
        else:
            offline_rb = _new_offline_rb(with_transform=True)
            build_offline_buffer(
                offline_rb, args.offline_dataset_path,
                action_scaler=action_scaler, state_standardizer=state_standardizer,
                image_keys=image_keys, bonus=args.stage_reward_bonus,
                mode=shaping_mode, gamma=args.gamma, num_demos=args.offline_num_demos,
                stage_cache=args.offline_stage_cache, potential=potential,
                subgoal=subgoal, way_steps=args.subgoal_way_steps,
                act_feat_seqs=_offline_act_feat_seqs,
                base_policy=base_policy, base_mode=args.offline_base_mode,
                base_device=args.device,)
```
改为：
```python
        else:
            offline_rb = _new_offline_rb(with_transform=True)
            if is_libero:
                build_libero_offline_buffer(
                    offline_rb, lerobot_root=lerobot_root,
                    suite=args.libero_suite, task_id=args.libero_task_id,
                    action_scaler=action_scaler, state_standardizer=state_standardizer,
                    base_policy=base_policy, base_mode=args.offline_base_mode,
                    base_device=args.device, image_size=img_h, num_demos=args.offline_num_demos)
            else:
                build_offline_buffer(
                    offline_rb, args.offline_dataset_path,
                    action_scaler=action_scaler, state_standardizer=state_standardizer,
                    image_keys=image_keys, bonus=args.stage_reward_bonus,
                    mode=shaping_mode, gamma=args.gamma, num_demos=args.offline_num_demos,
                    stage_cache=args.offline_stage_cache, potential=potential,
                    subgoal=subgoal, way_steps=args.subgoal_way_steps,
                    act_feat_seqs=_offline_act_feat_seqs,
                    base_policy=base_policy, base_mode=args.offline_base_mode,
                    base_device=args.device,)
```
（`img_h` 已在 622-627 行 `img_c, img_h, img_w = obs0[image_keys[0]].shape[1:]` 得到=84;libero 用它当 image_size。）

- [ ] **Step 6: import 冒烟(两 env 都要能 import,libero 分支不触发 robosuite-1.5)**

Run:
```bash
conda run -p /mnt/mnt/data/envs/resfit-libero python -c "import resfit.rl_finetuning.chunk_residual.train_chunk_residual as T; import resfit.rl_finetuning.chunk_residual.libero_offline; print('import ok')"
conda run -p /mnt/mnt/data/envs/residual python -c "import resfit.rl_finetuning.chunk_residual.train_chunk_residual as T; print('dexmg env import ok')"
```
Expected: 两行各打印 ok（无 ImportError）

- [ ] **Step 7: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline_validate.py
git commit -m "feat(libero-offline): wire into train (validate relax + env_family offline branch + cache sig)"
```

---

### Task 8: 回归(两 env 零回归)

**Files:** 无新增(只跑测试)

- [ ] **Step 1: residual env 全量回归(robosuite-1.5,dexmg 不回归)**

Run: `conda run -p /mnt/mnt/data/envs/residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: PASS（≥ 392 passed,新增 libero_offline 测试也计入;0 failed）

- [ ] **Step 2: resfit-libero env libero 系列回归**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline_validate.py resfit/rl_finetuning/chunk_residual/tests/test_libero_env_wrapper.py resfit/rl_finetuning/chunk_residual/tests/test_libero_obs.py resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py -q`
Expected: PASS（全绿）

- [ ] **Step 3: 提交(若回归中顺手修了任何东西;否则跳过)**

```bash
git commit -am "test(libero-offline): regression green on both envs" || true
```

---

### Task 9: opt-in 真 serve live smoke(默认 skip,需用户授权 + 真 GPU/serve)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_libero_offline_smoke.py`(`@pytest.mark.skipif` 默认跳过)

- [ ] **Step 1: 写默认 skip 的 live smoke(文档化真跑命令)**

```python
# tests/test_libero_offline_smoke.py
import os
import pytest

RUN = os.environ.get("LIBERO_OFFLINE_SMOKE") == "1"

@pytest.mark.skipif(not RUN, reason="需真 serve + GPU;设 LIBERO_OFFLINE_SMOKE=1 且先起 serve 才跑")
def test_build_task8_offline_buffer_base_policy():
    """真 pi0_libero serve 在 8000;建 task8(libero_10/8)offline buffer(base_policy)验 len>0。"""
    import torch
    from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
    from resfit.rl_finetuning.chunk_residual.libero_offline import (
        build_libero_offline_buffer, count_libero_offline_transitions)
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_libero_scalers, build_base_policy
    import argparse
    root = "/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero"
    stats = root + "/meta/stats.json"
    a = argparse.Namespace(env_family="libero", libero_suite="libero_10", libero_task_id=8,
                           base_policy_type="pi05", base_action_mode="queue", chunk_length=1,
                           pi0_host="127.0.0.1", pi0_port=8000, pi0_action_dim=7, pi0_execute_horizon=30,
                           pi0_prompt="", pi0_image_key_map=None, pi0_kai0_path="/mnt/mnt/data/kai0_new4090")
    act_scaler, state_std = build_libero_scalers(stats, "cpu", action_scale=0.05, min_range_per_dim=0.1)
    base = build_base_policy(a, "cpu")
    cap = count_libero_offline_transitions(root, "libero_10", 8, num_demos=2)
    rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=cap, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority", batch_size=4)
    build_libero_offline_buffer(rb, lerobot_root=root, suite="libero_10", task_id=8,
                                action_scaler=act_scaler, state_standardizer=state_std,
                                base_policy=base, base_mode="base_policy", base_device="cpu",
                                image_size=84, num_demos=2)
    assert len(rb) > 0
```

- [ ] **Step 2: 确认默认 skip**

Run: `conda run -p /mnt/mnt/data/envs/resfit-libero python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_offline_smoke.py -q`
Expected: `1 skipped`

- [ ] **Step 3: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/tests/test_libero_offline_smoke.py
git commit -m "test(libero-offline): opt-in live smoke (default skip)"
```

- [ ] **Step 4: (用户授权后手动)真跑端到端**

先起 serve（GPU 1）:
```bash
cd /mnt/mnt/data/chj/openpi && GPUS=1 PORT=8000 bash scripts/serve_pi0_libero_29999.sh &
```
再真训练（task8,offline_fraction + demo_bc_coef 都开,小步数）:
```bash
PYTHONPATH=/mnt/mnt/data/resfit:/mnt/mnt/data/wam-b1k/third_party/LIBERO MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME=/mnt/mnt/data/chj/openpi WANDB_MODE=offline \
conda run -p /mnt/mnt/data/envs/resfit-libero python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --env_family libero --libero_suite libero_10 --libero_task_id 8 \
  --libero_stats_json /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json \
  --base_policy_type pi05 --base_action_mode queue --chunk_length 1 --pi0_action_dim 7 \
  --pi0_host 127.0.0.1 --pi0_port 8000 \
  --offline_fraction 0.5 --demo_bc_coef 0.1 --offline_base_mode base_policy \
  --offline_buffer_cache /tmp/libero_task8_offcache --offline_num_demos 3 \
  --action_scale 0.05 --total_env_steps 60 --learning_starts 20 \
  --eval_num_envs 1 --eval_num_episodes 1 --output_dir /tmp/libero_offline_smoke --device cuda
```
Expected: 打印 `[offline] 灌装 N 条 demo transition`（或命中缓存）+ `[offline] 混采 ...` + 训练跑到 `done. best success_rate=...`，无 NaN、exit 0。

---

## Self-Review(对照 spec)

- **spec §4.1 三函数**:Task 1(language)、Task 2(find episodes)、Task 3(read demo)、Task 4(_demo_to_transitions)、Task 5(base_policy 出 base)、Task 6(build + count)——✅ 全覆盖。
- **spec §4.2 接线**:Task 7(validate 放开 + env_family 分流 + `_libero_offline_signature` + 复用缓存助手)——✅。
- **spec §4.3 混采/BC 不改**:计划未改训练循环——✅(靠 td 同构,Task 4 测了同构)。
- **三命门**:① 无翻转 → Task 4/Task 5 都有"非对称图未翻转"断言;② reward 末帧+1 → Task 4 断言;③ 同源缩放 → Task 4/5 用 scaler/standardizer、Task 9 用 build_libero_scalers——✅。
- **spec §7 测试**:read/resolve/build(gt)/base_policy(stub)/validate/非对称图/回归/opt-in smoke——✅ 全覆盖。
- **占位扫描**:每个代码 step 都有完整可跑代码,无 TBD——✅。
- **类型一致**:`_img_chw_uint8`/`AGENTVIEW_KEY`/`WRIST_KEY`/`build_libero_offline_buffer` 签名在 Task 4–7 间一致;`image_size=img_h=84` 一致——✅。
