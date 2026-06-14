# act_feat LeRobot 数据源 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 act_feat 加 `--data_source lerobot`,从 LeRobot 数据集(parquet+视频)读 demo 帧(no-stage),免 raw-hdf5 依赖,解锁 pouring/lifttray。

**Architecture:** 新增 `lerobot_demo_source.py`(按集读帧:images CHW float01 + state raw + actions),在 `read_per_demo_states`(缓存)和 `build_offline_buffer`(离线 buffer)两个数据读取触点上用 `--data_source` 切换;lerobot 路无 stage(stage_id≡0)、无 eef_piece。默认 hdf5 逐位不变。

**Tech Stack:** Python, PyTorch, `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`(video_backend=pyav), pytest。

参考 spec:`docs/superpowers/specs/2026-06-14-act-feat-lerobot-data-source-design.md`。LeRobot API(`ds.episode_data_index["from"/"to"]`、`ds[i]` 返回 `{observation.images.*: CHW float01, observation.state, action}`)已由 `verify_lerobot_image_consistency.py`(Phase0)实跑确认。测试目录 `resfit/rl_finetuning/chunk_residual/tests/`。命令从仓库根用 `conda run -n residual`。

---

## File Structure
- Create `resfit/rl_finetuning/chunk_residual/lerobot_demo_source.py` — 按集读 LeRobot 帧(纯读取,惰性 import LeRobotDataset)。
- Modify `train_hiql_value.py` — `read_per_demo_states` act_feat 加 lerobot 分支 + `--data_source/--lerobot_root` parser + 守卫。
- Modify `offline_stage_replay.py` — `build_offline_buffer` 加 lerobot/no-stage 分支。
- Modify `train_hiql_gc_value.py`/`train_hiql_high_actor.py`/`train_chunk_residual.py` — parser 透传 `--data_source/--lerobot_root`。
- Create tests + 一个 opt-in 可跑性 smoke。

---

## Task 1: `lerobot_demo_source.py`(按集读帧)

**Files:** Create `resfit/rl_finetuning/chunk_residual/lerobot_demo_source.py`; Test `resfit/rl_finetuning/chunk_residual/tests/test_lerobot_demo_source.py`

- [ ] **Step 1: 写失败测试**(stub ds,不解真视频)

```python
import torch
from resfit.rl_finetuning.chunk_residual.lerobot_demo_source import (
    lerobot_episode_frames, lerobot_episode_count, to_chw01,
)


class _StubDS:
    def __init__(self):
        self.episode_data_index = {"from": torch.tensor([0, 3]), "to": torch.tensor([3, 5])}
    def __getitem__(self, i):
        return {"observation.images.agentview": torch.full((3, 4, 4), float(i)),
                "observation.state": torch.tensor([float(i)] * 18),
                "action": torch.tensor([float(i)] * 7)}


def test_to_chw01_variants():
    assert to_chw01(torch.zeros(3, 4, 4)).shape == (3, 4, 4)            # CHW 原样
    hwc = torch.zeros(4, 4, 3); assert to_chw01(hwc).shape == (3, 4, 4)  # HWC->CHW
    u8 = torch.full((3, 4, 4), 255.0); assert float(to_chw01(u8).max()) <= 1.0  # uint8 尺度->[0,1]


def test_episode_count_and_frames():
    ds = _StubDS()
    assert lerobot_episode_count(ds) == 2
    fr = lerobot_episode_frames(ds, 0, ["observation.images.agentview"], "observation.state", "action")
    assert fr["images"]["observation.images.agentview"].shape == (3, 3, 4, 4)   # (T=3,C,H,W)
    assert fr["state"].shape == (3, 18) and fr["actions"].shape == (3, 7)
    # 第 1 集(帧 3..5) → T=2
    fr1 = lerobot_episode_frames(ds, 1, ["observation.images.agentview"], "observation.state", "action")
    assert fr1["state"].shape == (2, 18)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_lerobot_demo_source.py -v`
Expected: FAIL（ModuleNotFoundError）。

- [ ] **Step 3: 写实现**

```python
"""按集读 LeRobot 数据集的 demo 帧(act_feat 数据源;纯读取,惰性 import LeRobotDataset)。

帧格式与 raw-hdf5 路对齐到 ACT 输入:images=(T,3,84,84) float[0,1]、state=(T,Dp) raw、actions=(T,Da)。
LeRobot 图本就是 CHW float[0,1](见 verify_lerobot_image_consistency.py Phase0);to_chw01 兼容 HWC/uint8 防御。
"""
from __future__ import annotations

import torch


def to_chw01(v) -> torch.Tensor:
    """单帧图 -> (C,H,W) float[0,1]。兼容 CHW/HWC、uint8/float。"""
    v = torch.as_tensor(v).float()
    if v.ndim == 3 and v.shape[0] in (1, 3):        # CHW
        pass
    elif v.ndim == 3:                                # HWC
        v = v.permute(2, 0, 1)
    if float(v.max()) > 1.5:                          # uint8 尺度
        v = v / 255.0
    return v


def open_lerobot(repo_id, root):
    """打开本地 LeRobot 数据集(pyav 解码,见 project_resfit_pouring_traylift_residual)。"""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id, root=root, video_backend="pyav")


def lerobot_episode_count(ds) -> int:
    return int(len(ds.episode_data_index["from"]))


def lerobot_episode_frames(ds, ep_idx, image_keys, proprio_key="observation.state",
                           action_key="action") -> dict:
    """读第 ep_idx 集所有帧 -> {images:{k:(T,3,84,84)f01}, state:(T,Dp), actions:(T,Da)}。"""
    lo = int(ds.episode_data_index["from"][ep_idx])
    hi = int(ds.episode_data_index["to"][ep_idx])
    imgs = {k: [] for k in image_keys}
    states, actions = [], []
    for i in range(lo, hi):
        s = ds[i]
        for k in image_keys:
            imgs[k].append(to_chw01(s[k]))
        states.append(torch.as_tensor(s[proprio_key], dtype=torch.float32).reshape(-1))
        actions.append(torch.as_tensor(s[action_key], dtype=torch.float32).reshape(-1))
    return {"images": {k: torch.stack(imgs[k]) for k in image_keys},
            "state": torch.stack(states),
            "actions": torch.stack(actions)}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_lerobot_demo_source.py -v`
Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/lerobot_demo_source.py resfit/rl_finetuning/chunk_residual/tests/test_lerobot_demo_source.py
git commit -m "feat: lerobot_demo_source per-episode frame reader (act_feat data source)"
```

---

## Task 2: `read_per_demo_states` lerobot 分支 + flags

**Files:** Modify `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`; Test `tests/test_read_per_demo_states_act_feat.py`、`tests/test_act_feat_cli_wiring.py`

- [ ] **Step 1: 写失败测试**(append 到 `tests/test_read_per_demo_states_act_feat.py`)

```python
def test_act_feat_lerobot_source(tmp_path):
    import torch
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states

    class _StubDS:
        def __init__(self):
            self.episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([3])}
        def __getitem__(self, i):
            return {"observation.images.agentview": torch.zeros(3, 4, 4),
                    "observation.state": torch.full((18,), 5.0), "action": torch.zeros(7)}

    class _RecExtractor:
        captured = {}
        def embed_batch(self, ro):
            _RecExtractor.captured["proprio"] = ro["observation.state"].clone()
            b = ro["observation.state"].shape[0]
            return torch.cat([torch.zeros(b, 4), torch.as_tensor(ro["observation.state"], dtype=torch.float32)], -1)

    class _StubStd:
        def standardize(self, x):
            return (torch.as_tensor(x, dtype=torch.float32) - 1.0) / 2.0

    seqs, std, stats = read_per_demo_states(
        "ignored.hdf5", "repo/id", state_mode="act_feat", num_demos=None,
        act_feat_cache=None, act_extractor=_RecExtractor(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        state_standardizer=_StubStd(), data_source="lerobot", _lerobot_ds=_StubDS())
    assert std is None and seqs[0].shape == (3, 22)
    # proprio 经 dataset-标准化:(5-1)/2 = 2.0
    assert torch.allclose(_RecExtractor.captured["proprio"], torch.full((3, 18), 2.0))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py::test_act_feat_lerobot_source -v`
Expected: FAIL（`read_per_demo_states` 无 `data_source`/`_lerobot_ds` 参数）。

- [ ] **Step 3: 实现**

3a. `read_per_demo_states` 签名加参数(在现有 act_feat 参数之后):
```python
def read_per_demo_states(hdf5_path, dataset_id, state_mode="eef", num_demos=None,
                         device="cpu", cache_path=None,
                         act_feat_cache=None, act_extractor=None,
                         act_image_keys=None, act_ckpt_id=None, act_proprio_key="observation.state",
                         pooling="mean", state_standardizer=None,
                         data_source="hdf5", lerobot_root=None, _lerobot_ds=None,
                         _raw_obs_seqs=None):
```

3b. 在 act_feat build 分支里,把 `raw_seqs = _raw_obs_seqs if ... else _build_raw_obs_seqs(...)` 改成按 data_source 取(放在 proprio 标准化之前):
```python
        if _raw_obs_seqs is not None:
            raw_seqs = _raw_obs_seqs
        elif data_source == "lerobot":
            from resfit.rl_finetuning.chunk_residual.lerobot_demo_source import (
                open_lerobot, lerobot_episode_count, lerobot_episode_frames)
            ds = _lerobot_ds if _lerobot_ds is not None else open_lerobot(dataset_id, lerobot_root)
            n = lerobot_episode_count(ds)
            if num_demos is not None:
                n = min(n, num_demos)
            raw_seqs = []
            for ep in range(n):
                fr = lerobot_episode_frames(ds, ep, act_image_keys, act_proprio_key)
                raw_seqs.append({**fr["images"], act_proprio_key: fr["state"]})
        else:
            raw_seqs = _build_raw_obs_seqs(hdf5_path, act_image_keys, act_proprio_key, num_demos)
        # 下面 proprio dataset-标准化 + embed_batch 不变(命门 B)
```
(注:lerobot 的 `raw_seqs` 已是张量 dict;proprio 标准化那段 `std.standardize(torch.as_tensor(ro[act_proprio_key]))` 原样适用。真 build 路若 `state_standardizer is None` 仍从 `LeRobotDatasetMetadata(dataset_id, root=lerobot_root)` 建——给 metadata 加 `root=lerobot_root if data_source=="lerobot" else None`。)

3c. parser 加 flags(`build_parser`):
```python
    p.add_argument("--data_source", choices=["hdf5", "lerobot"], default="hdf5",
                   help="act_feat 数据源:hdf5(默认)|lerobot(从 LeRobot 数据集读,no-stage)")
    p.add_argument("--lerobot_root", default=None, help="--data_source lerobot 的本地数据根目录")
```

3d. 守卫(扩 `validate_act_feat_cfg` 或加 `validate_data_source_cfg`):
```python
def validate_data_source_cfg(args):
    import os
    if getattr(args, "data_source", "hdf5") != "lerobot":
        return
    assert getattr(args, "state_mode", None) == "act_feat", "--data_source lerobot 仅支持 --state_mode act_feat"
    assert args.lerobot_root and os.path.isdir(args.lerobot_root), \
        "--data_source lerobot 需 --lerobot_root(存在的本地数据目录)"
```
并在 main() 开头调用之。`setup_act_feat`/`read_per_demo_states` 调用处把 `data_source=args.data_source, lerobot_root=args.lerobot_root` 透传。`setup_act_feat` build extractor 的 image_keys 仍取 `act.config.image_features`,`ActFeatureExtractor(..., proprio_dim=<实际本体维>)`——实际本体维 = lerobot 时该数据集 state 维(可从 `LeRobotDatasetMetadata(dataset_id, root).features["observation.state"]["shape"][0]` 取;非 act_feat-online 必需,仅修 feature_dim 的 cosmetic,可先传该值)。

- [ ] **Step 4: 跑测试 + 回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py -v`
Expected: 新 lerobot 测试 + 既有全 PASS（默认 hdf5 不变）。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py
git commit -m "feat(act_feat): read_per_demo_states lerobot data source branch + flags + guard"
```

---

## Task 3: `build_offline_buffer` lerobot/no-stage 分支

**Files:** Modify `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`; Test `tests/test_build_offline_buffer_rel.py`

lerobot 路:逐集读 LeRobot 帧(图/state/action),stage_id≡0(不 replay、不开 h5py),base_action/subgoal 同 hdf5 路。

- [ ] **Step 1: 写失败测试**(append 到 `tests/test_build_offline_buffer_rel.py`)

```python
def test_build_offline_buffer_lerobot_no_stage():
    import numpy as np
    import torch
    from tensordict import TensorDict
    from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import build_offline_buffer

    class _StubDS:
        def __init__(self):
            self.episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([4])}
        def __getitem__(self, i):
            return {"observation.images.agentview": torch.zeros(3, 4, 4),
                    "observation.state": torch.zeros(18), "action": torch.zeros(7)}

    class _StubScaler:
        def scale(self, x): return torch.as_tensor(x, dtype=torch.float32)
    class _StubStd:
        def standardize(self, x): return torch.as_tensor(x, dtype=torch.float32)
    class _StubSubgoal:
        state_mode = "act_feat"
        def subgoal_waypoint(self, b, t): return torch.zeros(b.shape[0], 10)

    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=100, device="cpu"))
    n = build_offline_buffer(
        rb, "ignored.hdf5", action_scaler=_StubScaler(), state_standardizer=_StubStd(),
        image_keys=["observation.images.agentview"], bonus=1.0, mode="none", gamma=0.99,
        subgoal=_StubSubgoal(), way_steps=2,
        act_feat_seqs=[np.zeros((4, 530), np.float32)],
        data_source="lerobot", lerobot_repo_id="repo/id", lerobot_root="/x", _lerobot_ds=_StubDS(),
        base_mode="gt")
    assert n == 3                                  # T=4 → 3 transitions
    td = rb.sample(1)
    assert "observation.subgoal" in td["obs"].keys()
    assert float(td["obs"]["observation.stage_id"].abs().max()) == 0.0   # no-stage
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_build_offline_buffer_rel.py::test_build_offline_buffer_lerobot_no_stage -v`
Expected: FAIL（无 `data_source`/`lerobot_*` 参数）。

- [ ] **Step 3: 实现**

3a. `build_offline_buffer` 签名加参数:
```python
def build_offline_buffer(rb, dataset_path, *, action_scaler, state_standardizer,
                         image_keys, bonus, mode, gamma, num_demos=None,
                         stage_cache=None, potential=None, subgoal=None, way_steps=25,
                         act_feat_seqs=None,
                         data_source="hdf5", lerobot_repo_id=None, lerobot_root=None, _lerobot_ds=None,
                         base_policy=None, base_mode="gt", base_device="cpu") -> int:
```

3b. lerobot 路单独走一条循环(在 hdf5 的 `with h5py.File(...)` 之外),避免开 h5py。在函数体顶部(`_act_feat_subgoal` 等定义之后)加:
```python
    if data_source == "lerobot":
        return _build_offline_lerobot(
            rb, action_scaler=action_scaler, state_standardizer=state_standardizer,
            image_keys=image_keys, bonus=bonus, mode=mode, gamma=gamma, num_demos=num_demos,
            subgoal=subgoal, way_steps=way_steps, act_feat_seqs=act_feat_seqs,
            repo_id=lerobot_repo_id, root=lerobot_root, ds=_lerobot_ds,
            base_policy=base_policy, base_mode=base_mode, base_device=base_device)
```
并新增模块级 `_build_offline_lerobot(...)`(镜像 hdf5 循环,但帧来自 lerobot、stage_id≡0、无 rel):
```python
def _build_offline_lerobot(rb, *, action_scaler, state_standardizer, image_keys, bonus, mode,
                           gamma, num_demos, subgoal, way_steps, act_feat_seqs, repo_id, root, ds,
                           base_policy, base_mode, base_device) -> int:
    import numpy as np
    import torch
    from tensordict import TensorDict
    from resfit.rl_finetuning.chunk_residual.lerobot_demo_source import (
        open_lerobot, lerobot_episode_count, lerobot_episode_frames)
    assert subgoal is None or getattr(subgoal, "state_mode", "") == "act_feat", \
        "lerobot 数据源仅支持 act_feat subgoal(无 eef_piece rel)"
    if ds is None:
        ds = open_lerobot(repo_id, root)
    n = lerobot_episode_count(ds)
    if num_demos is not None:
        n = min(n, num_demos)
    added = 0
    for ep in range(n):
        fr = lerobot_episode_frames(ds, ep, image_keys, "observation.state", "action")
        T = fr["state"].shape[0]
        if T < 2:
            continue
        state_n = state_standardizer.standardize(fr["state"].float()).cpu()         # (T,Dp) std
        instant = np.zeros(T, dtype=np.int8)                                        # no-stage
        fld = transition_fields(instant, bonus=bonus, mode=mode, gamma=gamma, success=True,
                                potential=None, state_seq=state_n, rel_piece_seq=None)
        act_n = action_scaler.scale(fr["actions"].float()).cpu()
        base_n = act_n if base_mode == "gt" else _demo_base_actions_lerobot(
            base_policy, fr["images"], image_keys, action_scaler, base_device)
        imgs = {k: fr["images"][k] for k in image_keys}                             # (T,3,84,84) f01
        sid = torch.as_tensor(fld["stage_id"], dtype=torch.float32)
        nsid = torch.as_tensor(fld["next_stage_id"], dtype=torch.float32)
        subgoal_z = None
        if subgoal is not None:
            s530 = torch.as_tensor(act_feat_seqs[ep], dtype=torch.float32)          # (T,530) 已标准化
            assert s530.shape[0] == T, f"act_feat 缓存帧数 {s530.shape[0]} != demo {T} (ep{ep})"
            way = np.minimum(np.arange(T) + way_steps, T - 1)
            subgoal_z = subgoal.subgoal_waypoint(s530, s530[way]).cpu()
        for t in range(T - 1):
            curr = {"observation.state": state_n[t], "observation.base_action": base_n[t],
                    "observation.stage_id": sid[t:t + 1]}
            nxt = {"observation.state": state_n[t + 1], "observation.base_action": base_n[t + 1],
                   "observation.stage_id": nsid[t:t + 1]}
            if subgoal_z is not None:
                curr["observation.subgoal"] = subgoal_z[t]; nxt["observation.subgoal"] = subgoal_z[t + 1]
            for k in image_keys:
                curr[k] = imgs[k][t]; nxt[k] = imgs[k][t + 1]
            td = TensorDict({"obs": TensorDict(curr, batch_size=[]),
                             "next": TensorDict({"obs": TensorDict(nxt, batch_size=[]),
                                                 "done": torch.tensor(bool(fld["done"][t])),
                                                 "reward": torch.tensor(float(fld["reward"][t]), dtype=torch.float32)},
                                                batch_size=[]),
                             "action": act_n[t],
                             "max_stage": torch.tensor(float(fld["max_stage"][t]), dtype=torch.float32),
                             "_priority": torch.tensor(10.0, dtype=torch.float32)}, batch_size=[]).unsqueeze(0)
            rb.add(td); added += 1
    return added


def _demo_base_actions_lerobot(base_policy, images, image_keys, action_scaler, device):
    """逐帧用冻结 base_policy 现算 base_action(缩放后)。images[k]=(T,3,84,84) f01。"""
    import torch
    base_policy.eval()
    T = images[image_keys[0]].shape[0]
    outs = []
    with torch.no_grad():
        for t in range(T):
            raw = {k: images[k][t:t + 1].to(device) for k in image_keys}
            a = base_policy.select_action(raw)                # (1,Da) 原尺度
            outs.append(action_scaler.scale(a.float().cpu())[0])
    return torch.stack(outs)
```
(`transition_fields`/`STATE18_KEYS` 等已在文件中 import;`_demo_base_actions_lerobot` 与现有 `_demo_base_actions` 同义但喂 lerobot 图。base_mode=base_policy 的 lerobot 实测细节由 smoke 暴露;首版 smoke 可用 `--offline_base_mode gt` 验通主链,再切 base_policy。)

- [ ] **Step 4: 跑测试 + 回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_build_offline_buffer_rel.py -v`
Expected: 新 lerobot 测试 + 既有 eef_piece/act_feat(hdf5)全 PASS。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/offline_stage_replay.py resfit/rl_finetuning/chunk_residual/tests/test_build_offline_buffer_rel.py
git commit -m "feat(act_feat): build_offline_buffer lerobot/no-stage branch"
```

---

## Task 4: parser 透传(gc_value / high_actor / train_chunk_residual)

**Files:** Modify `train_hiql_gc_value.py`、`train_hiql_high_actor.py`、`train_chunk_residual.py`;Test `tests/test_act_feat_cli_wiring.py`

- [ ] **Step 1: 写失败测试**(append 到 `tests/test_act_feat_cli_wiring.py`)

```python
def test_data_source_flags_present_and_default_hdf5():
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser as gc
    from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser as ha
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser as tc
    for p in (gc(), ha()):
        a = p.parse_args(["--hdf5", "h", "--dataset", "d"] + (["--gc_value_ckpt", "g"] if "gc_value_ckpt" in
                          [x.dest for x in p._actions] else []))
        assert a.data_source == "hdf5" and a.lerobot_root is None
    a = tc().parse_args(["--task", "TwoArmThreePieceAssembly", "--dataset", "d"])
    assert a.data_source == "hdf5" and a.lerobot_root is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py -k data_source -v`
Expected: FAIL（parser 无 `--data_source`）。

- [ ] **Step 3: 实现**
- 三个 `build_parser` 各加(放在 act_feat flags 附近;train_hiql_value 已在 Task2 加过,gc_value/high_actor/train_chunk_residual 这里加):
```python
    p.add_argument("--data_source", choices=["hdf5", "lerobot"], default="hdf5",
                   help="act_feat 数据源:hdf5(默认)|lerobot(no-stage)")
    p.add_argument("--lerobot_root", default=None, help="--data_source lerobot 本地数据根目录")
```
- `train_hiql_gc_value.main` / `train_hiql_high_actor.main`:把 `read_per_demo_states(...)` 调用加 `data_source=args.data_source, lerobot_root=args.lerobot_root`;`setup_act_feat`/`validate_data_source_cfg` 同 Task2(从 train_hiql_value import 复用)。
- `train_chunk_residual.main`:`validate_data_source_cfg(args)`;build_offline_buffer 调用加 `data_source=args.data_source, lerobot_repo_id=args.dataset, lerobot_root=args.lerobot_root`;lerobot 时 `offline_dataset_path`/`offline_stage_cache` 不用(传 None 容忍);`setup_act_feat` 透传 data_source/lerobot_root 给其内部 read_per_demo_states(若它建缓存)。

- [ ] **Step 4: 跑测试 + 回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py -v`
Expected: 全 PASS;默认 hdf5 不变。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py
git commit -m "feat(act_feat): thread --data_source/--lerobot_root through gc_value/high_actor/train_chunk_residual"
```

---

## Task 5: 可跑性 smoke(opt-in,真 LeRobot + ACT base,pouring)

**Files:** Create `resfit/rl_finetuning/chunk_residual/verify_act_feat_lerobot_smoke.py`(或一个 driver 脚本)

- [ ] **Step 1: 写 smoke 脚本/driver**

一个 bash driver(仓库根 `run_act_feat_lerobot_smoke.sh`),用少量集 + 几十步跑通 pouring 全链:
```bash
#!/usr/bin/env bash
set -u; GPU=${1:?<gpu>}
export PATH="/root/miniconda3/bin:$PATH" CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"
BASE=$(ls -d /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/bc_run_*dexmg-two-arm-pouring_act/best 2>/dev/null | head -1)
ROOT=resfit/dataset/ankile/dexmg-two-arm-pouring
DS=ankile/dexmg-two-arm-pouring
CACHE=/tmp/pouring_act_feat_smoke.npz
echo "BASE=$BASE"
# ① gc_value(act_feat + lerobot,少量集)
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" --act_base_ckpt "$BASE" \
  --hdf5 NA --dataset "$DS" --num_demos 4 --steps 200 \
  --act_feat_cache "$CACHE" --output /tmp/pouring_gc_smoke.pt
# ② high_actor
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" --act_base_ckpt "$BASE" \
  --hdf5 NA --dataset "$DS" --num_demos 4 --steps 200 \
  --act_feat_cache "$CACHE" --gc_value_ckpt /tmp/pouring_gc_smoke.pt --output /tmp/pouring_ha_smoke.pt
# ③ 主训(subgoal + lerobot + no-stage,少步;先 base_mode gt 验主链)
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" --dataset "$DS" \
  --data_source lerobot --lerobot_root "$ROOT" --subgoal_conditioned --reward_shaping none \
  --offline_base_mode gt --offline_num_demos 4 --demo_bc_coef 0.1 --action_scale 0.05 --offline_fraction 0.5 \
  --gc_value_ckpt /tmp/pouring_gc_smoke.pt --high_actor_ckpt /tmp/pouring_ha_smoke.pt --act_feat_cache "$CACHE" \
  --offline_buffer_cache /tmp/pouring_offcache_smoke --total_env_steps 40 --learning_starts 10 \
  --wandb_project dexmg-chunk-residual --wandb_name pouring_lerobot_smoke --output_dir /tmp/pouring_smoke_out
echo "SMOKE rc=$?"
```

- [ ] **Step 2: 跑(opt-in,GPU 授权后)**

Run: `bash run_act_feat_lerobot_smoke.sh <free_gpu>`(WANDB 可 disabled)。
Expected:gc_value/high_actor 出 ckpt(state_dim = dim_model+36),主训 offline buffer(lerobot,stage_id 0)→ rollout(subgoal_online lerobot 在线提特征)→ eval 全通、V 有限、不崩;打印 `BASE=` 非空(ACT base 找到并加载)。**若 ACT base 加载失败 / 维度/相机键不匹配 / 视频解码失败,在此暴露并修。**

- [ ] **Step 3: Commit**

```bash
git add run_act_feat_lerobot_smoke.sh resfit/rl_finetuning/chunk_residual/verify_act_feat_lerobot_smoke.py 2>/dev/null
git commit -m "test: opt-in pouring lerobot act_feat runnability smoke (gt base first)"
```

---

## Self-Review 结论
- **Spec 覆盖**:§3.1 reader→Task1;§3.2 read_per_demo_states lerobot→Task2;§3.3 build_offline_buffer lerobot/no-stage→Task3;§3.4 parser 透传+守卫→Task2/4;§4 测试→各 Task TDD + Task5 smoke;§5 文件→全覆盖。
- **占位扫描**:Task3 的 `_demo_base_actions_lerobot` 与 base_mode=base_policy 实测细节标注由 smoke 暴露(首版 smoke 用 gt base 验主链)——这是真 ACT/env 依赖的合理边界,非占位;其余给真实代码。
- **类型一致**:`lerobot_episode_frames` 返回 `{images,state,actions}` 在 Task1 定义、Task2/3 用;`data_source/lerobot_root/lerobot_repo_id/_lerobot_ds` 参数名各 Task 一致;`act_feat_seqs` 沿用既有(上一 plan)。
- **范围**:单一计划;act_feat+no-stage;hdf5 逐位不变;eef_piece+lerobot 禁止。
- **风险**:视频-渲染 ~0.1 gap(已接受,结果明标);ACT base 加载 + 维度/相机泛化由 smoke 兜。
