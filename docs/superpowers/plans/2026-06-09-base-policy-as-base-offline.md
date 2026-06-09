# base-policy-as-base 离线 buffer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给离线 demo buffer 加一个 flag 新模式 `--offline_base_mode base_policy`,让 `observation.base_action` 由冻结 base policy 现算(而非现在的 GT-as-base),使 BC 锚向专家、critic offline 锚与在线流形对齐;默认 `gt` 时与现状逐位等价。

**Architecture:** 改动全部局部化在 `offline_stage_replay.py`(新增 `_demo_base_actions` helper + `build_offline_buffer` 增 `base_policy/base_mode/base_device` 参数,切换 base_action 来源)与 `train_chunk_residual.py`(新 CLI flag + 缓存签名条件键 + 调用处透传 + queue 断言)。BC loss / critic 混采 / 在线 rollout / subgoal / eval 一律不动。

**Tech Stack:** Python, PyTorch, lerobot ACTPolicy(`select_action` 内部 normalize_inputs + per-env action queue), tensordict ReplayBuffer, pytest。

**测试基建参照:** `resfit/rl_finetuning/chunk_residual/tests/test_build_offline_buffer_rel.py`(`_FakeRb`/`_IdScaler`/`_IdStd`/tiny-hdf5/monkeypatch make_replay_env)、`tests/test_demo_bc.py`(CHW float 图像 obs、签名/CLI 测试范式)。

**关键事实(已勘察,实现者照用):**
- 在线 queue 产 base_action:`chunk_env_wrapper.py:94-97` `base_action = base_policy.select_action(raw_obs)` → `action_scaler.scale`;`reset()`(:131)清队列。
- `ACTPolicy.select_action`(`resfit/lerobot/policies/act/modeling_act.py:155`)内部 `normalize_inputs(batch)`(:164)+ 每 `n_action_steps` 空队列才前向、否则弹队列(:202-219),**有状态 → 必须每 demo reset 后逐帧顺序调**。
- env 运行时 obs 格式(`resfit/dexmg/...dexmg.py` `_process_obs`):`observation.state`=原始18维 float32(未标准化);图像 uint8 HWC → **float32 CHW ÷255**,键 `observation.images.{cam}`。
- 离线现有数据:`grp["obs/{k}"]`(STATE18_KEYS 各分量)、`grp["obs/{cam}_image"]`(T,H,W,3 uint8);`_hdf5_image_key("observation.images.agentview")=="agentview_image"`;`assemble_state18(obs)`→(T,18) 原始。
- 当前 `build_offline_buffer`(`offline_stage_replay.py`):`act_n=action_scaler.scale(GT)`(:218),transition 里 `observation.base_action=act_n[t]`(:237)/`act_n[t+1]`(:240),`action=act_n[t]`(:255)。
- 调用处 `train_chunk_residual.py:553`;`base_policy` 已在 :443 之前由 `build_base_policy` 加载并在 scope;`image_keys=list(base_policy.config.image_features.keys())`(:451);缓存签名 `_offline_buffer_signature`(:167);CLI 在 `build_parser`(`--demo_bc_coef` 在 :279 一带)。

---

### Task 1: `_demo_base_actions` helper(逐帧顺序复现 queue,产 scale 后 base_action)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`(在 `build_offline_buffer` 上方新增函数)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py`(新建)

- [ ] **Step 1: Write the failing test**

新建 `tests/test_base_policy_as_base.py`:

```python
"""base-policy-as-base 离线模式:base_action 由冻结 base policy 现算(非 GT-as-base)。"""
import h5py
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual import offline_stage_replay as osr
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import STATE18_KEYS, save_stage_cache


class _IdScaler:
    def scale(self, a): return torch.as_tensor(a, dtype=torch.float32)
    def unscale(self, a): return a


class _IdStd:
    def standardize(self, s): return torch.as_tensor(s, dtype=torch.float32)


class _FakeRb:
    def __init__(self): self.items = []
    def add(self, td): self.items.append(td)


class _FakeBase:
    """假 base policy:reset 计数;select_action 第 i 次返回全 (i+1) 的 (1,7) 动作,记录见到的 key。"""
    def __init__(self, action_dim=7):
        self.reset_calls = 0
        self.select_calls = 0
        self.seen_keys = []
        self.action_dim = action_dim
    def reset(self): self.reset_calls += 1
    def select_action(self, raw_obs):
        self.seen_keys.append(set(raw_obs.keys()))
        i = self.select_calls
        self.select_calls += 1
        return torch.full((1, self.action_dim), float(i + 1))


def _make_tiny_hdf5(path, T=4):
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("states", data=np.zeros((T, 5), dtype=np.float32))
        g.create_dataset("actions", data=np.zeros((T, 7), dtype=np.float32))
        for k, d in STATE18_KEYS:
            g.create_dataset(f"obs/{k}", data=np.arange(T * d, dtype=np.float32).reshape(T, d))
        g.create_dataset("obs/agentview_image", data=np.zeros((T, 4, 4, 3), dtype=np.uint8))
        g.attrs["model_file"] = "dummy_model"
    return T


def test_demo_base_actions_order_reset_format_scale(tmp_path):
    hdf5 = str(tmp_path / "tiny.hdf5")
    T = _make_tiny_hdf5(hdf5)
    fake = _FakeBase(action_dim=7)
    with h5py.File(hdf5, "r") as f:
        grp = f["data/demo_0"]
        base_n = osr._demo_base_actions(
            fake, grp, image_keys=["observation.images.agentview"],
            action_scaler=_IdScaler(), device="cpu")
    assert fake.reset_calls == 1                      # 每 demo reset 一次
    assert fake.select_calls == T                     # 逐帧顺序调 T 次
    assert base_n.shape == (T, 7)
    # 每帧 select_action 返回 (i+1) → IdScaler 透传 → 第 t 行全 (t+1)
    for t in range(T):
        assert torch.allclose(base_n[t], torch.full((7,), float(t + 1)))
    # raw_obs 必含原始 state + 图像 key
    for keys in fake.seen_keys:
        assert "observation.state" in keys
        assert "observation.images.agentview" in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py::test_demo_base_actions_order_reset_format_scale -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_demo_base_actions'`

- [ ] **Step 3: Write minimal implementation**

在 `offline_stage_replay.py` 中 `def build_offline_buffer(` **上方**新增(文件顶部已 import `numpy as np`、`torch`、`STATE18_KEYS`、`assemble_state18`;`_hdf5_image_key` 在本文件 :111):

```python
def _demo_base_actions(base_policy, grp, image_keys, action_scaler, device):
    """逐帧顺序复现在线 queue 语义,返回 scale 后的 base_action (T, action_dim)。

    与在线 chunk_env_wrapper._base_chunk_flat(queue,:94-97)对齐:每 demo 先 base_policy.reset()
    清 ACT action queue,再按帧顺序调 select_action(内部自管队列,空了才前向、否则弹队列),
    故必须顺序、不可批。raw_obs 严格对齐 env 运行时 _process_obs:observation.state=原始18维
    float32(未标准化);图像 uint8 HWC → float32 CHW /255。action_scaler.scale 把原始尺度 base
    动作转到与 act_n 同一缩放空间。
    """
    state_raw = assemble_state18({k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS})  # (T,18) 原始
    state_t = torch.as_tensor(np.asarray(state_raw), dtype=torch.float32)
    T = state_t.shape[0]
    imgs = {}
    for k in image_keys:
        arr = np.asarray(grp[f"obs/{_hdf5_image_key(k)}"][()])          # (T,H,W,3) uint8
        imgs[k] = torch.as_tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # (T,3,H,W)
    base_policy.reset()
    out = []
    for t in range(T):
        raw_obs = {"observation.state": state_t[t:t + 1].to(device)}
        for k in image_keys:
            raw_obs[k] = imgs[k][t:t + 1].to(device)
        a_raw = base_policy.select_action(raw_obs)                      # (1, action_dim) 原始尺度
        out.append(a_raw.detach().to("cpu"))
    base_raw = torch.cat(out, dim=0)                                    # (T, action_dim)
    return action_scaler.scale(base_raw)                               # (T, action_dim)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py::test_demo_base_actions_order_reset_format_scale -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/offline_stage_replay.py resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py
git commit -m "feat(base-as-base): _demo_base_actions 逐帧顺序复现 queue 产 base_action"
```

---

### Task 2: `build_offline_buffer` 接 `base_mode`(切换 base_action 来源)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`(`build_offline_buffer` 签名 + base_action 来源)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py`(追加)

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_base_policy_as_base.py`(复用上方 `_make_tiny_hdf5`/`_IdScaler`/`_IdStd`/`_FakeRb`/`_FakeBase`):

```python
def _build(tmp_path, base_mode, base_policy):
    hdf5 = str(tmp_path / "tiny.hdf5")
    T = _make_tiny_hdf5(hdf5)
    stage_cache = str(tmp_path / "stages.npz")
    save_stage_cache(stage_cache, {"demo_0": np.arange(T, dtype=np.int8)})
    rb = _FakeRb()
    osr.build_offline_buffer(
        rb, hdf5, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
        image_keys=["observation.images.agentview"], bonus=1.0, mode="staged",
        gamma=0.99, stage_cache=stage_cache, potential=None,
        base_policy=base_policy, base_mode=base_mode, base_device="cpu")
    return rb, T


def test_base_policy_mode_anchors_to_base_not_gt(tmp_path):
    fake = _FakeBase(action_dim=7)
    rb, T = _build(tmp_path, base_mode="base_policy", base_policy=fake)
    # GT actions 全 0 → act(=GT)=0;base_action 应为 fake 的 (t+1),≠ GT
    for t in range(T - 1):
        ba = rb.items[t]["obs"]["observation.base_action"]
        act = rb.items[t]["action"]
        assert torch.allclose(act, torch.zeros(7))                  # action 仍存 GT(0)
        assert torch.allclose(ba, torch.full((7,), float(t + 1)))   # base_action = base policy 现算
        bc_target = act - ba                                        # 隐含残差目标 ≠ 0
        assert not torch.allclose(bc_target, torch.zeros(7))


def test_gt_mode_byte_equivalent_base_equals_action(tmp_path):
    rb, T = _build(tmp_path, base_mode="gt", base_policy=None)
    for t in range(T - 1):
        ba = rb.items[t]["obs"]["observation.base_action"]
        act = rb.items[t]["action"]
        assert torch.allclose(ba, act)                              # GT-as-base:base==action(逐位等价)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -v`
Expected: FAIL（`build_offline_buffer` 不接受 `base_policy`/`base_mode`/`base_device` → `TypeError: unexpected keyword argument`）

- [ ] **Step 3: Write minimal implementation**

在 `offline_stage_replay.py` 改 `build_offline_buffer` 签名,加三个参数(放在现有关键字参数末尾,保持默认值向后兼容):

```python
def build_offline_buffer(rb, dataset_path, *, action_scaler, state_standardizer,
                         image_keys, bonus, mode, gamma, num_demos=None,
                         stage_cache=None, potential=None, subgoal=None, way_steps=25,
                         base_policy=None, base_mode="gt", base_device="cpu") -> int:
```

在 `act_n = action_scaler.scale(...)`(当前 :218-219)**之后**、`for t in range(T - 1):` 之前,插入 base_action 来源选择:

```python
                if base_mode == "base_policy":
                    if base_policy is None:
                        raise ValueError("base_mode='base_policy' 需传 base_policy")
                    base_n = _demo_base_actions(base_policy, grp, image_keys,
                                                action_scaler, base_device)  # (T,D) base 现算
                else:
                    base_n = act_n                                            # gt:GT-as-base(逐位等价)
```

把 transition 里的两处 base_action 从 `act_n` 换成 `base_n`(当前 :237、:240),`action`(:255)**保持 `act_n[t]` 不变**:

```python
                    curr = {"observation.state": state_n[t],
                            "observation.base_action": base_n[t],
                            "observation.stage_id": sid[t:t + 1]}
                    nxt = {"observation.state": state_n[t + 1],
                           "observation.base_action": base_n[t + 1],
                           "observation.stage_id": nsid[t:t + 1]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -v`
Expected: PASS（3 个测试全过）

- [ ] **Step 5: 跑既有 offline buffer 回归确认 gt 逐位等价**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_build_offline_buffer_rel.py resfit/rl_finetuning/chunk_residual/tests/test_offline_stage_replay_smoke.py -v`
Expected: PASS（默认 base_mode="gt" → 既有行为不变）

- [ ] **Step 6: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/offline_stage_replay.py resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py
git commit -m "feat(base-as-base): build_offline_buffer 接 base_mode 切换 base_action 来源(默认 gt 逐位等价)"
```

---

### Task 3: CLI flag `--offline_base_mode` + 缓存签名条件键 + 调用处透传

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(`build_parser` 加 flag;`_offline_buffer_signature` 加条件键;:553 调用处透传)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py`(追加)

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_base_policy_as_base.py`:

```python
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    build_parser, _offline_buffer_signature)


def _sig_args(extra):
    base = ["--task", "TwoArmThreePieceAssembly", "--offline_dataset_path", "/tmp/x.hdf5"]
    return build_parser().parse_args(base + extra)


def test_cli_offline_base_mode_default_gt():
    assert build_parser().parse_args([]).offline_base_mode == "gt"


def test_cli_offline_base_mode_parses():
    assert build_parser().parse_args(["--offline_base_mode", "base_policy"]).offline_base_mode \
        == "base_policy"


def test_signature_gt_has_no_base_mode_key():
    img = ["observation.images.agentview"]
    s = _offline_buffer_signature(_sig_args([]), img, 100, "staged")
    assert "offline_base_mode" not in s          # gt 默认 → 不加键 → 旧缓存向后兼容


def test_signature_base_policy_keys_and_distinguish_base():
    img = ["observation.images.agentview"]
    sa = _offline_buffer_signature(
        _sig_args(["--offline_base_mode", "base_policy", "--base_wandb_id", "/tmp/baseA"]),
        img, 100, "staged")
    sb = _offline_buffer_signature(
        _sig_args(["--offline_base_mode", "base_policy", "--base_wandb_id", "/tmp/baseB"]),
        img, 100, "staged")
    assert sa.get("offline_base_mode") == "base_policy"
    assert "base_wandb_id" in sa
    assert sa != sb                              # 换 base → 不同签名 → 强制重建
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -k "cli_offline_base_mode or signature_gt_has_no_base or signature_base_policy_keys" -v`
Expected: FAIL（无 `offline_base_mode` 属性 / 签名无该键）

- [ ] **Step 3: Write minimal implementation**

(a) `build_parser` 里,在 `--demo_bc_coef`(:279)附近新增:

```python
    p.add_argument("--offline_base_mode", choices=["gt", "base_policy"], default="gt",
                   help="离线 buffer 的 base_action 来源:gt(默认,逐位等价=GT-as-base,残差目标0)|"
                        "base_policy(冻结 base 现算 base_action,action 仍存 GT;bc_target=GT-base,"
                        "锚向专家 + critic offline 锚对齐在线流形)。base_policy 需 queue 模式。")
```

(b) `_offline_buffer_signature`（:167）末尾、`return sig` 之前,照搬 `potential_source=="hiql"` 的条件写法新增:

```python
    if args.offline_base_mode != "gt":
        sig["offline_base_mode"] = args.offline_base_mode
        sig["base_policy_type"] = getattr(args, "base_policy_type", "act")
        sig["base_wandb_id"] = (os.path.abspath(args.base_wandb_id)
                                if args.base_wandb_id and os.path.isdir(args.base_wandb_id)
                                else args.base_wandb_id)
```

(c) `build_offline_buffer(...)` 调用处(:553)透传(在现有 `subgoal=subgoal, way_steps=...` 之后):

```python
                subgoal=subgoal, way_steps=args.subgoal_way_steps,
                base_policy=base_policy, base_mode=args.offline_base_mode,
                base_device=args.device,)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -v`
Expected: PASS（全部通过）

- [ ] **Step 5: 跑既有签名回归(确认向后兼容)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_build_offline_buffer_rel.py -k signature -v`
Expected: PASS（gt 默认不加键 → `test_signature_stage_source_backward_compatible` 仍过）

- [ ] **Step 6: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py
git commit -m "feat(base-as-base): --offline_base_mode flag + 缓存签名条件键 + 调用处透传"
```

---

### Task 4: base_policy 模式的 queue 断言 + pi0 成本告警

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(参数校验区,offline buffer 构建块之前)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py`(追加,用一个小校验函数)

- [ ] **Step 1: Write the failing test**

为可单测,把校验抽成纯函数 `_validate_offline_base_mode(args)`。追加测试:

```python
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _validate_offline_base_mode
import pytest


def test_base_policy_mode_requires_queue():
    a = _sig_args(["--offline_base_mode", "base_policy",
                   "--base_action_mode", "replan", "--chunk_length", "2"])
    with pytest.raises(AssertionError):
        _validate_offline_base_mode(a)


def test_base_policy_mode_ok_with_queue():
    a = _sig_args(["--offline_base_mode", "base_policy",
                   "--base_action_mode", "queue", "--chunk_length", "1"])
    _validate_offline_base_mode(a)        # 不抛


def test_gt_mode_skips_validation():
    a = _sig_args(["--base_action_mode", "replan", "--chunk_length", "2"])
    _validate_offline_base_mode(a)        # gt 默认 → 不校验、不抛
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -k "queue or skips_validation" -v`
Expected: FAIL（`_validate_offline_base_mode` 未定义 → ImportError）

- [ ] **Step 3: Write minimal implementation**

在 `train_chunk_residual.py` 中 `_offline_buffer_signature` 附近(模块级函数)新增:

```python
def _validate_offline_base_mode(args):
    """base_policy 模式需 queue(chunk_length==1 且 base_action_mode=="queue");gt 模式跳过。"""
    if args.offline_base_mode != "base_policy":
        return
    assert args.base_action_mode == "queue" and args.chunk_length == 1, \
        ("--offline_base_mode base_policy 需 queue 模式"
         "(--base_action_mode queue --chunk_length 1);当前 "
         f"base_action_mode={args.base_action_mode!r} chunk_length={args.chunk_length}")
    if getattr(args, "base_policy_type", "act") == "pi05":
        print("[offline-base] WARN: pi05 base 走 websocket 逐帧推理(~23.8 万次),"
              "首次建 buffer 很慢;--offline_buffer_cache 缓存后秒读")
```

并在 `main()`/训练入口里、构建 offline buffer 之前(:520 之前任意参数校验处)调用一次:

```python
    _validate_offline_base_mode(args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py -v`
Expected: PASS（全部通过)

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_base_policy_as_base.py
git commit -m "feat(base-as-base): base_policy 模式 queue 断言 + pi0 成本告警"
```

---

### Task 5: 真数据格式 gate（真 base policy 端到端验证 obs 格式)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/verify_base_as_base.py`（独立验证脚本,controller 跑)

> 说明:Task 1-4 的单测用假 base policy,验的是接线/顺序/存储/签名,**验不到真 lerobot ACT 的图像 layout/归一化是否吃得下**。本 task 用真 base policy 在真数据上端到端跑 `_demo_base_actions`,作为格式 gate。需 GPU + 本地 base ckpt + 数据集,故是脚本(非 pytest)。

- [ ] **Step 1: Write the verify script**

新建 `resfit/rl_finetuning/chunk_residual/verify_base_as_base.py`:

```python
"""真 base policy 在真 demo 上跑 _demo_base_actions,验 obs 格式 + base_action 合理。

跑法(本机):
  cd /mnt/mnt/data/resfit && CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl WANDB_MODE=offline \
  conda run -n residual --no-capture-output python -m \
  resfit.rl_finetuning.chunk_residual.verify_base_as_base
"""
import h5py
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser, build_base_policy
from resfit.rl_finetuning.chunk_residual import offline_stage_replay as osr
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import sorted_demo_keys

HDF5 = "resfit/dataset/two_arm_three_piece_assembly.hdf5"


def main():
    args = build_parser().parse_args([
        "--task", "TwoArmThreePieceAssembly",
        "--base_wandb_id", "/mnt/mnt/data/resfit/resfit/out/piecce/best",
        "--dataset", "ankile/dexmg-two-arm-three-piece-assembly",
        "--base_action_mode", "queue", "--chunk_length", "1", "--base_n_action_steps", "10",
        "--action_scale", "0.05",
        "--offline_dataset_path", HDF5, "--device", "cuda",
    ])
    base_policy = build_base_policy(args, device="cuda")
    base_policy.config.n_action_steps = 10
    image_keys = list(base_policy.config.image_features.keys())
    # 与训练同款 action_scaler(train_chunk_residual.py:363-365)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from resfit.rl_finetuning.utils.normalization import ActionScaler
    meta = LeRobotDatasetMetadata(args.dataset)
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device="cuda")

    with h5py.File(HDF5, "r") as f:
        demos = sorted_demo_keys(list(f["data"].keys()))[:3]
        for ep in demos:
            grp = f[f"data/{ep}"]
            base_n = osr._demo_base_actions(base_policy, grp, image_keys, action_scaler, "cuda")
            gt = action_scaler.scale(torch.as_tensor(grp["actions"][()], dtype=torch.float32))
            bc = (gt - base_n)                          # 隐含残差目标 = GT - base
            assert torch.isfinite(base_n).all(), f"{ep}: base_n 非有限"
            assert base_n.shape == gt.shape, f"{ep}: shape {base_n.shape} vs GT {gt.shape}"
            print(f"{ep} T={len(base_n)} base|mean|={base_n.abs().mean():.4f} "
                  f"bc_target|mean|={bc.abs().mean():.4f} bc_target|max|={bc.abs().max():.4f}")
            assert bc.abs().mean() > 1e-4, f"{ep}: bc_target ~0(base≈GT?格式可能错或 base 完美)"
    print("VERIFY OK: 真 obs 格式吃得下,base_action 有限、bc_target=GT-base 非零有界")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Controller 跑验证脚本**

Run:
```bash
cd /mnt/mnt/data/resfit && CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl WANDB_MODE=offline HF_HUB_OFFLINE=1 \
conda run -n residual --no-capture-output python -m \
resfit.rl_finetuning.chunk_residual.verify_base_as_base > /tmp/verify_base_as_base.log 2>&1; \
echo "EXIT=$?"; tail -20 /tmp/verify_base_as_base.log
```
Expected: `VERIFY OK ...`,各 demo 打印有限的 base|mean| 与非零 bc_target|mean|,`EXIT=0`

- [ ] **Step 3: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/verify_base_as_base.py
git commit -m "test(base-as-base): 真 base policy 端到端 obs 格式 gate 脚本"
```

---

## 完成后(超出本 plan 代码范围,执行阶段处理)

代码全绿 + 真数据 gate 过后:用 `--offline_base_mode base_policy --demo_bc_coef <coef> --offline_buffer_cache <新缓存>` 重建一份"锚向专家"的 offline buffer 并起新臂(或重启 B-staged);与现有 gt 缓存臂 A/B。重启 vs 平行臂、系数取值由用户定(不在本 plan)。

## 全量回归(最终 task 后)

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 既有 119 passed 基础上 + 本 plan 新增测试全绿,无 failure(baseline flag 关 → 逐位等价)。
