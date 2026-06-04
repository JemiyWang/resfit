# pi05 Base-Policy Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `residual-offpolicy-rl` 的冻结基座策略可在 ACT 与 pi05 之间用配置开关切换（step 级路径），并打通 robomimic 上微调 pi05 基座的链路。

**Architecture:** 方案 A——在 `train_residual_td3` 的基座加载处插一个由 `base_policy.type` 驱动的分发层；pi05 走 **websocket 跨进程**：GPU 端在 kai0 uv 环境起 pi05 server，residual 端用轻量 `openpi-client` 的 `WebsocketClientPolicy` + `Pi05PolicyAdapter.from_policy(client)`（补一个可配置视角映射），它已实现 step 级路径所需的 `select_action / reset / config.image_features`。RL 训练栈、残差组合、Q 网络一律不改。ACT 默认行为字节级不变。

**为何 websocket（2026-06-04 gate 结论）:** `residual` 环境纯 torch 2.7.1 无 flax，openpi 顶层硬 import flax，同进程不可行；无现成统一环境。详见 spec §11。

**Tech Stack:** Python, PyTorch, Hydra dataclass config, openpi/pi05 (JAX+PyTorch 推理, 在 /data2/kai0), LeRobot 数据格式, robomimic, pytest。

**关键约定（可配置，第一版默认值）:**
- 目标任务：robomimic 单臂 **Lift**（可换 Can 等）。
- action 维度：`action_dim = 7`（robomimic 单臂 OSC：末端位姿 + 夹爪）。
- robomimic 视角 → pi05 槽位映射：`observation.images.agentview` → `base`，`observation.images.robot0_eye_in_hand` → 左手腕，右手腕槽位缺省（由 openpi transform 补零图 + mask）。
- prompt：`"pick up the cube"`（微调与推理必须用同一句）。
- pi05 训练 config 名（GPU serve 端用）：`pi05_robomimic_lift`。
- pi05 websocket：`host=127.0.0.1`、`port=8000`。

**跨进程拓扑:** GPU 端在 kai0 uv 环境跑 `scripts/serve_policy.py` 起 pi05 websocket server（加载微调 ckpt）；residual 端纯 torch，pip 装轻量 `openpi-client`，用 `WebsocketClientPolicy` + `Pi05PolicyAdapter.from_policy`。residual 端不装/不 import flax/openpi。serve 端口与 config 由 Phase 3 / serve 任务管理。

---

## File Structure

**residual 端依赖:**
- 在 conda env `residual` 安装轻量 `openpi-client`（`/data2/kai0/packages/openpi-client`，无 flax/torch）。

**修改（residual-offpolicy-rl）:**
- `resfit/rl_finetuning/config/residual_td3.py` — `BasePolicyConfig` 加 `type` 与 pi05 字段（host/port 等）。
- `resfit/rl_finetuning/scripts/train_residual_td3.py` — 按 `type` 分发加载基座 + 设 `actor_name`。
- `resfit/rl_finetuning/wrappers/residual_env_wrapper.py` — 放宽 `base_policy` 类型注解。
- 注：`resfit/lerobot/utils/load_policy.py` **不改**（pi05 不走它，走新工厂）。

**新建（residual-offpolicy-rl）:**
- `resfit/lerobot/policies/pi05/__init__.py`
- `resfit/lerobot/policies/pi05/load_pi05.py` — pi05 加载工厂 `load_pi05_base_policy(cfg, device)`：连 websocket server + `Pi05PolicyAdapter.from_policy`。
- `resfit/rl_finetuning/scripts/eval_pi05_base.py` — Phase 1 gate 的纯基座 eval（连 server）。
- `tests/policies/pi05/test_import_smoke.py` — websocket 路径 import 验证（无 flax）。
- `tests/policies/pi05/test_load_pi05.py` — 工厂分发单测（mock client + adapter）。
- `tests/rl_finetuning/test_base_policy_switch.py` — 开关分发、视角映射、ACT 回归测试。

**修改（kai0）:**
- `/data2/kai0/resfit_pi05/pi05_policy_adapter.py` — 视角映射 `_IMAGE_KEYS` 改为可配置实例参数（默认值向后兼容 dsrl_pi05）；如 Task 1 发现顶层连带 import flax，则把 openpi import 惰性化。
- `/data2/kai0/resfit_pi05/tests/test_pi05_policy_adapter_image_map.py` — 视角映射可配置单测。

**微调 + serve 链路（Phase 0/1，含外部脚本与 openpi config）:**
- 复用 `resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`（不改，命令调用）。
- `/data2/kai0/src/openpi/training/config.py` — 新增 `Pi05RobomimicDataConfig` + `TrainConfig(name="pi05_robomimic_lift")`。
- 复用 `/data2/kai0/scripts/serve_policy.py`（不改，GPU 端起 websocket server）。

---

## Task 1: residual 端装 openpi-client + websocket 路径 import 验证（前置 gate）

> 背景：同进程方案已被 2026-06-04 gate 否决（residual 无 flax，openpi 顶层 import flax）。
> 本任务验证 **websocket 路径** 的前提：residual 端能装轻量 openpi-client、能 import
> WebsocketClientPolicy 与 Pi05PolicyAdapter（走 from_policy，不触发 flax）。

**Files:**
- Test: `tests/policies/pi05/test_import_smoke.py`（residual-offpolicy-rl）

- [ ] **Step 1: 在 residual 装 openpi-client（轻量，无 flax/torch）**

先记录现有 numpy 版本（openpi-client 要求 numpy<2.0，避免降级破坏 residual）：
Run: `conda run -n residual python -c "import numpy; print(numpy.__version__)"`

安装（editable）：
Run: `conda run -n residual pip install -e /data2/kai0/packages/openpi-client`
Expected: 成功安装 openpi-client 及 dm-tree/msgpack/websockets/pillow/tree。

安装后复查 numpy 未被意外降级到不兼容版本：
Run: `conda run -n residual python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"`
若 numpy 被降级且导致 torch 报错：改用 `pip install -e /data2/kai0/packages/openpi-client --no-deps`，再单独 `pip install "msgpack>=1.0.5" "websockets>=11.0" dm-tree tree pillow`，并把此偏差记入报告。

- [ ] **Step 2: 写 import smoke 测试（验证 websocket 路径不碰 flax）**

```python
# tests/policies/pi05/test_import_smoke.py
import importlib
import sys
from pathlib import Path

KAI0_ROOT = Path("/data2/kai0")  # 为 import resfit_pi05.* 提供包根


def test_websocket_path_imports_without_flax():
    """websocket 跨进程方案前提：residual 端能 import client + adapter，且不触发 flax。"""
    if str(KAI0_ROOT) not in sys.path:
        sys.path.insert(0, str(KAI0_ROOT))

    # 轻量 client（pip 安装后可直接 import，依赖里无 flax/torch）
    from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: F401

    # adapter 顶层 import 干净；走 from_policy 路径不会 import openpi.training.config
    mod = importlib.import_module("resfit_pi05.pi05_policy_adapter")
    assert hasattr(mod, "Pi05PolicyAdapter")

    # 关键断言：import adapter 不应连带把 flax 拉进来
    assert "flax" not in sys.modules, "import adapter 触发了 flax，from_policy 路径不该如此"
```

- [ ] **Step 3: 运行**

Run: `conda run -n residual python -m pytest tests/policies/pi05/test_import_smoke.py -v`
Expected: PASS。

若 `assert "flax" not in sys.modules` 失败（说明 adapter 顶层确实连带 import 了 openpi）：
在 Task 2 里把 `pi05_policy_adapter.py` 中对 openpi 的 import 移进 `from_checkpoint` 函数体（惰性化），
使 from_policy 路径不碰 flax；然后回到本步重跑。其它 import 失败按错误原文排查依赖安装。

- [ ] **Step 4: Commit**

```bash
git add tests/policies/pi05/test_import_smoke.py
git commit -m "test: pi05 websocket-path import smoke (no flax in residual)"
```

---

## Task 2: Pi05PolicyAdapter 视角映射可配置（kai0）

把写死的 `_IMAGE_KEYS` 变成可传入的实例参数，默认值保持现状以不破坏 `dsrl_pi05`。

**Files:**
- Modify: `/data2/kai0/resfit_pi05/pi05_policy_adapter.py`（`_IMAGE_KEYS`:10-14；`__init__`:19-39；`from_policy`:41-51；`from_checkpoint`:53-77；`_to_openpi_obs`:131-145）
- Test: `/data2/kai0/resfit_pi05/tests/test_pi05_policy_adapter_image_map.py`

- [ ] **Step 1: 写失败测试**

```python
# /data2/kai0/resfit_pi05/tests/test_pi05_policy_adapter_image_map.py
import numpy as np
from resfit_pi05.pi05_policy_adapter import Pi05PolicyAdapter


class _FakePolicy:
    """记录收到的 obs，并返回固定 chunk。"""
    def __init__(self):
        self.last_obs = None

    def infer(self, obs):
        self.last_obs = obs
        return {"actions": np.zeros((50, 32), dtype=np.float32)}


def test_default_image_map_backward_compatible():
    adapter = Pi05PolicyAdapter.from_policy(
        _FakePolicy(), prompt="p", action_dim=7, device="cpu",
    )
    assert list(adapter.config.image_features.keys()) == [
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]


def test_custom_image_map_used_for_keys_and_obs():
    fake = _FakePolicy()
    image_key_map = {
        "observation.images.agentview": "base",
        "observation.images.robot0_eye_in_hand": "left_wrist",
    }
    adapter = Pi05PolicyAdapter.from_policy(
        fake, prompt="p", action_dim=7, device="cpu",
        image_key_map=image_key_map,
    )
    # config.image_features 暴露 robomimic 风格的键，供 env wrapper 取图
    assert list(adapter.config.image_features.keys()) == list(image_key_map.keys())

    raw_obs = {
        "observation.state": np.zeros((1, 9), dtype=np.float32),
        "observation.images.agentview": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        "observation.images.robot0_eye_in_hand": np.zeros((1, 224, 224, 3), dtype=np.uint8),
    }
    adapter.select_action(raw_obs)
    # 内部 openpi obs 用映射后的槽位名
    assert set(fake.last_obs["images"].keys()) == {"base", "left_wrist"}
```

- [ ] **Step 2: 运行验证失败**

Run: `cd /data2/kai0 && conda run -n residual python -m pytest resfit_pi05/tests/test_pi05_policy_adapter_image_map.py -v`
Expected: FAIL（`from_policy` / `__init__` 不接受 `image_key_map`）。

- [ ] **Step 3: 实现——把 `_IMAGE_KEYS` 变默认，注入实例参数**

把模块常量改名为默认表，并在三个构造入口加 `image_key_map` 参数：

```python
# 顶部（原 10-14 行）：默认映射（保持向后兼容）
_DEFAULT_IMAGE_KEYS = {
    "observation.images.top_head": "top_head",
    "observation.images.hand_left": "hand_left",
    "observation.images.hand_right": "hand_right",
}
```

```python
# __init__（原 19-39 行）：新增 image_key_map 参数并存为实例属性
    def __init__(
        self,
        policy: Any,
        *,
        prompt: str,
        action_dim: int,
        device: str | torch.device,
        execute_horizon: int | None = 30,
        image_key_map: dict[str, str] | None = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if execute_horizon is not None and execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {execute_horizon}")

        self.policy = policy
        self.prompt = prompt
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.execute_horizon = int(execute_horizon) if execute_horizon is not None else None
        self.image_key_map = dict(image_key_map) if image_key_map else dict(_DEFAULT_IMAGE_KEYS)
        self.config = _ResfitPolicyConfig(image_features=dict.fromkeys(self.image_key_map))
        self._action_queues: list[deque[np.ndarray]] = []
```

```python
# from_policy（原 41-51 行）：透传 image_key_map
    @classmethod
    def from_policy(
        cls,
        policy: Any,
        *,
        prompt: str,
        action_dim: int,
        device: str | torch.device,
        execute_horizon: int | None = 30,
        image_key_map: dict[str, str] | None = None,
    ) -> "Pi05PolicyAdapter":
        return cls(
            policy, prompt=prompt, action_dim=action_dim, device=device,
            execute_horizon=execute_horizon, image_key_map=image_key_map,
        )
```

```python
# from_checkpoint（原 53-77 行）：在签名加 image_key_map 并透传给 from_policy
    @classmethod
    def from_checkpoint(
        cls,
        *,
        config_name: str,
        checkpoint_dir: str | Path,
        prompt: str,
        action_dim: int,
        device: str | torch.device,
        execute_horizon: int | None = 30,
        image_key_map: dict[str, str] | None = None,
    ) -> "Pi05PolicyAdapter":
        torch_device = torch.device(device)
        train_config = _get_openpi_config(config_name)
        policy = _create_openpi_policy(
            train_config,
            Path(checkpoint_dir),
            pytorch_device=str(torch_device),
        )
        return cls.from_policy(
            policy,
            prompt=prompt,
            action_dim=action_dim,
            device=torch_device,
            execute_horizon=execute_horizon,
            image_key_map=image_key_map,
        )
```

```python
# _to_openpi_obs（原 131-145 行）：用 self.image_key_map 替换模块常量
    def _to_openpi_obs(self, raw_obs, env_index: int) -> dict:
        if _STATE_KEY not in raw_obs:
            raise KeyError(f"missing observation state key: {_STATE_KEY}")

        obs = {
            "state": _state_to_numpy(raw_obs[_STATE_KEY], env_index),
            "images": {},
            "prompt": self.prompt,
        }

        for raw_key, image_name in self.image_key_map.items():
            if raw_key in raw_obs:
                obs["images"][image_name] = _image_to_numpy(raw_obs[raw_key], env_index, raw_key)

        return obs
```

- [ ] **Step 4: 运行验证通过**

Run: `cd /data2/kai0 && conda run -n residual python -m pytest resfit_pi05/tests/test_pi05_policy_adapter_image_map.py -v`
Expected: PASS（两个测试）。

- [ ] **Step 5: 跑 dsrl_pi05 既有 adapter 测试，确认未破坏**

Run: `cd /data2/kai0 && conda run -n residual python -m pytest resfit_pi05/tests -v`
Expected: 全部 PASS（默认映射行为不变）。

- [ ] **Step 6: Commit（在 kai0 仓）**

```bash
cd /data2/kai0
git add resfit_pi05/pi05_policy_adapter.py resfit_pi05/tests/test_pi05_policy_adapter_image_map.py
git commit -m "feat(pi05-adapter): make camera-view mapping configurable"
```

---

## Task 3: residual 侧 pi05 加载工厂（websocket 客户端）

连 pi05 websocket server，包成"长得像 ACTPolicy"的 step 级基座对象。

**Files:**
- Create: `resfit/lerobot/policies/pi05/__init__.py`
- Create: `resfit/lerobot/policies/pi05/load_pi05.py`
- Test: `tests/policies/pi05/test_load_pi05.py`

- [ ] **Step 1: 写失败测试（monkeypatch 注入假 client + 假 adapter，不依赖真 server）**

```python
# tests/policies/pi05/test_load_pi05.py
import types

from resfit.lerobot.policies.pi05 import load_pi05


class _FakeClient:
    created_with = None

    def __init__(self, host, port):
        type(self).created_with = {"host": host, "port": port}


class _FakeAdapter:
    created_with = None

    @classmethod
    def from_policy(cls, policy, **kwargs):
        cls.created_with = {"policy": policy, **kwargs}
        inst = cls()
        inst.config = types.SimpleNamespace(
            image_features=dict.fromkeys(kwargs["image_key_map"])
        )
        return inst


def test_load_pi05_builds_ws_client_and_adapter(monkeypatch):
    monkeypatch.setattr(load_pi05, "_import_ws_client", lambda: _FakeClient)
    monkeypatch.setattr(load_pi05, "_import_pi05_adapter", lambda: _FakeAdapter)

    cfg = types.SimpleNamespace(
        type="pi05",
        host="127.0.0.1",
        port=8000,
        prompt="pick up the cube",
        action_dim=7,
        execute_horizon=30,
        image_key_map={
            "observation.images.agentview": "base",
            "observation.images.robot0_eye_in_hand": "left_wrist",
        },
        kai0_paths=["/data2/kai0"],
    )
    policy = load_pi05.load_pi05_base_policy(cfg, device="cpu")

    assert _FakeClient.created_with == {"host": "127.0.0.1", "port": 8000}
    assert isinstance(_FakeAdapter.created_with["policy"], _FakeClient)
    assert _FakeAdapter.created_with["action_dim"] == 7
    assert _FakeAdapter.created_with["prompt"] == "pick up the cube"
    assert _FakeAdapter.created_with["image_key_map"] == cfg.image_key_map
    assert list(policy.config.image_features.keys()) == list(cfg.image_key_map.keys())
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n residual python -m pytest tests/policies/pi05/test_load_pi05.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现工厂**

```python
# resfit/lerobot/policies/pi05/__init__.py
from resfit.lerobot.policies.pi05.load_pi05 import load_pi05_base_policy

__all__ = ["load_pi05_base_policy"]
```

```python
# resfit/lerobot/policies/pi05/load_pi05.py
"""加载 pi05 基座策略：连 pi05 websocket server，用 Pi05PolicyAdapter.from_policy 包成 step 级基座。"""
from __future__ import annotations

import sys


def _import_ws_client():
    """openpi-client 的 WebsocketClientPolicy（独立函数便于测试 monkeypatch）。"""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: WPS433
    return WebsocketClientPolicy


def _import_pi05_adapter():
    """从 kai0 import Pi05PolicyAdapter（独立函数便于测试 monkeypatch）。"""
    from resfit_pi05.pi05_policy_adapter import Pi05PolicyAdapter  # noqa: WPS433
    return Pi05PolicyAdapter


def load_pi05_base_policy(cfg, device):
    """连 pi05 websocket server 并包成 step 级 base_policy。

    cfg 需含字段：host, port, prompt, action_dim, execute_horizon, image_key_map, kai0_paths。
    返回对象提供 select_action / reset / config.image_features（step 级路径所需）。
    """
    for p in getattr(cfg, "kai0_paths", []):
        if p not in sys.path:
            sys.path.insert(0, p)

    WebsocketClientPolicy = _import_ws_client()
    Pi05PolicyAdapter = _import_pi05_adapter()

    client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)
    return Pi05PolicyAdapter.from_policy(
        client,
        prompt=cfg.prompt,
        action_dim=cfg.action_dim,
        device=str(device),
        execute_horizon=cfg.execute_horizon,
        image_key_map=dict(cfg.image_key_map),
    )
```

- [ ] **Step 4: 运行验证通过**

Run: `conda run -n residual python -m pytest tests/policies/pi05/test_load_pi05.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add resfit/lerobot/policies/pi05/__init__.py resfit/lerobot/policies/pi05/load_pi05.py tests/policies/pi05/test_load_pi05.py
git commit -m "feat: pi05 websocket-client base-policy load factory"
```

---

## Task 4: 配置开关 `BasePolicyConfig.type` 与 pi05 字段

**Files:**
- Modify: `resfit/rl_finetuning/config/residual_td3.py`（`BasePolicyConfig`:36-40）
- Test: `tests/rl_finetuning/test_base_policy_switch.py`（本任务先建文件 + 第一个测试）

- [ ] **Step 1: 写失败测试**

```python
# tests/rl_finetuning/test_base_policy_switch.py
from resfit.rl_finetuning.config.residual_td3 import BasePolicyConfig


def test_default_type_is_act():
    cfg = BasePolicyConfig()
    assert cfg.type == "act"


def test_pi05_fields_exist_with_defaults():
    cfg = BasePolicyConfig(type="pi05")
    assert cfg.type == "pi05"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.action_dim == 7
    assert cfg.execute_horizon == 30
    assert cfg.prompt == "pick up the cube"
    assert "observation.images.agentview" in cfg.image_key_map
    assert cfg.kai0_paths == ["/data2/kai0"]
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_base_policy_switch.py -v`
Expected: FAIL（`type` 字段不存在）。

- [ ] **Step 3: 扩展 `BasePolicyConfig`**

```python
# resfit/rl_finetuning/config/residual_td3.py（替换原 36-40 行 BasePolicyConfig）
from dataclasses import dataclass, field


@dataclass
class BasePolicyConfig:
    # 基座类型开关："act"（默认，旧行为不变）或 "pi05"
    type: str = "act"

    # --- ACT 路径（从 W&B 加载，原有字段保持不变）---
    wandb_id: str = "TODO"
    wt_type: str = "best"
    wt_version: str = "latest"

    # --- pi05 路径（type == "pi05" 时使用，websocket 客户端）---
    host: str = "127.0.0.1"                 # pi05 websocket server 地址（GPU serve 端）
    port: int = 8000                        # pi05 websocket server 端口
    prompt: str = "pick up the cube"        # 与微调时一致
    action_dim: int = 7                     # robomimic 单臂动作维度
    execute_horizon: int = 30               # 每次推理实际执行步数（截断 50 步 chunk）
    image_key_map: dict = field(default_factory=lambda: {
        "observation.images.agentview": "base",
        "observation.images.robot0_eye_in_hand": "left_wrist",
    })
    kai0_paths: list = field(default_factory=lambda: ["/data2/kai0"])
    # 注：pi05 的 config_name/checkpoint_dir 属于 GPU serve 端（见 Task 9b），不在此配置
```

- [ ] **Step 4: 运行验证通过**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_base_policy_switch.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/config/residual_td3.py tests/rl_finetuning/test_base_policy_switch.py
git commit -m "feat: add base_policy.type switch and pi05 config fields"
```

---

## Task 5: `train_residual_td3` 加载分发 + actor_name 分发

**Files:**
- Modify: `resfit/rl_finetuning/scripts/train_residual_td3.py`（加载段:258-270；actor_name 段:272-278）

- [ ] **Step 1: 把"加载基座"抽成一个可测的分发函数**

在脚本里（或同目录新建 `base_policy_factory.py`，本计划放脚本内最小改动）新增：

```python
# train_residual_td3.py，在 import 区下方新增
def build_base_policy(cfg, device):
    """按 cfg.base_policy.type 分发加载基座，返回 (policy, actor_name)。"""
    bp = cfg.base_policy
    btype = getattr(bp, "type", "act")

    if btype == "act":
        from resfit.lerobot.utils.load_policy import load_policy
        from resfit.lerobot.utils.wandb_io import download_policy_from_wandb  # 原脚本已用
        policy_dir, _ = download_policy_from_wandb(
            bp.wandb_id, step=bp.wt_type, artifact_version=bp.wt_version,
        )
        policy = load_policy(policy_dir)
        policy.to(device)
        policy.eval()
        return policy, "residual_act"

    if btype == "pi05":
        from resfit.lerobot.policies.pi05 import load_pi05_base_policy
        assert bp.host and bp.port, "pi05 base policy requires host/port (websocket server)"
        # 连 pi05 websocket server（需 GPU 端先按 Task 9b 起好 server）
        policy = load_pi05_base_policy(bp, device)
        # 残差 actor 只吃 observation.base_action 向量，与基座类型解耦，复用 residual_act
        return policy, "residual_act"

    raise ValueError(f"Unknown base_policy.type: {btype}")
```

> 注：`download_policy_from_wandb` 的真实 import 路径以原脚本 258 行处为准；上面 import 行按原脚本调整。

- [ ] **Step 2: 用分发函数替换原加载段（258-270）**

把原来的：

```python
    policy_dir, _ = download_policy_from_wandb(
        cfg.base_policy.wandb_id,
        step=cfg.base_policy.wt_type,
        artifact_version=cfg.base_policy.wt_version,
    )
    base_policy: ACTPolicy = load_policy(policy_dir)
    base_policy.to(device)
    base_policy.eval()
    eval_base_policy: ACTPolicy = load_policy(policy_dir)
    eval_base_policy.to(device)
    eval_base_policy.eval()
```

替换为：

```python
    base_policy, inferred_actor_name = build_base_policy(cfg, device)
    eval_base_policy, _ = build_base_policy(cfg, device)
```

- [ ] **Step 3: 替换 actor_name 判断段（272-278）**

把原来的：

```python
    base_cfg = base_policy.config
    if isinstance(base_cfg, ACTConfig):
        cfg.actor_name = "residual_act"
    else:
        raise ValueError(f"Unknown base policy type: {type(base_cfg)}")
```

替换为：

```python
    cfg.actor_name = inferred_actor_name
```

- [ ] **Step 4: 语法/导入自检（不跑全训练）**

Run: `conda run -n residual python -c "import ast; ast.parse(open('resfit/rl_finetuning/scripts/train_residual_td3.py').read()); print('OK')"`
Expected: `OK`。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/scripts/train_residual_td3.py
git commit -m "feat: dispatch base policy load by type (act/pi05)"
```

---

## Task 6: 放宽 env wrapper 类型注解（不绑死 ACT）

**Files:**
- Modify: `resfit/rl_finetuning/wrappers/residual_env_wrapper.py`（import:20；类型注解:40）

- [ ] **Step 1: 改类型注解为结构化协议（鸭子类型）**

把 import（20 行）的硬依赖去掉对注解的强绑定，并把 40 行注解放宽：

```python
# 顶部新增（typing 区）
from typing import Protocol


class BasePolicyProtocol(Protocol):
    config: object
    def select_action(self, raw_obs: dict): ...
    def reset(self, env_ids=None): ...
```

```python
# __init__ 注解（原 40 行）
        base_policy: "BasePolicyProtocol",
```

> 保留原 `from resfit.lerobot.policies.act.modeling_act import ACTPolicy` 这行不删（其它地方可能引用），只是不再用它做唯一注解。

- [ ] **Step 2: 自检导入**

Run: `conda run -n residual python -c "from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper; print('OK')"`
Expected: `OK`。

- [ ] **Step 3: Commit**

```bash
git add resfit/rl_finetuning/wrappers/residual_env_wrapper.py
git commit -m "refactor: loosen base_policy type to structural protocol"
```

---

## Task 7: 集成 smoke test（mock pi05 跑通 wrapper）

确认开关切到 pi05 时，wrapper 的 `select_action / reset / config.image_features` 全程不报错，且残差组合维度对。

**Files:**
- Test: `tests/rl_finetuning/test_base_policy_switch.py`（追加集成用例）

- [ ] **Step 1: 追加测试**

```python
# tests/rl_finetuning/test_base_policy_switch.py 追加
import sys
import numpy as np

if "/data2/kai0" not in sys.path:  # 为 import resfit_pi05.* 提供包根
    sys.path.insert(0, "/data2/kai0")
from resfit_pi05.pi05_policy_adapter import Pi05PolicyAdapter


class _FakePi05Policy:
    def infer(self, obs):
        # 返回 50x32 的固定 chunk，前 7 维为可辨识值
        chunk = np.zeros((50, 32), dtype=np.float32)
        chunk[:, :7] = 0.5
        return {"actions": chunk}


def test_pi05_adapter_drives_step_loop_shapes():
    adapter = Pi05PolicyAdapter.from_policy(
        _FakePi05Policy(), prompt="pick up the cube", action_dim=7, device="cpu",
        execute_horizon=30,
        image_key_map={
            "observation.images.agentview": "base",
            "observation.images.robot0_eye_in_hand": "left_wrist",
        },
    )
    # env wrapper 取图用的 key
    assert list(adapter.config.image_features.keys()) == [
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
    ]

    B = 2
    raw_obs = {
        "observation.state": np.zeros((B, 9), dtype=np.float32),
        "observation.images.agentview": np.zeros((B, 224, 224, 3), dtype=np.uint8),
        "observation.images.robot0_eye_in_hand": np.zeros((B, 224, 224, 3), dtype=np.uint8),
    }
    a1 = adapter.select_action(raw_obs)
    assert tuple(a1.shape) == (B, 7)
    assert np.allclose(a1.cpu().numpy(), 0.5)

    # 连续调用走队列，不重复 infer 直到 execute_horizon 用尽
    for _ in range(5):
        adapter.select_action(raw_obs)
    adapter.reset()  # 清队列不报错
    a2 = adapter.select_action(raw_obs)
    assert tuple(a2.shape) == (B, 7)
```

- [ ] **Step 2: 运行**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_base_policy_switch.py -v`
Expected: PASS（全部用例）。

- [ ] **Step 3: ACT 回归——确认默认路径未被破坏**

Run: `conda run -n residual python -m pytest tests/rl_finetuning -k "act or base_policy" -v`
Expected: PASS。若仓内已有 ACT 相关测试，一并跑确认绿。

- [ ] **Step 4: Commit**

```bash
git add tests/rl_finetuning/test_base_policy_switch.py
git commit -m "test: pi05 step-loop integration smoke + act regression"
```

---

## Task 8 (Phase 0): robomimic → LeRobot 数据转换

**Files:**
- 复用: `resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`（不改）

- [ ] **Step 1: 选定任务与源数据**

确认 robomimic Lift demo HDF5 路径（如 `~/robomimic/datasets/lift/ph/image.hdf5`）。记录其 `actions` 维度与可用图像键（`agentview_image`, `robot0_eye_in_hand_image`）。

Run: `conda run -n residual python -c "import h5py; f=h5py.File('<LIFT_HDF5>','r'); d=f['data/demo_0']; print('action', d['actions'].shape); print('obs', list(d['obs'].keys()))"`
Expected: 打印 action 维度（应为 7）与 obs 键含 `agentview_image` / `robot0_eye_in_hand_image`。

- [ ] **Step 2: 跑转换**

Run:
```bash
conda run -n residual python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
  --dataset <LIFT_HDF5> \
  --output_dir /data2/datasets/lerobot/robomimic_lift \
  --repo_id local/robomimic_lift
```
Expected: 生成 LeRobot 数据集目录，无报错。

- [ ] **Step 3: 校验产物**

Run: `conda run -n residual python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds=LeRobotDataset('local/robomimic_lift', root='/data2/datasets/lerobot/robomimic_lift'); print(ds.meta.stats['action']['mean'].shape); print(ds.features.keys())"`
Expected: action 维度 7；features 含图像与 state 键。

- [ ] **Step 4: 记录映射结论**

把"robomimic 图像键 → pi05 槽位"映射与 action 维度写进 `docs/superpowers/specs/2026-06-04-pi05-base-policy-switch-design.md` 的 §10，供 Phase 1 data config 与 `BasePolicyConfig.image_key_map` 对齐。

- [ ] **Step 5: Commit（仅文档；数据集不入库）**

```bash
git add docs/superpowers/specs/2026-06-04-pi05-base-policy-switch-design.md
git commit -m "docs: record robomimic->lerobot mapping for pi05 finetune"
```

---

## Task 9 (Phase 1): pi05 微调 config + 训练

> 本任务依赖 openpi 内部 API。**实现前先读** `/data2/kai0/src/openpi/training/config.py` 中一个现成例子（推荐 `LeRobotLiberoDataConfig` 与其 `TrainConfig`），照其字段名仿写，避免猜 API。

**Files:**
- Modify: `/data2/kai0/src/openpi/training/config.py`（新增 `Pi05RobomimicDataConfig` 与 `TrainConfig`）

- [ ] **Step 1: 仿写一个 robomimic data config**

参照 `LeRobotLiberoDataConfig`，新增一个数据配置类，要点（字段名以现成例子为准）：
- `repo_id = "local/robomimic_lift"`；
- 数据 transforms：把 LeRobot 图像键 `observation.images.agentview` / `robot0_eye_in_hand` 映射到 pi05 模型槽位 `base_0_rgb` / `left_wrist_0_rgb`，右手腕槽位 pad 零图 + `image_mask=False`；
- `default_prompt = "pick up the cube"`；
- action 维度 7（模型内部 pad 到 32）。

- [ ] **Step 2: 注册 TrainConfig**

在 `_CONFIGS` 列表新增（参照 `pi05_aloha` 那条，name 与 model 用 pi05）：
```python
TrainConfig(
    name="pi05_robomimic_lift",
    model=pi0_config.Pi0Config(pi05=True),
    data=Pi05RobomimicDataConfig(repo_id="local/robomimic_lift",
                                 default_prompt="pick up the cube"),
)
```

- [ ] **Step 3: 计算归一化统计（kai0 uv 环境，非 residual）**

Run: `cd /data2/kai0 && uv run scripts/compute_norm_states_fast.py --config-name pi05_robomimic_lift`
（脚本名见 kai0 CLAUDE.md；kai0 用 uv 环境跑训练，不是 conda residual。）
Expected: 生成 norm stats assets，无报错。

- [ ] **Step 4: 跑微调（kai0 uv 环境，小步先验证能起训）**

Run: `cd /data2/kai0 && uv run scripts/train.py pi05_robomimic_lift --exp_name=pi05_lift_v1`
Expected: 训练正常迭代、loss 下降、按 save_interval 落 checkpoint。记下产出的 checkpoint 目录（供 Task 9b serve 用）。

- [ ] **Step 5: Commit（kai0 仓）**

```bash
cd /data2/kai0
git add src/openpi/training/config.py
git commit -m "feat(openpi): add pi05_robomimic_lift train config"
```

---

## Task 9b (serve): GPU 端起 pi05 websocket server

residual 端连 pi05 前（Task 10/11 都需要），GPU 端用 kai0 现成 serve 脚本起 server 加载微调 ckpt。这是运维步骤，不产代码、不 commit。

**Files:**
- 复用: `/data2/kai0/scripts/serve_policy.py`（不改，命令调用）

- [ ] **Step 1: 确认 serve 参数**

Run: `cd /data2/kai0 && uv run scripts/serve_policy.py --help`
Expected: 看到 policy config / checkpoint dir / port 等参数的真实名称（下面命令按此对齐）。

- [ ] **Step 2: 起 server（kai0 uv 环境，常驻进程）**

Run:
```bash
cd /data2/kai0 && uv run scripts/serve_policy.py \
  --policy.config pi05_robomimic_lift \
  --policy.dir <PI05_CKPT> \
  --default-prompt "pick up the cube" \
  --port 8000
```
Expected: 打印 pi05 加载完成并监听 `:8000`。保持常驻（后台或独立终端）。

- [ ] **Step 3: residual 端连通性自检**

在 residual 端另一终端：
```bash
conda run -n residual python -c "
import sys; sys.path.insert(0, '/data2/kai0')
from openpi_client.websocket_client_policy import WebsocketClientPolicy
c = WebsocketClientPolicy(host='127.0.0.1', port=8000)
print('connected OK')
"
```
Expected: 打印 `connected OK`（构造时会与 server 握手取 metadata）。失败则检查 server 是否在跑、端口是否一致。

---

## Task 10 (Phase 1 gate): 单独 eval pi05 基座成功率

**这是硬门槛**：基座成功率太低则不进入 Phase 3（参考 Coffee "基座全 0%" 教训）。

**Files:**
- Create: `resfit/rl_finetuning/scripts/eval_pi05_base.py`

- [ ] **Step 1: 写一个最小 eval 脚本**

在 residual 端，用 `load_pi05_base_policy`（连 Task 9b 起好的 websocket server）得到 base_policy，在 robomimic Lift 跑 N 个 episode，纯基座（无残差），统计成功率。脚本复用仓内现有 robomimic 评估环境构造（参照 `train_residual_td3.py` 的 env 创建段）。脚本接收 `--host/--port/--n_episodes`，构造一个 `BasePolicyConfig(type="pi05", host=..., port=...)` 传给 `load_pi05_base_policy`。

- [ ] **Step 2: 跑 eval（需 Task 9b 的 server 在跑）**

Run:
```bash
conda run -n residual python resfit/rl_finetuning/scripts/eval_pi05_base.py \
  --host 127.0.0.1 --port 8000 --n_episodes 50
```
Expected: 打印成功率。

- [ ] **Step 3: Gate 判定**

- 成功率达可用阈值（建议 ≥ 30%，具体按任务难度定）→ 进入 Task 11。
- 否则**停下**：排查 domain gap / prompt / 视角映射 / 归一化，回到 Task 8-9 迭代，不要硬上 RL。

- [ ] **Step 4: Commit**

```bash
git add resfit/rl_finetuning/scripts/eval_pi05_base.py
git commit -m "feat: standalone pi05 base-policy eval (phase-1 gate)"
```

---

## Task 11 (Phase 3): pi05 基座 + 残差 RL 端到端

**Files:**
- Create: 一个 pi05 实验配置/启动脚本（参照仓内现有 ACT 残差启动方式）。

- [ ] **Step 0: 确认 pi05 server 在跑**

Task 9b 的 websocket server 必须常驻（GPU 端 kai0 uv）。residual 端连通性自检通过后再继续。

- [ ] **Step 1: 配置切到 pi05**

设 `base_policy.type=pi05`、`host=127.0.0.1`、`port=8000`、`action_dim=7`、`image_key_map` 与 Task 8 结论一致、`prompt` 与微调一致。

- [ ] **Step 2: 短跑冒烟（少量 env step）**

Run: 用很小的 `total_steps` 跑一次，确认：加载分发走 pi05、能连上 server、rollout 不崩、残差组合维度对、能落 checkpoint。
Expected: 正常迭代，无维度/设备/连接错误。

- [ ] **Step 3: 正式训练**

Run: 正常配置启动残差 RL 训练。
Expected: 训练不崩；成功率相对纯基座有提升趋势。

- [ ] **Step 4: ACT 回归对照（确认开关无副作用）**

Run: 用 `base_policy.type=act` 跑一遍既有 ACT 实验配置的短冒烟，确认行为与改造前一致。
Expected: 与历史一致。

- [ ] **Step 5: Commit**

```bash
git add <pi05 实验配置/脚本>
git commit -m "feat: pi05 base policy + residual RL end-to-end config"
```

---

## Self-Review 备注

- **架构**：pi05 走 websocket 跨进程（spec §11）。GPU 端 kai0 uv 起 server（Task 9b），residual 端轻量 openpi-client + `Pi05PolicyAdapter.from_policy` 当 step 级基座。
- **Spec 覆盖**：§5.1 开关→Task 4/5/6；§5.2 微调链路→Task 8/9；§5.3 视角映射→Task 2/3（step 级，已确认无需补 model/normalize）；§7 测试→Task 1/2/3/7；§8 分阶段→Task 8/9/9b/10/11；§9 风险（环境/吞吐/server 生命周期/gate）→Task 1（环境）/Task 9b（server）/Task 10（gate）；§11 跨进程→Task 1/3/9b。
- **已知非精确处**（依赖外部库 API 的必要核实，非占位，均在任务内标注"先读 --help/现成例子"）：
  - Task 9 的 openpi data config 字段名（对照 `LeRobotLiberoDataConfig`）。
  - Task 9b 的 `serve_policy.py` 参数名（先 `--help`）。
  - `WebsocketClientPolicy(host, port)` 构造签名（探查确认，实现时复核）。
- **类型一致**：`build_base_policy` 返回 `(policy, actor_name)`，pi05 复用 `"residual_act"`（残差 actor 与基座解耦，仅吃 `observation.base_action`）。pi05 config 字段统一为 `host/port/prompt/action_dim/execute_horizon/image_key_map/kai0_paths`（不含 config_name/checkpoint_dir）。
