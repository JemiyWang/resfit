# pi0 基座跨进程链路冒烟 eval 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `eval_pi05_base.py` 第一版（冒烟），验证 serve(官方 pi0_aloha_sim) -> websocket -> Pi05PolicyAdapter -> three-piece dexmg env 跨进程链路在 robomimic 上跑通。

**Architecture:** 脚本拆成可独立测的纯逻辑单元（`check_action` 健全性断言、`run_smoke` episode 循环骨架）+ 真环境装配（`build_smoke_env` / `build_pi0_base_policy` / `main`）。纯逻辑用 fake 替身做 TDD 单测；真 server + 真 mujoco env 留手动集成 smoke。冒烟直接用裸 `create_vectorized_env`，不套残差用的 `BasePolicyVecEnvWrapper`（后者要 action_scaler/state_standardizer 且会叠加残差）。

**Tech Stack:** Python, numpy, torch, gymnasium, pytest;conda env `residual`;openpi-client websocket。

---

## 已核实事实（2026-06-04，写代码依据，勿再猜）

- obs dict 用 LeRobot 式 key：`observation.state`、`observation.images.agentview`、
  `observation.images.robot0_eye_in_hand`、`observation.images.robot1_eye_in_hand`。
- three-piece：action 14 维、3 视角；env_name（即 task）= `"TwoArmThreePieceAssembly"`。
- `create_vectorized_env(env_name, num_envs, device="cpu", camera_size=84, render_size=None, debug=False, video_key="observation.images.agentview") -> VectorizedEnvWrapper`
  （`resfit/dexmg/environments/dexmg.py`）。返回的 `vec_env`：`reset(**kwargs) -> (obs_dict, info)`，
  `step(actions_torch) -> (obs_dict, reward, terminated, truncated, info)`；reward/terminated/truncated 是 torch tensor[B]。
- `Pi05PolicyAdapter.select_action(obs) -> torch.Tensor[B, action_dim]`（截取前 action_dim 维），
  内部维护 chunk 队列；`reset(env_ids=None)` 清队列；`config.image_features` 是 image_key_map 的键集合。
- `load_pi05_base_policy(cfg, device)`（`resfit/lerobot/policies/pi05/load_pi05.py`）：cfg 需有
  `host/port/prompt/action_dim/execute_horizon/image_key_map/kai0_paths`。
- 现有测试布局：`tests/rl_finetuning/test_base_policy_switch.py`，fake 内层策略示例
  `_FakePi05Policy.infer(obs)->{"actions": np.zeros((50,32))}`。

## File Structure

- Create: `resfit/rl_finetuning/scripts/eval_pi05_base.py`
  - `check_action(arr, action_dim, abs_limit=5.0)` — 纯函数健全性断言
  - `run_smoke(env, base_policy, n_episodes, max_steps, action_dim)` — episode 循环骨架（env/policy 注入）
  - `format_report(report)` — 把诊断 dict 转成可打印字符串
  - `build_smoke_env(device, camera_size=84)` — 真 dexmg env 装配（薄封装）
  - `build_pi0_base_policy(host, port, device)` — 构造 BasePolicyConfig + 连 server
  - `main(argv=None)` — argparse + 装配 + 跑 run_smoke + 打印
- Create: `tests/rl_finetuning/test_eval_pi05_base.py`
  - 覆盖 `check_action`、`run_smoke`、`main`（monkeypatch 掉 build_* 真依赖）

测试命令统一：`conda run -n residual python -m pytest <path> -v`

---

## Task 1: check_action 健全性断言

**Files:**
- Create: `resfit/rl_finetuning/scripts/eval_pi05_base.py`
- Test: `tests/rl_finetuning/test_eval_pi05_base.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/rl_finetuning/test_eval_pi05_base.py
import numpy as np
import pytest

from resfit.rl_finetuning.scripts.eval_pi05_base import check_action


def test_check_action_accepts_valid():
    a = np.zeros((1, 14), dtype=np.float32)
    check_action(a, action_dim=14)  # should not raise


def test_check_action_rejects_wrong_dim():
    a = np.zeros((1, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="last dim"):
        check_action(a, action_dim=14)


def test_check_action_rejects_non_2d():
    a = np.zeros((14,), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        check_action(a, action_dim=14)


def test_check_action_rejects_nan():
    a = np.zeros((1, 14), dtype=np.float32)
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        check_action(a, action_dim=14)


def test_check_action_rejects_out_of_range():
    a = np.full((1, 14), 99.0, dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds limit"):
        check_action(a, action_dim=14)


def test_check_action_accepts_torch_tensor():
    import torch
    a = torch.zeros((1, 14))
    check_action(a, action_dim=14)  # should not raise
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_eval_pi05_base.py -v`
Expected: FAIL，`ModuleNotFoundError` 或 `ImportError: cannot import name 'check_action'`。

- [ ] **Step 3: 写最小实现**

```python
# resfit/rl_finetuning/scripts/eval_pi05_base.py
"""Smoke eval: drive a pi0/pi05 base policy (via websocket) on the three-piece dexmg env.

First version validates the cross-process link only (connect + step loop + sanity asserts),
not success rate. Grows into the Task 10 base-policy gate later.
"""
from __future__ import annotations

import numpy as np


def check_action(arr, action_dim, abs_limit=5.0):
    """Assert a base-policy action chunk-step is sane; raise ValueError otherwise."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"action must be 2-D [B, dim], got shape {a.shape}")
    if a.shape[-1] != action_dim:
        raise ValueError(f"action last dim must be {action_dim}, got {a.shape[-1]}")
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError(f"action must be floating dtype, got {a.dtype}")
    if not np.isfinite(a).all():
        raise ValueError("action contains NaN/Inf")
    peak = float(np.abs(a).max())
    if peak > abs_limit:
        raise ValueError(f"action abs max {peak:.3f} exceeds limit {abs_limit}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_eval_pi05_base.py -v`
Expected: 6 passed。

- [ ] **Step 5: 提交**

```bash
git -C /data2/RL/residual-offpolicy-rl add resfit/rl_finetuning/scripts/eval_pi05_base.py tests/rl_finetuning/test_eval_pi05_base.py
git -C /data2/RL/residual-offpolicy-rl commit -m "feat(eval_pi05_base): add check_action sanity asserts"
```

---

## Task 2: run_smoke episode 循环骨架

**Files:**
- Modify: `resfit/rl_finetuning/scripts/eval_pi05_base.py`
- Test: `tests/rl_finetuning/test_eval_pi05_base.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/rl_finetuning/test_eval_pi05_base.py
import torch
from resfit.rl_finetuning.scripts.eval_pi05_base import run_smoke


class _FakeEnv:
    """Vec env returning fixed obs; done after `done_after` steps."""
    def __init__(self, done_after=3):
        self._done_after = done_after
        self._t = 0
        self._obs = {
            "observation.state": torch.zeros((1, 18)),
            "observation.images.agentview": torch.zeros((1, 3, 84, 84)),
        }

    def reset(self, **kwargs):
        self._t = 0
        return self._obs, {}

    def step(self, action):
        self._t += 1
        term = torch.tensor([self._t >= self._done_after])
        trunc = torch.tensor([False])
        return self._obs, torch.tensor([0.0]), term, trunc, {}


class _FakePolicy:
    def __init__(self, action):
        self._a = action
        self.reset_calls = 0

    def reset(self, env_ids=None):
        self.reset_calls += 1

    def select_action(self, obs):
        return self._a


def test_run_smoke_runs_all_episodes():
    pol = _FakePolicy(torch.zeros((1, 14)))
    env = _FakeEnv(done_after=3)
    report = run_smoke(env, pol, n_episodes=2, max_steps=10, action_dim=14)
    assert len(report["episodes"]) == 2
    assert report["episodes"] == [3, 3]            # ends on terminated, not max_steps
    assert len(report["infer_times"]) == 6         # 2 episodes * 3 steps
    assert report["action_min"] == 0.0
    assert report["action_max"] == 0.0
    assert pol.reset_calls >= 2                     # reset once per episode


def test_run_smoke_propagates_bad_action():
    pol = _FakePolicy(torch.full((1, 14), 99.0))   # out of range
    env = _FakeEnv(done_after=3)
    with pytest.raises(ValueError, match="exceeds limit"):
        run_smoke(env, pol, n_episodes=1, max_steps=10, action_dim=14)


def test_run_smoke_stops_at_max_steps():
    pol = _FakePolicy(torch.zeros((1, 14)))
    env = _FakeEnv(done_after=1000)                # never done
    report = run_smoke(env, pol, n_episodes=1, max_steps=5, action_dim=14)
    assert report["episodes"] == [5]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_eval_pi05_base.py -k run_smoke -v`
Expected: FAIL，`ImportError: cannot import name 'run_smoke'`。

- [ ] **Step 3: 写最小实现**

```python
# 追加到 resfit/rl_finetuning/scripts/eval_pi05_base.py
import time


def _is_done(terminated, truncated):
    def _any(x):
        return bool(x.any()) if hasattr(x, "any") else bool(x)
    return _any(terminated) or _any(truncated)


def run_smoke(env, base_policy, n_episodes, max_steps, action_dim):
    """Run pure-base-policy rollouts; collect diagnostics. Raises on insane actions."""
    report = {
        "episodes": [],
        "infer_times": [],
        "action_min": float("inf"),
        "action_max": float("-inf"),
    }
    for _ in range(n_episodes):
        obs, _info = env.reset()
        base_policy.reset()
        steps = 0
        for _t in range(max_steps):
            t0 = time.perf_counter()
            action = base_policy.select_action(obs)
            report["infer_times"].append(time.perf_counter() - t0)
            check_action(action, action_dim)
            a = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
            report["action_min"] = min(report["action_min"], float(a.min()))
            report["action_max"] = max(report["action_max"], float(a.max()))
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if _is_done(terminated, truncated):
                break
        report["episodes"].append(steps)
    return report
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_eval_pi05_base.py -v`
Expected: 9 passed（6 from Task 1 + 3 here）。

- [ ] **Step 5: 提交**

```bash
git -C /data2/RL/residual-offpolicy-rl add resfit/rl_finetuning/scripts/eval_pi05_base.py tests/rl_finetuning/test_eval_pi05_base.py
git -C /data2/RL/residual-offpolicy-rl commit -m "feat(eval_pi05_base): add run_smoke episode loop"
```

---

## Task 3: 真环境装配 + main

**Files:**
- Modify: `resfit/rl_finetuning/scripts/eval_pi05_base.py`
- Test: `tests/rl_finetuning/test_eval_pi05_base.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/rl_finetuning/test_eval_pi05_base.py
import resfit.rl_finetuning.scripts.eval_pi05_base as evalmod
from resfit.rl_finetuning.scripts.eval_pi05_base import format_report


def test_format_report_runs():
    report = {"episodes": [3, 3], "infer_times": [0.01, 0.02], "action_min": -0.1, "action_max": 0.1}
    text = format_report(report)
    assert "episodes" in text.lower()
    assert "3" in text


def test_main_wires_env_and_policy(monkeypatch):
    captured = {}

    def fake_build_env(device, camera_size=84):
        captured["env_device"] = device
        return _FakeEnv(done_after=2)

    def fake_build_policy(host, port, device):
        captured["host"] = host
        captured["port"] = port
        return _FakePolicy(torch.zeros((1, 14)))

    monkeypatch.setattr(evalmod, "build_smoke_env", fake_build_env)
    monkeypatch.setattr(evalmod, "build_pi0_base_policy", fake_build_policy)

    evalmod.main(["--host", "1.2.3.4", "--port", "9999", "--n_episodes", "2", "--max_steps", "5"])

    assert captured["host"] == "1.2.3.4"
    assert captured["port"] == 9999
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_eval_pi05_base.py -k "format_report or main_wires" -v`
Expected: FAIL，`ImportError: cannot import name 'format_report'`。

- [ ] **Step 3: 写最小实现**

```python
# 追加到 resfit/rl_finetuning/scripts/eval_pi05_base.py
import argparse

import torch

ACTION_DIM = 14
TASK_NAME = "TwoArmThreePieceAssembly"
PROMPT = "assemble the three pieces"
IMAGE_KEY_MAP = {
    "observation.images.agentview": "base",
    "observation.images.robot0_eye_in_hand": "left_wrist",
    "observation.images.robot1_eye_in_hand": "right_wrist",
}


def format_report(report):
    n = len(report["episodes"])
    times = report["infer_times"]
    avg_ms = (sum(times) / len(times) * 1000.0) if times else 0.0
    return (
        f"smoke report: {n} episodes, lengths={report['episodes']}, "
        f"total_steps={sum(report['episodes'])}, "
        f"avg_infer={avg_ms:.1f}ms, "
        f"action_range=[{report['action_min']:.3f}, {report['action_max']:.3f}]"
    )


def build_smoke_env(device, camera_size=84):
    """Bare three-piece vec env (no residual wrapper)."""
    from resfit.dexmg.environments.dexmg import create_vectorized_env
    return create_vectorized_env(
        env_name=TASK_NAME, num_envs=1, device=device, camera_size=camera_size
    )


def build_pi0_base_policy(host, port, device):
    """Connect to the pi0/pi05 websocket server and wrap as a step-level base policy."""
    from resfit.rl_finetuning.config.residual_td3 import BasePolicyConfig
    from resfit.lerobot.policies.pi05 import load_pi05_base_policy

    cfg = BasePolicyConfig(
        type="pi05",
        host=host,
        port=port,
        action_dim=ACTION_DIM,
        execute_horizon=30,
        prompt=PROMPT,
        image_key_map=dict(IMAGE_KEY_MAP),
    )
    return load_pi05_base_policy(cfg, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description="pi0/pi05 base-policy cross-process smoke eval (three-piece)")
    parser.add_argument("--host", default="127.0.0.1", help="pi0 websocket server host")
    parser.add_argument("--port", type=int, default=8000, help="pi0 websocket server port")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device} host={args.host} port={args.port}")

    try:
        base_policy = build_pi0_base_policy(args.host, args.port, device)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"failed to connect base policy at {args.host}:{args.port} -- "
            f"is the pi0 server running? (see plan Task 9b). cause: {exc}"
        ) from exc

    env = build_smoke_env(device)
    report = run_smoke(env, base_policy, args.n_episodes, args.max_steps, ACTION_DIM)
    print(format_report(report))
    print("[smoke] PASS: cross-process link ran without crashing.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n residual python -m pytest tests/rl_finetuning/test_eval_pi05_base.py -v`
Expected: 11 passed。

- [ ] **Step 5: 提交**

```bash
git -C /data2/RL/residual-offpolicy-rl add resfit/rl_finetuning/scripts/eval_pi05_base.py tests/rl_finetuning/test_eval_pi05_base.py
git -C /data2/RL/residual-offpolicy-rl commit -m "feat(eval_pi05_base): add env/policy wiring + main"
```

---

## Task 4: 手动集成 smoke（GPU + 真 server，无自动测试）

> 这是运维 / 手动验证步骤，不产代码、不 commit。前三个 Task 的单测全绿后执行。

- [ ] **Step 1: GPU 端起官方 pi0 server（kai0 uv 环境）**

先确认参数名：`cd /data2/kai0 && uv run scripts/serve_policy.py --help`
然后起 server（常驻，独立终端）：
```bash
cd /data2/kai0 && uv run scripts/serve_policy.py \
  --policy.config pi0_aloha_sim \
  --policy.dir gs://openpi-assets/checkpoints/pi0_aloha_sim \
  --default-prompt "assemble the three pieces" \
  --port 8000
```
Expected: 打印 pi0 加载完成并监听 `:8000`。

- [ ] **Step 2: residual 端跑冒烟**

```bash
cd /data2/RL/residual-offpolicy-rl && \
MUJOCO_GL=egl conda run -n residual python -m resfit.rl_finetuning.scripts.eval_pi05_base \
  --host 127.0.0.1 --port 8000 --n_episodes 3 --max_steps 200
```
Expected: 打印 smoke report（episodes 长度、avg_infer、action_range）+ `[smoke] PASS`。
不看成功率（官方 aloha 基座在 three-piece 上必然约 0）。

- [ ] **Step 3: 判读结果**

冒烟**通过**判据（全部满足）：
- 没有连接异常（连不上会报 "is the pi0 server running?"）。
- 没有 `check_action` 断言失败（action shape=14、无 NaN、范围合理）。
- `env.step` 不崩，3 个 episode 跑完。
- avg_infer 时延在合理范围（单次推理几十~几百 ms，过大说明 server/网络异常）。

若 `check_action` 报 "action last dim must be 14"：说明 pi0_aloha_sim 输出维度 < 14 或 adapter
截取异常 —— 核对 server 端 config 的 action_dim，必要时调 `ACTION_DIM` / adapter 截取逻辑。
若连接失败：核对 server 是否在跑、端口一致、防火墙。

- [ ] **Step 4: 记录结论**

把冒烟结果（通过 / 失败 + 现象）记到
`docs/superpowers/2026-06-04-pi05-base-policy-switch-HANDOFF.md` 的进度区，供后续 Task 8/9 接手。

---

## Self-Review

- **Spec 覆盖**：§3 三单元 -> Task 1(check_action)/Task 2(run_smoke)/Task 3(main+build_*)；
  §5 错误处理 -> Task 3 main 的连接异常包装 + check_action 断言信息 + Task 4 Step 3 判读；
  §6 测试 -> Task 1/2/3 单测 + Task 4 手动集成；§8 待核实依赖 -> 已在“已核实事实”全部钉死。
- **占位符**：无 TBD/TODO；每个 code step 有完整代码。
- **类型一致**：`check_action(arr, action_dim, abs_limit)`、`run_smoke(env, base_policy, n_episodes, max_steps, action_dim)`、
  `build_smoke_env(device, camera_size)`、`build_pi0_base_policy(host, port, device)`、`main(argv)`、
  `format_report(report)` 在测试与实现中签名一致；report dict 键 `episodes/infer_times/action_min/action_max` 一致。
- **测试计数**：Task 1=6、Task 2=+3=9、Task 3=+2=11，与各 Step 4 Expected 对得上。
