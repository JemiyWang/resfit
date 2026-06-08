# HIQL 分层 Phase 3：低层子目标条件化 + buffer + 执行接线 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把残差低层（QAgent 的 raw `Actor` + `Critic`）改成**子目标条件**——条件在 10 维潜子目标 `z` 上（复用 `--stage_conditioned` 的 `append_*` 模板，把 stage one-hot 换成 z）；offline transition 用 demo 真航点 `z=φ(s_t,s_{t+k})`，online 用冻结高层 `z=π^h(s,g)`；TD3 训法、base policy 全不变。先做"喂真航点 z"消融证伪命门假设，再接学出来的高层做 A/B。

**Architecture:** 关键约束——给 actor/critic 的 `observation.state` **恒 18 维**（不含物体 pose）；object-aware 的 30 维 eef_piece 只经 `info["rel_piece"]` 喂 value/高层；**z（10 维）是唯一进策略的 object-aware 信号**。新增运行时 helper `hiql_subgoal.py`（载入 Phase 1 `gc_value.pt` + Phase 2 `high_actor.pt`，在线算 z、离线算航点 z）。改 `stage_utils`/`actor.py`/`q_agent.py` 加子目标条件，改 `offline_stage_replay.build_offline_buffer` 注入 `observation.subgoal`，改 `train_chunk_residual.py` 接线。

**Tech Stack:** PyTorch、TensorDict/torchrl replay buffer、pytest。conda env `residual`，命令从仓库根跑，训练/eval 要 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`。

**前置：** Phase 1 产 `gc_value.pt`、Phase 2 产 `high_actor.pt`；③a' object-aware（env 在 `state_mode=eef_piece` 经 `info["rel_piece"]` 透出 rel）已就绪。

---

## File Structure

| 文件 | 改动 |
|---|---|
| `off_policy/rl/stage_utils.py` | 加 `append_subgoal`（纯函数，拼 z 向量）。 |
| `off_policy/rl/actor.py` | `Actor` 加 `subgoal_conditioned`/`subgoal_dim`：`prop_dim += subgoal_dim`、`forward` append `obs["observation.subgoal"]`。 |
| `off_policy/rl/q_agent.py` | `QAgent` 加 `subgoal_conditioned`/`subgoal_dim`：`critic_prop_dim`、`_critic_prop`、构造 Actor 传参。 |
| `chunk_residual/hiql_subgoal.py`（新） | `HiqlSubgoal`：载 gc_value+high_actor，`subgoal_online(state_std, rel_raw, goal30)`、`subgoal_waypoint(s30_base, s30_target)`、`build_state30`。 |
| `chunk_residual/offline_stage_replay.py` | `build_offline_buffer` 加 `subgoal=` 参，per-transition 注入 `observation.subgoal=φ(s_t,s_{t+k})`。 |
| `chunk_residual/train_chunk_residual.py` | 新 flag、`lowdim_keys` 加 subgoal、构造 QAgent 传 subgoal、act 循环注入 z 到 obs/next_obs、buffer 签名加键、`build_offline_buffer` 传 subgoal、`add_chunk_transition` 透传 subgoal。 |
| `tests/test_hiql_subgoal_wiring.py`（新） | append_subgoal / Actor / QAgent / HiqlSubgoal 单测。 |

**不动**：Phase 1/2 模块（只 import）、`hiql_value.py`/`hiql_potential.py`、`residual_flow_actor.py`（`--actor flow` 的子目标化留作可选后续，本 plan 只接 `--actor raw`，与现有 stage_conditioned 同口径）。

---

## Task 1: append_subgoal 纯函数

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/stage_utils.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hiql_subgoal_wiring.py
import numpy as np
import torch


def test_append_subgoal():
    from resfit.rl_finetuning.off_policy.rl.stage_utils import append_subgoal
    prop = torch.zeros(4, 18)
    z = torch.ones(4, 10)
    out = append_subgoal(prop, z)
    assert out.shape == (4, 28)
    assert torch.equal(out[:, 18:], z)
    # 跨设备对齐:z 在 cpu, prop 在 cpu(GPU 不可用时退化为同设备)
    assert out.device == prop.device
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_append_subgoal -q`
Expected: FAIL（`ImportError: append_subgoal`）

- [ ] **Step 3: 写最小实现**（追加到 `stage_utils.py`，在 `append_stage` 之后）

```python
def append_subgoal(prop: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
    """把 10 维潜子目标 z 拼到 prop 末尾:[B, P] -> [B, P + rep_dim]。

    z 对齐到 prop 的 device(env 给的 z 可能在 CPU,prop 在 GPU)。与 append_stage 同模板,
    但 z 已是连续向量,无需 one-hot。
    """
    return torch.cat([prop, subgoal.to(prop.device)], dim=-1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_append_subgoal -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/stage_utils.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py
git commit -m "feat(hiql-low): append_subgoal(拼 z 向量到 prop)"
```

---

## Task 2: Actor 子目标条件化

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/actor.py:69-92, 164-183`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py`

- [ ] **Step 1: 写失败测试**

```python
def test_actor_subgoal_conditioned_dim():
    from resfit.rl_finetuning.off_policy.rl.actor import Actor
    from resfit.rl_finetuning.config.rlpd import ActorConfig
    cfg = ActorConfig()
    cfg.spatial_emb = 0
    # residual_actor=True 会 +action_dim;subgoal_conditioned 再 +subgoal_dim
    a = Actor(repr_dim=64, patch_repr_dim=8, prop_dim=18, action_dim=12, cfg=cfg,
              residual_actor=True, subgoal_conditioned=True, subgoal_dim=10)
    # prop_dim = 18 + 12(base_action) + 10(subgoal)
    assert a.prop_dim == 40
    obs = {
        "feat": torch.zeros(2, 64),
        "observation.state": torch.zeros(2, 18),
        "observation.base_action": torch.zeros(2, 12),
        "observation.subgoal": torch.ones(2, 10),
    }
    dist = a.forward(obs, std=0.1)
    assert dist.mean.shape == (2, 12)
```

> 注：若 `ActorConfig()` 无参构造失败，按本仓库 `ActorConfig` 的必填项补默认（参考 `q_agent` 测试或 `config/rlpd.py`）；关键是 `spatial_emb=0`、`use_layer_norm` 有默认。

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_actor_subgoal_conditioned_dim -q`
Expected: FAIL（`TypeError: unexpected keyword 'subgoal_conditioned'`）

- [ ] **Step 3: 改实现**

在 `actor.py` 的 `Actor.__init__` 签名（69-72 行）加两个参：

```python
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: ActorConfig, residual_actor: bool = False,
                 stage_conditioned: bool = False, num_stages: int = 0,
                 stage_budget: "list[float] | None" = None,
                 subgoal_conditioned: bool = False, subgoal_dim: int = 0):
```

在 `__init__` 体里存属性（紧接 `self.stage_budget = stage_budget` 之后）：

```python
        self.subgoal_conditioned = subgoal_conditioned
        self.subgoal_dim = subgoal_dim
```

在 `if stage_conditioned: self.prop_dim += num_stages` 之后追加：

```python
        if subgoal_conditioned:
            # 子目标条件:10 维潜 z 作为额外 prop 维度喂入(与 stage one-hot 同位置)
            self.prop_dim += subgoal_dim
```

在 `forward`（179-181 行 `if self.stage_conditioned: ...` 之后）追加：

```python
            if self.subgoal_conditioned:
                all_input.append(obs["observation.subgoal"].to(feat.device))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_actor_subgoal_conditioned_dim -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py
git commit -m "feat(hiql-low): Actor 子目标条件(prop += z;forward append subgoal)"
```

---

## Task 3: QAgent 子目标条件化

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/q_agent.py:28-39, 68-69, 90-102, 267-273`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py`

- [ ] **Step 1: 写失败测试**

```python
def test_qagent_critic_prop_subgoal():
    # 只测 _critic_prop 拼接逻辑(免构造整个 QAgent),用 SimpleNamespace 伪装
    from types import SimpleNamespace
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    fake = SimpleNamespace(stage_conditioned=False, subgoal_conditioned=True, num_stages=0)
    obs = {"observation.state": torch.zeros(3, 18), "observation.subgoal": torch.ones(3, 10)}
    prop = QAgent._critic_prop(fake, obs)
    assert prop.shape == (3, 28)
    assert torch.equal(prop[:, 18:], torch.ones(3, 10))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_qagent_critic_prop_subgoal -q`
Expected: FAIL（`AttributeError`：`_critic_prop` 里访问 `self.subgoal_conditioned` 不存在 / 未拼接）

- [ ] **Step 3: 改实现**

`QAgent.__init__` 签名（28-39 行）加参：

```python
        stage_budget: "list[float] | None" = None,
        subgoal_conditioned: bool = False,
        subgoal_dim: int = 0,
    ):
```

存属性（68-69 行 `self.num_stages = num_stages` 之后）：

```python
        self.subgoal_conditioned = subgoal_conditioned
        self.subgoal_dim = subgoal_dim
```

`critic_prop_dim`（91 行）改为同时计入 subgoal：

```python
        critic_prop_dim = (prop_dim
                           + (self.num_stages if self.stage_conditioned else 0)
                           + (self.subgoal_dim if self.subgoal_conditioned else 0))
```

构造 Actor（99-102 行）传参：

```python
        self.actor = Actor(repr_dim, patch_repr_dim, prop_dim, action_dim, cfg.actor,
                           residual_actor=residual_actor,
                           stage_conditioned=self.stage_conditioned, num_stages=self.num_stages,
                           stage_budget=stage_budget,
                           subgoal_conditioned=self.subgoal_conditioned, subgoal_dim=self.subgoal_dim)
```

`_critic_prop`（267-273 行）追加 subgoal 拼接，并 import `append_subgoal`（文件顶部 `from ...stage_utils import append_stage` 改为 `append_stage, append_subgoal`）：

```python
    def _critic_prop(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        prop = obs["observation.state"]
        if self.stage_conditioned:
            prop = append_stage(prop, obs["observation.stage_id"], self.num_stages)
        if self.subgoal_conditioned:
            prop = append_subgoal(prop, obs["observation.subgoal"])
        return prop
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_qagent_critic_prop_subgoal -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/q_agent.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py
git commit -m "feat(hiql-low): QAgent 子目标条件(critic_prop_dim/_critic_prop/Actor 传参)"
```

---

## Task 4: 运行时 helper HiqlSubgoal

封装"载冻结 gc_value+high_actor、在线算 z、离线算航点 z、拼 30 维 state"，online/offline 同源标准化。

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py`

- [ ] **Step 1: 写失败测试**

```python
def test_hiql_subgoal_shapes(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF, save_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor, save_high_actor
    from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal
    # 造冻结 gc_value + high_actor 存盘
    gc = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=32)
    gp = str(tmp_path / "gc.pt")
    save_gc_value(gp, gc, v_stats={"min": -1, "max": 0, "mean": -0.5},
                  mean=torch.zeros(18), std=torch.ones(18), dataset_id="ds",
                  state_mode="eef_piece", rel_piece_stats=(np.zeros(12), np.ones(12)))
    ha = HighActor(state_dim=30, rep_dim=10, hidden=32)
    hp = str(tmp_path / "ha.pt")
    save_high_actor(hp, ha, gc_value_ckpt=gp, way_steps=25, beta=1.0)

    goal30 = torch.zeros(30)
    sg = HiqlSubgoal.from_ckpts(gp, hp, goal30=goal30, device="cpu")
    assert sg.rep_dim == 10
    # build_state30: 18 std state + 标准化 rel(12) -> 30
    s30 = sg.build_state30(torch.zeros(2, 18), np.zeros((2, 12)))
    assert s30.shape == (2, 30)
    # 在线 z = high_actor(s30, goal30)
    z = sg.subgoal_online(torch.zeros(2, 18), np.zeros((2, 12)))
    assert z.shape == (2, 10)
    # 离线航点 z = gc_value.phi(s30_base, s30_target)
    zw = sg.subgoal_waypoint(torch.zeros(3, 30), torch.ones(3, 30))
    assert zw.shape == (3, 10)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_hiql_subgoal_shapes -q`
Expected: FAIL（`ModuleNotFoundError: hiql_subgoal`）

- [ ] **Step 3: 写实现**

```python
# hiql_subgoal.py
"""HIQL 分层路运行时 helper(Phase 3):载冻结 gc_value+high_actor,在线/离线算潜子目标 z。

约束:给 actor/critic 的 observation.state 恒 18 维;30 维 eef_piece(18+12 标准化 rel)只用于
算 z;z(10 维)是唯一进策略的 object-aware 信号。online/offline 用同一套 rel mean/std 标准化。
设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import load_high_actor


class HiqlSubgoal:
    def __init__(self, gc_value, high_actor, goal30, rel_mean, rel_std, device="cpu"):
        self.vf = gc_value.to(device).eval()
        self.ha = high_actor.to(device).eval()
        for m in (self.vf, self.ha):
            for p in m.parameters():
                p.requires_grad_(False)
        self.rep_dim = gc_value.rep_dim
        self.device = device
        self.goal30 = torch.as_tensor(np.asarray(goal30), dtype=torch.float32, device=device).reshape(-1)
        self.rel_mean = torch.as_tensor(np.asarray(rel_mean), dtype=torch.float32, device=device)
        self.rel_std = torch.as_tensor(np.asarray(rel_std), dtype=torch.float32, device=device)

    @classmethod
    def from_ckpts(cls, gc_value_ckpt, high_actor_ckpt, *, goal30, device="cpu"):
        gc, info = load_gc_value(gc_value_ckpt, map_location=device)
        ha, _ = load_high_actor(high_actor_ckpt, map_location=device)
        assert info["state_mode"] == "eef_piece", "分层路 gc_value 须 eef_piece(object-aware)"
        return cls(gc, ha, goal30, info["rel_piece_mean"], info["rel_piece_std"], device=device)

    def build_state30(self, state_std, rel_raw):
        """18 维已标准化 state(tensor [B,18]) + raw rel_piece([B,12] np/tensor) -> [B,30] tensor。"""
        x = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        rel = torch.as_tensor(np.asarray(rel_raw), dtype=torch.float32, device=self.device)
        if rel.ndim == 1:
            rel = rel.unsqueeze(0)
        rel_n = (rel - self.rel_mean) / self.rel_std
        return torch.cat([x, rel_n], dim=-1)

    @torch.no_grad()
    def subgoal_online(self, state_std, rel_raw):
        """在线:z = π^h(s30, goal30)(取分布均值,确定性)。返回 [B, rep_dim]。"""
        s30 = self.build_state30(state_std, rel_raw)
        g30 = self.goal30.unsqueeze(0).expand(s30.shape[0], -1)
        return self.ha(s30, g30).mean

    @torch.no_grad()
    def subgoal_waypoint(self, s30_base, s30_target):
        """离线:z = φ(base=s_t, target=s_{t+k})(真航点)。返回 [B, rep_dim]。"""
        b = torch.as_tensor(s30_base, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(s30_target, dtype=torch.float32, device=self.device)
        return self.vf.phi(b, t)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_hiql_subgoal_shapes -q`
Expected: PASS

- [ ] **Step 5: 跑全 wiring 单测 + 提交**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py -q`
Expected: 4 个测试全 PASS

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_subgoal.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py
git commit -m "feat(hiql-low): HiqlSubgoal 运行时 helper(在线 z=π^h;离线 z=φ 航点)"
```

---

## Task 5: offline buffer 注入 observation.subgoal

`build_offline_buffer` 已为 eef_piece 算 `rel_seq`（T,12）。复用它拼 30 维、用 `subgoal.subgoal_waypoint` 算 per-transition 航点 z，存进 curr/nxt 的 `observation.subgoal`。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py:156-244`

- [ ] **Step 1: 改 `build_offline_buffer` 签名**（156-158 行）加 `subgoal=None`、`way_steps=25`：

```python
def build_offline_buffer(rb, dataset_path, *, action_scaler, state_standardizer,
                         image_keys, bonus, mode, gamma, num_demos=None,
                         stage_cache=None, potential=None, subgoal=None, way_steps=25) -> int:
```

- [ ] **Step 2: 强制 rel 序列在 subgoal 模式也算**（176 行 `need_rel` 改为）：

```python
    need_rel = (potential is not None and getattr(potential, "state_mode", "eef") == "eef_piece") \
               or (subgoal is not None)
```

- [ ] **Step 3: 算 per-demo 30 维 state + 航点 z**（在 220-221 行 `sid/nsid` 之后、`for t in range(T-1)` 之前追加）：

```python
                subgoal_z = None
                if subgoal is not None:
                    assert rel_seq is not None, "subgoal 模式需 eef_piece rel(env replay 已算)"
                    rel_n = (torch.as_tensor(rel_seq, dtype=torch.float32) - subgoal.rel_mean.cpu()) \
                        / subgoal.rel_std.cpu()
                    s30 = torch.cat([state_n, rel_n[:T]], dim=-1)            # (T,30)
                    way = np.minimum(np.arange(T) + way_steps, T - 1)        # k 步航点(裁到末态)
                    subgoal_z = subgoal.subgoal_waypoint(s30, s30[way]).cpu()  # (T,10)
```

- [ ] **Step 4: 把 z 写进 transition**（在 224-229 行 `curr`/`nxt` 字典里，`for t` 循环内，stage_id 行之后追加）：

```python
                    if subgoal_z is not None:
                        curr["observation.subgoal"] = subgoal_z[t]
                        nxt["observation.subgoal"] = subgoal_z[min(t + 1, T - 1)]
```

> 注：`curr`/`nxt` 是 `for t in range(T-1)` 内构造的 dict（223-229 行）；上面两行加在该 dict 字面量构造之后、`for k in image_keys` 之前，或直接在 dict 里加键。保持与 stage_id 同结构（`subgoal_z[t]` 形状 `[10]`）。

- [ ] **Step 5: 跑现有 offline 相关单测确认不破坏**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -k "offline or stage or hdf5" -q`
Expected: 现有测试仍 PASS（`subgoal=None` 时行为逐位等价旧版）。

- [ ] **Step 6: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/offline_stage_replay.py
git commit -m "feat(hiql-low): build_offline_buffer 注入 observation.subgoal(真航点 z)"
```

---

## Task 6: train_chunk_residual 接线（flag + 构造 + act 循环 + buffer）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`（多处，见各步行号）

- [ ] **Step 1: 加 CLI flag**（`build_parser`，紧邻 261-264 行的 `--stage_conditioned` 之后）

```python
    p.add_argument("--subgoal_conditioned", action="store_true",
                   help="把 HIQL 潜子目标 z(10维)喂进 actor/critic(分层路 Phase 3;默认关=baseline)")
    p.add_argument("--gc_value_ckpt", default=None, help="Phase 1 gc_value.pt(--subgoal_conditioned 需)")
    p.add_argument("--high_actor_ckpt", default=None, help="Phase 2 high_actor.pt(在线提 z;消融模式可不给)")
    p.add_argument("--subgoal_way_steps", type=int, default=25, help="offline 真航点 z 的 k 步")
```

- [ ] **Step 2: 构造 HiqlSubgoal + goal30**（在构造 QAgent 之前，446-453 行 stage 断言附近追加）

```python
    subgoal = None
    if args.subgoal_conditioned:
        assert args.actor == "raw", "subgoal-conditioning 第一版只支持 --actor raw"
        assert env_state_mode == "eef_piece", "--subgoal_conditioned 需 gc_value 为 eef_piece + env 透出 rel"
        assert args.gc_value_ckpt and args.high_actor_ckpt, \
            "--subgoal_conditioned 需 --gc_value_ckpt 与 --high_actor_ckpt"
        from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal
        from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
        # goal30 = demo 末态(30 维)均值(在线高层的固定任务目标)
        seqs, _, _ = read_per_demo_states(args.offline_dataset_path, args.dataset, "eef_piece",
                                          num_demos=args.offline_num_demos)
        goal30 = np.stack([s[-1] for s in seqs], axis=0).mean(axis=0)
        subgoal = HiqlSubgoal.from_ckpts(args.gc_value_ckpt, args.high_actor_ckpt,
                                         goal30=goal30, device=args.device)
        print(f"[hiql-subgoal] on; rep_dim={subgoal.rep_dim} "
              f"gc={args.gc_value_ckpt} high={args.high_actor_ckpt}")
```

> 注：`env_state_mode` 由 `--potential_source hiql` 决定（382 行）。subgoal 路要 env 在 eef_piece 透出 rel，但**不**需要 PBS 整形；若未开 potential，需让 env 也走 eef_piece。最简：当 `--subgoal_conditioned` 时把 384 行 `state_mode=env_state_mode` 改为 `state_mode=("eef_piece" if (args.subgoal_conditioned or env_state_mode=="eef_piece") else "eef")`，并相应把 `env_state_mode="eef_piece"`。在 Step 2 之前（384 行附近）加：
> ```python
>     if args.subgoal_conditioned:
>         env_state_mode = "eef_piece"
> ```
> （置于 382 行 `env_state_mode = ...` 之后、384 行 `create_vectorized_env` 之前。）

- [ ] **Step 3: lowdim_keys 加 subgoal**（434 行）

```python
    lowdim_keys = ["observation.state", "observation.base_action", "observation.stage_id"]
    if args.subgoal_conditioned:
        lowdim_keys.append("observation.subgoal")
```

- [ ] **Step 4: 构造 QAgent 传 subgoal**（454-458 行）

```python
    agent = QAgent(obs_shape=(img_c, img_h, img_w), prop_shape=(state_dim,),
                   action_dim=action_dim, rl_cameras=image_keys,
                   cfg=cfg.agent, residual_actor=True,
                   stage_conditioned=args.stage_conditioned, num_stages=num_stages,
                   stage_budget=stage_budget,
                   subgoal_conditioned=args.subgoal_conditioned,
                   subgoal_dim=(subgoal.rep_dim if subgoal is not None else 0))
```

- [ ] **Step 5: build_offline_buffer 传 subgoal**（508-513 行）

```python
            build_offline_buffer(
                offline_rb, args.offline_dataset_path,
                action_scaler=action_scaler, state_standardizer=state_standardizer,
                image_keys=image_keys, bonus=args.stage_reward_bonus,
                mode=shaping_mode, gamma=args.gamma, num_demos=args.offline_num_demos,
                stage_cache=args.offline_stage_cache, potential=potential,
                subgoal=subgoal, way_steps=args.subgoal_way_steps)
```

并在 buffer 签名（`_offline_buffer_signature`，192-200 行附近）加键，使开/关 subgoal 缓存不串：

```python
    if args.subgoal_conditioned:
        sig["subgoal"] = True
        sig["gc_value_ckpt"] = os.path.abspath(args.gc_value_ckpt)
        sig["high_actor_ckpt"] = os.path.abspath(args.high_actor_ckpt)
        sig["subgoal_way_steps"] = int(args.subgoal_way_steps)
```

> （`_offline_buffer_signature` 需能拿到 `args`，它已是首参；直接在函数体末尾 `return sig` 前加上述块。）

- [ ] **Step 6: act 循环注入 z 到 obs/next_obs**（546-553 行）

在 `while` 循环体顶部、`agent.act` 之前，给 obs 注入当前 z；`env.step` 之后给 next_obs 注入 z'。需要当前/下一步的 raw rel_piece——`env.reset()`/`env.step()` 的 `info["rel_piece"]` 透出（`ChunkResidualEnvWrapper._extract_rel`）。改为：

```python
    obs, reset_info = env.reset()
    cur_rel = reset_info.get("rel_piece") if args.subgoal_conditioned else None
    ...
    while env_steps <= total:
        if args.subgoal_conditioned:
            obs["observation.subgoal"] = subgoal.subgoal_online(
                obs["observation.state"], cur_rel).to(obs["observation.state"].device)
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, eval_mode=False, stddev=args.stddev, cpu=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        if args.subgoal_conditioned:
            next_rel = info.get("rel_piece")
            next_obs["observation.subgoal"] = subgoal.subgoal_online(
                next_obs["observation.state"], next_rel).to(next_obs["observation.state"].device)
        add_chunk_transition(obs=obs, next_obs=next_obs, combined_action=info["scaled_action"],
                             reward=reward, done=done, info=info, image_keys=image_keys,
                             lowdim_keys=lowdim_keys, online_rb=online_rb)
        ...
        obs = next_obs
        if args.subgoal_conditioned:
            cur_rel = next_rel
```

> `add_chunk_transition` 已按 `lowdim_keys` 取 obs 字段（55-76 行），`observation.subgoal` 进了 `lowdim_keys`（Step 3）即自动存入 transition，无需改该函数。eval 路（`eval_env`）若要也用 z，同法在 eval rollout 注入；首版 eval 可先用 `subgoal_online`（与训练一致）。

- [ ] **Step 7: 冒烟（`--smoke`，关键:① 端到端首跑）**

先训出 `gc_value.pt`（Phase 1）与 `high_actor.pt`（Phase 2）。再：

```bash
cd /data2/RL/residual-offpolicy-rl && CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual --no-capture-output python -u \
  -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id dexmg-threepiece-bc/7zklm69g \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --offline_dataset_path deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/three_piece_stages.npz \
  --subgoal_conditioned --gc_value_ckpt outputs_chunk/three_piece_gc_value.pt \
  --high_actor_ckpt outputs_chunk/three_piece_high_actor.pt \
  --smoke
```
Expected: 打印 `[hiql-subgoal] on; rep_dim=10 ...`；offline buffer 用 subgoal 重建（含 observation.subgoal）；跑起若干 step 不报错。

- [ ] **Step 8: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat(hiql-low): train_chunk_residual 接子目标条件(flag/构造/act循环/buffer)"
```

---

## Phase 3 验证 Gate / A/B（spec §6）

**先做"喂真航点 z"消融（最省钱的命门证伪）：** offline 半边已是真航点 z；先确认子目标条件这条链对残差**有反应**——

- [ ] **消融 A/B**：A=`--stage_conditioned`（现状证伪基线）；B=`--subgoal_conditioned`（offline 真航点 z + online 高层 z）。两臂其余同 nas10 best（`--offline_fraction 0.5 --stage_balanced` 等）。看 B 是否优于 A。
- [ ] 若 B 未优于 A（且非 scale/质量问题）→ **命门假设证伪**（action_scale 0.05 残差吃不动 z），止于此，不再深挖；记 memory。
- [ ] 若 B 优于 A → 正式 A/B：A=online 也喂"近似真航点"（用 demo 检索的航点，仅作上界参照）/ B=学出来的高层 `π^h`。指标：`eval success_rate` 后 5 点滑动均值 + per-stage 到达率（`eval_stage_reach.py`），重点 stage3 回退、reach3→reach4。判读看后 5 点滑动均值（eval 方差大）。

---

## 自查记录（writing-plans self-review）

- **Spec 覆盖**：覆盖 spec §4.4（子目标条件化、append_subgoal、`--subgoal_conditioned` 与 stage 互斥由 `--actor raw` + 各自 flag 保证）、§4.5（数据流：offline 真航点 z / online 高层 z / 执行注入；18 维 policy state vs 30 维 value state；z 为唯一 object-aware 信号）、§4.6 全部文件、§5 wiring 单测 + smoke、§4.7 Phase 3 gate（先真航点消融）、§6 A/B。`residual_flow_actor` 的子目标化按 spec 标注为可选后续，本 plan 限 `--actor raw`。
- **占位符**：无 TBD/TODO；每步给完整代码与精确行号/命令。Task 2 的 `ActorConfig()` 构造留了一处"按本仓库必填项补默认"的提示——这是对未读全的 `config/rlpd.py` 的诚实标注，执行者据实补；不影响实现主体。
- **类型一致**：`observation.subgoal` 维度 = `rep_dim`(=10) 在 Actor/QAgent/HiqlSubgoal/buffer 一致；`HiqlSubgoal.subgoal_online(state_std, rel_raw)`/`subgoal_waypoint(s30_base, s30_target)`/`build_state30` 签名在 Task 4 与 Task 5/6 调用一致；`append_subgoal(prop, subgoal)` 在 Task 1/3 一致；`build_offline_buffer(..., subgoal=, way_steps=)` 在 Task 5/6 一致；`from_ckpts(gc, high, *, goal30, device)` 在 Task 4/6 一致；复用 Phase 1 `load_gc_value`/`vf.phi(base, target)`、Phase 2 `load_high_actor`/`HighActor.forward(s,g)` 签名一致。
```
