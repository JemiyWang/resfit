# A2 子目标条件 V(s,z) Potential 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给残差 TD3 低层新增一种 goal-conditioned PBS 势函数 Φ(s)=V_gc(s,z)·scale 的 reward shaping(z=high_actor 子目标 rep),在线+离线一致,只改 reward、不动 TD3 actor。

**Architecture:** 复用已有 `gc_value`(`GoalConditionedVF`)+ `HiqlSubgoal`(出 z)。新增 `GoalConditionedVF.value_from_rep(s,z)` 直接吃 rep;新增 `GcSubgoalPotential` 把 V 映射成 Φ;在线 shaping 在主循环 `add_chunk_transition` 前用同一个 z_start 算;离线 shaping 扩展 `transition_rewards` 逐帧用同一 z_t 算双 V。新 `--potential_source hiql_subgoal`,绑定 `--subgoal_conditioned` + `renorm_subgoal=True`,默认 `stage` 逐位等价。

**Tech Stack:** Python, PyTorch, TensorDict, pytest。设计见 `docs/superpowers/specs/2026-06-29-a2-subgoal-potential-design.md`。

## Global Constraints

- 默认 `--potential_source stage`(potential=None)行为**逐位等价**,旧实验不受影响。
- A2 仅在 `--reward_shaping potential` + `--subgoal_conditioned` + `renorm_subgoal=True` 三者同时成立时启用;否则报错。
- 喂 value 的 z 必须在半径 `sqrt(rep_dim)` 球面(由 `renorm_subgoal=True` 保证),与 `goal_encoder` 的 rep 同空间。
- 单步 PBS:同一个 transition 的 `phi_start`/`phi_next` 必须用**同一个 z**;`done` 时 `phi_next=0`。
- Φ 取双 critic 均值 `(v1+v2)/2`(与 HIQL 低层 advantage 口径一致)。
- scale 复用 `gc_value.pt` 内 `v_stats`:`auto_scale=(num_stages-1)/max(vmax-vmin,1e-6)`,`scale=auto_scale·phi_scale`。
- 所有路径相对仓库根 `/mnt/mnt/data/resfit`;模块根 `resfit/rl_finetuning/chunk_residual/`(下文简称 `CR/`)。
- pytest 跑法:`cd /mnt/mnt/data/resfit && python -m pytest <path> -v`。

---

### Task 1: `GoalConditionedVF.value_from_rep(s, z)`

直接用 rep z(跳过 goal_encoder)算 value;与正常 `forward(s, goal_encoder(g,s))` 逐位等价。

**Files:**
- Modify: `CR/hiql_gc_value.py`(`GoalConditionedVF` 加方法,在 `forward` 之后,约 `:94`)
- Test: `CR/tests/test_hiql_gc_value.py`(追加)

**Interfaces:**
- Consumes: 现有 `GoalConditionedVF`:`self.v1`/`self.v2`(`_mlp(state_dim+rep_dim,...)`),`forward(s,g)`,`phi(s,g)=goal_encoder(g,s)`(归一化到半径 `sqrt(rep_dim)`)。
- Produces: `GoalConditionedVF.value_from_rep(s, z) -> (v1, v2)`,`s:[B,state_dim]`、`z:[B,rep_dim]`,返回两个 `[B]` 张量。

- [ ] **Step 1: 写失败测试**

在 `CR/tests/test_hiql_gc_value.py` 末尾追加:

```python
def test_value_from_rep_matches_forward():
    """value_from_rep(s, phi(s,g)) 必须逐位等于 forward(s,g)。"""
    import torch
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    torch.manual_seed(0)
    m = GoalConditionedVF(state_dim=6, rep_dim=4, hidden=16).eval()
    s = torch.randn(3, 6)
    g = torch.randn(3, 6)
    with torch.no_grad():
        z = m.phi(s, g)                      # goal_encoder(g, s) -> rep(半径 sqrt(rep_dim))
        v1a, v2a = m.value_from_rep(s, z)
        v1b, v2b = m(s, g)
    assert torch.allclose(v1a, v1b, atol=1e-6)
    assert torch.allclose(v2a, v2b, atol=1e-6)
    assert v1a.shape == (3,)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_value_from_rep_matches_forward -v`
Expected: FAIL（`AttributeError: 'GoalConditionedVF' object has no attribute 'value_from_rep'`)

- [ ] **Step 3: 实现**

在 `CR/hiql_gc_value.py` 的 `GoalConditionedVF.forward`(`:92-94`)之后加:

```python
    def value_from_rep(self, s, z):
        """跳过 goal_encoder,直接用 rep z(=phi 输出空间)算 value。

        s:[B,state_dim] 已标准化 state;z:[B,rep_dim] 归一化 rep(须在半径 sqrt(rep_dim) 球面,
        与 phi(s,g) 同空间)。返回 (v1, v2)。供 A2 子目标 potential 用(z=high_actor 子目标 rep)。
        """
        x = torch.cat([s, z], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_value_from_rep_matches_forward -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat: GoalConditionedVF.value_from_rep for A2 subgoal potential"
```

---

### Task 2: `GcSubgoalPotential` 类

把冻结 gc_value + 子目标 z 映射成 PBS 势函数 Φ(s,z)=mean(V)·scale;scale 由 ckpt 的 v_stats 算。

**Files:**
- Modify: `CR/hiql_potential.py`(在 `HiqlPotential` 之后追加)
- Test: `CR/tests/test_hiql_potential.py`(追加)

**Interfaces:**
- Consumes: Task 1 的 `GoalConditionedVF.value_from_rep`;`hiql_gc_value.load_gc_value(path)->(model, info)`,`info["v_stats"]={"min","max","mean"}`。
- Produces:
  - `GcSubgoalPotential(gc_value, scale, device="cpu")`,属性 `.scale`、`.model`、`.is_subgoal=True`。
  - `GcSubgoalPotential.phi(s, z) -> Tensor[B]`(=`0.5*(v1+v2)*scale`)。
  - `GcSubgoalPotential.from_ckpt(gc_value_ckpt, *, num_stages, phi_scale=1.0, device="cpu")`。

- [ ] **Step 1: 写失败测试**

在 `CR/tests/test_hiql_potential.py` 末尾追加:

```python
def test_gc_subgoal_potential_phi_is_mean_v_times_scale():
    import torch
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    from resfit.rl_finetuning.chunk_residual.hiql_potential import GcSubgoalPotential
    torch.manual_seed(0)
    m = GoalConditionedVF(state_dim=6, rep_dim=4, hidden=16).eval()
    pot = GcSubgoalPotential(m, scale=2.0)
    assert pot.is_subgoal is True
    s = torch.randn(1, 6)
    z = torch.randn(1, 4)
    with torch.no_grad():
        v1, v2 = m.value_from_rep(s, z)
        expect = 0.5 * (v1 + v2) * 2.0
    assert torch.allclose(pot.phi(s, z), expect, atol=1e-6)


def test_gc_subgoal_potential_auto_scale_from_vstats(tmp_path):
    """from_ckpt 用 v_stats 算 auto_scale=(num_stages-1)/(vmax-vmin)。"""
    import torch
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF, save_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_potential import GcSubgoalPotential
    m = GoalConditionedVF(state_dim=6, rep_dim=4, hidden=16).eval()
    ckpt = tmp_path / "gc.pt"
    save_gc_value(str(ckpt), m, v_stats={"min": -4.0, "max": 0.0, "mean": -2.0},
                  mean=[0.0] * 6, std=[1.0] * 6, dataset_id="dummy",
                  state_mode="eef_piece", rep_dim=4, hidden=16)
    pot = GcSubgoalPotential.from_ckpt(str(ckpt), num_stages=5, phi_scale=1.0)
    # auto_scale = (5-1)/(0-(-4)) = 4/4 = 1.0
    assert abs(pot.scale - 1.0) < 1e-6
```

> 注:`save_gc_value` 的精确签名见 `CR/hiql_gc_value.py:300`。若某关键字参数名不符,以该函数实际签名为准微调本测试的 `save_gc_value(...)` 调用(只调用方,不改产品代码)。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k gc_subgoal -v`
Expected: FAIL（`ImportError`/`cannot import name 'GcSubgoalPotential'`)

- [ ] **Step 3: 实现**

在 `CR/hiql_potential.py` 文件末尾(`HiqlPotential` 类之后)追加:

```python
class GcSubgoalPotential:
    """A2:把冻结 goal-conditioned value 当 PBS 势函数 Φ(s,z)=mean(V(s,z))·scale。

    与 HiqlPotential(单状态 V(s))正交:这里 phi 第二参是子目标 rep z(不是 rel_piece),
    用 value_from_rep 直接吃 z(z 须在半径 sqrt(rep_dim) 球面,renorm_subgoal=True 保证)。
    """

    def __init__(self, gc_value, scale, device="cpu"):
        self.model = gc_value.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scale = float(scale)
        self.device = device
        self.is_subgoal = True

    @classmethod
    def from_ckpt(cls, gc_value_ckpt, *, num_stages, phi_scale=1.0, device="cpu"):
        from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
        model, info = load_gc_value(gc_value_ckpt, map_location=device)
        vmin, vmax = info["v_stats"]["min"], info["v_stats"]["max"]
        phi_range = max(int(num_stages) - 1, 1)
        auto_scale = phi_range / max(vmax - vmin, 1e-6)
        return cls(model, scale=auto_scale * phi_scale, device=device)

    @torch.no_grad()
    def phi(self, s, z):
        """s:[B,state_dim] 已标准化 state;z:[B,rep_dim] 子目标 rep -> [B] 势函数值。"""
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        v1, v2 = self.model.value_from_rep(s, z)
        return 0.5 * (v1 + v2) * self.scale
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k gc_subgoal -v`
Expected: PASS（2 passed)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_potential.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat: GcSubgoalPotential (goal-conditioned PBS potential) for A2"
```

---

### Task 3: 在线单步 shaping helper `gc_subgoal_shaping`

封装"同一个 z_start 算双 V → potential_shaping"的纯逻辑,供主循环调用且可单测。

**Files:**
- Modify: `CR/hiql_potential.py`(加模块级函数)
- Test: `CR/tests/test_hiql_potential.py`(追加)

**Interfaces:**
- Consumes: 现有 `potential_shaping(phi_start, phi_next, *, bonus, gamma, done)`(`CR/hiql_potential.py:13`);任意有 `.phi(s, z)` 的 potential。
- Produces: `gc_subgoal_shaping(potential, s_start, s_end, z_start, *, bonus, gamma, done) -> float`。

- [ ] **Step 1: 写失败测试**

在 `CR/tests/test_hiql_potential.py` 末尾追加:

```python
def test_gc_subgoal_shaping_uses_same_z_and_telescopes():
    import torch
    from resfit.rl_finetuning.chunk_residual.hiql_potential import gc_subgoal_shaping

    class FakePot:               # phi = s[0] + 10*z[0],便于手算
        def phi(self, s, z):
            return torch.tensor([float(s.reshape(-1)[0]) + 10.0 * float(z.reshape(-1)[0])])

    s_start = torch.tensor([1.0]); s_end = torch.tensor([2.0]); z = torch.tensor([0.5])
    # phi_a = 1 + 5 = 6 ; phi_b = 2 + 5 = 7 ; F = 1*(0.9*7 - 6) = 0.3
    F = gc_subgoal_shaping(FakePot(), s_start, s_end, z, bonus=1.0, gamma=0.9, done=False)
    assert abs(F - 0.3) < 1e-6


def test_gc_subgoal_shaping_done_zeros_next():
    import torch
    from resfit.rl_finetuning.chunk_residual.hiql_potential import gc_subgoal_shaping

    class FakePot:
        def phi(self, s, z):
            return torch.tensor([float(s.reshape(-1)[0])])

    # done -> phi_next=0 -> F = bonus*(0 - phi_a) = 2*(0 - 3) = -6
    F = gc_subgoal_shaping(FakePot(), torch.tensor([3.0]), torch.tensor([9.0]),
                           torch.tensor([0.0]), bonus=2.0, gamma=0.99, done=True)
    assert abs(F - (-6.0)) < 1e-6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k gc_subgoal_shaping -v`
Expected: FAIL（`cannot import name 'gc_subgoal_shaping'`)

- [ ] **Step 3: 实现**

在 `CR/hiql_potential.py` 的 `potential_shaping`(`:13-19`)之后加:

```python
def gc_subgoal_shaping(potential, s_start, s_end, z_start, *, bonus, gamma, done):
    """A2 单步 shaping:F = bonus·(γ·Φ(s_end,z_start) − Φ(s_start,z_start))。

    phi_start 与 phi_next 用**同一个 z_start**(保证单步 PBS);done 时 phi_next=0
    (由 potential_shaping 处理)。返回 float。
    """
    phi_a = potential.phi(s_start, z_start)
    phi_b = potential.phi(s_end, z_start)
    return potential_shaping(phi_a, phi_b, bonus=bonus, gamma=gamma, done=done)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k gc_subgoal_shaping -v`
Expected: PASS（2 passed)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_potential.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat: gc_subgoal_shaping single-step PBS helper for A2 online"
```

---

### Task 4: 离线 `transition_rewards` 逐帧子目标分支

`transition_rewards`/`transition_fields` 支持"逐帧、同一 z_t 的双 V" gc 分支;`offline_stage_replay` 把已算的 `subgoal_z`(T,rep)+ `s_sub`(T,vf.state_dim)传进去。

**Files:**
- Modify: `CR/offline_hdf5_buffer.py`(`transition_rewards`/`transition_fields` 加 `subgoal_z_seq`/`gc_state_seq` 参数 + gc 分支)
- Modify: `CR/offline_stage_replay.py`(把 `subgoal_z`/`s_sub` 计算挪到 `transition_fields` 调用前并透传)
- Test: `CR/tests/test_offline_hdf5_buffer.py`(追加)

**Interfaces:**
- Consumes: Task 2 的 `GcSubgoalPotential`(有 `.is_subgoal`、`.phi(s,z)`);现有 `potential_shaping`。
- Produces: `transition_rewards(..., gc_state_seq=None, subgoal_z_seq=None)` 与 `transition_fields(..., gc_state_seq=None, subgoal_z_seq=None)`:当 `getattr(potential,"is_subgoal",False)` 为真时,第 t 个 transition 用 `phi(gc_state_seq[t], subgoal_z_seq[t])` 与 `phi(gc_state_seq[t+1], subgoal_z_seq[t])`(同 z_t)。

- [ ] **Step 1: 写失败测试**

在 `CR/tests/test_offline_hdf5_buffer.py` 末尾追加:

```python
def test_transition_rewards_gc_subgoal_same_z_per_transition():
    import numpy as np, torch
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import transition_rewards

    class FakeGcPot:                       # phi(s,z) = s[0] + 100*z[0]
        is_subgoal = True
        def phi(self, s, z):
            s = torch.as_tensor(s).reshape(-1); z = torch.as_tensor(z).reshape(-1)
            return torch.tensor([float(s[0]) + 100.0 * float(z[0])])

    # T=3 帧,非末步 done=False;末 transition done=True
    instant = np.array([0, 0, 1])
    gc_state = torch.tensor([[1.0], [2.0], [3.0]])      # s_t[0] = 1,2,3
    subgoal_z = torch.tensor([[0.1], [0.2], [0.3]])     # z_t[0] = 0.1,0.2,0.3
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.9,
                           success=True, potential=FakeGcPot(),
                           gc_state_seq=gc_state, subgoal_z_seq=subgoal_z)
    # t=0(done=F): phi_a=1+10=11, phi_b=2+10=12, F=0.9*12-11=-0.2, base=0 -> -0.2
    # t=1(done=T): phi_a=2+20=22, phi_b=0, F=0-22=-22, base=1 -> -21.0
    assert abs(float(r[0]) - (-0.2)) < 1e-5
    assert abs(float(r[1]) - (-21.0)) < 1e-5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_offline_hdf5_buffer.py::test_transition_rewards_gc_subgoal_same_z_per_transition -v`
Expected: FAIL（`transition_rewards() got an unexpected keyword argument 'gc_state_seq'`)

- [ ] **Step 3a: 给 `transition_rewards` 加 gc 分支**

`CR/offline_hdf5_buffer.py:189`,把签名与函数体改为(新增 `gc_state_seq`/`subgoal_z_seq` 参数 + gc 分支):

```python
def transition_rewards(instant_stages, *, bonus: float, mode: str,
                       gamma: float, success: bool = True,
                       potential=None, state_seq=None, rel_piece_seq=None,
                       act_feat_seq=None, gc_state_seq=None, subgoal_z_seq=None) -> np.ndarray:
    """一条 demo 的 T 帧瞬时 stage → T-1 个 transition 的总 reward。

    与线上 cl=1 一致:每步 reward = base 稀疏 + shaping。
    potential=None:Φ=闩锁 stage(现状)。potential.is_subgoal(A2):Φ=V(s_t,z_t),逐帧同一 z_t
    算双 V(gc_state_seq/subgoal_z_seq 长度=T)。否则(单状态/act_feat):Φ=potential.phi(state_seq)。
    """
    latch = latch_from_instant(instant_stages)
    T = len(latch)
    rewards = np.empty(T - 1, dtype=np.float32)
    is_subgoal = getattr(potential, "is_subgoal", False)
    phi = None
    if potential is not None and not is_subgoal:
        if getattr(potential, "state_mode", "eef") == "act_feat":
            assert act_feat_seq is not None and len(act_feat_seq) == T, \
                "act_feat potential 模式需 act_feat_seq 且长度=T"
            phi = potential.phi(act_feat_seq, None)
        else:
            assert state_seq is not None and len(state_seq) == T, \
                "potential 模式需 state_seq 且长度=T"
            assert rel_piece_seq is None or len(rel_piece_seq) == T, \
                "rel_piece_seq 长度须 = T"
            phi = potential.phi(state_seq, rel_piece_seq)            # [T]
    elif is_subgoal:
        assert gc_state_seq is not None and len(gc_state_seq) == T, \
            "gc subgoal potential 需 gc_state_seq 且长度=T"
        assert subgoal_z_seq is not None and len(subgoal_z_seq) == T, \
            "gc subgoal potential 需 subgoal_z_seq 且长度=T"
    for t in range(T - 1):
        done = success and (t == T - 2)
        base = float(done)
        if potential is None:
            shaped = shaping_reward(int(latch[t]), int(latch[t + 1]),
                                    mode=mode, bonus=bonus, gamma=gamma, done=done)
        elif is_subgoal:
            z_t = subgoal_z_seq[t]                                   # 同一个 z_t
            phi_a = potential.phi(gc_state_seq[t], z_t)
            phi_b = potential.phi(gc_state_seq[t + 1], z_t)
            shaped = potential_shaping(phi_a, phi_b,
                                       bonus=bonus, gamma=gamma, done=done)
        else:
            shaped = potential_shaping(phi[t], phi[t + 1],
                                       bonus=bonus, gamma=gamma, done=done)
        rewards[t] = base + shaped
    return rewards
```

- [ ] **Step 3b: 让 `transition_fields` 透传新参数**

`CR/offline_hdf5_buffer.py:148`,签名加 `gc_state_seq=None, subgoal_z_seq=None`,并在调用 `transition_rewards` 时传下去:

```python
def transition_fields(instant_stages, *, bonus: float, mode: str,
                      gamma: float, success: bool = True,
                      potential=None, state_seq=None, rel_piece_seq=None,
                      act_feat_seq=None, gc_state_seq=None, subgoal_z_seq=None) -> dict:
```

并把内部 `"reward": transition_rewards(...)` 调用补上:

```python
        "reward": transition_rewards(instant, bonus=bonus, mode=mode,
                                     gamma=gamma, success=success,
                                     potential=potential, state_seq=state_seq,
                                     rel_piece_seq=rel_piece_seq,
                                     act_feat_seq=act_feat_seq,
                                     gc_state_seq=gc_state_seq,
                                     subgoal_z_seq=subgoal_z_seq),
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_offline_hdf5_buffer.py::test_transition_rewards_gc_subgoal_same_z_per_transition -v`
Expected: PASS

- [ ] **Step 5: 在 `offline_stage_replay.py` 把 subgoal_z/s_sub 挪前并透传**

`CR/offline_stage_replay.py`:当前 `subgoal_z`/`s_sub` 在 `transition_fields`(`:308`)**之后**算(`:325-338`)。把这段计算**整体上移到 `transition_fields` 调用之前**,得到 `s_sub`(T,vf.state_dim)与 `subgoal_z`(T,rep);然后在 `transition_fields(...)` 调用里,当 `getattr(potential, "is_subgoal", False)` 为真时传:

```python
                _gc_state_seq = s_sub if (subgoal is not None and getattr(potential, "is_subgoal", False)) else None
                _subgoal_z_seq = subgoal_z if (subgoal is not None and getattr(potential, "is_subgoal", False)) else None
                fld = transition_fields(instant, bonus=bonus, mode=mode,
                                        gamma=gamma, success=True,
                                        potential=potential, state_seq=state_n,
                                        rel_piece_seq=rel_seq,
                                        act_feat_seq=act_feat_seq,
                                        gc_state_seq=_gc_state_seq,
                                        subgoal_z_seq=_subgoal_z_seq)
```

> 注:`subgoal_z`/`s_sub` 原计算块(`s_sub=cat([state_n,rel_n])` 或 `_af_by_ep[ep]`;`way=min(arange(T)+way_steps,T-1)`;`subgoal_z=subgoal.subgoal_waypoint(s_sub, s_sub[way])`)逻辑**不变**,只是位置上移到 `transition_fields` 之前;原 `:325-338` 处删除重复块(保留下游 `curr/nxt["observation.subgoal"]=subgoal_z[t/t+1]` 对 `subgoal_z` 的引用)。`lerobot` 数据源分支(`build_offline_buffer_from_lerobot`,`:377+`)同理上移并透传(若该路也支持 subgoal)。

- [ ] **Step 6: 跑离线 smoke 回归确认未破坏既有路径**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_offline_hdf5_buffer.py resfit/rl_finetuning/chunk_residual/tests/test_offline_stage_replay_smoke.py -v`
Expected: PASS（既有用例全过 + 新用例过)

- [ ] **Step 7: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py resfit/rl_finetuning/chunk_residual/offline_stage_replay.py resfit/rl_finetuning/chunk_residual/tests/test_offline_hdf5_buffer.py
git commit -m "feat: offline transition_rewards gc-subgoal per-transition same-z shaping (A2)"
```

---

### Task 5: CLI 开关 / 构造 / 断言 / 缓存签名

新增 `--potential_source hiql_subgoal` + `--gc_potential_scale`;构造 `GcSubgoalPotential`;三项断言;在线主循环插 shaping;离线签名纳入新键。

**Files:**
- Modify: `CR/train_chunk_residual.py`(argparse `:620`;校验函数;potential 构造 `:757-769` 附近;主循环 `:1129-1137`;`_offline_buffer_signature` `:245-269`)
- Test: `CR/tests/test_hiql_potential.py`(追加 wiring/断言级测试,纯函数化部分)

**Interfaces:**
- Consumes: Task 2 `GcSubgoalPotential.from_ckpt`;Task 3 `gc_subgoal_shaping`;现有 `subgoal`(`HiqlSubgoal`)、`compute_online_subgoal`、`add_chunk_transition`。
- Produces: `--potential_source` 增加 `"hiql_subgoal"`;`args.gc_potential_scale`(默认 1.0);在线/离线在 `hiql_subgoal` 下产出 A2 shaping。

- [ ] **Step 1: argparse 加选项**

`CR/train_chunk_residual.py:620`,`--potential_source` 的 `choices` 加 `"hiql_subgoal"`:

```python
    p.add_argument("--potential_source", choices=["stage", "hiql", "hiql_subgoal"], default="stage",
                   help="potential 模式的 Φ 源:stage|hiql(单状态V)|hiql_subgoal(goal-cond V(s,z),A2)")
```

紧邻 `--hiql_value_ckpt`(`:622`)后加:

```python
    p.add_argument("--gc_potential_scale", type=float, default=1.0,
                   help="A2(hiql_subgoal)的 phi_scale,乘到 auto_scale 上(默认 1.0)")
```

- [ ] **Step 2: 写断言的失败测试**

`HiqlSubgoal` 路的参数校验集中在 main 里。为可测,新增一个纯校验函数 `validate_hiql_subgoal_args(args)` 并单测。在 `CR/tests/test_hiql_potential.py` 追加:

```python
def test_validate_hiql_subgoal_requires_subgoal_and_renorm():
    import types, pytest
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import validate_hiql_subgoal_args
    base = dict(potential_source="hiql_subgoal", reward_shaping="potential",
                subgoal_conditioned=True, renorm_subgoal=True, gc_value_ckpt="x")
    validate_hiql_subgoal_args(types.SimpleNamespace(**base))      # ok,不报错
    for bad in (dict(subgoal_conditioned=False), dict(renorm_subgoal=False),
                dict(reward_shaping="none"), dict(gc_value_ckpt=None)):
        d = dict(base); d.update(bad)
        with pytest.raises(AssertionError):
            validate_hiql_subgoal_args(types.SimpleNamespace(**d))
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k validate_hiql_subgoal -v`
Expected: FAIL（`cannot import name 'validate_hiql_subgoal_args'`)

- [ ] **Step 4: 实现校验函数**

在 `CR/train_chunk_residual.py` 模块级(靠近其他 helper,如 `:204` `compute_online_subgoal` 附近)加:

```python
def validate_hiql_subgoal_args(args):
    """A2(--potential_source hiql_subgoal)的前置校验。其余 potential_source 不受影响。"""
    if getattr(args, "potential_source", "stage") != "hiql_subgoal":
        return
    assert args.reward_shaping == "potential", \
        "--potential_source hiql_subgoal 需 --reward_shaping potential"
    assert getattr(args, "subgoal_conditioned", False), \
        "--potential_source hiql_subgoal 需 --subgoal_conditioned(提供 gc_value/high_actor/z)"
    assert getattr(args, "renorm_subgoal", False), \
        "--potential_source hiql_subgoal 需 renorm_subgoal=True(z 须在 sqrt(rep_dim) 球面)"
    assert getattr(args, "gc_value_ckpt", None), \
        "--potential_source hiql_subgoal 需 --gc_value_ckpt"
```

并在 main 早期校验段(与现有 `--potential_source hiql` 的 assert 同处,`:761` 附近)调用一次 `validate_hiql_subgoal_args(args)`。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k validate_hiql_subgoal -v`
Expected: PASS

- [ ] **Step 6: 构造 gc_potential**

`CR/train_chunk_residual.py:761` 现有 `if args.potential_source == "hiql":` 构造 `HiqlPotential` 的块之后,补 A2 分支(在 `subgoal` 已构造之后——`subgoal` 在 `:948` 构造,故把本块放在 `subgoal` 构造之后、主循环之前):

```python
    gc_potential = None
    if args.potential_source == "hiql_subgoal":
        from resfit.rl_finetuning.chunk_residual.hiql_potential import GcSubgoalPotential
        gc_potential = GcSubgoalPotential.from_ckpt(
            args.gc_value_ckpt, num_stages=num_stages,
            phi_scale=args.gc_potential_scale, device=args.device)
        potential = None        # A2 不走 wrapper 的 potential 路径(wrapper 逐位等价无 shaping)
```

> `num_stages` 与 `HiqlPotential.from_ckpt` 同处可得(`NUM_STAGES` 已 import,`:46`)。离线构造时 `potential` 入参改用 `gc_potential`(见 Step 8)。

- [ ] **Step 7: 主循环插在线 shaping**

`CR/train_chunk_residual.py`,在 `done = terminated | truncated`(`:1134`)之后、`add_chunk_transition`(`:1135`)之前(此时 `next_obs["observation.subgoal"]`、`done`、`next_rel` 均已就绪),加:

```python
        if gc_potential is not None:
            _pf = base_policy.last_prefix_feat() if subgoal.state_mode == "pi0_feat" else None
            _s_start = subgoal.encode_state(obs, rel_raw=cur_rel, prefix_feat=_pf)
            _s_end = subgoal.encode_state(next_obs, rel_raw=next_rel, prefix_feat=_pf)
            _z_start = obs["observation.subgoal"]
            from resfit.rl_finetuning.chunk_residual.hiql_potential import gc_subgoal_shaping
            _shape = gc_subgoal_shaping(gc_potential, _s_start, _s_end, _z_start,
                                        bonus=args.stage_reward_bonus, gamma=args.gamma,
                                        done=bool(done.any()))
            reward = reward + _shape
```

> `obs["observation.subgoal"]` 是 step 前算的 chunk 起点 z(`:1122`),与 `_s_start` 同源;`done` 时 `phi_next=0`,`_s_end`(autoreset 后 obs)不被使用。

- [ ] **Step 8: 离线传 gc_potential**

找到主训里构造离线 buffer 调 `build_offline_buffer*`/`offline_stage_replay` 的位置(`potential=` 入参,`:1071`/`:1079` 附近),当 `args.potential_source == "hiql_subgoal"` 时把 `potential` 入参传 `gc_potential`(其余 source 仍传原 `potential`):

```python
                    potential=(gc_potential if args.potential_source == "hiql_subgoal" else potential),
```

- [ ] **Step 9: 缓存签名纳入新键**

`CR/train_chunk_residual.py:_offline_buffer_signature`(`:245`),在 `if args.potential_source == "hiql":` 分支旁加:

```python
    if args.potential_source == "hiql_subgoal":
        sig["potential_source"] = "hiql_subgoal"
        sig["gc_value_ckpt"] = os.path.abspath(args.gc_value_ckpt) if args.gc_value_ckpt else None
        sig["gc_potential_scale"] = round(float(args.gc_potential_scale), 8)
        sig["subgoal_way_steps"] = int(args.subgoal_way_steps)
```

- [ ] **Step 10: 跑相关单测 + 提交**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py resfit/rl_finetuning/chunk_residual/tests/test_env_family_wiring.py -v`
Expected: PASS

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat: wire --potential_source hiql_subgoal (A2 online+offline shaping)"
```

---

### Task 6: 全量回归 + 分阶段真 env smoke

**Files:** 无新增产品代码;只跑验证。

- [ ] **Step 1: 全量单测零回归**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全过(基线 400+ passed;新增用例计入,0 failed)。若有失败,定位是否 A2 改动引入,修复后重跑。

- [ ] **Step 2: 默认路径逐位等价抽检**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_golden_defaults.py resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -v`
Expected: PASS（确认默认 `stage`/wrapper 行为未变)。

- [ ] **Step 3: 真 env 短 smoke(eef_piece 或 act_feat)**

挑一个已有 subgoal_conditioned 跑过、ckpt/cache 齐全的任务(如 `three_piece_hiqlv512_sg15`)。在其 run 脚本基础上加 `--potential_source hiql_subgoal --reward_shaping potential --gc_potential_scale 1.0`,设 `--smoke`(主循环 `total=2*chunk_length`)跑一遍。

确认(看日志/wandb):在线 reward 含**非零且量级合理**的 shaping 项(与纯 base 稀疏 reward 有区别);离线 buffer 构建完成(`buffer_meta.json` 哨兵落地);训练不崩、loss 有限。

> 具体任务的 ckpt/cache/脚本路径由执行者按现有 run 脚本定位;smoke 命令不进 git,只验证。

- [ ] **Step 4: 提交 smoke 记录(可选)**

若 smoke 产生需留档的日志,按项目惯例放仓库根 `*.log` 并:

```bash
cd /mnt/mnt/data/resfit
git add <smoke>.log && git commit -m "test: A2 hiql_subgoal smoke log"
```

---

### Task 7: A/B 实验(用户授权后跑)

**Files:** 无代码改动;跑实验对比。

- [ ] **Step 1: 主对照 B0(关 shaping)**

同任务、同 ckpt/cache,`--subgoal_conditioned` + `--reward_shaping none`,跑满设定步数。

- [ ] **Step 2: 实验 A(A2)**

同上但 `--potential_source hiql_subgoal --reward_shaping potential --gc_potential_scale 1.0`(必要时扫 `gc_potential_scale`)。

- [ ] **Step 3: 对比**

eval success 曲线 + 训练稳定性,A vs B0。若想另比"V(s,z) shaping vs stage shaping",再加 B1 = `--potential_source stage`。

> A/B 为长跑实验,需用户授权 GPU/时长后再启动;本计划不自动触发。

---

## 落地顺序与依赖

Task 1 → 2 → 3 → 4 → 5 严格顺序(后者依赖前者的接口);Task 6 在 1-5 全绿后跑;Task 7 待用户授权。每个 Task 自带 TDD + 提交;每个 flag 默认等价,先落码+测试,再重训/跑生效。
