# HIQL 对齐三改 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 resfit 分层路对 HIQL 的三处偏离收敛——① value 加 LayerNorm、② 高层 AWR target clamp-to-goal、③ eval 子目标重归一化——各落一个 flag、默认逐位等价于现状,验证只到离线 gate。

**Architecture:** 三处独立 flag(`--use_layer_norm` / `--target_mode` / 构造参 `renorm_subgoal`),老行为默认。①改 value 网结构有级联(重训 gc_value→在其上重训 high_actor);②③不级联。每改先 TDD 落码,再跑 A/B 训练 + 重跑对应 `verify_*` gate,证据写回各自 plan。设计见 `docs/superpowers/specs/2026-06-10-hiql-alignment-deltas-design.md`。

**Tech Stack:** PyTorch、NumPy、pytest。conda env `residual`,所有命令从仓库根 `/mnt/mnt/data/resfit` 跑。

**前置(均已就绪):** 基线两件套 `outputs_chunk/three_piece_gc_value_geom.pt` + `three_piece_high_actor_geom.pt`(geom、1006 demos、238821 transitions,各过 gate);`outputs_chunk/three_piece_state30.npz`(回放缓存,重训 high_actor 传它免 MuJoCo 回放);`outputs_chunk/three_piece_stages.npz`;hdf5 `resfit/dataset/two_arm_three_piece_assembly.hdf5`;dataset `ankile/dexmg-two-arm-three-piece-assembly`。

---

## File Structure

| 文件 | 改动 |
|---|---|
| `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py` | ③:`HiqlSubgoal.__init__`/`from_ckpts` 加 `renorm_subgoal`;`subgoal_online` 末投球面。 |
| `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py` | ②:新纯函数 `sample_high_goal_target`;`train_high_actor` 加 `target_mode`/`high_p_randomgoal`;`save/load_high_actor` 透传(带默认)。 |
| `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py` | ② CLI:`--target_mode`/`--high_p_randomgoal`。 |
| `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py` | ①:`_mlp(use_layer_norm=)`;`RelativeGoalEncoder`/`GoalConditionedVF`/`train_gc_value` 透传;`save/load_gc_value` 存/重建 `use_layer_norm`。 |
| `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py` | ① CLI:`--use_layer_norm`。 |
| `resfit/rl_finetuning/chunk_residual/verify_high_actor.py` | ②:`--goal_mode {final,near}` + `evaluate_near`/`report_near`(near-goal 塌缩诊断)。 |
| `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py` | ③ 单测(追加)。 |
| `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py` | ② 单测(追加)。 |
| `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py` | ① 单测(追加)。 |

**不动**:`hiql_value.py`、`hiql_potential.py`、`q_agent.py`、`train_chunk_residual.py`、`sample_gc_goals`/`train_gc_value` 的 stage_entry/geom 逻辑。

---

## Task 1: ③ eval 子目标重归一化(`HiqlSubgoal`)

最便宜、不重训。`renorm_subgoal=True` 时把在线子目标 z 投到半径 sqrt(rep_dim)(对齐 HIQL `evaluation.py:114`),默认 False 保持现状。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_hiql_subgoal_wiring.py` 末尾)

```python
def test_subgoal_online_renorm_projects_to_sphere():
    """renorm_subgoal=True:在线子目标 z 投到半径 sqrt(rep_dim)(对齐 HIQL eval);
    默认 False 保持取 .mean 原值(范数一般 != sqrt(rep))。"""
    import numpy as np
    import torch
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
    from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal

    torch.manual_seed(0)
    vf = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=32)
    ha = HighActor(state_dim=30, rep_dim=10, hidden=32)
    # 让 mean 明显偏离球面:放大 mean_net 末层权重
    with torch.no_grad():
        ha.mean_net[-1].weight.mul_(5.0)
    goal30, rel_mean, rel_std = np.zeros(30, np.float32), np.zeros(12, np.float32), np.ones(12, np.float32)

    sg_off = HiqlSubgoal(vf, ha, goal30, rel_mean, rel_std)                       # 默认 False
    sg_on = HiqlSubgoal(vf, ha, goal30, rel_mean, rel_std, renorm_subgoal=True)
    state_std, rel_raw = np.ones((4, 18), np.float32), np.ones((4, 12), np.float32)

    z_on = sg_on.subgoal_online(state_std, rel_raw)
    z_off = sg_off.subgoal_online(state_std, rel_raw)
    sqrt_rep = float(np.sqrt(10))
    assert torch.allclose(z_on.norm(dim=-1), torch.full((4,), sqrt_rep), atol=1e-4)
    # 关掉时范数不被强制贴球面(本构造下明显偏离)
    assert (z_off.norm(dim=-1) - sqrt_rep).abs().max() > 1e-2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py::test_subgoal_online_renorm_projects_to_sphere -q`
Expected: FAIL(`TypeError: __init__() got an unexpected keyword argument 'renorm_subgoal'`)

- [ ] **Step 3: 改 `HiqlSubgoal`**

`__init__` 签名加 `renorm_subgoal=False` 并存属性(改 `hiql_subgoal.py:29` 那行及函数体):

```python
    def __init__(self, gc_value, high_actor, goal30, rel_mean, rel_std, device="cpu",
                 renorm_subgoal=False):
        self.vf = gc_value.to(device).eval()
        self.ha = high_actor.to(device).eval()
        for m in (self.vf, self.ha):
            for p in m.parameters():
                p.requires_grad_(False)
        self.rep_dim = gc_value.rep_dim
        self.device = device
        self.renorm_subgoal = renorm_subgoal
        self.goal30 = torch.as_tensor(np.asarray(goal30), dtype=torch.float32, device=device).reshape(-1)
        self.rel_mean = torch.as_tensor(np.asarray(rel_mean), dtype=torch.float32, device=device)
        self.rel_std = torch.as_tensor(np.asarray(rel_std), dtype=torch.float32, device=device)
```

`from_ckpts` 透传(改 `hiql_subgoal.py:42` 起的 classmethod):签名加 `renorm_subgoal=False`,末行返回改成
`return cls(gc, ha, goal30, info["rel_piece_mean"], info["rel_piece_std"], device=device, renorm_subgoal=renorm_subgoal)`。

`subgoal_online` 末投球面(改 `hiql_subgoal.py:62-66`):

```python
    @torch.no_grad()
    def subgoal_online(self, state_std, rel_raw):
        """在线:z = π^h(s30, goal30)(取分布均值);renorm_subgoal 时投到半径 sqrt(rep_dim)。"""
        s30 = self.build_state30(state_std, rel_raw)
        g30 = self.goal30.unsqueeze(0).expand(s30.shape[0], -1)
        z = self.ha(s30, g30).mean
        if self.renorm_subgoal:
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return z
```

- [ ] **Step 4: 跑测试确认通过 + 不回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py -q`
Expected: 全 PASS(含原有 `test_hiql_subgoal_shapes` 等)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_subgoal.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py
git commit -m "feat(hiql-align): ③ HiqlSubgoal renorm_subgoal 在线子目标投球面(对齐 HIQL eval)"
```

---

## Task 2: ② 纯函数 `sample_high_goal_target`(HIQL GCSDataset 高段)

逐字照 `gc_dataset.py:122-135`:traj goal 线性插值落 [si+1, final](永不命中 current);traj target=min(si+way, goal);random goal(prob `high_p_randomgoal`)的 target=min(si+way, final)。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_hiql_high_actor.py`)

```python
def test_sample_high_goal_target_traj_branch():
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import sample_high_goal_target
    last_idx_of = {0: 99}
    si = np.zeros(3000, dtype=np.int64)          # 全从 t=0 出发
    tj = np.zeros(3000, dtype=np.int64)
    rng = np.random.default_rng(0)
    goal, target = sample_high_goal_target(si, tj, last_idx_of, rng,
                                           way_steps=25, n_total=100, high_p_randomgoal=0.0)
    # traj goal ∈ [si+1, final],永不命中 current(=si=0)
    assert goal.min() >= 1 and goal.max() <= 99
    # target = min(si+way, goal) = min(25, goal)
    assert np.all(target == np.minimum(25, goal))
    assert np.all(target <= goal)

def test_sample_high_goal_target_random_branch():
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import sample_high_goal_target
    last_idx_of = {0: 99}
    si = np.full(4000, 10, dtype=np.int64)
    tj = np.zeros(4000, dtype=np.int64)
    rng = np.random.default_rng(1)
    goal, target = sample_high_goal_target(si, tj, last_idx_of, rng,
                                           way_steps=25, n_total=100, high_p_randomgoal=1.0)
    # 全 random:goal ∈ [0,100),target = min(10+25, 99) = 35(恒定)
    assert goal.min() >= 0 and goal.max() < 100
    assert np.all(target == 35)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_sample_high_goal_target_traj_branch resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_sample_high_goal_target_random_branch -q`
Expected: FAIL(`ImportError: sample_high_goal_target`)

- [ ] **Step 3: 写实现**(追加到 `hiql_high_actor.py`,在 `awr_weight` 之后、`train_high_actor` 之前)

```python
def sample_high_goal_target(s_idx, traj_id, last_idx_of, rng, *, way_steps,
                            n_total, high_p_randomgoal=0.0):
    """HIQL GCSDataset 高段(逐字照 gc_dataset.py:122-135),返回 (goal_idx, target_idx)。

    traj goal: 线性插值 round(min(si+1,final)·d + final·(1−d)) ∈ [si+1, final],永不命中 current;
    traj target = min(si+way, traj_goal)。random goal(prob high_p_randomgoal)的 target=min(si+way, final)。
    """
    si = np.asarray(s_idx, dtype=np.int64)
    B = len(si)
    final = np.array([last_idx_of[int(t)] for t in traj_id], dtype=np.int64)
    dist = rng.random(B)
    traj_goal = np.round(np.minimum(si + 1, final) * dist + final * (1 - dist)).astype(np.int64)
    traj_target = np.minimum(si + way_steps, traj_goal)
    rand_goal = rng.integers(0, n_total, size=B)
    rand_target = np.minimum(si + way_steps, final)
    pick = rng.random(B) < high_p_randomgoal
    goal = np.where(pick, rand_goal, traj_goal)
    target = np.where(pick, rand_target, traj_target)
    return goal.astype(np.int64), target.astype(np.int64)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py -k sample_high_goal_target -q`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git commit -m "feat(hiql-align): ② sample_high_goal_target(HIQL GCSDataset 高段 clamp-to-goal 采样)"
```

---

## Task 3: ② `train_high_actor` 加 `target_mode` + `save/load_high_actor` schema

`fixed_waypoint`(默认)RNG 抽取顺序与现状不变(wi 先、gi 后),逐位等价;`clamp_to_goal` 走 Task 2 采样。非法 mode 抛 ValueError(证明开关接通)。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_hiql_high_actor.py`)

```python
def test_train_high_actor_rejects_bad_target_mode():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    vf, _ = train_gc_value(data, steps=1, batch_size=8, rep_dim=4, hidden=16, seed=0)
    with pytest.raises(ValueError):
        train_high_actor(data, vf, steps=1, batch_size=8, hidden=16, target_mode="bogus")

def test_train_high_actor_clamp_collapses_to_near_goal():
    """clamp_to_goal:goal 近(dist<way)时子目标应收敛到 goal 而非固定 +way 航点。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor
    seq = np.arange(40).reshape(40, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    vf, _ = train_gc_value(data, steps=3000, batch_size=32, rep_dim=8, hidden=64,
                           lr=1e-3, ema=0.01, seed=0)
    ha = train_high_actor(data, vf, way_steps=10, beta=1.0, steps=4000, batch_size=32,
                          hidden=64, lr=1e-3, seed=0, target_mode="clamp_to_goal")
    states = data["states"]
    st = states[5:6]
    g_near = states[8:9]          # dist=3 < way=10 -> 子目标应≈goal(states[8])
    with torch.no_grad():
        z_pred = ha(st, g_near).mean
        z_goal = vf.phi(st, states[8:9])     # 落 goal
        z_way = vf.phi(st, states[15:16])    # 固定 +10 航点
    assert (z_pred - z_goal).norm() < (z_pred - z_way).norm()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py -k "bad_target_mode or clamp_collapses" -q`
Expected: FAIL(`TypeError: train_high_actor() got an unexpected keyword argument 'target_mode'`)

- [ ] **Step 3: 改 `train_high_actor` 与 `save/load_high_actor`**

`train_high_actor` 签名加两参 + mode 校验 + 采样分支(替换 `hiql_high_actor.py:39-79` 整个函数):

```python
def train_high_actor(data, vf, *, way_steps=25, beta=1.0, lr=3e-4,
                     batch_size=256, steps=50_000, hidden=256, seed=0,
                     target_mode="fixed_waypoint", high_p_randomgoal=0.0):
    """AWR 抽高层 π^h。vf:冻结 GoalConditionedVF。复用 Phase 1 的 data(build_gc_data)。

    优势 Ã^h = min V(s_{t+k},g) − min V(s_t,g);回归目标 z=vf.phi(s_t, s_{t+k})。返回训练后的 HighActor。
    target_mode:
      - 'fixed_waypoint'(默认):wi=min(si+way,demo末),goal 混采 sample_gc_goals(与现状逐位等价)。
      - 'clamp_to_goal':HIQL GCSDataset 高段——goal 线性插值轨迹态,wi=min(si+way, goal)。
    """
    if target_mode not in ("fixed_waypoint", "clamp_to_goal"):
        raise ValueError(f"unknown target_mode: {target_mode!r}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vf = copy.deepcopy(vf).eval()
    for p in vf.parameters():
        p.requires_grad_(False)
    states = data["states"]
    D = states.shape[1]
    rep_dim = vf.rep_dim
    ha = HighActor(D, rep_dim, hidden)
    opt = torch.optim.Adam(ha.parameters(), lr=lr)
    s_idx, traj_id = data["s_idx"], data["traj_id"]
    last_arr = np.array([data["last_idx_of"][int(d)] for d in traj_id], dtype=np.int64)
    n = len(s_idx)
    bs = min(batch_size, n)
    for _ in range(steps):
        b = rng.integers(0, n, size=bs)
        si, tj = s_idx[b], traj_id[b]
        if target_mode == "fixed_waypoint":
            wi = np.minimum(si + way_steps, last_arr[b])
            gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"],
                                 rng, n_total=len(states))
        else:
            gi, wi = sample_high_goal_target(si, tj, data["last_idx_of"], rng,
                                             way_steps=way_steps, n_total=len(states),
                                             high_p_randomgoal=high_p_randomgoal)
        s, sw, g = states[si], states[wi], states[gi]
        with torch.no_grad():
            vs1, vs2 = vf(s, g)
            vw1, vw2 = vf(sw, g)
            adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
            w = awr_weight(adv, beta)
            z_tgt = vf.phi(s, sw)
        dist = ha(s, g)
        logp = dist.log_prob(z_tgt).sum(-1)
        loss = -(w * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ha
```

`save_high_actor` 加两个带默认的关键字(改 `hiql_high_actor.py:82-94`):

```python
def save_high_actor(path, model, *, gc_value_ckpt, way_steps, beta,
                    target_mode="fixed_waypoint", high_p_randomgoal=0.0):
    """存 high_actor.pt:权重 + 维度 + gc_value_ckpt/way_steps/beta + target_mode/high_p_randomgoal。"""
    torch.save({
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "rep_dim": model.rep_dim,
        "hidden": model.hidden,
        "log_std_min": model.log_std_min,
        "log_std_max": model.log_std_max,
        "gc_value_ckpt": gc_value_ckpt,
        "way_steps": way_steps,
        "beta": beta,
        "target_mode": target_mode,
        "high_p_randomgoal": high_p_randomgoal,
    }, path)
```

`load_high_actor` info 带回新键(改 `hiql_high_actor.py:97-106` 的 info 段):

```python
    info = {k: ckpt[k] for k in ("gc_value_ckpt", "way_steps", "beta")}
    info["target_mode"] = ckpt.get("target_mode", "fixed_waypoint")
    info["high_p_randomgoal"] = ckpt.get("high_p_randomgoal", 0.0)
    return model, info
```

- [ ] **Step 4: 跑测试确认通过 + 全模块单测(含旧 roundtrip 不破)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py -q`
Expected: 全 PASS(`test_high_actor_save_load_roundtrip`/`test_train_high_actor_predicts_forward_waypoint` 仍绿——save_high_actor 旧签名因新参带默认不受影响)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git commit -m "feat(hiql-align): ② train_high_actor target_mode=clamp_to_goal + save/load 透传"
```

---

## Task 4: ② 训练 CLI flag(`train_hiql_high_actor.py`)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py`

- [ ] **Step 1: 加 flag 并透传**

`build_parser` 加(在 `--seed` 后,`hiql_high_actor.py` 现 CLI 的 `train_hiql_high_actor.py:38` 附近):

```python
    p.add_argument("--target_mode", choices=["fixed_waypoint", "clamp_to_goal"],
                   default="fixed_waypoint",
                   help="高层 AWR 航点:fixed_waypoint(原口径,恒 +way)| clamp_to_goal(HIQL,近 goal 塌到 goal)")
    p.add_argument("--high_p_randomgoal", type=float, default=0.0,
                   help="clamp_to_goal 下高层 goal 取 random 的概率(HIQL 默认 0)")
```

`main` 里 `train_high_actor` 调用加两参(改 `train_hiql_high_actor.py:60-62`):

```python
    ha = train_high_actor(data, vf, way_steps=args.way_steps, beta=args.beta, lr=args.lr,
                          batch_size=args.batch_size, steps=args.steps, hidden=args.hidden,
                          seed=args.seed, target_mode=args.target_mode,
                          high_p_randomgoal=args.high_p_randomgoal)
```

`save_high_actor` 调用加两参(改 `train_hiql_high_actor.py:63-64`):

```python
    save_high_actor(args.output, ha, gc_value_ckpt=args.gc_value_ckpt,
                    way_steps=args.way_steps, beta=args.beta,
                    target_mode=args.target_mode, high_p_randomgoal=args.high_p_randomgoal)
```

`print` 行加上 target_mode(改 `train_hiql_high_actor.py:58-59`):在格式串末尾追加 ` target_mode={args.target_mode}`。

- [ ] **Step 2: 冒烟(parser 解析 + 默认值)**

Run: `conda run -n residual python -c "from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser; a=build_parser().parse_args(['--hdf5','x','--dataset','y','--stage_cache','z','--gc_value_ckpt','w']); print(a.target_mode, a.high_p_randomgoal)"`
Expected: 打印 `fixed_waypoint 0.0`

- [ ] **Step 3: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py
git commit -m "feat(hiql-align): ② train_hiql_high_actor --target_mode/--high_p_randomgoal"
```

---

## Task 5: ① value LayerNorm(`hiql_gc_value.py`)

`use_layer_norm=True`:`_mlp` 每隐层 `Linear→GELU→LayerNorm`(忠实 HIQL `LayerNormMLP`),施于 `v1/v2` 与 `RelativeGoalEncoder.net`;默认 False = 现状 `Linear→ReLU`。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_hiql_gc_value.py`)

```python
def test_mlp_layer_norm_modules_present():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import _mlp
    m_plain = _mlp(8, 16, 4, use_layer_norm=False)
    m_ln = _mlp(8, 16, 4, use_layer_norm=True)
    has = lambda m, t: any(isinstance(x, t) for x in m.modules())
    assert not has(m_plain, torch.nn.LayerNorm) and has(m_plain, torch.nn.ReLU)
    assert has(m_ln, torch.nn.LayerNorm) and has(m_ln, torch.nn.GELU)
    x = torch.randn(3, 8)
    assert m_ln(x).shape == (3, 4)            # 前向 shape 不变

def test_gc_value_layer_norm_forward_and_phi_sphere():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    vf = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, use_layer_norm=True)
    assert vf.use_layer_norm is True
    s, g = torch.randn(5, 30), torch.randn(5, 30)
    v1, v2 = vf(s, g)
    assert v1.shape == (5,) and v2.shape == (5,)
    z = vf.phi(s, g)
    assert torch.allclose(z.norm(dim=-1), torch.full((5,), float(np.sqrt(10))), atol=1e-4)

def test_gc_value_layer_norm_save_load_roundtrip(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, save_gc_value, load_gc_value)
    model = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, use_layer_norm=True)
    p = str(tmp_path / "gc_ln.pt")
    save_gc_value(p, model, v_stats={"min": -3.0, "max": 0.0, "mean": -1.0},
                  mean=torch.zeros(30), std=torch.ones(30),
                  dataset_id="ds", state_mode="eef_piece",
                  rel_piece_stats=(np.zeros(12), np.ones(12)))
    m2, _ = load_gc_value(p)                  # 须按 use_layer_norm=True 重建,否则 load_state_dict 失配报错
    assert m2.use_layer_norm is True
    s, g = torch.randn(4, 30), torch.randn(4, 30)
    v1a, _ = model(s, g); v1b, _ = m2(s, g)
    assert torch.allclose(v1a, v1b, atol=1e-6)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -k "layer_norm" -q`
Expected: FAIL(`_mlp() got an unexpected keyword argument 'use_layer_norm'`)

- [ ] **Step 3: 改 `_mlp` / `RelativeGoalEncoder` / `GoalConditionedVF` / `train_gc_value` / `save+load_gc_value`**

`_mlp`(改 `hiql_gc_value.py:16-22`):

```python
def _mlp(in_dim, hidden, out_dim, n_hidden=2, use_layer_norm=False):
    act = nn.GELU if use_layer_norm else nn.ReLU
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), act()]
        if use_layer_norm:
            layers += [nn.LayerNorm(hidden)]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)
```

`RelativeGoalEncoder.__init__`(改 `hiql_gc_value.py:28-32`):

```python
    def __init__(self, state_dim, rep_dim=10, hidden=256, use_layer_norm=False):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.net = _mlp(2 * state_dim, hidden, rep_dim, use_layer_norm=use_layer_norm)
```

`GoalConditionedVF.__init__`(改 `hiql_gc_value.py:47-54`):

```python
    def __init__(self, state_dim, rep_dim=10, hidden=256, use_layer_norm=False):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.hidden = hidden
        self.use_layer_norm = use_layer_norm
        self.goal_encoder = RelativeGoalEncoder(state_dim, rep_dim, hidden, use_layer_norm=use_layer_norm)
        self.v1 = _mlp(state_dim + rep_dim, hidden, 1, use_layer_norm=use_layer_norm)
        self.v2 = _mlp(state_dim + rep_dim, hidden, 1, use_layer_norm=use_layer_norm)
```

`train_gc_value` 签名加 `use_layer_norm=False`(改 `hiql_gc_value.py:152-154`)并把构造改为
`model = GoalConditionedVF(D, rep_dim, hidden, use_layer_norm=use_layer_norm)`(改 `hiql_gc_value.py:165`)。

`save_gc_value` payload 加键(改 `hiql_gc_value.py:210-220` 的 payload dict,加一行):
`"use_layer_norm": model.use_layer_norm,`

`load_gc_value` 重建带该键(改 `hiql_gc_value.py:229`):

```python
    model = GoalConditionedVF(ckpt["state_dim"], ckpt["rep_dim"], ckpt["hidden"],
                              use_layer_norm=ckpt.get("use_layer_norm", False))
```

- [ ] **Step 4: 跑测试确认通过 + 全模块单测(旧 roundtrip 不破)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -q`
Expected: 全 PASS(旧 `test_gc_value_save_load_roundtrip` 仍绿——`load_gc_value` 用 `get(默认 False)` 兼容无键旧档)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-align): ① GoalConditionedVF use_layer_norm(LN+GELU,对齐 HIQL LayerNormMLP)"
```

---

## Task 6: ① value 训练 CLI flag(`train_hiql_gc_value.py`)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`

- [ ] **Step 1: 加 flag 并透传**

`build_parser` 加(在 `--goal_future_mode` 后,`train_hiql_gc_value.py:61` 附近):

```python
    p.add_argument("--use_layer_norm", type=int, default=0,
                   help="value/rep 用 LN+GELU(1,对齐 HIQL LayerNormMLP)| 裸 ReLU(0,现状)")
```

`main` 里 `train_gc_value` 调用加参(改 `train_hiql_gc_value.py:78-81`):末尾加 `, use_layer_norm=bool(args.use_layer_norm)`。
`print` 行(`train_hiql_gc_value.py:75-77`)末尾追加 ` use_layer_norm={bool(args.use_layer_norm)}`。

- [ ] **Step 2: 冒烟(parser 解析)**

Run: `conda run -n residual python -c "from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser; a=build_parser().parse_args(['--hdf5','x','--dataset','y','--stage_cache','z']); print(a.use_layer_norm)"`
Expected: 打印 `0`

- [ ] **Step 3: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py
git commit -m "feat(hiql-align): ① train_hiql_gc_value --use_layer_norm"
```

---

## Task 7: ② verify 近 goal 塌缩诊断(`verify_high_actor.py --goal_mode near`)

现 `verify_high_actor` 恒用 g=末态(远 goal),clamp 行为退化成 +way、看不出塌缩。加 `--goal_mode near`:对每个 t 取距离 `dist` 的 goal(随机 1..2·way),验证子目标最近邻偏移 `j*−t ≈ min(way, dist)`——近 goal 落 goal、远 goal 落 +way。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/verify_high_actor.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`

- [ ] **Step 1: 写失败测试**(追加到 `test_hiql_high_actor.py`;只验诊断 bookkeeping,不依赖 MuJoCo)

```python
def test_evaluate_near_expected_off_is_clamped():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
    from resfit.rl_finetuning.chunk_residual.verify_high_actor import evaluate_near
    vf = GoalConditionedVF(state_dim=2, rep_dim=4, hidden=16)
    ha = HighActor(state_dim=2, rep_dim=4, hidden=16)
    seqs = [np.stack([np.arange(30), np.arange(30)], axis=1).astype(np.float32)]  # 一条 T=30 demo
    rng = np.random.default_rng(0)
    rows = evaluate_near(ha, vf, seqs, way_steps=10, n_per_demo=20, rng=rng)
    assert len(rows) > 0
    for r in rows:
        assert r["gj"] <= r["T"] - 1
        assert r["dist"] == r["gj"] - r["t"]
        assert r["expected_off"] == min(10, r["dist"])     # clamp 期望偏移
        assert 0 <= r["jstar"] <= r["T"] - 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_evaluate_near_expected_off_is_clamped -q`
Expected: FAIL(`ImportError: evaluate_near`)

- [ ] **Step 3: 加 `evaluate_near` / `report_near` + CLI flag**

`evaluate_near`(追加到 `verify_high_actor.py`,在 `evaluate` 函数之后):

```python
def evaluate_near(ha, vf, seqs, way_steps, n_per_demo, rng):
    """近 goal 塌缩诊断:每个起点 t 取距离 dist∈[1,2·way] 的 goal=s_{min(t+dist,末)},
    求子目标 z 的 demo 内 φ 最近邻 j*;clamp-to-goal 期望 off=j*-t ≈ min(way, dist)。"""
    rows = []
    for di, seq in enumerate(seqs):
        T = len(seq)
        if T < 4:
            continue
        ts = np.unique(np.linspace(0, T - 2, min(T - 1, n_per_demo)).astype(int))
        for t in ts:
            t = int(t)
            dist = int(rng.integers(1, 2 * way_steps + 1))
            gj = min(t + dist, T - 1)
            z = predict_z(ha, seq[t:t + 1], seq[gj:gj + 1])[0]
            base = np.broadcast_to(seq[t], (T, seq.shape[1]))
            phi = phi_of(vf, base, seq)
            jstar = int(np.linalg.norm(phi - z, axis=-1).argmin())
            rows.append(dict(demo=di, t=t, T=int(T), gj=int(gj), dist=int(gj - t),
                             jstar=jstar, off=int(jstar - t),
                             expected_off=int(min(way_steps, gj - t))))
    return rows


def report_near(rows, way_steps):
    near = [r for r in rows if r["dist"] < way_steps]
    far = [r for r in rows if r["dist"] >= way_steps]
    near_err = np.array([abs(r["off"] - r["dist"]) for r in near]) if near else np.array([np.nan])
    far_off = np.array([r["off"] for r in far]) if far else np.array([np.nan])
    print("\n============ ② clamp-to-goal 近 goal 塌缩诊断 ============")
    print(f"采样 {len(rows)} 个 (t, goal@dist);near(dist<{way_steps})={len(near)} far(dist>={way_steps})={len(far)}")
    print(f"近 goal: |off − dist| median={np.nanmedian(near_err):.1f}(≈0 表示子目标落在 goal 上)")
    print(f"远 goal: off median={np.nanmedian(far_off):.1f}(≈way_steps={way_steps} 表示落 +way 航点)")
    ok = (np.nanmedian(near_err) <= max(2, 0.25 * way_steps)) and \
         (abs(np.nanmedian(far_off) - way_steps) <= max(3, 0.3 * way_steps))
    print(f"clamp 诊断: {'PASS' if ok else 'FAIL'}(近 goal 塌到 goal & 远 goal 落 +way)")
    return ok
```

`build_parser` 加(在 `--out` 前):

```python
    p.add_argument("--goal_mode", choices=["final", "near"], default="final",
                   help="final(原 gate,g=末态)| near(clamp-to-goal 近 goal 塌缩诊断)")
```

`main` 末段分流(改 `verify_high_actor.py:198-200`):

```python
    if args.goal_mode == "near":
        rng = np.random.default_rng(0)
        rows = evaluate_near(ha, vf, seqs, hinfo["way_steps"], args.n_per_demo, rng)
        report_near(rows, hinfo["way_steps"])
    else:
        rows, z_norms = evaluate(ha, vf, seqs, hinfo["way_steps"], args.n_per_demo)
        ok, off, off_room, ratio, room = report(rows, z_norms, hinfo["way_steps"], vf.rep_dim)
        make_plot(rows, off, off_room, ratio, room, seqs, hinfo["way_steps"], args.out)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_evaluate_near_expected_off_is_clamped -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/verify_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git commit -m "feat(hiql-align): ② verify_high_actor --goal_mode near(clamp 塌缩诊断)"
```

---

## Task 8: A/B 训练 + 离线 gate 重验 + 证据写回(执行,无 TDD)

> 重训用全量 1006 demos;high_actor 务必传 `--state30_cache outputs_chunk/three_piece_state30.npz` 免 MuJoCo 回放。基线 `_geom` 两件套不删,作对照。

- [ ] **Step 1: 跑全量单测,确认三改全绿**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py -q`
Expected: all passed

- [ ] **Step 2: ② clamp A/B —— 在现 geom value 上重训 high_actor(clamp)**

```bash
cd /mnt/mnt/data/resfit && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --state30_cache outputs_chunk/three_piece_state30.npz \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_geom.pt \
  --way_steps 25 --target_mode clamp_to_goal \
  --output outputs_chunk/three_piece_high_actor_geom_clamp.pt 2>&1 | tee three_piece_high_actor_geom_clamp.log
```
Expected: 打印 `target_mode=clamp_to_goal` 与 `saved`。

- [ ] **Step 3: ② 重验 —— near 诊断(clamp 应过)+ final gate(对照基线)**

对 clamp 与 fixed 基线各跑一次 near 诊断(同参,只换 `--pt`),对照两组数字:

```bash
# A) clamp ckpt
cd /mnt/mnt/data/resfit && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.verify_high_actor \
  --pt outputs_chunk/three_piece_high_actor_geom_clamp.pt --num_demos 40 --goal_mode near
# B) fixed 基线 ckpt
cd /mnt/mnt/data/resfit && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.verify_high_actor \
  --pt outputs_chunk/three_piece_high_actor_geom.pt --num_demos 40 --goal_mode near
```
Expected:A(clamp)near 诊断 PASS——近 goal `|off−dist|` median≈0、远 goal off≈25;B(fixed 基线)近 goal `|off−dist|` 明显更大(不塌 goal,恒 +25)。把两组数字记下对照。

- [ ] **Step 4: ① LN A/B —— 重训 gc_value(LN)再在其上重训 high_actor(fixed,隔离 LN 单变量)**

```bash
cd /mnt/mnt/data/resfit && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --goal_future_mode geometric --use_layer_norm 1 \
  --output outputs_chunk/three_piece_gc_value_geom_ln.pt 2>&1 | tee three_piece_gc_value_geom_ln.log

cd /mnt/mnt/data/resfit && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --state30_cache outputs_chunk/three_piece_state30.npz \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_geom_ln.pt \
  --way_steps 25 \
  --output outputs_chunk/three_piece_high_actor_geom_ln.pt 2>&1 | tee three_piece_high_actor_geom_ln.log
```
Expected:gc_value LN 打印 `use_layer_norm=True` 与 `v_stats`;high_actor 挂 LN value 存盘。

- [ ] **Step 5: ① 重验 —— Phase 1 gate(LN vs geom 基线)+ Phase 2 gate(LN value 上 high_actor 不破)**

```bash
cd /mnt/mnt/data/resfit && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.verify_gc_value \
  --pt outputs_chunk/three_piece_gc_value_geom_ln.pt --num_demos 40
cd /mnt/mnt/data/resfit && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.verify_high_actor \
  --pt outputs_chunk/three_piece_high_actor_geom_ln.pt --num_demos 40
```
Expected:LN value 的三诊断不劣于 geom 基线(目标侧早期尖坑、`V(s,g=s)` 负尾、状态侧不退化);LN 上 high_actor 的 final gate 仍 PASS。与 geom 基线数字逐项对照。

- [ ] **Step 6: 证据写回各自 plan 的 gate 段**

把 Step 3/5 的 A/B 数字(带日期、产物名)追加到:
- `docs/superpowers/plans/2026-06-08-hiql-hierarchy-residual-phase2-high-actor.md` 的 Gate 段 —— 记 clamp near 诊断结果 + 与 fixed 基线对照。
- `docs/superpowers/plans/2026-06-08-hiql-hierarchy-residual-phase1-gc-value.md` 的 Gate 段 —— 记 LN vs geom 三诊断对照。
并在本 plan 末追加一段"A/B 结论"(哪改有效、是否纳入后续 Phase 3 默认两件套)。

- [ ] **Step 7: 提交证据**

```bash
git add docs/superpowers/plans/ outputs_chunk/*.log
git commit -m "docs(hiql-align): A/B 离线 gate 证据回填(clamp near 诊断 + LN vs geom)"
```

---

## 自查记录(writing-plans self-review)

- **Spec 覆盖**:§4.1 LN→Task 5/6;§4.2 clamp→Task 2/3/4;§4.3 renorm→Task 1;§5 验证→Task 7(near 诊断)+ Task 8(A/B 重验 + 写回);§6 接口/单测→Task 1-7 单测;§7 已知遗留差异不实现(本轮范围外,无对应 Task,符合 spec)。
- **占位符**:无;每步完整代码/命令。
- **类型一致**:`sample_high_goal_target(s_idx,traj_id,last_idx_of,rng,*,way_steps,n_total,high_p_randomgoal)` 在 Task 2 定义、Task 3 调用一致;`train_high_actor(...,target_mode,high_p_randomgoal)` Task 3 定义、Task 4 调用一致;`save_high_actor(...,target_mode=,high_p_randomgoal=)` Task 3 定义、Task 4 调用一致;`_mlp(...,use_layer_norm=)`、`GoalConditionedVF(...,use_layer_norm=)`、`train_gc_value(...,use_layer_norm=)` Task 5 定义、Task 6 调用一致;`HiqlSubgoal(...,renorm_subgoal=)`、`from_ckpts(...,renorm_subgoal=)` Task 1 内一致;`evaluate_near`/`report_near` Task 7 定义、`predict_z`/`phi_of` 复用现有。
- **向后兼容**:`save_high_actor`/`save_gc_value` 新参带默认,旧 caller(现有单测)不破;`load_*` 用 `get(默认老值)` 兼容旧 ckpt。
- **默认等价**:②`fixed_waypoint` 保持 `wi`先`gi`后的 RNG 顺序;①`use_layer_norm=False` 走原 `Linear→ReLU`;③`renorm_subgoal=False` 取 `.mean` 原值——三者默认逐位等价现状。

---

## A/B 结论（2026-06-10,离线 gate）

Task1-7 落码完成(subagent 两段审查 + 整体复审 Ready,33 单测绿,默认等价实测参数差=0.0)。Task8 A/B 重训 + 离线 gate 重验已跑完(detached,各限 8 线程防抢核;driver `run_hiql_align_ab.sh` + 看门狗 `run_hiql_align_verify.sh`)。

- **② clamp-to-goal —— 纳入 Phase 3 默认。** near 诊断 `|off−dist|` median:**clamp 1.0 vs fixed 基线 12.0**(远 goal 都 25.0);clamp PASS、基线 FAIL。解掉"恒 +25"局限。证据写回 phase2 plan gate 段。
- **① LayerNorm —— 暂不纳入。** LN value 与 geom 基线 Gate 均 PASS;LN 在状态/目标单调性、折扣量级(-87.6 比 -82.1 更贴解析 -90.6)边际更好,但 **V(s,g=s)≈0 负尾反而更深(-0.96→-3.20、std 0.17→0.31)**;LN 上重训 high_actor final gate 仍 PASS(没带崩高层)。离线证据不足以换默认,留待在线 rollout 再议。证据写回 phase1 plan gate 段。
- **③ renorm —— 纯运行时开关 + 单测已落,接进 rollout 属 Phase 3。**

**Phase 3 默认两件套(本轮结论):** `three_piece_gc_value_geom.pt`(① 维持 geom、不 LN)+ `three_piece_high_actor_geom_clamp.pt`(② clamp)。`_ln` 三件产物保留作对照,不删。
