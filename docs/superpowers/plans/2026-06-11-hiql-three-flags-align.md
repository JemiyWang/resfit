# HIQL 三处对齐(done-mask / 优势 min→mean / goal 表征 concat→goal-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 resfit 分层路与 HIQL 仍未对齐的三处(done-mask、高层优势 min→mean、goal 表征 concat→goal-only)各做成一个 flag,CLI 默认翻 HIQL、底层函数默认=旧行为(逐位等价)、旧 ckpt 可回退。

**Architecture:** 沿用项目"加 flag、CLI 默认翻 HIQL、底层签名默认=旧行为、旧值显式可回退、save provenance、load `get` 回退"惯例(参照已落地的 `--value_loss_mode`/`--use_layer_norm`)。三个 flag 各自独立、逐条可 A/B。只动 offline 两件套(`gc_value`+`high_actor`),不碰 `train_chunk_residual`。

**Tech Stack:** PyTorch、numpy、pytest;文件均在 `resfit/rl_finetuning/chunk_residual/`。

**关键约定(三处一致):**
- **底层函数签名默认 = 旧行为**(`done_aware`/`concat`/`min`),保证调用方不传时与现状逐位等价。
- **CLI(`argparse`)默认 = 新行为**(`hiql`/`goal_only`/`mean`),即对齐 HIQL。
- 这是 `project_resfit_chunk_residual_golden_defaults` 记的"只改 CLI 默认、底层签名默认不动"模式,本计划严格遵循。

**参考 spec:** `docs/superpowers/specs/2026-06-11-hiql-three-flags-align-design.md`

---

## File Structure(要改的文件 + 职责)

- `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py` — value 模型 + 训练 + save/load。改:`RelativeGoalEncoder`/`GoalConditionedVF` 加 `rep_mode`(③);`train_gc_value` 加 `value_mask_mode`(①)+`value_rep_mode`(③);`save_gc_value`/`load_gc_value` 记/回退两键。
- `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py` — high_actor 训练 + save/load。改:`train_high_actor` 加 `adv_agg`(②);`save_high_actor`/`load_high_actor` 记/回退 `adv_agg`。
- `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py` — gc_value CLI。改:加 `--value_mask_mode`(默认 `hiql`)+`--value_rep_mode`(默认 `goal_only`),透传 + echo。
- `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py` — high_actor CLI。改:加 `--adv_agg`(默认 `mean`),透传 + echo。
- `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py` — ① 与 ③ 的测试。
- `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py` — ② 的测试。

**执行顺序:** Task 1(①)→ Task 2(②)→ Task 3(③)→ Task 4(回归)。Task 3 改 `GoalConditionedVF` 构造签名,放最后以免影响前两个 task 的等价基线。所有 `pytest` 从 repo 根 `/mnt/mnt/data/resfit` 运行。

---

## Task 1: ① done-mask flag(`--value_mask_mode`)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`(`train_gc_value` 签名 + mask 分支 `:208`;`save_gc_value` `:243`;`load_gc_value` `:267`)
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`(CLI + 透传 + echo)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_hiql_gc_value.py` 末尾追加:

```python
def test_value_mask_mode_default_equivalence():
    """底层默认(不传)与显式 'done_aware' 逐位等价(默认路径未被破坏)。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(steps=300, batch_size=16, rep_dim=8, hidden=32, lr=1e-3, ema=0.01, seed=0)
    m0, _ = train_gc_value(data, **kw)
    m1, _ = train_gc_value(data, value_mask_mode="done_aware", **kw)
    for a, b in zip(m0.state_dict().values(), m1.state_dict().values()):
        assert torch.equal(a, b)


def test_value_mask_mode_hiql_differs_and_validates():
    """'hiql' 模式去掉 (1-done) → 训出与 'done_aware' 不同;非法值报错。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(steps=300, batch_size=16, rep_dim=8, hidden=32, lr=1e-3, ema=0.01, seed=0)
    m_done, _ = train_gc_value(data, value_mask_mode="done_aware", **kw)
    m_hiql, _ = train_gc_value(data, value_mask_mode="hiql", **kw)
    diffs = [not torch.equal(a, b) for a, b in
             zip(m_done.state_dict().values(), m_hiql.state_dict().values())]
    assert any(diffs)
    with pytest.raises(ValueError):
        train_gc_value(data, value_mask_mode="bogus", **kw)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_value_mask_mode_default_equivalence resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_value_mask_mode_hiql_differs_and_validates -v`
Expected: FAIL(`train_gc_value() got an unexpected keyword argument 'value_mask_mode'`)

- [ ] **Step 3: 改 `train_gc_value`(`hiql_gc_value.py`)**

签名(`:168-171`)加 `value_mask_mode="done_aware"`:

```python
def train_gc_value(data, *, gamma=0.99, expectile=0.7, ema=0.005, lr=3e-4,
                   batch_size=256, steps=50_000, rep_dim=10, hidden=256, seed=0,
                   future_mode="stage_entry", use_layer_norm=False,
                   value_loss_mode="shared_min", value_mask_mode="done_aware"):
```

函数体开头(紧挨现有 `value_loss_mode` 校验之后,`:182-183` 附近)加校验:

```python
    if value_mask_mode not in ("done_aware", "hiql"):
        raise ValueError(f"unknown value_mask_mode: {value_mask_mode!r}")
```

把 `mask` 那行(`:208`)换成分支:

```python
        if value_mask_mode == "done_aware":
            mask = (1.0 - success) * (1.0 - done_all[b])
        else:  # "hiql"
            mask = 1.0 - success
```

- [ ] **Step 4: 改 `save_gc_value` / `load_gc_value`(`hiql_gc_value.py`)**

`save_gc_value` 签名(`:243-245`)加 `value_mask_mode="done_aware"`,payload(`:249-261`)加一键:

```python
def save_gc_value(path, model, *, v_stats, mean, std, dataset_id,
                  state_mode="eef_piece", rel_piece_stats=None,
                  value_loss_mode="shared_min", value_mask_mode="done_aware"):
    payload = {
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "rep_dim": model.rep_dim,
        "hidden": model.hidden,
        "use_layer_norm": model.use_layer_norm,
        "v_stats": v_stats,
        "mean": mean,
        "std": std,
        "dataset_id": dataset_id,
        "state_mode": state_mode,
        "value_loss_mode": value_loss_mode,
        "value_mask_mode": value_mask_mode,
    }
    if rel_piece_stats is not None:
        payload["rel_piece_mean"], payload["rel_piece_std"] = rel_piece_stats
    torch.save(payload, path)
```

`load_gc_value`(`:276` 后)加一行 info:

```python
    info["value_mask_mode"] = ckpt.get("value_mask_mode", "done_aware")
```

- [ ] **Step 5: 改 CLI(`train_hiql_gc_value.py`)**

在 `--value_loss_mode`(`:64`)之后加(注意 **CLI 默认 `hiql`**):

```python
    p.add_argument("--value_mask_mode", choices=["done_aware", "hiql"], default="hiql",
                   help="value TD mask:done_aware=(1-success)(1-done)旧行为;hiql=1-success(对齐HIQL,默认)")
```

`train_gc_value(...)` 调用(`main` 里)末尾加 `value_mask_mode=args.value_mask_mode`;`save_gc_value(...)` 调用末尾加 `value_mask_mode=args.value_mask_mode`;echo 行(`:80-83`)的 f-string 末尾追加 ` value_mask_mode={args.value_mask_mode}`。

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -v`
Expected: PASS(新增 2 个 + 原有全过)

- [ ] **Step 7: Commit**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git -C /mnt/mnt/data/resfit commit -m "feat(hiql): --value_mask_mode flag(默认 hiql 去 done-mask,底层默认 done_aware 等价)"
```

---

## Task 2: ② 高层优势 min→mean flag(`--adv_agg`)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`(`train_high_actor` 签名 `:61-63` + adv 分支 `:106`;`save_high_actor` `:118`;`load_high_actor` `:136`)
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py`(CLI + 透传 + echo)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_hiql_high_actor.py` 末尾追加:

```python
def test_adv_agg_default_equivalence():
    """底层默认(不传)与显式 'min' 逐位等价(默认路径未被破坏)。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    vf, _ = train_gc_value(data, steps=300, batch_size=16, rep_dim=8, hidden=32,
                           lr=1e-3, ema=0.01, seed=0)
    kw = dict(way_steps=5, beta=1.0, steps=300, batch_size=16, hidden=32, lr=1e-3, seed=0)
    h0 = train_high_actor(data, vf, **kw)
    h1 = train_high_actor(data, vf, adv_agg="min", **kw)
    for a, b in zip(h0.state_dict().values(), h1.state_dict().values()):
        assert torch.equal(a, b)


def test_adv_agg_mean_differs_and_validates():
    """'mean' 聚合与 'min' 训出不同;非法值报错。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    vf, _ = train_gc_value(data, steps=300, batch_size=16, rep_dim=8, hidden=32,
                           lr=1e-3, ema=0.01, seed=0)
    kw = dict(way_steps=5, beta=1.0, steps=300, batch_size=16, hidden=32, lr=1e-3, seed=0)
    h_min = train_high_actor(data, vf, adv_agg="min", **kw)
    h_mean = train_high_actor(data, vf, adv_agg="mean", **kw)
    diffs = [not torch.equal(a, b) for a, b in
             zip(h_min.state_dict().values(), h_mean.state_dict().values())]
    assert any(diffs)
    with pytest.raises(ValueError):
        train_high_actor(data, vf, adv_agg="bogus", **kw)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_adv_agg_default_equivalence resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_adv_agg_mean_differs_and_validates -v`
Expected: FAIL(`train_high_actor() got an unexpected keyword argument 'adv_agg'`)

- [ ] **Step 3: 改 `train_high_actor`(`hiql_high_actor.py`)**

签名(`:61-63`)加 `adv_agg="min"`:

```python
def train_high_actor(data, vf, *, way_steps=25, beta=1.0, lr=3e-4,
                     batch_size=256, steps=50_000, hidden=256, seed=0,
                     target_mode="fixed_waypoint", high_p_randomgoal=0.0,
                     adv_agg="min"):
```

函数体开头(紧挨现有 `target_mode` 校验 `:71-72` 之后)加校验:

```python
    if adv_agg not in ("min", "mean"):
        raise ValueError(f"unknown adv_agg: {adv_agg!r}")
```

把 adv 那行(`:106`)换成分支:

```python
            if adv_agg == "min":
                adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
            else:  # "mean"
                adv = 0.5 * (vw1 + vw2) - 0.5 * (vs1 + vs2)
```

- [ ] **Step 4: 改 `save_high_actor` / `load_high_actor`(`hiql_high_actor.py`)**

`save_high_actor` 签名(`:118-119`)加 `adv_agg="min"`,payload(`:121-133`)加一键 `"adv_agg": adv_agg,`:

```python
def save_high_actor(path, model, *, gc_value_ckpt, way_steps, beta,
                    target_mode="fixed_waypoint", high_p_randomgoal=0.0, adv_agg="min"):
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
        "adv_agg": adv_agg,
    }, path)
```

`load_high_actor`(`:146` 后)加一行:

```python
    info["adv_agg"] = ckpt.get("adv_agg", "min")
```

- [ ] **Step 5: 改 CLI(`train_hiql_high_actor.py`)**

在 `--high_p_randomgoal`(`:42`)之后加(注意 **CLI 默认 `mean`**):

```python
    p.add_argument("--adv_agg", choices=["min", "mean"], default="mean",
                   help="高层优势双 critic 聚合:min=min(vw)-min(vs)旧行为;mean=均值(对齐HIQL,默认)")
```

`train_high_actor(...)` 调用(`main` 里)末尾加 `adv_agg=args.adv_agg`;`save_high_actor(...)` 调用末尾加 `adv_agg=args.adv_agg`;echo 行(`:67-69`)f-string 末尾追加 ` adv_agg={args.adv_agg}`。

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py -v`
Expected: PASS(新增 2 个 + 原有全过)

- [ ] **Step 7: Commit**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git -C /mnt/mnt/data/resfit commit -m "feat(hiql): --adv_agg flag(默认 mean 对齐HIQL,底层默认 min 等价)"
```

---

## Task 3: ③ goal 表征 concat→goal-only flag(`--value_rep_mode`)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`(`RelativeGoalEncoder` `:40-52`;`GoalConditionedVF` `:62-77`;`train_gc_value` 签名 + 构造 `:188`;`save_gc_value` payload;`load_gc_value` 重建)
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`(CLI + 透传 + echo)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_hiql_gc_value.py` 末尾追加:

```python
def test_rep_mode_default_concat_and_dims():
    """默认(不传)= concat:goal 编码器第一层吃 2*state_dim;goal_only 吃 state_dim。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import RelativeGoalEncoder
    enc_default = RelativeGoalEncoder(state_dim=30, rep_dim=10, hidden=64)
    assert enc_default.net[0].in_features == 60          # concat[g,s]
    enc_go = RelativeGoalEncoder(state_dim=30, rep_dim=10, hidden=64, rep_mode="goal_only")
    assert enc_go.net[0].in_features == 30               # goal-only


def test_goal_only_phi_ignores_state():
    """goal_only 下 phi(s,g) 只依赖 g(扰动 s 不变);concat 下扰动 s 会变。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    g = torch.randn(4, 30)
    s1 = torch.randn(4, 30)
    s2 = torch.randn(4, 30)
    vf_go = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="goal_only")
    assert torch.allclose(vf_go.phi(s1, g), vf_go.phi(s2, g), atol=1e-6)
    vf_cat = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="concat")
    assert not torch.allclose(vf_cat.phi(s1, g), vf_cat.phi(s2, g), atol=1e-4)
    with pytest.raises(ValueError):
        GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="bogus")


def test_value_rep_mode_default_equivalence():
    """train_gc_value 底层默认(不传)与显式 'concat' 逐位等价。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(steps=300, batch_size=16, rep_dim=8, hidden=32, lr=1e-3, ema=0.01, seed=0)
    m0, _ = train_gc_value(data, **kw)
    m1, _ = train_gc_value(data, value_rep_mode="concat", **kw)
    for a, b in zip(m0.state_dict().values(), m1.state_dict().values()):
        assert torch.equal(a, b)


def test_gc_value_rep_mode_save_load_roundtrip(tmp_path):
    """save/load 往返保 value_rep_mode 并按之重建维度;旧档(无键)回退 concat。"""
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, save_gc_value, load_gc_value)
    vf = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="goal_only")
    p = tmp_path / "v.pt"
    save_gc_value(str(p), vf, v_stats={"min": 0.0, "max": 0.0, "mean": 0.0},
                  mean=torch.zeros(30), std=torch.ones(30), dataset_id="x")
    m, info = load_gc_value(str(p))
    assert info["value_rep_mode"] == "goal_only"
    assert m.goal_encoder.net[0].in_features == 30
    # 旧档:删掉键 → 回退 concat(2*state_dim)
    ckpt = torch.load(str(p), weights_only=False)
    del ckpt["value_rep_mode"]
    cat = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64, rep_mode="concat")
    ckpt["state_dict"] = cat.state_dict()
    torch.save(ckpt, str(p))
    m2, info2 = load_gc_value(str(p))
    assert info2["value_rep_mode"] == "concat"
    assert m2.goal_encoder.net[0].in_features == 60
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_rep_mode_default_concat_and_dims -v`
Expected: FAIL(`RelativeGoalEncoder.__init__() got an unexpected keyword argument 'rep_mode'`)

- [ ] **Step 3: 改 `RelativeGoalEncoder`(`hiql_gc_value.py:40-52`)**

```python
class RelativeGoalEncoder(nn.Module):
    """φ:rep_mode='concat' 吃 concat([g,s]);'goal_only' 只吃 g。归一化到半径 sqrt(rep_dim)。"""

    def __init__(self, state_dim, rep_dim=10, hidden=256, use_layer_norm=False, rep_mode="concat"):
        super().__init__()
        if rep_mode not in ("concat", "goal_only"):
            raise ValueError(f"unknown rep_mode: {rep_mode!r}")
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.rep_mode = rep_mode
        in_dim = state_dim if rep_mode == "goal_only" else 2 * state_dim
        self.net = _mlp(in_dim, hidden, rep_dim, use_layer_norm=use_layer_norm)

    def forward(self, targets, bases):
        inp = targets if self.rep_mode == "goal_only" else torch.cat([targets, bases], dim=-1)
        rep = self.net(inp)
        rep = rep / (rep.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return rep
```

- [ ] **Step 4: 改 `GoalConditionedVF`(`hiql_gc_value.py:62-77`)**

`__init__` 加 `rep_mode="concat"`,存 `self.rep_mode`,透传给 `goal_encoder`:

```python
    def __init__(self, state_dim, rep_dim=10, hidden=256, use_layer_norm=False, rep_mode="concat"):
        super().__init__()
        if rep_mode not in ("concat", "goal_only"):
            raise ValueError(f"unknown rep_mode: {rep_mode!r}")
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.hidden = hidden
        self.use_layer_norm = use_layer_norm
        self.rep_mode = rep_mode
        self.goal_encoder = RelativeGoalEncoder(state_dim, rep_dim, hidden,
                                                use_layer_norm=use_layer_norm, rep_mode=rep_mode)
        self.v1 = _mlp(state_dim + rep_dim, hidden, 1, use_layer_norm=use_layer_norm)
        self.v2 = _mlp(state_dim + rep_dim, hidden, 1, use_layer_norm=use_layer_norm)
```

`phi`/`forward` 不变(`forward` 仍 `concat[s, phi(s,g)]`,value 头输入维度 `state_dim+rep_dim` 不变)。

- [ ] **Step 5: 改 `train_gc_value`(`hiql_gc_value.py`)**

签名加 `value_rep_mode="concat"`(接在 Task 1 已加的 `value_mask_mode` 之后):

```python
def train_gc_value(data, *, gamma=0.99, expectile=0.7, ema=0.005, lr=3e-4,
                   batch_size=256, steps=50_000, rep_dim=10, hidden=256, seed=0,
                   future_mode="stage_entry", use_layer_norm=False,
                   value_loss_mode="shared_min", value_mask_mode="done_aware",
                   value_rep_mode="concat"):
```

校验(接 `value_mask_mode` 校验之后)+ 构造透传(`:188`):

```python
    if value_rep_mode not in ("concat", "goal_only"):
        raise ValueError(f"unknown value_rep_mode: {value_rep_mode!r}")
```

```python
    model = GoalConditionedVF(D, rep_dim, hidden, use_layer_norm=use_layer_norm,
                              rep_mode=value_rep_mode)
```

(`target = copy.deepcopy(model)` 自动继承 `rep_mode`,无需改。)

- [ ] **Step 6: 改 `save_gc_value` / `load_gc_value`(`hiql_gc_value.py`)**

`save_gc_value` 的 payload 加一键(从 model 读,放在 `"use_layer_norm"` 那行附近):

```python
        "value_rep_mode": model.rep_mode,
```

`load_gc_value`(`:270-271`)按 `value_rep_mode` 重建,并加 info:

```python
    value_rep_mode = ckpt.get("value_rep_mode", "concat")
    model = GoalConditionedVF(ckpt["state_dim"], ckpt["rep_dim"], ckpt["hidden"],
                              use_layer_norm=ckpt.get("use_layer_norm", False),
                              rep_mode=value_rep_mode)
```

```python
    info["value_rep_mode"] = value_rep_mode
```

- [ ] **Step 7: 改 CLI(`train_hiql_gc_value.py`)**

在 `--value_mask_mode`(Task 1 加的)之后加(注意 **CLI 默认 `goal_only`**):

```python
    p.add_argument("--value_rep_mode", choices=["concat", "goal_only"], default="goal_only",
                   help="goal 编码器输入:concat=[g,s]旧行为;goal_only=只吃 g(对齐HIQL,默认)")
```

`train_gc_value(...)` 调用末尾加 `value_rep_mode=args.value_rep_mode`;echo 行末尾追加 ` value_rep_mode={args.value_rep_mode}`。(`save_gc_value` 从 model 读 `rep_mode`,**不必**加调用参数。)

- [ ] **Step 8: 跑测试确认通过**

Run: `pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -v`
Expected: PASS(新增 4 个 + Task 1 的 2 个 + 原有全过)

- [ ] **Step 9: Commit**

```bash
git -C /mnt/mnt/data/resfit add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git -C /mnt/mnt/data/resfit commit -m "feat(hiql): --value_rep_mode flag(默认 goal_only 对齐HIQL,底层默认 concat 等价,旧 ckpt 回退)"
```

---

## Task 4: 全量回归 + CLI 默认核对

**Files:**(只读验证,不改代码)

- [ ] **Step 1: 全量单测**

Run: `pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全绿(此前 294 passed 基础上 +8 新测试 = 302 passed 量级,0 failed)

- [ ] **Step 2: 核对 CLI 默认已翻 HIQL**

Run:
```bash
cd /mnt/mnt/data/resfit && python -c "
from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser as bg
from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser as bh
a = bg().parse_args(['--hdf5','x','--dataset','y','--stage_cache','z'])
b = bh().parse_args(['--hdf5','x','--dataset','y','--stage_cache','z','--gc_value_ckpt','c'])
print('value_mask_mode', a.value_mask_mode)
print('value_rep_mode', a.value_rep_mode)
print('adv_agg', b.adv_agg)
assert (a.value_mask_mode, a.value_rep_mode, b.adv_agg) == ('hiql','goal_only','mean')
print('OK: CLI 默认已全部对齐 HIQL')
"
```
Expected: 打印三个值 = `hiql` / `goal_only` / `mean`,末行 `OK: CLI 默认已全部对齐 HIQL`

- [ ] **Step 3: 核对底层函数默认仍是旧行为(等价基线未漂)**

Run:
```bash
cd /mnt/mnt/data/resfit && python -c "
import inspect
from resfit.rl_finetuning.chunk_residual.hiql_gc_value import train_gc_value
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor
g = inspect.signature(train_gc_value).parameters
h = inspect.signature(train_high_actor).parameters
assert g['value_mask_mode'].default == 'done_aware'
assert g['value_rep_mode'].default == 'concat'
assert h['adv_agg'].default == 'min'
print('OK: 底层函数默认仍是旧行为(done_aware/concat/min)')
"
```
Expected: `OK: 底层函数默认仍是旧行为(done_aware/concat/min)`

- [ ] **Step 4: Commit(若 Step 1-3 无需改代码,本步跳过;有微调则提交)**

```bash
git -C /mnt/mnt/data/resfit status --short
# 如有改动:
# git -C /mnt/mnt/data/resfit add -A && git -C /mnt/mnt/data/resfit commit -m "test(hiql): 三 flag 全量回归 + 默认核对"
```

---

## 重训命令(留用户本机跑,不在本计划自动执行)

默认翻 HIQL 后需重训两件套才生效。基线两件套保留不删作对照。

```bash
# 1) 重训 value(三 flag 默认即对齐 HIQL:hiql mask + goal_only rep + 已默认的 LN/value_loss/geom)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --output outputs_chunk/three_piece_gc_value_hiql3.pt

# 2) 级联重训 high_actor(挂新 value;adv_agg 默认 mean;state30 缓存免回放)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --state30_cache outputs_chunk/three_piece_state30.npz \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_hiql3.pt \
  --output outputs_chunk/three_piece_high_actor_hiql3.pt

# 3) 离线 verify(确认不退化)
python -m resfit.rl_finetuning.chunk_residual.verify_gc_value --ckpt outputs_chunk/three_piece_gc_value_hiql3.pt ...
python -m resfit.rl_finetuning.chunk_residual.verify_high_actor --ckpt outputs_chunk/three_piece_high_actor_hiql3.pt ...
```

---

## Self-Review(写完即查)

**1. Spec coverage:**
- spec §4.1 ①done-mask → Task 1 ✓
- spec §4.2 ②min→mean → Task 2 ✓
- spec §4.3 ③concat→goal_only(只 goal 侧、state 不动)→ Task 3 ✓
- spec §4.4 ckpt 兼容(save provenance + load get 回退 + ③重建维度)→ Task 1 Step4 / Task 2 Step4 / Task 3 Step6 ✓
- spec §7 单测清单(逐位等价/手算/结构/save-load 往返/旧档回退)→ Task 1-3 各 Step1 + Task 4 ✓
- spec §5 级联重训命令 → "重训命令"段 ✓(留用户跑)
- spec §8 scope(不碰 train_chunk_residual)→ File Structure 未列该文件 ✓

**2. Placeholder scan:** 无 TBD/TODO;每个代码步给完整代码;verify 命令的 `...` 仅为留用户补 hdf5/stage_cache 参数(非本计划执行范围,已在段首注明"留用户本机跑")。

**3. Type consistency:** 三处 flag 命名贯穿一致(`value_mask_mode`/`value_rep_mode`/`adv_agg`);choices 一致(`done_aware`/`hiql`、`concat`/`goal_only`、`min`/`mean`);底层默认(旧)与 CLI 默认(新)在 Task 4 Step2/3 双向核对;`GoalConditionedVF(rep_mode=)` 在 Task 3 Step4 定义、Step5/6 与测试 Step1 用法一致。
