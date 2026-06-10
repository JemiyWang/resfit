# HIQL value-loss 对齐实施计划(per-critic 目标 + adv 门控 expectile)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `train_gc_value` 加 `value_loss_mode` flag,`'hiql'` 模式把双 critic TD 目标改成 per-critic(不取 min)、expectile 改成 adv 门控的两参形式,对齐参考 HIQL `compute_value_loss`;默认 `'shared_min'` 与现状逐位等价。

**Architecture:** 纯 PyTorch 改动,集中在 `hiql_gc_value.py`(新增两参 expectile + 损失分支 + 双 critic 分化度诊断)、`train_hiql_gc_value.py`(CLI flag)、`verify_gc_value.py`(诊断接线)。模型结构不变,改的只是训练损失;flag-gated、老行为默认、逐位等价单测护栏。级联重训(value→high_actor)+ 离线 gate 重验是重活,留用户在本机跑(Task 7)。

**Tech Stack:** PyTorch、NumPy、pytest。conda env `residual`,命令从仓库根 `/mnt/mnt/data/resfit` 跑。设计见 `docs/superpowers/specs/2026-06-10-hiql-value-loss-align-design.md`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`(改) | 新增 `expectile_loss_weighted`(两参,门控用 adv);`train_gc_value` 加 `value_loss_mode`;`save/load_gc_value` 记/读 provenance;新增 `critic_divergence` 纯诊断。 |
| `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`(改) | 加 `--value_loss_mode {shared_min,hiql}` + 透传 + 启动日志 echo。 |
| `resfit/rl_finetuning/chunk_residual/verify_gc_value.py`(改) | 调 `critic_divergence`,在 gate 报告里打印"⑥ 双 critic 分化度"。 |
| `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`(改) | Task 1–5 的新单测。 |

**不动**:`hiql_value.py`(单任务 ③a 仍用其单参 `expectile_loss`,保持正交)、`hiql_high_actor.py`、`hiql_subgoal.py`、其余 Phase 2/3 代码。

---

## Task 1: 两参 expectile(`expectile_loss_weighted`)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**(追加到测试文件末尾)

```python
def test_expectile_loss_weighted_gates_on_adv():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import expectile_loss_weighted
    from resfit.rl_finetuning.chunk_residual.hiql_value import expectile_loss
    tau = 0.7
    # 门控用 adv:正/负/0(0 归入 >=0 -> tau),平方的是 diff
    adv = torch.tensor([1.0, -1.0, 0.0])
    diff = torch.tensor([2.0, 2.0, 2.0])
    out = expectile_loss_weighted(adv, diff, tau)
    expected = (0.7 * 4 + 0.3 * 4 + 0.7 * 4) / 3       # weight=[.7,.3,.7], diff²=4, 取 mean
    assert abs(float(out) - expected) < 1e-6
    # adv 与 diff 异号时,与"用 diff 自门控"的单参版不同(证明门控真的看 adv)
    adv2, diff2 = torch.tensor([-1.0]), torch.tensor([2.0])
    assert abs(float(expectile_loss_weighted(adv2, diff2, tau)) - 1.2) < 1e-6   # 0.3*4
    assert abs(float(expectile_loss(diff2, tau)) - 2.8) < 1e-6                  # 0.7*4
    # adv==diff 时退化为标准 expectile,两者一致
    z = torch.tensor([1.5, -0.5, 2.0])
    assert torch.allclose(expectile_loss_weighted(z, z, tau), expectile_loss(z, tau), atol=1e-7)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_expectile_loss_weighted_gates_on_adv -q`
Expected: FAIL(`ImportError: cannot import name 'expectile_loss_weighted'`)

- [ ] **Step 3: 写最小实现**(追加到 `hiql_gc_value.py`,放在文件顶部 `expectile_loss` import 之后、`_mlp` 之前)

```python
def expectile_loss_weighted(adv, diff, expectile):
    """两参 expectile(逐字照 HIQL hiql.py:20-22):门控看 adv 符号,平方的是 diff。

    weight = tau      if adv >= 0   # adv=q-V_target,正=该转移是赚的 -> 把 V 往上拉
    weight = 1 - tau  if adv <  0
    与单参 expectile_loss 的区别:门控量(adv)与被平方量(diff)解耦。adv==diff 时两者等价。
    返回标量(.mean())。
    """
    weight = torch.where(adv >= 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_expectile_loss_weighted_gates_on_adv -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-vloss): expectile_loss_weighted(两参 adv 门控 expectile)"
```

---

## Task 2: `train_gc_value` 加 `value_loss_mode`(默认逐位等价 + hiql 分支)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`(`train_gc_value`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**(追加到测试文件末尾)

```python
def test_train_gc_value_loss_mode_default_equiv_and_hiql_differs():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(12).reshape(12, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    kw = dict(gamma=0.99, expectile=0.7, ema=0.01, lr=1e-3,
              batch_size=8, steps=300, rep_dim=8, hidden=32, seed=0)
    # 默认 == 显式 shared_min,逐张量 atol=0(证明默认分支没被动过)
    m_def, _ = train_gc_value(data, **kw)
    m_sm, _ = train_gc_value(data, value_loss_mode="shared_min", **kw)
    for k, a in m_def.state_dict().items():
        assert torch.equal(a, m_sm.state_dict()[k]), f"default vs shared_min 不等: {k}"
    # hiql 模式确实走了不同分支 -> 权重与 shared_min 不同
    m_hi, _ = train_gc_value(data, value_loss_mode="hiql", **kw)
    diff = max(float((m_hi.state_dict()[k] - m_sm.state_dict()[k]).abs().max())
               for k in m_sm.state_dict())
    assert diff > 1e-6, "hiql 与 shared_min 应训出不同权重"


def test_train_gc_value_hiql_learns_progress():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    model, v_stats = train_gc_value(
        data, gamma=0.99, expectile=0.7, ema=0.01, lr=1e-3,
        batch_size=9, steps=2000, rep_dim=8, hidden=64, seed=0, value_loss_mode="hiql")
    assert v_stats["max"] > v_stats["min"]
    states = data["states"]
    g = states[9].repeat(10, 1)
    with torch.no_grad():
        v1, v2 = model(states, g)
        v = torch.minimum(v1, v2)
    assert v[-3:].mean() > v[:3].mean()


def test_train_gc_value_rejects_bad_loss_mode():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    with pytest.raises(ValueError):
        train_gc_value(data, steps=1, batch_size=4, rep_dim=4, hidden=16,
                       value_loss_mode="bogus", seed=0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -q -k "loss_mode or hiql_learns"`
Expected: FAIL(`TypeError: train_gc_value() got an unexpected keyword argument 'value_loss_mode'`)

- [ ] **Step 3: 写实现**(替换 `hiql_gc_value.py` 里**整个** `train_gc_value` 函数为下面版本;★ `shared_min` 分支必须是现状原文,只新增 `value_loss_mode` 参数、顶部校验、与 `else` 的 hiql 分支)

```python
def train_gc_value(data, *, gamma=0.99, expectile=0.7, ema=0.005, lr=3e-4,
                   batch_size=256, steps=50_000, rep_dim=10, hidden=256, seed=0,
                   future_mode="stage_entry", use_layer_norm=False,
                   value_loss_mode="shared_min"):
    """在扁平 GC 数据上训 action-free expectile goal-conditioned value。

    reward r(s,g)=0 if s==g else -1;到达 goal 或 demo 末步都截断 bootstrap。
    EMA target。返回 (model, v_stats)。future_mode 透传给 sample_gc_goals。
    value_loss_mode:
      - 'shared_min'(默认,现状):两 critic 都回归 y=r+γ·mask·min(nv1,nv2),残差自门控 expectile。
      - 'hiql'(对齐参考):per-critic 目标 q_i=r+γ·mask·nv_i(不取 min)、adv=q−V_target 门控的
        两参 expectile、当前态 V 走 target 网。复刻 HIQL compute_value_loss 的 expectile+双 critic
        两处(mask 仍含 (1-done),done-mask 不在本轮范围)。
    """
    if value_loss_mode not in ("shared_min", "hiql"):
        raise ValueError(f"unknown value_loss_mode: {value_loss_mode!r}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    states = data["states"]
    D = states.shape[1]
    model = GoalConditionedVF(D, rep_dim, hidden, use_layer_norm=use_layer_norm)
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(data["s_idx"])
    bs = min(batch_size, n)
    s_idx, sn_idx, traj_id = data["s_idx"], data["sn_idx"], data["traj_id"]
    done_all = data["done"]
    for _ in range(steps):
        b = rng.integers(0, n, size=bs)
        si, sni, tj = s_idx[b], sn_idx[b], traj_id[b]
        gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"],
                             rng, n_total=len(states),
                             future_mode=future_mode, discount=gamma)
        s = states[si]
        s_next = states[sni]
        g = states[gi]
        success = torch.tensor(si == gi, dtype=torch.float32)
        reward = success - 1.0
        mask = (1.0 - success) * (1.0 - done_all[b])
        if value_loss_mode == "shared_min":
            with torch.no_grad():
                nv1, nv2 = target(s_next, g)
                nv = torch.minimum(nv1, nv2)
                y = reward + gamma * mask * nv
            v1, v2 = model(s, g)
            loss = expectile_loss(y - v1, expectile) + expectile_loss(y - v2, expectile)
        else:  # "hiql"
            with torch.no_grad():
                nv1, nv2 = target(s_next, g)
                nv = torch.minimum(nv1, nv2)
                q = reward + gamma * mask * nv
                v1t, v2t = target(s, g)            # 当前态 V 走 target 网
                v_t = 0.5 * (v1t + v2t)
                adv = q - v_t
                q1 = reward + gamma * mask * nv1   # per-critic 目标,不取 min
                q2 = reward + gamma * mask * nv2
            v1, v2 = model(s, g)
            loss = (expectile_loss_weighted(adv, q1 - v1, expectile)
                    + expectile_loss_weighted(adv, q2 - v2, expectile))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), model.parameters()):
                tp.mul_(1.0 - ema).add_(ema * mp)
    with torch.no_grad():
        gl = np.array([data["last_idx_of"][int(d)] for d in traj_id], dtype=np.int64)
        vv1, vv2 = model(states[s_idx], states[gl])
        vv = torch.minimum(vv1, vv2)
        v_stats = {"min": float(vv.min()), "max": float(vv.max()), "mean": float(vv.mean())}
    return model, v_stats
```

> ★ 等价性要点:`hiql` 分支新增的 `target(s, g)` 前向**不消耗 rng**;`b`/`sample_gc_goals` 的 rng 调用在分支之前、两支共用 → 默认(shared_min)与现状逐位等价(单测 1 的 `torch.equal` 护栏)。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -q -k "loss_mode or hiql_learns or bad_loss_mode"`
Expected: PASS(3 个)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-vloss): train_gc_value --value_loss_mode hiql(per-critic 目标+adv 门控,默认等价)"
```

---

## Task 3: `save/load_gc_value` 记 `value_loss_mode`(provenance)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`(`save_gc_value`/`load_gc_value`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**(追加到测试文件末尾)

```python
def test_gc_value_save_load_value_loss_mode(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, save_gc_value, load_gc_value)
    import torch as _t
    model = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64)
    p = str(tmp_path / "gc_vlm.pt")
    save_gc_value(p, model, v_stats={"min": -3.0, "max": 0.0, "mean": -1.0},
                  mean=_t.zeros(30), std=_t.ones(30), dataset_id="ds",
                  state_mode="eef_piece", rel_piece_stats=(np.zeros(12), np.ones(12)),
                  value_loss_mode="hiql")
    _m, info = load_gc_value(p)
    assert info["value_loss_mode"] == "hiql"
    # 旧档(无该键)回退 shared_min
    ckpt = _t.load(p, weights_only=False)
    del ckpt["value_loss_mode"]
    _t.save(ckpt, p)
    _m2, info2 = load_gc_value(p)
    assert info2["value_loss_mode"] == "shared_min"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_gc_value_save_load_value_loss_mode -q`
Expected: FAIL(`TypeError: save_gc_value() got an unexpected keyword argument 'value_loss_mode'`)

- [ ] **Step 3: 写实现**(改 `save_gc_value` 签名 + payload;改 `load_gc_value` info)

`save_gc_value` 签名加参数 `value_loss_mode="shared_min"`:
```python
def save_gc_value(path, model, *, v_stats, mean, std, dataset_id,
                  state_mode="eef_piece", rel_piece_stats=None,
                  value_loss_mode="shared_min"):
```
在 payload dict 里(`"state_mode": state_mode,` 之后)加一行:
```python
        "value_loss_mode": value_loss_mode,
```
`load_gc_value` 里,在 `info["state_mode"] = ckpt.get("state_mode", "eef_piece")` 之后加一行:
```python
    info["value_loss_mode"] = ckpt.get("value_loss_mode", "shared_min")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_gc_value_save_load_value_loss_mode -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-vloss): save/load_gc_value 记 value_loss_mode(provenance,旧档回退)"
```

---

## Task 4: CLI `--value_loss_mode`

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`

- [ ] **Step 1: 加 flag**(在 `build_parser` 里,`--use_layer_norm` 那个 `add_argument` 之后)

```python
    p.add_argument("--value_loss_mode", choices=["shared_min", "hiql"], default="shared_min",
                   help="value 损失:shared_min(现状,两 critic 钉 min 共享目标 + 残差 expectile)| "
                        "hiql(对齐参考,per-critic 目标不取 min + adv 门控两参 expectile)")
```

- [ ] **Step 2: 透传 + echo**

在 `main()` 里 `train_gc_value(...)` 调用加实参 `value_loss_mode=args.value_loss_mode`(放在 `use_layer_norm=bool(args.use_layer_norm)` 同处):
```python
    model, v_stats = train_gc_value(
        data, gamma=args.gamma, expectile=args.expectile, ema=args.ema, lr=args.lr,
        batch_size=args.batch_size, steps=args.steps, rep_dim=args.rep_dim,
        hidden=args.value_hidden, seed=args.seed, future_mode=args.goal_future_mode,
        use_layer_norm=bool(args.use_layer_norm), value_loss_mode=args.value_loss_mode)
```
把启动日志那行的 f-string 末尾补上 `value_loss_mode`:
```python
    print(f"[hiql_gc] demos={len(seqs)} transitions={len(data['s_idx'])} "
          f"state_dim={data['states'].shape[1]} rep_dim={args.rep_dim} "
          f"goal_future_mode={args.goal_future_mode} use_layer_norm={bool(args.use_layer_norm)} "
          f"value_loss_mode={args.value_loss_mode}")
```
并把 `save_gc_value(...)` 调用加实参 `value_loss_mode=args.value_loss_mode`(放在 `rel_piece_stats=rel_stats` 同处):
```python
    save_gc_value(args.output, model, v_stats=v_stats,
                  mean=standardizer._mean.cpu(), std=standardizer._std.cpu(),
                  dataset_id=args.dataset, state_mode="eef_piece", rel_piece_stats=rel_stats,
                  value_loss_mode=args.value_loss_mode)
```

- [ ] **Step 3: 冒烟(只验 CLI 接通,不训练)**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value --help 2>&1 | grep -A2 value_loss_mode`
Expected: 打印出 `--value_loss_mode {shared_min,hiql}` 及其 help 文本,无 import 报错。

- [ ] **Step 4: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py
git commit -m "feat(hiql-vloss): train_hiql_gc_value --value_loss_mode(透传+echo+provenance)"
```

---

## Task 5: 双 critic 分化度诊断(`critic_divergence` + verify 接线)

把"双 critic 是否真分化"量化为可复跑诊断——这是本次改动的核心收益指标(shared_min 下两 critic 趋同 → corr≈1;hiql 下应更分化)。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`(加纯函数 `critic_divergence`)
- Modify: `resfit/rl_finetuning/chunk_residual/verify_gc_value.py`(接线打印)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**(追加到测试文件末尾)

```python
def test_critic_divergence_keys_and_range():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, critic_divergence)
    torch.manual_seed(0)
    vf = GoalConditionedVF(state_dim=4, rep_dim=4, hidden=16)
    seqs = [np.random.RandomState(1).randn(6, 4).astype(np.float32),
            np.random.RandomState(2).randn(5, 4).astype(np.float32)]
    d = critic_divergence(vf, seqs)
    assert set(d) == {"corr", "mean_abs_diff"}
    assert -1.0 <= d["corr"] <= 1.0
    assert d["mean_abs_diff"] >= 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_critic_divergence_keys_and_range -q`
Expected: FAIL(`ImportError: cannot import name 'critic_divergence'`)

- [ ] **Step 3: 写实现**(追加到 `hiql_gc_value.py` 末尾)

```python
@torch.no_grad()
def critic_divergence(model, seqs):
    """双 critic 在真态上的分化度诊断:corr(v1, v2) 与 mean|v1−v2|。

    g 取各 demo 末态(与 verify 状态侧同口径)。corr 越低、|Δ| 越大 = 两 critic 越分化 =
    min(v1,v2) ensemble 越有意义。shared_min 下两 critic 易趋同(corr→1);hiql per-critic 目标应更分化。
    返回 {'corr': float, 'mean_abs_diff': float}。
    """
    v1s, v2s = [], []
    for seq in seqs:
        s = torch.as_tensor(np.ascontiguousarray(seq), dtype=torch.float32)
        g = torch.as_tensor(np.ascontiguousarray(np.broadcast_to(seq[-1], seq.shape)),
                            dtype=torch.float32)
        a, b = model(s, g)
        v1s.append(a.numpy())
        v2s.append(b.numpy())
    v1, v2 = np.concatenate(v1s), np.concatenate(v2s)
    return {"corr": float(np.corrcoef(v1, v2)[0, 1]),
            "mean_abs_diff": float(np.abs(v1 - v2).mean())}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_critic_divergence_keys_and_range -q`
Expected: PASS

- [ ] **Step 5: 接进 verify_gc_value.py**

在 `verify_gc_value.py` 顶部 import 改为带上 `critic_divergence`:
```python
from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value, critic_divergence
```
在 `main()` 里 `res = evaluate(model, seqs, args.gamma)` 之后、`report(res, len(seqs))` 之前加:
```python
    div = critic_divergence(model, seqs)
    print(f"⑥ 双 critic 分化度 [hiql 应更分化]: corr(v1,v2)={div['corr']:.4f} "
          f"mean|v1-v2|={div['mean_abs_diff']:.4f}  "
          f"(value_loss_mode={info.get('value_loss_mode', 'shared_min')})")
```

- [ ] **Step 6: 语法自检 + 提交**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m py_compile resfit/rl_finetuning/chunk_residual/verify_gc_value.py && echo OK`
Expected: `OK`(诊断逻辑已由 Step 4 单测覆盖;在真 .pt 上的实跑由 Task 7 gate 触发)

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/verify_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-vloss): critic_divergence 双 critic 分化度诊断 + verify 接线"
```

---

## Task 6: 全模块单测复跑(回归护栏)

**Files:** 无新增,只跑测试。

- [ ] **Step 1: 跑全模块单测**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -q`
Expected: 全 PASS(原有 + 本次新增 6 个测试,无失败)。

- [ ] **Step 2(可选): 跑相邻基线回归**(确认没碰坏 Phase 2/3)

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全 PASS(`hiql_value.py` 单参 expectile 未动 → 单任务 ③a/Phase2/Phase3 测试不受影响)。

- [ ] **Step 3: 提交(若 Step 1/2 有任何 lint/格式微调)**

```bash
git add -A && git commit -m "test(hiql-vloss): 全模块单测复跑绿" || echo "无改动,跳过"
```

---

## Task 7: 级联重训 + 离线 gate 重验(重活,留用户在本机跑)

代码改的是损失而非结构,但训出的权重不同 → 必须重训 value、级联重训 high_actor、重跑两 gate,与现 `_geom` 基线 A/B。**这步耗时(value 训练含 EGL 回放、high_actor 吃 state30 缓存),不在 subagent 自动执行范围,交用户。** 基线 `_geom` 两件套保留不删。

> EGL 单卡:多卡机上建议前置 `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8` 防渲染漏卡/多核颠簸(见 memory `project_resfit_egl_render_samecard`)。

- [ ] **Step 1: 重训 value(hiql 模式,挂 geometric)**

```bash
cd /mnt/mnt/data/resfit && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --goal_future_mode geometric --value_loss_mode hiql \
  --output outputs_chunk/three_piece_gc_value_geom_hiqlvloss.pt 2>&1 | tee three_piece_gc_value_geom_hiqlvloss.log
```
Expected: 启动行含 `value_loss_mode=hiql`;末尾 `v_stats` 的 `max>min`。

- [ ] **Step 2: 级联重训 high_actor(挂新 value,target_mode 默认 fixed_waypoint 隔离单变量)**

```bash
cd /mnt/mnt/data/resfit && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --state30_cache outputs_chunk/three_piece_state30.npz \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_geom_hiqlvloss.pt \
  --way_steps 25 \
  --output outputs_chunk/three_piece_high_actor_geom_hiqlvloss.pt 2>&1 | tee three_piece_high_actor_geom_hiqlvloss.log
```
Expected: 无异源 ckpt / state_dim 报错(新 value 同 dataset、state_dim=30),正常存盘。

- [ ] **Step 3: 重验 value gate + 双 critic 分化度(新臂 vs geom 基线)**

```bash
cd /mnt/mnt/data/resfit && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.verify_gc_value \
  --pt outputs_chunk/three_piece_gc_value_geom_hiqlvloss.pt --num_demos 40
# 对照基线:
cd /mnt/mnt/data/resfit && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.verify_gc_value \
  --pt outputs_chunk/three_piece_gc_value_geom.pt --num_demos 40
```
判据:hiql 臂 ①状态侧 median spearman、⑤目标侧 median 不劣于基线(Gate 仍 PASS);**⑥ 双 critic `corr(v1,v2)` 明显低于基线 / `mean|v1-v2|` 明显高于基线**(核心收益:ensemble 不再退化)。若 ⑥ 无差异 → 记录"已知:共享 adv 门控下分化有限",不追加改动。

- [ ] **Step 4: 重验 high_actor gate(判据不变,确认 value 改动未破坏高层)**

```bash
cd /mnt/mnt/data/resfit && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.verify_high_actor \
  --pt outputs_chunk/three_piece_high_actor_geom_hiqlvloss.pt \
  --num_demos 40 \
  --out outputs_chunk/high_actor_verify_geom_hiqlvloss.png
```
（`--gc_value` 不传 → 自动读 high_actor ckpt 内嵌的 `gc_value_ckpt` 路径,即新 hiqlvloss value。）
判据(沿用 Phase 2 gate):前向步 median≈25、前向占比高、塌末态≈0、off/expected≈1.0。

- [ ] **Step 5: 把 A/B 数值与结论写回本 plan 的下方 Gate 段**(带日期、产物名、⑥ 分化度对照数字)。

---

## Phase 验证 Gate(本改动)

- [ ] 单测全绿:Task 6 复跑 `pytest test_hiql_gc_value.py -q` → 全 PASS(含新增 6 测)。
- [ ] 默认逐位等价:`value_loss_mode='shared_min'` 与现状 `torch.equal`(单测 `test_train_gc_value_loss_mode_default_equiv_and_hiql_differs`)。
- [ ] (用户跑)hiql 臂离线 gate:value gate 不劣于 geom 基线 + ⑥ 双 critic 分化度对照;high_actor gate 判据不变。证据(数值/log/图)落 `outputs_chunk/` 并回填此段。

---

## 自查记录(writing-plans self-review)

- **Spec 覆盖**:覆盖 spec §4.1(`expectile_loss_weighted`=Task 1)、§4.2(`train_gc_value` value_loss_mode=Task 2)、§4.3(CLI+schema=Task 3/4)、§5(verify 加 v1/v2 分化度=Task 5,gate 重验=Task 7)、§6 接口与单测清单(Task 1–5 单测一一对应)、§7 风险(默认等价=Task 2 单测护栏、adv 不带梯度=Task 2 实现 no_grad 范围)。done-mask/min-vs-mean/concat-φ 属 spec §2 非目标,无对应 task(正确)。
- **占位符**:无 TBD/TODO;每步含完整代码或精确命令。
- **类型一致**:`expectile_loss_weighted(adv, diff, expectile)` 签名在 Task 1 定义、Task 2 调用一致;`value_loss_mode` 取值 `{'shared_min','hiql'}` 在 Task 2/3/4 一致;`critic_divergence(model, seqs)` 返回 `{'corr','mean_abs_diff'}` 在 Task 5 定义与 verify 接线/单测一致;`save_gc_value(..., value_loss_mode=)`/`load_gc_value` info 键在 Task 3/4 自洽。
