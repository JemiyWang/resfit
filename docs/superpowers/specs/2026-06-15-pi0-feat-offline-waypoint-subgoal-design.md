# pi0_feat 离线 waypoint 子目标 设计

**日期**: 2026-06-15
**作者**: subagent-driven 前置 brainstorming
**状态**: 待实现

## 目标

让 `chunk_residual` 残差 RL 在 **LIBERO + base=pi0_libero** 下能跑 `--subgoal_conditioned` + `--offline_fraction>0` + `--demo_bc_coef`（BC 锚），与参照 run `libero10_task8_aligned_bp_bc01_h10`（`offline_fraction=0.5, demo_bc_coef=0.1`）对齐。

## 背景 / 问题

pi0_feat 分层（state_mode=pi0_feat：pi05 prefix 图像特征 2048 ⊕ LIBERO proprio 8 = 2056 当 HIQL value 的 state）目前**只实现了在线子目标**（`compute_online_subgoal` → `subgoal_online(obs, prefix_feat=base.last_prefix_feat())`）。离线子目标缺失，体现在两处：

1. `hiql_subgoal.py:133` `subgoal_waypoint` 硬 `assert self.state_mode != "pi0_feat"`，注释"pi0_feat 暂不支持离线 waypoint"。
2. `libero_offline.py` 的 `build_libero_offline_buffer` 签名无 `subgoal` 参数、全文件零 `observation.subgoal` 处理。

后果：`subgoal_conditioned + offline_fraction>0 + env_family=libero` 时，离线 buffer 的 transition 没有 `observation.subgoal`。而 agent 在 subgoal_conditioned 时把 `observation.subgoal` 列为必需 lowdim 键（`train_chunk_residual.py:690`）。混采时 `concat_mixed_batch`（`offline_hdf5_buffer.py:74`）只保留在线/离线**共有**键 → 在线有 subgoal、离线没有 → subgoal 被丢 → subgoal-conditioned agent 拿不到 subgoal 而崩。

**设计依据**：dexmg 的 `build_offline_buffer`（`offline_stage_replay.py`）已为 **act_feat**（同为"缓存特征态"）实现了离线子目标：逐 demo 用 `act_feat_seqs[ep]`（标准化缓存序列）算 `z=subgoal.subgoal_waypoint(s530, s530[way])` 并逐 transition 存 `curr/nxt["observation.subgoal"]`。本设计把该模式照搬到 pi0_feat + libero：缓存序列从 act_feat 换成 pi0_feat、env 从 dexmg 换 libero。

## 在线/离线子目标语义（保持 HIQL 不对称，不改）

- **在线**：`z = high_actor(s, goal).mean`（高层策略预测，不知未来）。
- **离线**：`z = vf.phi(s_t, s_{t+way})`（demo 知未来，用真航点 φ 表征）。

二者经**同一个** `vf.phi`、同一套标准化 → 同源。本设计的离线 z 走 `subgoal_waypoint`（即 `vf.phi`），与 dexmg 离线语义一字不差。

## 作用域

- **仅 LIBERO**：pi0_feat 只用于 pi0 base（=LIBERO）；dexmg base 是 ACT，永不触发 dexmg+pi0_feat。dexmg 路的 pi0_feat 离线子目标**不在本设计范围**。
- `way_steps` 用 `args.subgoal_way_steps`（默认 25 = high_actor 训练 way_steps = 参照 `subgoal_way_steps`），三处一致。

## 改动面（4 处，集中）

### 改动 1：`hiql_subgoal.py` — `subgoal_waypoint` 放开 pi0_feat

`subgoal_waypoint`（line 130-141）方法体本就通用：把输入转 tensor、`vf.phi(b, t)`，并在 line 140 断言 `b.shape[-1]==vf.state_dim and t.shape[-1]==vf.state_dim`。pi0_feat 下 `vf.state_dim=2056`，传入的缓存 seq 是 2056 维标准化态 → 自洽。

- **删除** line 133 的 `assert self.state_mode != "pi0_feat", "..."`。
- **更新 docstring**：注明 pi0_feat 传 2056 维已标准化的 pi0_feat 缓存序列（prefix_feat ⊕ proprio），与 eef_piece 的 30 维、act_feat 的 530 维并列。

不动 line 140 的 state_dim 断言（它本就用 `vf.state_dim` 通用校验）。

### 改动 2：`libero_offline.py` — 离线 buffer 接 subgoal 并存 observation.subgoal

**2a. `_demo_to_transitions`**（line ~_obs）：加参数 `subgoal_z=None`。在内部 `_obs(t)` 闭包里，当 `subgoal_z is not None` 时加键 `"observation.subgoal": subgoal_z[t]`。因 curr 用 `_obs(t)`、next.obs 用 `_obs(t+1)`，一处即覆盖 curr/nxt 两侧。

```python
def _demo_to_transitions(demo, *, action_scaler, state_standardizer, base_actions,
                         image_size, subgoal_z=None):
    ...
    def _obs(t):
        d = {"observation.state": state_std[t], "observation.base_action": base[t],
             "observation.stage_id": torch.zeros(1, dtype=torch.float32),
             AGENTVIEW_KEY: img_av[t], WRIST_KEY: img_wr[t]}
        if subgoal_z is not None:
            d["observation.subgoal"] = subgoal_z[t]
        return d
    ...
```

**2b. `build_libero_offline_buffer`**：签名加 `subgoal=None, way_steps=25, feat_seqs=None`。改 `for p in paths` 为 `for i, p in enumerate(paths)`（index 对齐缓存序列，含 T<2 跳过也不错位）。逐 demo 算 `subgoal_z` 后传给 `_demo_to_transitions`：

```python
def build_libero_offline_buffer(offline_rb, *, lerobot_root, suite, task_id,
                                action_scaler, state_standardizer,
                                base_policy, base_mode, base_device="cpu",
                                image_size=84, num_demos=None,
                                subgoal=None, way_steps=25, feat_seqs=None) -> None:
    ...
    if subgoal is not None and feat_seqs is None:
        raise ValueError("subgoal 离线子目标需 feat_seqs(pi0_feat 缓存 per-demo 标准化序列)")
    for i, p in enumerate(paths):
        demo = read_libero_demo(p)
        T = demo["state"].shape[0]
        if T < 2:
            print(f"[libero-offline] 跳过 {p}(T<2)")
            continue
        base_actions = (... if base_mode == "base_policy" else None)
        subgoal_z = None
        if subgoal is not None:
            seq = torch.as_tensor(feat_seqs[i], dtype=torch.float32)   # (T, 2056) 已标准化
            assert seq.shape[0] == T, \
                f"pi0_feat 缓存帧数 {seq.shape[0]} != demo {T} (ep#{i}, {p})"
            way = np.minimum(np.arange(T) + way_steps, T - 1)
            subgoal_z = subgoal.subgoal_waypoint(seq, seq[way]).cpu()  # (T, rep_dim)
        for td in _demo_to_transitions(demo, action_scaler=action_scaler,
                                       state_standardizer=state_standardizer,
                                       base_actions=base_actions, image_size=image_size,
                                       subgoal_z=subgoal_z):
            offline_rb.add(td.unsqueeze(0))
```

### 改动 3：`train_chunk_residual.py` — 传参 + 缓存签名 + 变量可见

**3a. 初始化**（line ~712，`_offline_act_feat_seqs = None` 旁）：加 `_pi0_seqs = None`。pi0_feat 分支（line ~752）已有 `_pi0_seqs, _pi0_stats, _pi0_cache_sig = load_pi0_feat_cache(...)`，把该赋值的左值与新初始化同名（确认是 `_pi0_seqs`，无需改分支，仅补 None 初始化使非 pi0_feat 分支也有定义）。

**3b. libero 离线构建调用**（line ~850）：加传参

```python
build_libero_offline_buffer(
    offline_rb, lerobot_root=lerobot_root,
    suite=args.libero_suite, task_id=args.libero_task_id,
    action_scaler=action_scaler, state_standardizer=state_standardizer,
    base_policy=base_policy, base_mode=args.offline_base_mode,
    base_device=args.device, image_size=img_h, num_demos=args.offline_num_demos,
    subgoal=subgoal, way_steps=args.subgoal_way_steps, feat_seqs=_pi0_seqs)
```

注：`subgoal` 仅在 `subgoal_conditioned` 时非 None；`_pi0_seqs` 仅在 pi0_feat 分支非 None。非 pi0_feat 的 subgoal（eef_piece/act_feat）+ libero 当前无此路径需求，`feat_seqs=None` 会触发改动 2b 的 ValueError（明确报错好过静默丢 subgoal）——属预期防护。

**3c. `_libero_offline_signature`**：subgoal_conditioned 时纳入 subgoal 配置，使换 gc_value/high_actor/way_steps 时缓存自动失效重建（防串用）。在 return dict 末尾按条件并入：

```python
def _libero_offline_signature(args, image_keys, offline_cap, image_size):
    sig = { ...现有字段... }
    if getattr(args, "subgoal_conditioned", False):
        sig["subgoal"] = True
        sig["gc_value_ckpt"] = os.path.abspath(args.gc_value_ckpt)
        sig["high_actor_ckpt"] = os.path.abspath(args.high_actor_ckpt)
        sig["subgoal_way_steps"] = int(args.subgoal_way_steps)
    return sig
```

非 subgoal 的 libero run 签名不变 → 既有 libero offcache 不失效。

## 同源不变量（实现/审查须逐条守住）

1. **demo 同序**：`build_libero_offline_buffer` 与 pi0_feat 缓存均用 `find_demo_episodes(lerobot_root, language)`（episode_index 升序）。`enumerate(paths)` 第 i 个 demo ↔ `feat_seqs[i]`。
2. **帧数对齐**：`feat_seqs[i].shape[0] == T`（assert，镜像 act_feat）。
3. **标准化态**：`feat_seqs` 已用 gc_value ckpt 的 feat_stats 标准化；`subgoal_waypoint` 吃标准化态（同 dexmg s30/s530），直接喂。
4. **way_steps 一致**：`args.subgoal_way_steps`（=high_actor 训练 way_steps=参照 25）。
5. **T<2 跳过安全**：index i 跟 paths 位置，跳过某 demo 不使用其 `feat_seqs[i]`，后续 index 不错位。
6. **num_demos 覆盖**：`offline_num_demos` 截断时 `paths[:n]`，要求 `len(feat_seqs) >= n`（i<n 同序）；不满足应 IndexError，由 2 的 assert 链上游 `feat_seqs[i]` 触发。
7. **离线/在线同 φ**：离线 `subgoal_waypoint=vf.phi`、在线 `subgoal_online` 也走同一 gc_value 的 φ，同源。

## 错误处理

- `subgoal is not None and feat_seqs is None` → `ValueError`（改动 2b）。
- `feat_seqs[i].shape[0] != T` → `AssertionError`（改动 2b）。
- `subgoal_waypoint` 维度不符 → 既有 line 140 `AssertionError`（state_dim 校验，通用，无需改）。

## 测试

1. **单元（`subgoal_waypoint` 放开）**：构造 pi0_feat 态的假 `HiqlSubgoal`（vf.state_dim=2056、rep_dim=10），喂 2056 维标准化 base/target → 返回 (B, 10)、finite、不抛 assert。
2. **单元（`build_libero_offline_buffer` 存 subgoal）**：假 subgoal（`subgoal_waypoint` 返回固定 rep_dim z）+ 2 条小 feat_seqs（与 stub demo 帧数一致）→ 灌入的 offline_rb 每 transition 的 `obs` 与 `next.obs` 均含 `observation.subgoal`、维=rep_dim、finite；curr 的 subgoal==z[t]、nxt==z[t+1]。
3. **单元（feat_seqs 缺失防护）**：`subgoal` 给但 `feat_seqs=None` → `ValueError`；`feat_seqs[i]` 帧数与 demo T 不符 → `AssertionError`。
4. **回归**：现有 eef_piece/act_feat/dexmg 离线测试不变；libero 无 subgoal 路（`subgoal=None`）逐位等价（新参数默认 None，`_obs` 不加键、签名不变）。
5. **集成 smoke（实现后人工跑，不入 pytest）**：`--smoke` rollout（libero task8 + subgoal_conditioned + offline_fraction=0.5 + pi0_feat ckpts）→ 离线 buffer 构建带 subgoal 落盘 + 2 步在线注入不报错；确认离线/在线 `observation.subgoal` 维度一致。

## 不做（YAGNI）

- dexmg + pi0_feat 离线子目标（无 base=pi0 的 dexmg 任务）。
- 非 fixed-waypoint 的离线 z（clamp_to_goal 等）——离线统一 `min(t+way, T-1)` 真航点，与 dexmg 现状一致。
- 改在线子目标路径（已实现且本设计不触碰）。
