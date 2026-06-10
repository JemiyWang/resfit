# 设计：HIQL value-loss 对齐(per-critic TD 目标 + adv 门控 expectile)

日期：2026-06-10
分支：chunk-residual-validation
关联文档：
- `docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md`(分层路总设计)。
- `docs/superpowers/plans/2026-06-08-hiql-hierarchy-residual-phase1-gc-value.md`(Phase 1,已过 gate,产 `gc_value_geom.pt`)。
- `docs/superpowers/specs/2026-06-10-hiql-alignment-deltas-design.md`(对齐三改;本 spec 即把其 §7 遗留项"expectile 权重符号"提为正式改动,但**独立成 spec/plan**,不并入三改 A/B 矩阵)。
- 参考实现 HIQL:`/mnt/mnt/data/wjm/residual/HIQL/src/agents/hiql.py`(`compute_value_loss` + `expectile_loss`)、`src/gc_dataset.py`、`src/special_networks.py`。
- memory:`project_resfit_hiql_hierarchy_progress`、`project_resfit_hiql_alignment_deltas`。

## 1. 背景与动机

resfit Phase 1 的 `train_gc_value`(`hiql_gc_value.py`)与参考 HIQL `compute_value_loss` 在 value 损失上有两处实质偏离,合称"value-loss 偏离",已在对齐三改 spec §7 记为遗留:

- **(甲)双 critic 的 TD 目标**:HIQL 让 `v1` 回归 `q1 = r + γ·mask·nv1`、`v2` 回归 `q2 = r + γ·mask·nv2`(**各回归各的、不取 min**),两个 critic 始终保持多样性,`min(v1,v2)` 才是有意义的保守 ensemble。resfit 现状两路都回归同一个 `y = r + γ·mask·min(nv1,nv2)`,两 critic 会越学越像 → `min(v1,v2)` 退化成单 critic。
- **(乙)expectile 门控依据**:HIQL `expectile_loss(adv, diff)` 是**两参**——权重看 `adv = q − V_target` 的符号(`q` 里 next 取 `min`、当前态 V 走 **target 网**、两 critic 共享一个 adv),平方的是 `q_i − v_i`(online)。resfit 复用的是单参 `expectile_loss(diff)`,直接用残差 `y − v_i` 自门控。因 resfit 的 `y` 恰等于 HIQL 的 `q`,(乙)的实际差异在"门控基线用 online 的 `v_i` 还是 target 的 `v_t`",偏稳定性。

本 spec 把(甲)+(乙)对齐到 HIQL `compute_value_loss`,做成 flag-gated A/B,默认 = 现状逐位等价,验证只到离线 gate 层。

> 注:这里说的"完整复刻"指(甲)per-critic 目标 +(乙)adv 门控两处;**mask 不在范围内**——HIQL `mask=1−success`,resfit 现状 `mask=(1−success)·(1−done)` 保持不变(见 §2 非目标)。故 `'hiql'` 模式 ≠ 字节级照搬 HIQL `compute_value_loss`,而是"对齐 expectile + 双 critic、沿用 resfit 的 done-mask"。

> action-free value 场景下,(甲)的双 critic 退化不算"错"(无对动作取 max,高估压力弱),但忠实对齐参考、且能恢复 ensemble 多样性,是本轮明确目标(用户已确认完整复刻)。

## 2. 目标 / 非目标

目标：
- `train_gc_value` 的 value 损失增 `value_loss_mode`(默认 `'shared_min'` = 现状逐位等价;`'hiql'` = 完整复刻 HIQL `compute_value_loss`)。
- CLI `--value_loss_mode`,贴项目"加 flag、老行为默认、逐条 A/B"惯例(参照 `--goal_future_mode`、`--use_layer_norm`、`--target_mode`)。
- ckpt schema 向后兼容:`save_gc_value` 记 `value_loss_mode`(provenance),`load_gc_value` 旧档回退 `'shared_min'`。
- 重训重验产 `_geom_hiqlvloss` 两件套,与现 `_geom` 基线出 A/B 对照证据(含 v1/v2 分化度诊断)。

非目标(本轮不做)：
- **done-mask**(#2):resfit `mask=(1−success)·(1−done)` vs HIQL `mask=1−success`。本轮 scope 收紧到 expectile+双 critic,mask 维持现状。
- **优势 min vs mean**(高层 actor `hiql_high_actor.py` 的 `adv=min(vw)−min(vs)`):属对齐三改 §7 另一条,不在此 spec。
- **concat-φ vs rep_type='state'**:resfit φ 现状更合理,不回退(对齐三改 §7 已记)。
- 在线 rollout 评测(验证只到离线 gate)。
- 改动 `hiql_value.py` 的单参 `expectile_loss`(单任务 ③a 仍用,保持正交)。

## 3. 关键设计事实(代码已核实)

- **resfit 现状**(`hiql_gc_value.py:178-201`):
  ```
  nv1, nv2 = target(s_next, g);  nv = torch.minimum(nv1, nv2)
  y = reward + gamma * mask * nv
  v1, v2 = model(s, g)
  loss = expectile_loss(y - v1, expectile) + expectile_loss(y - v2, expectile)
  ```
  其中 `reward = success - 1.0`、`mask = (1.0 - success) * (1.0 - done_all[b])`。
- **resfit 复用的单参 expectile**(`hiql_value.py:13-20`):`weight = where(diff < 0, 1-tau, tau); return (weight*diff²).mean()`。边界:`diff>=0` → `tau`。
- **HIQL `compute_value_loss`**(`hiql.py:85-105`):
  ```
  masks = 1 - rewards;  rewards = rewards - 1.0            # masks=1-success, rewards=success-1
  next_v1, next_v2 = target_value(s', g);  next_v = min(next_v1, next_v2)
  q = rewards + discount * masks * next_v
  v1_t, v2_t = target_value(s, g);  v_t = (v1_t + v2_t)/2
  adv = q - v_t
  q1 = rewards + discount*masks*next_v1;  q2 = rewards + discount*masks*next_v2
  v1, v2 = value(s, g)                                     # online
  loss = expectile_loss(adv, q1-v1, tau).mean() + expectile_loss(adv, q2-v2, tau).mean()
  ```
- **HIQL 两参 expectile**(`hiql.py:20-22`):`weight = where(adv >= 0, tau, 1-tau); return weight * diff²`。边界:`adv>=0` → `tau`。
- 两边 reward(0/−1)、γ、tau、EMA target、next 取 min,均一致;差异仅在"每 critic 回归目标"与"门控依据/基线网"。

## 4. 设计

### 4.1 新增模块内两参 expectile

放 `hiql_gc_value.py`(不改 `hiql_value.py`):

```python
def expectile_loss_weighted(adv, diff, expectile):
    """两参 expectile(逐字照 HIQL hiql.py:20-22):门控看 adv 符号,平方的是 diff。

    weight = tau      if adv >= 0      # adv=q-V_target,正=该转移是赚的→把 V 往上拉
    weight = 1 - tau  if adv <  0
    返回标量(.mean())。
    """
    weight = torch.where(adv >= 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()
```

### 4.2 `train_gc_value` 增 `value_loss_mode`

签名加 `value_loss_mode='shared_min'`(keyword,默认现状)。训练循环内按模式分两支:

- **`'shared_min'`(默认,代码原样保留)**:与现状字节一致——
  ```python
  nv1, nv2 = target(s_next, g)
  nv = torch.minimum(nv1, nv2)
  y = reward + gamma * mask * nv
  v1, v2 = model(s, g)
  loss = expectile_loss(y - v1, expectile) + expectile_loss(y - v2, expectile)
  ```
- **`'hiql'`(新)**:
  ```python
  with torch.no_grad():                   # 目标量全程 detach,只当常数/门控权重
      nv1, nv2 = target(s_next, g)
      nv = torch.minimum(nv1, nv2)
      q = reward + gamma * mask * nv
      v1t, v2t = target(s, g)             # 乙:当前态 V 走 target 网
      v_t = 0.5 * (v1t + v2t)
      adv = q - v_t
      q1 = reward + gamma * mask * nv1    # 甲:per-critic 目标,不取 min
      q2 = reward + gamma * mask * nv2
  v1, v2 = model(s, g)                     # online,带梯度(在 no_grad 块外)
  loss = expectile_loss_weighted(adv, q1 - v1, expectile) \
       + expectile_loss_weighted(adv, q2 - v2, expectile)
  ```
  **no_grad 范围**:`nv*`、`q`、`v_t`、`adv`、`q1`、`q2` 全在块内(detach);`v1,v2 = model(s,g)` 在块外带梯度。`q_i − v_i` 中 `q_i` 是常数、`v_i` 带梯度 → 梯度只从 `v_i` 流入;`adv` 仅作权重不回传(与 HIQL `target_value` 算 adv、本就无梯度一致)。`reward`、`mask` 与现状同(mask 含 `(1-done)`,本轮不动)。

**★ 等价性关键(RNG 顺序)**:两分支共用前面的 `b = rng.integers(...)` 与 `gi = sample_gc_goals(..., rng, ...)`;`'hiql'` 分支新增的 `target(s,g)` 前向**不消耗 rng**。default 分支语句一字不改 → 同 seed 与现状逐位等价。实现时 default 分支必须是"现状原文",新逻辑只进 `'hiql'` 分支。

### 4.3 CLI + schema

- `train_hiql_gc_value.py`:加 `--value_loss_mode`(`choices=['shared_min','hiql']`,默认 `'shared_min'`),透传给 `train_gc_value`;启动日志 echo(并入现 `goal_future_mode=.. use_layer_norm=..` 那行)。
- `save_gc_value`:payload 加 `"value_loss_mode"`(默认参数 `value_loss_mode='shared_min'`)。**纯 provenance**——只影响训练、不影响推理;模型结构不变,.pt shape 不变。
- `load_gc_value`:`info["value_loss_mode"] = ckpt.get("value_loss_mode", "shared_min")`,旧档回退现状。

### 4.4 级联与产物(留用户在本机跑,A/B)

改的是损失、非网络结构,但训出的权重不同 → 级联同对齐三改①:
- 重训 value:`--goal_future_mode geometric --value_loss_mode hiql` → `outputs_chunk/three_piece_gc_value_geom_hiqlvloss.pt`。
- 级联重训 high_actor(挂新 value,`--target_mode` 取默认 `fixed_waypoint` 隔离单变量)→ `outputs_chunk/three_piece_high_actor_geom_hiqlvloss.pt`(务必传 `--state30_cache outputs_chunk/three_piece_state30.npz` 免 MuJoCo 回放)。
- 基线 `_geom` 两件套不删,作对照。

## 5. 验证(只到离线 gate)

基线 = 现 `_geom` 两件套。新臂 `_geom_hiqlvloss` 单开 vs 基线:

- **`verify_gc_value.py`**(沿用现三诊断 + 新增一项):
  - 状态侧 `spearman(t, V(s,g=末态))`(期望仍 ≈0.99,不退化);
  - 目标侧 `spearman(g距离, V(起始,g))`(期望仍 ≈−0.98);
  - `V(s,g=s)` 自指(期望 ≈0,长尾不劣于 geom 基线);
  - **★ 新增 v1/v2 分化度诊断**:在真 demo 状态上算 `corr(v1, v2)` 与 `mean|v1−v2|`,对比 geom 基线。预期 `'hiql'` 模式两 critic 更分化(per-critic 目标拉开),量化本次改动的核心收益。无独立"PASS 阈值",作 A/B 对照数值报告。
- **`verify_high_actor.py`**(判据不变):在新 value 上重训的 high_actor,前向步 median≈25、前向占比高、塌末态≈0、off/expected≈1.0,确认 value-loss 改动未破坏高层。

把 A/B 数值与图落 `outputs_chunk/`,结论写回本 spec 同名 plan 的 gate 段(带日期、产物名)。全程无在线 rollout。

## 6. 接口与单测清单

- `hiql_gc_value.py`:
  - `expectile_loss_weighted(adv, diff, expectile)`(新)。
  - `train_gc_value(..., value_loss_mode='shared_min')`。
  - `save_gc_value(..., value_loss_mode='shared_min')` / `load_gc_value` info 带 `value_loss_mode`。
- `train_hiql_gc_value.py`:`--value_loss_mode {shared_min,hiql}`。
- 单测(`tests/test_hiql_gc_value.py` 追加):
  1. **`expectile_loss_weighted` 手算对拍**:构造 adv(含正、负、0)与 diff,验证 `adv>=0` 用 `tau`、`adv<0` 用 `1-tau`,门控用 adv、平方用 diff(与现状单参 `expectile_loss` 在 `adv==diff` 时一致、在 `adv≠diff` 时不同)。
  2. **默认逐位等价**:同 `data`/seed,`value_loss_mode='shared_min'` 训出的 `state_dict` 与不传该参(现状)逐张量 `allclose(atol=0)`。
  3. **`'hiql'` 学到单调 V**:仿 `test_train_gc_value_learns_progress`,一维进度链上 `V(s,g=末态)` 末段均值 > 起段均值,`v_stats['max']>v_stats['min']`。
  4. **save/load 往返**:存 `value_loss_mode='hiql'` → 读回 `info['value_loss_mode']=='hiql'`;旧档(payload 无该键)→ 回退 `'shared_min'`。

## 7. 风险

- **默认等价被污染**:`'hiql'` 分支若不慎插入 rng 调用、或改了 default 分支语句,基线 A/B 失真。缓解:default 分支保现状原文 + 单测 2 逐张量 `atol=0`。
- **adv 误带梯度**:`adv`/`v_t`/`q1`/`q2` 里的 `target(...)` 与中间量须全程 `no_grad`(只 `q_i - v_i` 的 `v_i` 走 online 带梯度)。缓解:单测 3 跑通即说明可训(梯度只从 `v_i` 流入);代码 review 确认 `no_grad` 包裹范围。
- **双 critic 仍部分趋同**:共享 adv 门控下,若 nv1/nv2 初始差异小,分化可能有限。缓解:验证靠 v1/v2 分化度诊断量化,而非假设;若分化弱,留作"已知收益有限"记录,不追加改动。
- **级联成本**:重训 value + high_actor + 两 verify。缓解:high_actor 用默认 fixed_waypoint 隔离单变量;`_geom` 基线保留不删。
