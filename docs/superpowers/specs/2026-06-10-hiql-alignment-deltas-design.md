# 设计：HIQL 对齐三改(value LayerNorm · 高层 clamp-to-goal · eval 子目标重归一化)

日期：2026-06-10
分支：chunk-residual-validation
关联文档：
- `docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md`(分层路总设计:gc value + AWR 高层 + 子目标条件残差低层)。
- `docs/superpowers/plans/2026-06-08-hiql-hierarchy-residual-phase1-gc-value.md`(Phase 1,已过 gate,产 `gc_value_geom.pt`)。
- `docs/superpowers/plans/2026-06-08-hiql-hierarchy-residual-phase2-high-actor.md`(Phase 2,已过 gate,产 `high_actor_geom.pt`)。
- `docs/superpowers/plans/2026-06-08-hiql-hierarchy-residual-phase3-lowlevel-wiring.md`(Phase 3,低层接线,未实现)。
- 参考实现 HIQL:`/mnt/mnt/data/HIQL/src/agents/hiql.py`、`src/special_networks.py`、`src/gc_dataset.py`、`jaxrl_m/evaluation.py`。
- memory:`project_resfit_hiql_hierarchy_progress`、`project_resfit_hiql_phi_is_not_hiql`、`project_resfit_object_aware_value`。

## 1. 背景与动机

把 resfit 已实现的分层路(`hiql_gc_value.py` Phase 1 + `hiql_high_actor.py` Phase 2 + `hiql_subgoal.py` Phase 3 helper)逐行对照参考项目 HIQL 后,发现三处对结果可能有实质影响、且修法明确的偏离。本 spec 只收这三处,作 flag-gated A/B,老行为默认,验证只到离线 gate 层(Phase 3 在线接线尚未实现,在线成功率比较留到 Phase 3 plan)。

三处(详见 §4):
1. **value 缺 LayerNorm**:HIQL 的 value/rep 用 `LayerNormMLP`(GELU + LayerNorm),论文明确强调 LayerNorm 对 goal-conditioned value 的稳定关键;resfit 的 `_mlp` 是裸 `Linear+ReLU`。
2. **高层 AWR 回归目标恒为固定 +k 航点**:resfit `wi=min(si+way,last)` 只截到 demo 末;HIQL `high_target=min(indx+way, high_goal)` 把航点 clamp 到当前采样 goal,近 goal 时子目标收敛到 goal。这正是 Phase 2 gate ⚠️ 自陈的局限。
3. **eval 子目标不重归一化**:HIQL rollout 把高层采样的 z 重投到半径 sqrt(rep_dim)(`evaluation.py:114`)再喂低层;resfit `subgoal_online` 取 `.mean` 直接喂,靠 z 范数≈sqrt(rep) 碰巧兜着。

## 2. 目标 / 非目标

目标:
- 三处改动各落一个 flag,**默认 = 现状逐位等价**,可单独 A/B 也可叠加,贴项目"加 flag、老行为默认、逐条 A/B"惯例(参照 `--goal_future_mode`、`--offline_base_mode`)。
- 每处改动重跑它对应的离线 verify_*(gate),与现 `_geom` 基线出 A/B 对照证据。
- ckpt schema 向后兼容:旧档无新键时 `load_*` 回退到老行为。

非目标(本轮不做,§7 仅作"已知遗留 HIQL 差异"记录):
- min vs mean 优势聚合、concat-relative φ vs HIQL 默认 goal-only(rep_type='state')、expectile 权重符号(残差 vs target-优势)。
- 低层换 HIQL AWR actor(分层路总设计已定:低层保留残差 off-policy RL)。
- 高层每步 receding-horizon 查询节奏、Phase 3 在线接线与在线成功率 A/B。
- 在线 rollout 评测(本轮验证只到离线 gate)。

## 3. 关键设计事实(代码已核实)

- **Phase 1 value**(`hiql_gc_value.py`):`_mlp` = `Linear→ReLU` ×2 + `Linear`,无 LN。`GoalConditionedVF` 含 `RelativeGoalEncoder`(concat[g,s]→MLP→rep_dim,归一化到 sqrt(rep_dim))+ 双 critic `v1/v2`。`save_gc_value`/`load_gc_value` 存/重建维度。EMA target、expectile 0.7、geom goal 混采(`future_mode='geometric'`,与 HIQL `geom_sample=1` 逐字一致)。
- **Phase 2 high actor**(`hiql_high_actor.py`):`train_high_actor` 里 `wi=np.minimum(si+way_steps, last_arr[b])`;goal 复用 `sample_gc_goals`(含 p_curr=0.2/p_traj=0.5/p_rand=0.3)。优势用 `min(v1,v2)`(本轮不改,§7)。回归目标 `z_tgt=vf.phi(s,sw)`。
- **Phase 3 helper**(`hiql_subgoal.py`):`subgoal_online` 返回 `ha(s30,g30).mean`,无重归一化;`HiqlSubgoal` 构造载冻结 gc_value + high_actor。
- **HIQL 参考**:
  - `special_networks.py` `LayerNormMLP`:每隐层 `Dense → GELU → LayerNorm`,输出层不带激活/LN;`RelativeRepresentation` bottleneck 末归一化到 sqrt(rep_dim)。actor/high_actor 是 `Policy`(裸 MLP,**不带 LN**)。
  - `gc_dataset.py` `GCSDataset.sample`:high 段 traj goal 恒用**线性插值** `round(min(indx+1,final)·d + final·(1−d))` ∈ [indx+1, final](**不用 geom**——`geom_sample` 只管低层 value `sample_goals`,故高层 goal 永不命中 current);`high_traj_target = min(indx+way, high_traj_goal)`;random goal 时 `high_random_target = min(indx+way, final)`;`high_p_randomgoal` 默认 0。
  - `evaluation.py:112-117`:`use_waypoints` 时每步 `cur_obs_goal = high_policy_fn(...)`,`use_rep` 下 `cur_obs_goal = cur_obs_goal/‖·‖·sqrt(dim)` 后喂低层。
- **基线产物**:现 `_geom` 两件套 `outputs_chunk/three_piece_gc_value_geom.pt` + `three_piece_high_actor_geom.pt`,均已过各自 gate。

## 4. 设计

### 4.0 总览:flag 与产物

| 文件 | 改动 | flag(默认=老行为) | 新产物 |
|---|---|---|---|
| `hiql_gc_value.py` | `_mlp` 加 `use_layer_norm`;`GoalConditionedVF`/`RelativeGoalEncoder`/`train_gc_value`/`save+load_gc_value` 透传 | `train_hiql_gc_value.py --use_layer_norm 0` | `gc_value_geom_ln.pt` |
| `hiql_high_actor.py` | 新纯函数 `sample_high_goal_target`;`train_high_actor` 加 `target_mode`;`save+load_high_actor` 透传 | `train_hiql_high_actor.py --target_mode fixed_waypoint` | `high_actor_geom_clamp.pt`;级联 `high_actor_geom_ln.pt` |
| `hiql_subgoal.py` | `HiqlSubgoal(..., renorm_subgoal=False)`;`subgoal_online` 末投球面 | 构造参 `renorm_subgoal=False` | 无(运行时) |

三处独立,LayerNorm 那条有级联(见 §4.1)。

### 4.1 改动①:value LayerNorm

`_mlp` 增参:

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

- **施加范围**:`GoalConditionedVF` 的 `v1/v2` 与 `RelativeGoalEncoder.net`(两者都过 `_mlp`)。`use_layer_norm` 经 `GoalConditionedVF(__init__)` → `RelativeGoalEncoder(__init__)` 与两个 value 头透传;`train_gc_value` 增同名参,默认 False。
- **★ 决定(已与用户确认)**:LN 分支同时把 `ReLU→GELU`,忠实 HIQL `LayerNormMLP` 整套配方。代价:A/B 把 "LN+GELU" 绑一起测,不单独拆 GELU。理由:目标是"贴 HIQL 的 value 网",论文稳定性论断针对的就是整套 LayerNormMLP;要纯隔离 LN 可后补第三臂,本轮不做。
- **high_actor 的 `mean_net` 不加 LN**:对齐 HIQL(Policy 是裸 MLP)。
- **schema**:`save_gc_value` payload 加 `"use_layer_norm"`;`load_gc_value` 用 `ckpt.get("use_layer_norm", False)` 重建,旧档回退老行为。
- **级联**:`use_layer_norm=1` 改了 value 网结构 → 必须重训 `gc_value`(产 `gc_value_geom_ln.pt`)→ high_actor 吃冻结 φ,须在该 ln value 上重训(产 `high_actor_geom_ln.pt`,`target_mode` 取默认 fixed_waypoint 以隔离 LN 单一变量)。Phase 1、Phase 2 两个 gate 都要重验。

### 4.2 改动②:高层 AWR target clamp-to-goal

**★ 决定(已与用户确认):clamp-to-goal 与"高层 goal 采样"是一体的 HIQL 机制,不可只 clamp target。** 若仍用 value 的 `sample_gc_goals`(含 0.2 current、0.3 random),goal=current 会把航点 clamp 成原地→教"stay",random goal 跨 demo 无意义。故 `target_mode='clamp_to_goal'` 同时把高层 goal 采样换成 HIQL `GCSDataset` 高段口径。

新增纯函数(逐字照 HIQL `gc_dataset.py` high 段):

```python
def sample_high_goal_target(s_idx, traj_id, last_idx_of, rng, *, way_steps,
                            n_total, high_p_randomgoal=0.0):
    """HIQL GCSDataset 高段:返回 (goal_idx, target_idx)。逐字照 gc_dataset.py:122-135。
    traj goal: 线性插值 round(min(si+1,final)·d + final·(1−d)) ∈ [si+1, final],永不命中 current。
    traj target = min(si+way, traj_goal);random goal(prob high_p_randomgoal)的 target=min(si+way, final)。
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

**★ traj-goal 用线性插值(非 geom)**:忠实 HIQL——其高段 goal 恒线性插值、与 `geom_sample` 无关,且 `min(si+1,·)` 保证 goal ≥ si+1 永不命中 current(若用 geom,offset 可为 0→回退 current,语义坏)。代价:高层 goal 分布与 resfit 已切 geom 的 value 不同源;这是 HIQL 的有意设计(value 与 high-goal 用不同采样),本轮照搬,不引 geom 选项以免增歧义。

`train_high_actor` 增 `target_mode`(默认 `'fixed_waypoint'`)与 `high_p_randomgoal=0.0`:
- `'fixed_waypoint'`(默认):完全保持现状——`gi=sample_gc_goals(...)`、`wi=np.minimum(si+way_steps, last_arr[b])`。**默认分支 RNG 抽取顺序不变,保证与现状同 seed 逐位等价。**
- `'clamp_to_goal'`:`gi, wi = sample_high_goal_target(si, tj, last_idx_of, rng, way_steps=way_steps, n_total=len(states), high_p_randomgoal=high_p_randomgoal)`。其余(优势、`z_tgt=vf.phi(s, sw)`、AWR 权重)不变。

**schema**:`save_high_actor` payload 加 `"target_mode"`、`"high_p_randomgoal"`;`load_high_actor` info 带回(`get` 默认 `'fixed_waypoint'`/0.0)。

### 4.3 改动③:eval 子目标重归一化

`HiqlSubgoal.__init__` 增 `renorm_subgoal=False`(并经 `from_ckpts` 透传);`subgoal_online` 末:

```python
z = self.ha(s30, g30).mean
if self.renorm_subgoal:
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
return z
```

纯运行时、不重训。属 Phase 3 plan 地盘——本 spec 只把开关 + 单测落好,实际接线(谁在 rollout 里传 `renorm_subgoal=True`)交 Phase 3。`subgoal_waypoint`(离线真航点 φ)不动(φ 已在球面上)。

## 5. 验证(只到离线 gate)

基线 = 现 `_geom` 两件套。各臂单开 vs 基线:

- **①LN**:重训 `gc_value_geom_ln.pt`,重跑 `verify_gc_value.py`,与 geom 基线比三诊断——目标侧早期尖坑、`V(s,g=s)` 负尾、状态侧不退化(期望 LN 不劣化、最好更平滑)。再在 ln value 上重训 `high_actor_geom_ln.pt`,复跑 `verify_high_actor.py`(判据不变,确认 LN 未破坏高层)。
- **②clamp**:在现 geom value 上重训 `high_actor_geom_clamp.pt`。`verify_high_actor.py` **判据改写**(见下),复跑确认通过。
- **③renorm**:`tests/test_hiql_subgoal.py` 加单测——`renorm_subgoal=True` 后 `‖z‖≈sqrt(rep_dim)`(atol 1e-4),`False` 时取 `.mean` 原值;`mean` 的范数一般 ≠ sqrt(rep)。无重训。

**②的 verify 判据改写(重要)**:clamp-to-goal **故意打破**当前 gate 的"前向步数恒命中 +25"。新判据:按 goal 距离 `dist=goal_idx−t` 分箱——
- 远 goal(`dist ≥ way_steps`):前向步 `j*−t` median ≈ way_steps(原判据,保留);
- 近 goal(`dist < way_steps`):子目标应收敛到 goal 邻域——前向步 `j*−t` ≈ `dist`,且 `‖z_pred − φ(s,goal)‖ < ‖z_pred − φ(s, s_{t+way})‖`;
- 仍 PASS 条件:前向占比高、塌末态 ~0、off/expected 不退化(沿用现 gate 其余项)。
诊断图:前向步 vs goal 距离散点,应贴 `min(way_steps, dist)` 折线。

每臂把 A/B 数值与图落到 `outputs_chunk/`,并把结论写回各自 plan 的 gate 段(带日期、产物名)。全程无在线 rollout。

## 6. 接口与单测清单

- `hiql_gc_value.py`:`_mlp(use_layer_norm=)`、`GoalConditionedVF(use_layer_norm=)`、`RelativeGoalEncoder(use_layer_norm=)`、`train_gc_value(use_layer_norm=)`、`save/load_gc_value` 带 `use_layer_norm` 键。单测:LN 构造前向 shape 不变 + save/load 往返保 `use_layer_norm`。
- `hiql_high_actor.py`:`sample_high_goal_target(...)`、`train_high_actor(target_mode=, high_p_randomgoal=)`、`save/load_high_actor` 带 `target_mode`/`high_p_randomgoal`。单测:`sample_high_goal_target` 的 `traj_goal ∈ [si+1, final]`(永不命中 current)、`traj_target=min(si+way, goal)`、random 分支占比 ≈ `high_p_randomgoal`;`target_mode='fixed_waypoint'` 与现状逐位等价(同 seed 同输出);save/load 往返。
- `hiql_subgoal.py`:`HiqlSubgoal(renorm_subgoal=)`、`from_ckpts(renorm_subgoal=)`。单测:renorm 球面、默认关。
- 训练 CLI:`train_hiql_gc_value.py --use_layer_norm`、`train_hiql_high_actor.py --target_mode/--high_p_randomgoal`。

## 7. 已知遗留 HIQL 差异(本轮不改,仅记录)

- **优势双 critic 聚合**:resfit high actor 用 `min(v1,v2)`(悲观);HIQL 用 `(v1+v2)/2`(均值)。
- **φ 表征语义**:resfit 恒 `concat[g,s]`(goal-relative-to-state);HIQL 默认 `rep_type='state'`(φ 只编码 goal 本身)。
- **expectile 权重符号**:resfit 权重按残差 `(y−v)` 符号(标准 expectile);HIQL 按 `adv=q−V_target` 符号(IQL 式),online/target 错位。
- **低层**:resfit 用残差 off-policy RL 替 HIQL AWR 低层 actor(分层路总设计如此)。
- **高层查询节奏**:HIQL eval 每步 receding-horizon 重算 z;resfit 节奏属 Phase 3。

## 8. 风险

- **②clamp 改了 gate 含义**:若沿用旧"命中 +25"判据会误判失败。缓解:§5 判据改写 + 分箱诊断,先想清"通过"长什么样再跑。
- **①LN 级联废两个 gate + geom 基线**:重训重验成本最高。缓解:LN 臂用默认 fixed_waypoint 隔离单变量;保留 geom 基线两件套不删,作对照。
- **schema 漂移**:新键须 `get(默认老值)`,旧 ckpt 不可载即回归。缓解:save/load 往返单测覆盖默认与显式两路。
- **fixed_waypoint 等价性**:②的默认分支必须与现状逐位等价(同 seed),否则 A/B 基线被污染。缓解:专门加"同 seed 等价"单测。
