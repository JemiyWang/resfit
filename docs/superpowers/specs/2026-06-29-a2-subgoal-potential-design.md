# A2:子目标条件 V(s,z) 作为 PBS 势函数(goal-conditioned subgoal potential)

- 日期:2026-06-29
- 状态:已评审,待实现(转 writing-plans)
- 分支:chunk-residual-validation

## 1. 背景与动机

resfit 的底层策略是**在线残差 TD3**,reward = 环境稀疏 reward + 可选 stage shaping。
现在的 reward shaping 有三种(`chunk_env_wrapper.shaping_reward`):`none` / `staged` / `potential`。
`potential` 模式下,势函数 Φ 的来源由 `--potential_source` 决定:

- `stage`:Φ = 整数 stage(里程碑)。
- `hiql`:Φ = **单状态** value `V(s)`(`hiql_value.ValueMLP`,只吃当前 state,无 goal),
  ③a 路线 A。

HIQL 低层策略判断动作好坏用的是 **goal-conditioned** 的 `V(s', 子目标) − V(s, 子目标)`,
其中 value 是 `V(s, g)`,高层低层共用、只是喂的 goal 不同(高层喂最终目标,低层喂子目标)。

**A2 的目标**:把 `potential` 模式新增一种 goal-conditioned 来源——
势函数 Φ(s) = `V_gc(s, z)`,其中 `V_gc` 是 resfit 已有的 goal-conditioned value
(`hiql_gc_value.GoalConditionedVF`,分层路 Phase 1 产物 `gc_value.pt`),
`z` 是 high_actor 在线/离线产出的**子目标 rep**。shaping 仍走标准 PBS:

```
Φ(s) = V_gc(s, z) · scale
F     = bonus · ( γ·Φ(s') − Φ(s) )        # done 时 Φ(s')=0
```

直觉:给残差 TD3 一个"朝当前子目标 z 推进就加分"的稠密、goal-conditioned 引导。

### 非目标(明确不做)

- **不替换 TD3 actor**。A2 只动 reward shaping;低层仍是 TD3(∇_a Q 寻优 + 探索),
  保留"在线残差突破 base policy"的能力。用 `V(s',z)−V(s,z)` 当 AWR advantage 直接驱动
  actor(方式 B)被否决:那是离线模仿型范式,会削弱在线突破能力,且与已有 critic 冗余。
- 不改 `gc_value` / `high_actor` 的训练;A2 只在主训阶段**加载并冻结**它们用于算 Φ。

### 与现状的关系:resfit 有两个 value function

| | 现状 `potential_source=hiql` | A2 `potential_source=hiql_subgoal` |
|---|---|---|
| value | `ValueMLP`(`hiql_value.py`),单状态 `V(s)` | `GoalConditionedVF`(`hiql_gc_value.py`),`V(s,g)` |
| 训练脚本 | `train_hiql_value.py`(③a) | `train_hiql_gc_value.py`(Phase 1) |
| 主训 ckpt 参数 | `--hiql_value_ckpt` | `--gc_value_ckpt`(经 `--subgoal_conditioned`) |
| 喂 goal | 否(隐含固定=轨迹终点) | 是(显式喂子目标 rep z) |

> 注意区分同目录下 `2026-06-28-actfeat-hiql-potential-design.md`(那是 act_feat 的**单状态**
> Φ=V(s),路线 A);本设计是 goal-conditioned 的 Φ=V(s,z),不同的 value、不同的 source。

## 2. 定位与依赖

A2 要算 `V_gc(s, z)`,需要 `gc_value` + `high_actor` + 在线/离线 z——这三样正是
`--subgoal_conditioned` 管线已提供的(`HiqlSubgoal` 持有 `self.vf=gc_value`、`self.ha=high_actor`;
在线 `compute_online_subgoal`、离线 `subgoal_waypoint`)。

**设计决定:A2 建立在 `--subgoal_conditioned` 之上。** 开 A2 必须同时开 subgoal_conditioned。
后果:同一个 z **同时**

- (a) concat 进 obs 当 condition(现状行为,不变);
- (b) 新增用于 `V(s,z)` reward shaping(A2 新增)。

**A/B 隔离**:实验组 = subgoal_conditioned + A2 shaping;对照组 = 同样 subgoal_conditioned 但不开
此 shaping。被隔离的唯一变量是 "V(s,z) shaping",z-concat 是两组共同基线。

**范围**:在线 + 离线 demo buffer 都做 shaping,保证同一 buffer 内 reward 口径一致
(与现状 `potential_source=hiql` 一致)。

## 3. 同源命门(正确性前提)

1. **z 与训练 rep 同空间**:`gc_value` 的 goal_encoder 把 rep 归一化到半径 `sqrt(rep_dim)`
   (`hiql_gc_value.py:59`);high_actor 的 z 在 `renorm_subgoal=True` 下也归一化到 `sqrt(rep_dim)`
   (`hiql_subgoal.py:137`)。两者在同一球面 → 喂 value 的 z 不 OOD。
   **因此 A2 必须断言 `renorm_subgoal=True`**(现已是默认)。
2. **scale 自洽**:scale 复用 `gc_value.pt` 内的 `v_stats`(min/max,`save_gc_value` 已存、
   `load_gc_value` info 已读),`auto_scale = (num_stages-1)/(v_max-v_min)`,口径同 `HiqlPotential.from_ckpt`。
3. **PBS 单步合法**:一个 chunk 的 `phi_start` 与 `phi_next` 必须用**同一个 z**(chunk 起点算的
   `z_start`)。跨 chunk z 随 `compute_online_subgoal` 滚动更新(与 obs 里 concat 的 z 完全一致);
   跨 chunk 边界 telescoping 不严格闭合,这是已知近似(同 HIQL 跨 segment 换子目标),可接受。

## 4. 组件改动(5 处)

### ① `hiql_gc_value.py`:`GoalConditionedVF.value_from_rep(s, z)`
现状 `forward(s,g)` 先 `goal_encoder(g,s)` 把目标**状态**编码成 rep,再 `cat[s,rep]→v1/v2`。
A2 的 z 已是 rep,新增一条**跳过 goal_encoder** 的前向:`cat[s, z] → (v1, v2)`。
value MLP 输入维度本就是 `state_dim+rep_dim`(`line 84`),无结构改动。Φ 取 `(v1+v2)/2`(mean,
与 HIQL 低层 advantage 口径一致)。

### ② `hiql_potential.py`:`GcSubgoalPotential`
类比 `HiqlPotential`:持有冻结 gc_value(复用 `HiqlSubgoal.vf`,已 eval/冻结)+ scale。
`phi(state_std, z) = value_from_rep(s, z).mean · scale`。
与 `HiqlPotential` 唯一区别:phi 第二参是 z(rep),不是 rel_piece。

### ③ `chunk_env_wrapper.py`:在线接入
wrapper 额外持有 subgoal,在 **chunk 起点**算一次 `z_start`(=`compute_online_subgoal` 的结果,
与 obs 里 concat 的 z 同一个):
```
phi_start = potential.phi(s_start, z_start)
phi_next  = potential.phi(s_end,   z_start)   # 同一个 z_start
F = potential_shaping(phi_start, phi_next, bonus, gamma, done)   # done 时 phi_next=0
```
时序模式照搬现状 `_start_phi_state`(reset 存、step 末为下个 chunk 更新)。

### ④ `offline_stage_replay.py`:离线接入
离线已为 subgoal 算好逐帧 hindsight z(`subgoal_waypoint→subgoal_z[t]`,rep)。
A2 把第 t 个 transition 的 shaping 用 `z_t`:`V(s_t,z_t)` 与 `V(s_{t+1},z_t)` 烤进 `reward[t]`。
复用现成 `shape_offline_rewards`,potential 换成 `GcSubgoalPotential` 并喂逐帧 z。

### ⑤ 开关 / 签名 / 兼容
- 新 `--potential_source hiql_subgoal`,断言:`--reward_shaping potential` +
  `--subgoal_conditioned` + `renorm_subgoal=True`。
- 离线 buffer 缓存签名纳入 `potential_source=hiql_subgoal` + `scale`
  (`gc_value_ckpt`/`subgoal_way_steps` 已在签名里)→ 改 ckpt/scale 自动失效重建。
- 默认 `stage`(potential=None)逐位等价;现有 `potential_source=hiql` 单状态路径不受影响。

## 5. 测试策略

### 单元测试(TDD,先写测试)
- **核心等价性**:对任意目标状态 g,`value_from_rep(s, goal_encoder(g,s))` 必须 `== forward(s, g)`。
  证明"跳过 goal_encoder 直接喂 rep"与正常路径逐位一致。
- `GcSubgoalPotential.phi(s,z) == value_from_rep(s,z).mean · scale`;scale 由 v_stats 算对。
- PBS 组合:`F = bonus·(γΦ'−Φ)`,done 时 Φ'=0。
- 单步同 z(wrapper 层):一个 chunk 的 phi_start/phi_next 用同一个 z_start。
- 在线 z 时序:reset/step 里 z_start 在 chunk 起点算、chunk 末更新。
- 离线:第 t 帧 shaping 用 z_t(start=s_t, next=s_{t+1})。
- 缓存签名:`potential_source=hiql_subgoal`+`scale` 进签名,改 ckpt/scale 失效重建。
- 断言:`renorm_subgoal=False` / 未开 `subgoal_conditioned` / `reward_shaping≠potential` 各报错。

### 回归 / 兼容
默认 `stage` 逐位等价;现有 `potential_source=hiql` 不受影响;全量 pytest 零回归(基线 400+ passed)。

### 真实端到端 smoke(分阶段)
挑一个已有 subgoal_conditioned 跑过的任务(如 `three_piece_hiqlv512_sg15` 或 threading),
复用其 `gc_value`/`high_actor` ckpt + cache,短步数 smoke 确认:在线 reward 含非零且量级合理的
shaping、离线 buffer 烤 shaping 成功(等 `buffer_meta` 哨兵)、训练不崩。

## 6. A/B 实验

对照组与实验组**除 shaping 外完全相同**(同任务、同 ckpt/cache、同 z-concat)。
主对照隔离 "V(s,z) shaping" 这一个变量:

| | 配置 | z-concat | V(s,z) shaping |
|---|---|---|---|
| 主对照 B0 | subgoal_conditioned + `--reward_shaping none` | 是 | 否(无任何 shaping) |
| 实验 A | subgoal_conditioned + `potential_source=hiql_subgoal` | 是 | 是 |

- 主对照 B0(关 shaping)回答"V(s,z) shaping 有没有增量";若另想回答"V(s,z) shaping vs
  stage shaping 谁更好",再加一组 B1 = `potential_source=stage`(可选)。
- 复用已有 ckpt/cache;bonus 沿用 1.0;scale 用 `auto_scale·phi_scale`(phi_scale 默认 1.0,
  需要时扫)。
- 指标:eval success 曲线 + 训练稳定性,对比 B0(必要时再对 B1)。

## 7. 落码顺序(给 writing-plans)

1. `value_from_rep` + 核心等价性测试
2. `GcSubgoalPotential` + scale(v_stats)
3. wrapper 在线接入 + z 时序/单步同 z 测试
4. `offline_stage_replay` 离线接入
5. 开关 / 签名 / 断言
6. 回归 + 分阶段 smoke
7. A/B 跑(用户授权后跑)

每个 flag 默认等价,先落码+测试,再重训/跑生效。
