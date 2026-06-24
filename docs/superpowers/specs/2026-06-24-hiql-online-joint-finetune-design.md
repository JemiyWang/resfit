# HIQL 在线联合微调 (V + high_actor) 设计

日期: 2026-06-24
分支: chunk-residual-validation
状态: 设计待用户复核

## 1. 背景与动机

resfit 当前的 HIQL 分层残差是**严格三段流水线**:

1. **Phase 1** `train_hiql_gc_value.py:117-189` → 离线训 goal-conditioned `V(s,g)`(双 critic + φ 编码器,IQL expectile),纯 offline demo,存 `gc_value.pt`。
2. **Phase 2** `train_hiql_high_actor.py:76-149` → AWR 训高层 `π_h(z|s,g)`,纯 offline demo,读冻结 `gc_value.pt`,存 `high_actor.pt`。
3. **Phase 3** `train_chunk_residual.py:555-1033`(主训)→ 在线训残差低层(TD3)。`HiqlSubgoal.from_ckpts()`(`hiql_subgoal.py:28-90`)把 `gc_value` 和 `high_actor` 全部 `requires_grad_(False)` 冻结只读;`q_agent.py:631-687` 的 `update()` 只动 actor/critic,**完全不碰 V 和 high_actor**。

**问题**: V 和 high_actor 只吃了 offline demo,online rollout 一点没回灌给它们。当残差策略在线访问的状态分布偏离离线 demo 几何时,冻结的 V/high_actor 无法适应。

**对照 HIQL** (`/mnt/mnt/data/wjm/residual/HIQL`): 它用 `loss = value_loss + actor_loss + high_actor_loss` 单 loss 单 backward 把三件套联合优化(`src/agents/hiql.py:124-169`),但**全程纯 offline,不碰 online 数据**。所以"像 HIQL 一样联合训"给的是**联合训练的结构**;真正让 V/高层吃到 online 数据的杠杆是**在 Phase 3 不再冻结它们、持续用 online(+offline 混采)更新**。

> 结构差异: resfit 的"低层"是残差 TD3 策略(base policy + 残差,带自己的 Q critic),**不是** HIQL 那个用 AWR + V 训的低层 actor。本设计**保留**残差范式,只把 V 和 high_actor 从"离线冻结"变成"在线联合微调"。

## 2. 目标与非目标

### 目标
- 在 Phase 3 主训 loop 内,解冻 `V(s,g)` 和 `high_actor`,用 online(+offline 混采)数据**持续更新**它们,与残差 TD3 在同一个在线循环里并存。
- Phase 1/2 离线产物保留为 **warm-start 初值**。
- 所有新行为走 flag,**默认 OFF 时与现状逐位等价**(resfit 金标准纪律)。

### 非目标(本 spec 不做)
- **不**解冻 φ goal 编码器(子目标潜空间 z 保持平稳;见 §4.1)。
- **不**把低层换成 HIQL 的 AWR 低层(残差 TD3 不动)。
- **不**把 env 任务奖励引入 V(保持 HIQL 势函数的 goal-conditioned 稀疏 reward 语义;见 §4.4)。
- **不**做课程式混采调度(固定 offline_fraction,YAGNI)。

## 3. 已敲定的三大决策(与用户确认)

1. **在线联合微调**: 保留残差 TD3 低层;Phase 3 内解冻 V 和 high_actor,在线 + offline 混采持续更新。
2. **冻 φ**: φ goal 编码器保持离线学到的几何不动(z 分布平稳);在线只更新 V 的 critic head 和 high_actor。
3. **online + offline 混采**: V/high_actor 的在线更新复用 RLPD 式混采,offline demo 锚住几何、防灾难性遗忘。

## 4. 架构设计

### 4.0 总体数据流(改动后)

```
Phase 1/2 (不变) ──► gc_value.pt / high_actor.pt  ──(warm-start)──┐
                                                                  ▼
Phase 3 主训 loop (每个 update tick):
  ┌─────────────────────────────────────────────────────────────┐
  │ (a) 残差 TD3: agent.update()           [不动, 零回归]         │
  │ (b) V 在线微调: IQL expectile          [新增, 冻 φ + EMA tgt] │
  │ (c) high_actor 在线微调: AWR(用当前 V) [新增]                │
  │  (b)(c) 数据 = online-HIQL buffer + offline_rb 混采           │
  └─────────────────────────────────────────────────────────────┘
  high_actor 实时出 z → compute_online_subgoal() 自动跟着变
```

### 4.1 冻 φ,只更新 V 的 critic head 与 high_actor

- `GoalConditionedVF`(`hiql_gc_value.py:63-94`)= φ(RelativeGoalEncoder) + 双 critic head。在线更新时 **φ 的参数 `requires_grad_(False)`**,只对 critic head 反传。
- 理由: 残差策略的 critic/actor 以 z(φ 输出)为**条件输入**(`q_agent.py:789` `subgoal_dim`)。冻 φ → z 分布平稳 → 残差 critic 不必追非平稳条件输入,联合微调最稳。
- high_actor 全量更新(它本来就是要在线重选 z 的那个)。

### 4.2 关键新组件: 独立 online-HIQL buffer

**问题**: 现有 `online_rb`(`train_chunk_residual.py:799-803`)是扁平 `TensorDictPrioritizedReplayBuffer` + `MultiStepTransform`(`rb_transforms.py:13-100`),只把 n-step reward 合进 `batch["next"]`,**不暴露中间帧 s_{t+1}..s_{t+k}**。而:
- V 的在线 loss 用 hindsight goal 采样(`sample_gc_goals` `hiql_gc_value.py:149-182`: p_curr=0.2 / p_traj=0.5 / p_rand=0.3,geometric future),需要**同轨未来状态**;
- high_actor 的 AWR(`train_high_actor` `hiql_high_actor.py:61-124`)需要 **s_{t+k}** 算 adv = `0.5(V1+V2)(s_w,g) - 0.5(V1+V2)(s,g)` 和回归目标 z_tgt = φ(base=s, target=s_w)。

两者都需要轨迹/未来状态,现有扁平 buffer 给不了。

**方案(低风险,推荐)**: 在 rollout 端维护 **per-episode 轨迹缓存**;**默认在 episode 结束时**对整条轨迹**批量预算**(完全仿照 `offline_stage_replay.build_offline_buffer` `offline_stage_replay.py:193-355`,其中 `:307-346` 预算 waypoint z 与 s_{t+k}),为该 episode 每条 transition 预算好:
  - hindsight goal 采样所需的轨迹索引/未来帧(供 V);
  - s_{t+k}(way state)与 z_tgt = φ(s,s_{t+k})(供 high_actor AWR);
  - reward/mask(见 §4.4)。

把这些写进一个**独立的 online-HIQL buffer**(只供 (b)(c) 两条更新用)。

> **零回归保证**: 残差 TD3 用的 `online_rb`(transition 字段见 `train_chunk_residual.py:59-80`)**一个字节不改**。online-HIQL buffer 是一条并行数据路,与 resfit 一贯的 offline/online 分桶风格一致。

这是本方案**工作量大头**。

### 4.3 抽共享 loss + 挂载点(保证同源)

- `train_gc_value`(`hiql_gc_value.py:185-277`,EMA `:269-271`,`expectile_loss_weighted` `:16-25`)与 `train_high_actor`(`hiql_high_actor.py:61-124`)的 loss 已是相对独立的函数。
- 把"**算 loss + 一步优化**"从"数据加载/采样循环"里抽成 buffer-agnostic 的 `value_update_step()` / `high_actor_update_step()`,**离线 trainer 和在线 loop 都调同一个函数** → 杜绝两套实现漂移(resfit 反复踩过的"同源"命门)。
- 挂载点: 主 loop `train_chunk_residual.py:940-966`,`agent.update()` 之后,按可配 cadence 触发。
- EMA 复用 `soft_update_params()`(`off_policy/common_utils/utils.py:57-59`)。

### 4.4 reward 语义: 同源稀疏 goal-conditioned(已与用户确认)

- 离线 V 用 goal-conditioned 稀疏 reward `r(s,g) = 0 if reached(g) else -1`(`hiql_gc_value.py` 约 `:241`),**不是** env 任务奖励。
- 在线微调**沿用同一套 goal-conditioned 稀疏 reward**,保持 offline/online 同源。
- 因此"online 数据的收益"= V/high_actor 看到残差策略**真实访问的状态分布**(修正离线几何与在线分布的 shift)+ 在线轨迹上**重拟合子目标**,而**非**引入新 reward 信号。
- env 任务奖励仍只走残差 TD3 的 Q,不进 V(避免破坏 HIQL 势函数语义、避免与 Q 重复)。

### 4.5 goal 来源(不变)

`self.goal`(`hiql_subgoal.py:29,47`)来自 `representative_goal30(seqs)`(`train_chunk_residual.py:779`,各 demo 末态 medoid),**全程恒定**,每 episode 不变。在线 hindsight 采样的 goal 仍按 `sample_gc_goals` 的混采分布产生(与离线一致)。

## 5. Flag 与默认值(向后兼容)

所有新行为走 flag,**默认 OFF → 与现状逐位等价**:

| flag | 默认 | 含义 |
|---|---|---|
| `--online_finetune_value` | off | 是否在线微调 V |
| `--online_finetune_high_actor` | off | 是否在线微调 high_actor |
| `--online_value_lr` | 1e-5(小) | V 在线 LR |
| `--online_high_actor_lr` | 1e-5(小) | high_actor 在线 LR |
| `--online_finetune_offline_fraction` | 复用现值 | V/高层在线更新的混采比 |
| `--online_finetune_every` | 1 | 每几个 update tick 更一次 |

- φ 冻结是第一版**默认且唯一**行为(不开放解冻口子)。
- buffer 签名(`train_chunk_residual.py:225-229`)需把上述 flag 纳入,避免静默复用旧 cache。

## 6. 验证策略

1. **回归单测**: 抽出的 `value_update_step` / `high_actor_update_step` 喂离线 batch,与原 `train_gc_value` / `train_high_actor` 逐位等价。
2. **golden-defaults 回归**: flag 全默认 off 时,整管线与现状逐位等价(沿用现有 294/302 那套测试,零回归)。
3. **A/B**: 在已跑通的 `three_piece` 或 `square` 配置上开 flag,对比 eval 曲线 + 监控 V(s,g) 的在线分布漂移/塌方指标(V 值范围、z 分布偏移)。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 在线 V/high_actor 把离线几何带崩(灾难性遗忘) | online+offline 混采锚住(§4.3);小 LR;EMA target |
| z 条件非平稳冲击残差 critic | 冻 φ(§4.1)使 z 分布平稳;high_actor 小 LR |
| online-HIQL buffer 改造引入回归 | 残差 online_rb 完全不动;新 buffer 独立;默认 off |
| 在线/离线 loss 实现漂移 | 抽共享 update 函数,两侧同源(§4.3)+ 逐位等价单测 |
| φ() 在线频繁调用的算力开销 | episode 结束时批量预算 z_tgt/waypoint;小 cadence |

## 8. 涉及文件清单(现状证据)

| 文件 | 关键位置 | 角色 / 改动 |
|---|---|---|
| `hiql_gc_value.py` | `:16-25` expectile / `:149-182` 采 goal / `:185-277` train(EMA `:269-271`) / `:63-94` VF | 抽 `value_update_step`(共享);φ 冻结 |
| `hiql_high_actor.py` | `:61-124` train(adv `:109-115`,waypoint `:98-107`) | 抽 `high_actor_update_step`(共享) |
| `offline_stage_replay.py` | `:193-355` build(`:307-346` 预算 z/s_{t+k}) | 仿照逻辑搬到 online 端 |
| `train_chunk_residual.py` | main `:555-1033` / 更新 loop `:940-966` / online_rb `:799-803` / 字段 `:59-80` / goal `:779` / 签名 `:225-229` / `compute_online_subgoal` `:176-184` / subgoal 加载 `:617-782` | 新增 online-HIQL buffer + 挂载两条在线更新 + flag/签名 |
| `hiql_subgoal.py` | `:28-90` / `subgoal_online` `:103-131` | 解冻 high_actor(条件);self.goal 不变 |
| `q_agent.py` | `update()` `:631-687` / `subgoal_dim` `:789` | **不动**(残差路零回归) |
| `off_policy/common_utils/utils.py` | `soft_update_params` `:57-59` | 复用做 EMA |
| `rb_transforms.py` | `MultiStepTransform` `:13-100` | 现有局限(不暴露中间帧)的根因 |

## 9. 与 2026-06-08 spec 的关系

`2026-06-08-hiql-hierarchy-residual-design.md:27-30` 明确把"高层 online 联合微调"列为**非目标**,并在 §7 留了"若分布漂移明显后续留 online 联合训口子"。本 spec 正是兑现那个口子的第一版(冻 φ 的保守版)。
