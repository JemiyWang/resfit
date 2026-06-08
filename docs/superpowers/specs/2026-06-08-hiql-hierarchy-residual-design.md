# 设计：HIQL 分层(goal-conditioned V + AWR 高层 + 子目标条件残差低层)嫁接进 chunk residual RL

日期：2026-06-08
分支：chunk-residual-validation
关联文档：
- `/mnt/mnt/data/HIQL/HIQL_paper_explained.md`(HIQL 论文详解)、HIQL 2307.11949。
- `docs/superpowers/specs/2026-06-07-hiql-value-design.md`(③a 单任务 V-as-Φ)、`2026-06-07-hiql-potential-design.md`(③b PBS 接线)、`2026-06-07-hiql-value-object-aware-design.md`(③a' eef_piece 30 维 object-aware state)。
- memory：`project_resfit_hiql_phi_is_not_hiql`(澄清 ③ 只借 value 配方、未用 HIQL 核心)、`project_resfit_gt_as_base_bug`(残差 relabel 的 GT-as-base 坑)、`project_resfit_object_aware_value`(③a' 同源命门)、`project_resfit_stage_offline`(stage 检测/offline buffer)。

## 1. 背景与动机

现状的"③ HIQL-Φ"(`hiql_value.py`+`hiql_potential.py`)**名不副实**：它只借了 HIQL 的 action-free expectile value 训练配方，把学出来的标量 `V(s)` 当 PBS 势函数 Φ 替掉整数阶梯 stage。HIQL 真正的核心——**分层(高层提子目标 + 低层到子目标)、用 V 中间层做潜子目标表征 φ、AWR 从 V 抽策略、goal-conditioning**——一条都没用上。最要害的错位：HIQL 里 V 是**策略本体**(靠 V 的相对差直接选子目标/动作)，resfit ③ 里 V 只是**奖励整形**(PBS 不变性 → 安全但结构上碰不到 HIQL 要解决的"远目标信噪比"问题)。

本 spec 把 HIQL 核心**真正嫁接**进 resfit 的残差 off-policy RL：保留冻结 base policy 与 TD3 式残差机制，新增/改造三件——goal-conditioned 价值、AWR 高层、子目标条件的残差低层。命门假设是：**给残差一个比 stage one-hot 更稠密的"k 步后子目标"方向目标，能帮到长程多阶段装配(尤其 stage3 精插)**。

与现有 ③(V-as-Φ)**正交**：`hiql_value.py`/`hiql_potential.py` 与 `--potential_source hiql` 那条路保持不变，本路另起模块。

## 2. 目标 / 非目标

目标：
- goal-conditioned 价值 `V(s, φ([g,s]))`(新模块)，含 10 维归一化瓶颈表征 φ；action-free expectile TD；建在 eef_piece 30 维 object-aware state 上。
- 高层 `π^h(z | s, g)`(新模块)，AWR 抽取，输出子目标潜向量 z=φ(s_{t+k},s_t)；离线训、online 冻结(首版)。
- 低层 = 现有残差 actor + critic，改成**子目标条件**(条件在 z 上)，复用 `stage_conditioned/append_stage` 钩子，低层抽取方式仍 TD3 不变。
- 执行：每步高层提 z、残差低层条件在 z 上，无时间抽象。
- 三阶段交付，每阶段带验证 gate；全程 default-off，不破坏现有 baseline 与 ③ 的 V-as-Φ 路。

非目标(本 spec 不做)：
- 低层**不**换成纯 AWR(避开残差 relabel 的 GT-as-base 坑)；保留 off-policy RL。
- base policy **不**感知子目标(始终冻结、只有残差对 z 响应)。
- 高层首版**不**做 online 联合微调(离线训、冻结)。
- **不**新做 object-aware state(③a' 假设已就绪，eef_piece 30 维 + rel mean/std 可用)。
- 不动 `hiql_value.py`/`hiql_potential.py`/`--potential_source hiql`(正交)。

## 3. 关键设计事实(代码已核实)

- **低层 RL = QAgent**(`off_policy/rl/q_agent.py`)：TD3 式 off-policy actor-critic。`Critic.q_value(feat, prop, action)`、`Actor.forward(obs, std)->TruncatedNormal`。actor/critic 都条件在 `feat(图像)+observation.state(本体)+observation.base_action` 上。
- **现成的条件化钩子**：`stage_conditioned=True` 时 `_critic_prop` 用 `append_stage(prop, obs["observation.stage_id"], num_stages)` 把 stage one-hot 拼到 prop 末尾喂 critic；actor 侧 `Actor(..., stage_conditioned=, num_stages=)` 同理。**子目标条件化照此模板，把 one-hot 换成 10 维 z 即可。**(注：`--stage_conditioned` 现已证伪——见风险 §8。)
- **残差 actor**(`residual_flow_actor.py`)：`forward` 里 `s_p = cat([compress(feat), observation.state])`，velocity 网络输入含 `s_p`。子目标 z 拼进 `s_p` 是最小改动点。另有非 flow 的 `Actor`(`--actor raw`)走同款 prop 条件。
- **base policy 冻结**(`build_base_policy`)，出 `base_action` chunk；最终动作 = base + 残差(scale 0.05、clip)。
- **offline buffer**(`offline_hdf5_buffer.py`/`train_chunk_residual.py:_offline_buffer_signature`)：demo 烤入 shaped reward/done/stage，签名变化即重建。本路需在 transition 里加子目标 z(及 goal g)→ 签名加键 → 触发重建。
- **state**：③a' eef_piece 30 维(18 本体 + 12 标准化 rel_piece)，online/offline 同源标准化(rel mean/std 存于 value ckpt)。
- **stage 机制**：`stage_detectors.NUM_STAGES`、stage cache npz、`offline_stage_replay`——供混合子目标的"stage 入口态"锚使用。

## 4. 设计

### 4.1 总体架构

冻结 base 不动。三个学习件：

1. **goal-conditioned 价值** `V(s, φ([g,s]))`(新文件 `hiql_gc_value.py`，**不动** `hiql_value.py`)：
   - `goal_encoder = RelativeRepresentation(rep_type='concat', bottleneck=True, rep_dim=10)` → φ([g,s])，归一化 10 维(照 HIQL `special_networks.py`)。
   - state encoder：lowdim 下恒等。value 头：`MLP(cat[s, φ]) -> 标量`，双 critic 集成 + EMA target。
   - 学习目标：action-free expectile TD(复用 `hiql_value.expectile_loss`/`discounted_target` 思路)，goal 混采(§4.2)。
2. **高层** `π^h(z | s, g)`(新文件 `hiql_high_actor.py`)：MLP 输出 10 维 z 上的高斯。AWR 抽取(§4.3)。离线训、冻结存 ckpt。
3. **低层** = QAgent 的 actor + critic，加子目标条件 z(§4.4)，TD3 训法不变。

φ 同时被高层(产 z=φ(s_{t+k}))与低层(条件在 z)共用——与 HIQL"高低层都用 value_goal 表征"一致。

### 4.2 goal-conditioned V 训练

- 元组 `(s_t, s_{t+1}, g)`；g 混采(HIQL 口径)：当前 0.2 / 未来 0.5 / 随机 0.3。**未来目标锚在 stage 入口态**(下一 stage 首次到达态，混合方案的语义锚)；随机目标取数据集任意状态。
- reward `r(s,g)=0 if s==g else -1`，到达即终止截断 bootstrap(同 HIQL/GCDataset)。
- TD：`y = r + γ·(1-done)·min(V̄1', V̄2')`，expectile τ=0.7，EMA target τ_ema=0.005。
- 产 `gc_value.pt`：权重 + 维度 + rep_dim + state_mode(eef_piece) + rel mean/std + dataset_id。

### 4.3 高层 AWR 训练

- 元组 `(s_t, s_{t+k}, g)`，k 步航点 + g 混采(同 §4.2)。
- AWR：`J = E[ exp(β·(V(s_{t+k},g) − V(s_t,g))) · log π^h(φ(s_{t+k},s_t) | s_t, g) ]`(HIQL 式 6)，β=`--high_beta`(默认 1)，exp 权重 clip 100。
- state-only，可离线一次性训冻结，产 `high_actor.pt`。
- 用冻结的 `gc_value.pt` 的 φ 算回归目标 z。

### 4.4 低层子目标条件化

- 新增 `--subgoal_conditioned`(与 `--stage_conditioned` 互斥)。`QAgent` 加 `subgoal_dim=rep_dim(10)`：
  - `_critic_prop`：`append_subgoal(prop, obs["observation.subgoal"])`(拼 10 维 z)。
  - `Actor`/`ResidualFlowActor.forward`：z 拼进 `s_p`。
- 训练时 z 来自数据(buffer 里存的 k 步航点 z=φ(s_{t+k},s_t))；online 执行时 z 来自高层。
- critic/actor 维度随之 +10；TD3 loss、demo_bc、target 网络一切不变。

### 4.5 数据流与执行

- offline buffer：build 时为每条 transition 计算 k 步航点状态 s_{t+k} 与 g，用冻结 φ 算 `z`，连同 g 存进 transition(z 10 维、g 30 维)；buffer 签名加 `subgoal_conditioned/gc_value_ckpt/way_steps/high_actor_ckpt`。
- online(`act`)：`g` = 任务目标，构造方式**与 V 训练时的目标同源**——取 eval 环境给定的最终装配成功态、按 eef_piece 30 维口径拼接并用同一套标准化(本体维取首帧、其余维取 target，沿用现有 eval goal 设定)。高层 `z=π^h(s,g)`(冻结)→ 写入 obs `observation.subgoal` → 残差 actor 条件在 z；`add_chunk_transition` 把 z(及 g)写进 transition。
- 无时间抽象：每步重提 z(与 HIQL 一致)。

### 4.6 涉及文件

| 文件 | 改动 |
|---|---|
| `hiql_gc_value.py`(新) | `GoalConditionedVF`(φ 瓶颈 + 双 critic + EMA)、goal 采样、`train_gc_value`、`save/load_gc_value`、`phi()` |
| `hiql_high_actor.py`(新) | `HighActor`(MLP→10 维高斯)、`train_high_actor`(AWR)、`save/load` |
| `train_hiql_gc_value.py`(新) | loader(复用 `assemble_state*`/StateStandardizer)+ stage 锚 + CLI |
| `train_hiql_high_actor.py`(新) | loader(k 步航点 + g)+ 用冻结 φ + CLI |
| `q_agent.py` | `subgoal_conditioned`/`subgoal_dim`；`append_subgoal`；`_critic_prop`、`act`、actor/critic 构造 |
| `off_policy/rl/stage_utils.py` | 加 `append_subgoal`(或复用 append 拼接) |
| `residual_flow_actor.py` / `Actor` | `forward` 把 z 拼进 `s_p` |
| `offline_hdf5_buffer.py` | 产 `(s_t, s_{t+k}, g)` + 用冻结 φ 算 z 存 transition |
| `train_chunk_residual.py` | 新 flag(`--subgoal_conditioned`/`--gc_value_ckpt`/`--high_actor_ckpt`/`--way_steps`/`--high_beta`)；接高层进 `act()`；z/g 进 transition；buffer 签名加键 |
| `hiql_value.py`/`hiql_potential.py` | **不动**(正交,V-as-Φ 路保留) |

### 4.7 分阶段(plan 按此切，每阶段独立可验证)

- **Phase 1**：goal-conditioned V + φ(`hiql_gc_value.py`+`train_hiql_gc_value.py`+单测)。Gate：V 对不同 g 有区分度；沿成功 demo 对正确 g 单调；φ 维度/归一化正确。
- **Phase 2**：高层 π^h(AWR，离线，`hiql_high_actor.py`+`train_hiql_high_actor.py`+单测)。Gate：高层提的子目标在 demo 上合理(最近邻可视化落在 k 步后附近)。
- **Phase 3**：低层子目标条件化 + buffer + 执行接线(+单测+smoke)。**先做"喂 demo 真航点 z"消融**(不接学出来的高层，z 直接取自数据)——若残差吃不动真子目标，则 stop(命门假设证伪)；过了再接学出来的高层做 A/B。

## 5. 测试方案(TDD)

- `GoalConditionedVF` forward/φ shape；goal 采样比例(0.2/0.5/0.3)与 stage 锚正确；expectile/TD 纯函数复用现有测。
- `train_gc_value` 小数据 loss 下降；save-load 往返一致。
- `HighActor` forward shape；`train_high_actor` AWR 权重数值(exp(β·adv)、clip 100)；用假 V 验证回归目标=φ(s_{t+k})。
- `append_subgoal` 拼接维度；QAgent `subgoal_conditioned` 下 critic/actor 输入维度 +10；`_critic_prop` 正确。
- buffer：transition 含 z(10)/g(30)，way_steps 取 s_{t+k} 与 stage 边界裁剪正确；签名加键触发重建。
- smoke：`--subgoal_conditioned` 端到端跑若干 step 不报错，打印高层接上的标记。

## 6. A/B 验证方案(Phase 3 后)

- 消融门(先)：A=`--stage_conditioned`(现状证伪基线)/B=`--subgoal_conditioned` 喂 **demo 真航点 z**。看 B 是否优于 A——验证"子目标条件对残差有用"。
- 正式 A/B(消融过后)：A=demo 真航点 z / B=学出来的高层 π^h 提 z。指标：`eval success_rate` 后 5 点滑动均值 + per-stage 到达率(`eval_stage_reach.py`)，重点 stage3 回退、reach3→reach4。

## 7. 风险与缓解

- **命门假设风险**：`--stage_conditioned` 已证伪 → action_scale 0.05 的残差可能吃不动 z，子目标条件同命。缓解：Phase 3 先跑"喂真航点 z"的便宜消融，证伪即止，不浪费高层/V 的工。
- **goal-conditioning 需物体 pose**：依赖 ③a' eef_piece 30 维(已假设就绪)；若 state 退回 eef-only，g 区分不了 piece 构型 → 本路失效。硬依赖，spec 明确写死。
- **残差 relabel 坑**：本路低层走 TD3、不 relabel 残差动作，刻意绕开 GT-as-base(`project_resfit_gt_as_base_bug`)。
- **action-free V 确定性假设**：随机环境会高估(HIQL 已知 caveat)；装配近确定性，接受。
- **base 不感知 z**：残差是唯一对 z 响应的部件，权限受 action_scale 限。接受(首版机制验证)。
- **buffer 重建**：签名加键 → 旧缓存失效需重建，避免错误复用。
- **高层冻结**：首版不 online 微调；若分布漂移明显，后续留 online 联合训口子(非目标)。
