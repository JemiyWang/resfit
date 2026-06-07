# 设计：离线 HIQL action-free value(模块 ③a)

日期：2026-06-07
分支：chunk-residual-validation
关联文档：`/data2/kai0/rlt/residual_flow_design_and_training.md` §21;
  `/data2/kai0/relative-work/relative_paper/HIQL_2307.11949_detailed_analysis.md`;Ng et al. 1999(PBS)。
关联：三模块 default-off 计划的 ③(HIQL-Φ)。③ 拆两步:
  - **③a(本 spec)= 离线训 value** —— 从成功 demo 学一个冻结的 action-free IQL expectile value V(s),产出 `value.pt`。
  - ③b(后续另开 spec)= 把 V 当 PBS 势函数 Φ 接进 reward shaping(online 的 chunk_env_wrapper + offline 的 buffer 构建两端)。
  前置:模块 ①(stage_budget)②(relabeling)已实现。

## 1. 背景与动机

现状 `--reward_shaping potential`(PBS, Ng 1999)的势函数 **Φ = 整数 latch stage**(0..num_stages-1,
见 `chunk_env_wrapper.py:23-40 shaping_reward`):`F = bonus·(γ·Φ(s') − Φ(s))`。它是**阶梯状**的 ——
只在跨 stage 的瞬间给一次脉冲,**stage 内部没有梯度**,对长程稀疏(尤其 stage3 精插)帮助有限。

HIQL 的思路是用 action-free 的 IQL value(只看状态序列、不依赖动作,expectile regression,信噪比高)学一个
**平滑的"离成功还有多近"**当 Φ:stage 内部也有连续梯度。**不做分层**(不学高层 goal 选择,goal 固定=轨迹终点)。
PBS 理论保证:Φ 任意(只要固定)都不改变最优策略,所以即便 V 不完美,最坏只是"shaping 没帮上忙",不会带歪。

本 spec(③a)只造**离线 value 训练**:产出一个冻结的 `value.pt`。把它接进 reward shaping 是 ③b。

## 2. 目标 / 非目标

目标:
- 一个独立离线训练脚本,从**纯成功 offline demo** 学 action-free IQL expectile value `V(s)`,
  lowdim state(18 维)输入,小 MLP;产出冻结 `value.pt`(含 state 归一化参数 + 训练集 V 统计)。
- 训练后 V 沿一条成功 demo 的进度**单调递增**(越接近终点 V 越大)。

非目标(本模块不做):
- 不接 reward shaping、不碰 `chunk_env_wrapper`/offline buffer 的 reward(那是 ③b)。
- 不在线更新 V(离线一次性训练,冻结)。
- 不用观测/图像/ViT(第一版仅 lowdim state)。
- 不吃在线/失败轨迹(第一版纯成功 demo;HIQL 吃次优数据的威力留后续)。
- 不做分层 / goal-conditioned policy / 高层子目标。

## 3. 关键设计事实(代码已核实)

- offline demo 数据:每条 demo 是一条**完整成功轨迹**,`done` 仅末步 True。obs 含
  `observation.state`(18 维,两臂 eef pos/quat/gripper,**已标准化**)。来源 hdf5
  `deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5` +
  现有 stage cache npz(见 `offline_stage_replay.py` / `offline_hdf5_buffer.py`)。
- ③a 学 value **只需 state 序列 + episode 边界**(goal = 轨迹终点),**不需要 stage**(stage 是 ③b/PBS 用的)。
  故 ③a 可写一个轻量 loader 直接读 hdf5 的 state + 每条 demo 的边界,构 `(s_t, s_{t+1}, done_t)`,
  **不依赖 stage cache、不跑 sim replay**(比 build_offline_buffer 轻得多)。
- **18 维 state 的来源(已核实)**:不是 hdf5 里的单一 dataset,而是 6 个分离 lowdim key 拼接 ——
  `robot0/1` 各 `eef_pos[3]+eef_quat[4]+gripper_qpos[2]`,由 `offline_hdf5_buffer.assemble_state18` + 常量
  `STATE18_KEYS` + `sorted_demo_keys` 完成(hdf5 路径 `data/demo_i/obs/<key>`,T<2 的 demo 跳过)。
- **标准化口径(已核实,关键)**:用 `StateStandardizer.from_dataset_stats(LeRobotDatasetMetadata(dataset).stats
  ["observation.state"])`(从数据集 metadata 拉 mean/std,**不是在 ③a 数据上自算**)。在线 obs(chunk_env_wrapper:105)
  与 offline buffer 用**同一套拼接顺序 + 同一个 StateStandardizer**,所以 ③b 推理时喂给 V 的 `obs.state` 本就已标准化、
  与 ③a 训练同分布 —— 这是 ③a/③b 对齐的根基。

## 4. 设计

### 4.1 value 网络(`hiql_value.py`:`ValueMLP`)

- 小 MLP:`state_dim(18) -> 256 -> 256 -> 1`,中间 ReLU(可选 LayerNorm);输出标量 `V(s)`。
- 纯 torch.nn.Module,可单测 forward shape。

### 4.2 学习目标:action-free IQL expectile value

- goal-reaching 内部 reward(**仅 value 训练自用,与 RL 主任务 reward / PBS 无关**):
  `r_t = 1 if done_t else 0`,其中 `done_t` = 该 transition 是否轨迹末步(到达终点=到 goal)。
- TD target(action-free,无动作):
  ```
  y_t = r_t + γ · (1 − done_t) · V_target(s_{t+1})
  ```
  done 时 `y_t = r_t = 1`(终止不 bootstrap)。
- expectile regression loss(纯函数 `expectile_loss`):
  ```
  u = y − V(s)
  L_τ(u) = |τ − 1[u < 0]| · u²        # τ=0.5 退化成 0.5·MSE;τ>0.5 对 u>0(低估)惩罚更重
  ```
  默认 τ=0.7(HIQL 常用 0.7~0.9)。
- target network `V_target`:EMA(polyak)或定期 hard copy,稳定 bootstrap。默认 EMA(τ_ema=0.005)。
- 学出的 `V(s) ≈ γ^(到终点步数)`,平滑递增到末态≈1;stage 内部有连续梯度。
- 备注:纯成功 demo 下 Monte-Carlo 折扣 return 也可,但用 IQL expectile + target net 更贴 HIQL、且为 ③ 后续
  吃失败数据留扩展口。

### 4.3 数据加载(`train_hiql_value.py`:loader)

- **复用** `assemble_state18` + `STATE18_KEYS` + `sorted_demo_keys`(offline_hdf5_buffer.py)逐 demo 读 6 个 key
  拼 (T,18);用 `StateStandardizer.from_dataset_stats(LeRobotDatasetMetadata(dataset).stats["observation.state"])`
  标准化(**与 RL 训练同源,不自算 mean/std**);把每条 demo 的标准化 state 序列交给纯函数 `build_transitions`
  构 `(s_t, s_{t+1}, done_t)`,`done_{T-1}=True`,T<2 跳过。
- 全部装进内存张量(state 18 维,极轻),普通 minibatch 采样。

### 4.4 产出 `value.pt`

- 存:`ValueMLP` 权重 + `state_dim`/`hidden` + **训练集上 V 的统计(min/max/mean)** + 记录用的 `dataset_id`
  与 state mean/std(仅作记录/自校验,**③b 不依赖**)。
- **标准化不进 value.pt 的依赖链**:V 在标准化 state 上训练;③b 推理时喂的 `obs.state` 本就被同一套 StateStandardizer
  标准化过(在线 wrapper + offline buffer 都已标准化、同源 dataset stats),故 value.pt 不需自带 norm 也能对齐。
- V 的**尺度归一化决策留给 ③b**(关系到 PBS 幅度与 bonus 的配合):③a 只产出 raw V 模型 + V 统计,
  ③b 据此把 V 线性缩放到与整数 stage Φ∈[0,num_stages-1] 可比的量级(或暴露 `--phi_scale`)。
  职责切分:③a 学 V,③b 决定怎么缩放接进 PBS。

### 4.5 训练超参(默认值,CLI 可覆盖)

- `--gamma 0.99`、`--expectile 0.7`、`--ema 0.005`、`--lr 3e-4`、`--batch_size 256`、
  `--steps 50000`(state 极轻,几分钟级)、`--value_hidden 256`。
- 独立脚本,与 RL 训练解耦;产出路径 `--output value.pt`。

### 4.6 正交性 / default-off

- ③a 是**独立离线脚本**,不进 RL 训练循环,不改任何现有训练/env 代码 -> 对 baseline **零影响**(不跑就不存在)。
- default-off 的等价性约束落在 ③b(接 Φ 时 `--potential_source` 默认 stage 逐位等价);③a 本身无需等价性单测。

## 5. 测试方案(TDD)

- `expectile_loss` 纯函数(light):τ=0.5 == 0.5·MSE;u>0 与 u<0 的非对称权重(τ>0.5 时 u>0 权重更大);shape。
- `discounted_target` 纯函数(light):done 时 y==r(无 bootstrap);非 done y==r+γ·V';批量正确。
- `ValueMLP` forward(light):输入 [B,18] -> 输出 [B,1];参数可训。
- `build_transitions`(light):假轨迹 list -> (s,s',done) 正确,末步 done=True,episode 边界正确,T<2 跳过。
- save-load 往返(light):value.pt 存取后权重 + state_dim/hidden + v_stats 一致。
- (state 标准化用现有 `StateStandardizer`,本模块不重测;loader 的 hdf5 读取靠 manual 实跑验证。)
- manual(集成,CPU):小数据训几百步,loss 下降;沿一条成功 demo,V 随进度**单调递增**、末态最高。

## 6. A/B 验证方案(属 ③b,不在本实现)

③a 的产物 value.pt 在 ③b 接进 `--reward_shaping potential --potential_source hiql` 后做 A/B
(A=整数 stage Φ;B=HIQL Φ),看 stage 内部 shaping 是否变密、stage3 回退是否降。本 spec 不涉及。

## 7. 文件改动清单

- `resfit/rl_finetuning/chunk_residual/hiql_value.py`(新)—— `ValueMLP` + `expectile_loss` +
  `discounted_target` + `build_transitions` + `train_value`(EMA target)+ `save_value`/`load_value`(纯逻辑,可单测)。
- `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`(新)—— `read_per_demo_states`(复用 assemble_state18
  + StateStandardizer)+ CLI + 串起 build_transitions/train_value/save_value。
- 测试:`resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`(新)。

## 8. 风险与缓解

- 纯成功 demo -> V 只在成功流形上准,online 偏离(件掉/没夹稳)时高估 —— **PBS 安全性兜底**(Φ 不准不改最优策略);
  若增益不明显,后续升级为吃失败轨迹(③ 第二步)。
- lowdim state 不含物体 pose -> 对精插"接近度"表达可能不足(接受,第一版机制验证);后续可升级观测/ViT,
  接口不变(③b 接的是 `V(obs)->标量`)。
- 尺度/归一化没配好 -> PBS 幅度异常:③b 用训练集 V 统计线性缩放 + `--phi_scale` 兜底。
- expectile τ 过高可能过保守:默认 0.7,可扫 0.7/0.9。
