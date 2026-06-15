# pouring / lifttray 接 act_feat 残差 RL —— 走纯 hdf5 路线(对齐 three_piece)

日期：2026-06-15
状态：设计已批准，待写实现 plan

## 1. 背景与目标

目标：启动 lift tray + pouring 两个任务的残差 RL 实验，配置参数与
`outputs_chunk/three_piece_actfeat_bp_bc01` 一致（state_mode=act_feat、
offline_base_mode=base_policy、subgoal_conditioned、chunk_length=1、actor_lr=1e-6、
queue+base_n_action_steps=10、action_scale=0.05、offline_fraction=0.5、demo_bc_coef=0.1、
total_env_steps=500k、no-stage）。

现有 pouring/lifttray 已有 **lerobot 路线**（`--data_source lerobot`，smoke 跑通），但建
act_feat 缓存需 pyav 解码视频，全集 ~311 分钟（5h+）。three_piece 自身走的是 **hdf5 路线**
（`--offline_dataset_path *.hdf5`，离线 cache 图像/state 直接读 hdf5）。

本设计：给 hdf5 路线补上 pouring/lifttray 的 state 拼装支持，让两任务走与 three_piece
**完全相同的代码路径**。这既提速 ~100x，又是真正"配置对齐 three_piece"。

## 2. 调查结论（设计依据，均已实测验证）

1. **同源金标准 = 在线 env 渲染图**。`verify_act_feat_online_consistency.py` 证 three_piece
   上"离线 hdf5 raw 图 ≈ 在线 robosuite env 渲染图"（~0.03 EGL 舍入噪声，frame0 像素级完美）。

2. **hdf5 raw 图与 lerobot 解码图差异 0.43 是视频有损压缩噪声，非几何 bug**。
   图像差异分解（pouring demo_0，3 相机各 ~21 帧）：identity mean=0.009/max=0.54，
   vflip mean=0.32、hflip mean=0.10、chan-rev mean=0.025——**identity 即对齐，无翻转/无通道交换**。
   mean 仅 0.9% + max 0.54 的形态 = 典型有损压缩（多数像素一致、高频处个别像素误差大）。
   推论：hdf5 raw = 无损图 = 在线 env 图；lerobot 是压缩版，与在线 env 有 0.9%mean/0.43max gap。
   故现有 lerobot 路线离线 cache 反而**不如** hdf5 同源于在线。

3. **提速来源**。act_feat 缓存构建中 pyav 解码占 99%（551s）、ACT forward 仅 1%（3.9s）。
   走 hdf5 省掉解码：全集 1009 demo 从 ~311 分钟降到 ~2 分钟。

4. **唯一阻碍 = state 拼法**。act_feat 的 hdf5 路线图像读取已是 `obs/{name}_image`
   （对所有 task 通用，几何 identity 已证）；但 proprio 用硬编码 18 维 three_piece 字段名的
   `assemble_state18`，pouring 字段是 `robot0_left/right_*`，会 KeyError。

5. **在线拼法精确语义**（dexmg `_get_expected_low_dim_keys` + `_process_obs`）：
   按 task 选 proprio key 集、**按 key 顺序完整维度 np.concatenate、不截断**。
   - pouring/coffee/cansort → humanoid 双臂手：robot0_right_{eef_pos,eef_quat,gripper_qpos}
     + robot0_left_{...}
   - 单臂 Panda 集（lift/can/square/threading 等，精确匹配）→ robot0_{...}
   - 默认 two-arm → robot0_{...} + robot1_{...}

6. **维度由 hdf5/env 字段自然决定（实测 4 task）**，统一走"完整 concat"对全部正确、对 18 维路零回归：

   | task | key 集 | gripper_qpos/臂 | 总维 |
   |---|---|---|---|
   | three_piece | robot0/robot1 | 2 | 18（== STATE18，逐位等价）|
   | threading | robot0/robot1 | 2 | 18（== STATE18，逐位等价）|
   | pouring | robot0_right/left | 11 | 36 |
   | lifttray | robot0/robot1 | 12 | 38 |

7. **state 拼法已反推验证**：hdf5 按上述拼出的 state 与 lerobot `observation.state` 逐帧
   max|diff|=5.96e-08（pouring/lifttray 各 4 demo，ep↔demo 保序）。

8. **ACT base 来源对等**：three_piece 的 `resfit/out/piecce/best` 与 pouring/lifttray 的
   `run_anw5pphu_best:v2` / `run_e0o0sckj_best:v4` 都是 LeRobot 框架训的 ACT；三任务在
   "ACT 训练源 / 离线 cache / 在线 env"三方同源待遇上一致。base_wandb_id 与 act_base_ckpt
   同一 ACT（同源、不触发警告）。

## 3. 设计

### 3.1 核心抽象（offline_hdf5_buffer.py）

复刻 dexmg 的 key 选择，离线侧自带一份（**不 import dexmg.py**，避免顶层 robosuite-1.5 依赖
被拉进纯读 hdf5 的 build 流程）；用一致性测试守护与 dexmg 不漂移：

- 常量 `LOW_DIM_KEYS_SINGLE / LOW_DIM_KEYS_MULTI / LOW_DIM_KEYS_HUMANOID`
- `expected_low_dim_keys(hint) -> list[str]`：hint 可为 dataset_id（如
  `ankile/dexmg-two-arm-pouring`）或 robosuite env_name（如 `TwoArmPouring`），小写后用
  与 dexmg 同款的包含/精确匹配；对现有 4 task 两种 hint 结果一致（测试守护）。
- `assemble_state_by_env(obs, hint) -> np.ndarray`：按 `expected_low_dim_keys(hint)` 顺序、
  每字段完整维度 `np.concatenate(axis=1)`，不截断。缺字段 KeyError（同源命门）。

保留 `STATE18_KEYS` / `assemble_state18` 不动（向后兼容 + 测试基准）。

### 3.2 改动点（env_hint 贯穿）

- `train_hiql_value.py`
  - `_build_raw_obs_seqs(..., env_hint=None)`：proprio 走 `assemble_state_by_env(obs, env_hint)`；
    env_hint=None 回退 `assemble_state18`（旧默认）。
  - act_feat build 调用点传 `env_hint=dataset_id`。
- `offline_stage_replay.py`
  - `_demo_base_actions(..., env_hint)`：base_policy 模式逐帧喂 ACT 的 `observation.state`
    走 `assemble_state_by_env`（pouring=36/lifttray=38，喂对应 ACT）。
  - `build_offline_buffer(..., env_hint)`：行 260 的 state_n 拼装走 `assemble_state_by_env`
    （act_feat+无 potential 时此 state_n 不进 reward，但 assemble_state18 会对 pouring KeyError，
    必须换）；`_demo_base_actions` 调用传 env_hint。
  - `train_chunk_residual` 调用 `build_offline_buffer` 处传 `env_hint=args.task`（如 TwoArmPouring）。

注：act_feat 主训 critic/actor 的 observation.state 从 act_feat 缓存（548/550 维）读，
不重拼 proprio——拼法改动通过缓存生效。eef/eef_piece 路线（仅 three_piece 用）完全不动。

### 3.3 向后兼容

- three_piece/threading：完整 concat 实测 = 18 维、与 STATE18 逐位等价 → 缓存/已训模型/在跑实验零回归。
- env_hint=None 路径保留旧 18 维行为。
- libero 路线（libero_env，另一套 schema）不受影响。

## 4. 测试

新增/扩充 chunk_residual 测试：
- `expected_low_dim_keys` 对 4 task（dataset_id 与 env_name 两种 hint）返回正确 key 集。
- `assemble_state_by_env` 维度：three_piece=18 / threading=18 / pouring=36 / lifttray=38。
- three_piece/threading：`assemble_state_by_env` 输出与 `assemble_state18` 逐位相等（np.array_equal）。
- 与 dexmg 一致性：与 `dexmg._get_expected_low_dim_keys` 比对（import 失败则 skip，避免硬依赖
  robosuite-1.5 阻塞 CI；硬编码期望值兜底）。

## 5. 启动脚本

新建 `run_act_feat_hdf5_full_bp.sh <pouring|lifttray>`（不动旧 lerobot 脚本）：
- gc_value / high_actor：`--hdf5 resfit/dataset/two_arm_{pouring,lift_tray}.hdf5 --dataset <ds>`，
  去 `--data_source lerobot --lerobot_root`。
- 主训：`--offline_dataset_path <hdf5> --task <TwoArmPouring|TwoArmLiftTray>`，去 lerobot 标志，
  其余超参对齐 three_piece_actfeat_bp_bc01。
- base_wandb_id 与 act_base_ckpt 同一 ACT（anw5pphu / e0o0sckj）。

## 6. 启动前 hard gate（破坏性/资源密集动作前必过）

1. 全量 `pytest`（chunk_residual）零回归。
2. 把 `verify_act_feat_online_consistency.py` 推广到 pouring/lifttray：
   - robots_map 加 TwoArmPouring / TwoArmLiftTray 的 robots；
   - state 拼装走 `assemble_state_by_env`；
   - 验"离线 hdf5 act_feat ≈ 在线 env act_feat"（atol≈0.05，与 three_piece 同判据）。
3. 小规模 hdf5 smoke（gc/high/主训 40 步）跑通。
全过后向用户确认 GPU 分配再启动全集 500k。

## 7. 非目标（YAGNI）

- 不删除/不重构现有 lerobot 路线与脚本（保留）。
- 不重构 dexmg.py 在线代码（用测试守护一致性，不动在跑路径）。
- 不改 eef/eef_piece、pi0_feat、libero 路线。
- 不重训 ACT base。

## 8. 风险与缓解

- **离线≠在线（同源命门）**：靠 §6.2 推广 online-consistency gate 在启动前数值验证拦截。
- **key 选择与 dexmg 漂移**：一致性测试守护；dexmg 改了 key 集测试会红。
- **hint 匹配歧义**（dataset_id vs env_name）：当前 4 task 两种 hint 均正确，测试覆盖；
  未来单臂 task 走 offline 需用 env_name（精确匹配）而非 dataset_id。
