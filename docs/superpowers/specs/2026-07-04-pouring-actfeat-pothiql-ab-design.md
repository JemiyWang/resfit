# Pouring ActFeat pothiql A/B (rollout) Design

## Goal

给 pouring 的 `--potential_source hiql`(pothiql,合法单状态势函数)换一个**输入信息更全的 value**:从现在的 `eef`(36 维纯本体)换成新训的**单状态 `act_feat` value**(548 维,图像+本体特征)。除 value 输入外,其余训练配置**逐字对齐** `pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0703`,形成一个只隔离"势函数 V 看到多少信息"的干净 A/B。

动机:pothiql 现用的 eef value 只看机器人自身状态,看不到物体/水/杯子,对"离成功多远"的估计被状态混叠抹平(pouring 尤甚)。给 V 补图像+本体信息,期望让 shaping 更有信息、进一步抬升成功率。合法性不受影响——仍是单状态 `Φ(s)`,PBS telescoping 成立。

## 这是"rollout / 执行",不是新机制

act_feat 单状态势函数的**机制早已设计+实现+单测**,见 [`2026-06-28-actfeat-hiql-potential-design.md`](2026-06-28-actfeat-hiql-potential-design.md)。该 spec 的 "Rollout" 段(训 `pouring_value_actfeat_hdf5.pt`、起 `_pothiql_actfeat` 新 run、把旧 `_pothiql` eef run 当 ablation)正是本 spec 要落地的下一步。

代码侧已就绪、无需改:
- 在线 `chunk_env_wrapper.py:138-139`:act_feat 势函数每步用冻结 ACT 编码器对实时 `raw_obs`(图像+本体)提特征算 `Φ`(`PotentialActFeatureEncoder`)。
- 离线 `offline_hdf5_buffer.py` `transition_rewards` 的 act_feat 分支:用 `act_feat_cache` 序列算 `Φ`,不重放图像过 ACT。
- 校验 `train_chunk_residual.py` `_validate_actfeat_potential_cache` / 缓存签名 / 错误快失败。
- 单测:`test_hiql_potential.py` / `test_chunk_env_wrapper.py` / `test_act_feat_cli_wiring.py` / `test_read_per_demo_states_act_feat.py`。

**但这条路从没端到端真跑过**(磁盘上不存在任何单状态 act_feat value;现有单状态 value 全是 eef,现有 act_feat value 全是 gc_value/subgoal 路)。故先 smoke 再正式跑。

## Non-Goals

- 不改 pothiql 分支代码逻辑(只喂它不同的 value ckpt + 已有的 `--act_feat_cache`)。
- 不动 potsubgoal(hiql_subgoal);它的塌方是"z 每步变破坏 telescoping"的正交问题,本 spec 不处理。
- 不改 actor/critic 的 `observation.state` 契约(仍低维本体状态,不变);act_feat 只喂势函数 V。
- 不在线微调势函数 value(与现有 pothiql 一致,势函数 value 冻结;subgoal 的 gc_value/high_actor 仍按 rerun0703 joint 在线微调,不受影响)。
- 本轮只做 pouring;lifttray/three_piece/threading 推迟。

## Scope 与变量隔离

- **对照(control)= 现有 `pouring_pothiql_joint_rerun0703`**(eef 势函数 → eval 稳定 ~0.94)。
- **处理(treatment)= 本 spec 的 `_pothiql_actfeat` run**(act_feat 势函数)。
- **唯一变量 = 势函数 V 的输入表征**(eef 36 维本体 vs act_feat 548 维图像+本体)。
- 势函数幅度自动可比:scale = auto_scale·phi_scale = phi_range/(vmax−vmin)·1.0,`auto_scale` 把不同 value 的动态范围都归一到同一 Φ 尺度,故 A/B 隔离的是"V 的信息量"而非"shaping 幅度"。

## 组件 1:训练单状态 act_feat value

- 工具:`train_hiql_value.py --state_mode act_feat`,**复用现有 `outputs_chunk/pouring_act_feat_hdf5.npz` 缓存**(不重提特征;缓存命中即秒读)。
- 产物:`outputs_chunk/pouring_value_actfeat_hdf5.pt`(`state_mode=act_feat`,`state_dim=548`,与现有 pouring act_feat 缓存/gc_value 同维)。
- 配方**镜像现有 eef `pouring_value_hdf5.pt`**:`hidden=256`、`dataset_id=ankile/dexmg-two-arm-pouring`、`train_value` 其余超参(gamma/expectile/steps/lr)取**默认**。
- 已知小缺口:`pouring_value_hdf5.pt` 的确切原始训练命令未存入任何脚本/log,只能从 ckpt 读回结构维度(state_mode/state_dim/hidden=256)并用 `train_value` 默认超参复刻。因 auto_scale 归一化幅度,value 配方的细小差异是二阶的,不污染 A/B 隔离度。若 act_feat value 明显欠拟合(548→256 容量偏小),`hidden` 可作为 plan 层调参旋钮,默认仍取 256 以最大化与 eef 对照的一致性。
- 同源命门:value 训在这份 act_feat 缓存上,在线 `PotentialActFeatureEncoder` 复现同一套特征(同 ACT 权重/image_keys/proprio_key/pooling/标准化)。由 `_validate_actfeat_potential_cache` 的签名+权重指纹校验强制;且这份缓存正是能跑通 potsubgoal 在线的那份,一致性有既有保障。

## 组件 2:端到端 smoke(先于正式 run)

因这条路从没真跑过,正式 500k 前做一次短 smoke,把潜在缺口抖出来:
- 离线:用少量 demo 建 offline buffer,确认 act_feat 势函数被烤进 reward、无报错、reward 非零且合理。
- 在线:跑若干 chunk step,确认在线 `PotentialActFeatureEncoder` 出特征、`Φ(s)/Φ(s')` 计算、`reward += bonus*(γΦ'−Φ)` 正常。
- eval:一次评测跑通、存 ckpt 无报错。
- 若需改代码,**只允许"加在现有 act_feat 分派之后、对 eef/eef_piece/potsubgoal/全量测试零回归"的增量改动**。守卫:现有 eef pothiql、potsubgoal、`pytest` 全量必须保持绿。

## 组件 3:A/B 正式 run 脚本

复制 `run_pouring_pothiql_joint_rerun0703.sh` → `run_pouring_pothiql_actfeat_joint.sh`,**只改 4 处**,其余(`--chunk_length 1 --base_action_mode queue --base_n_action_steps 10 --action_scale 0.05 --actor_lr 1e-6 --actor raw --reward_shaping potential --potential_source hiql --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 --subgoal_conditioned --gc_value_ckpt ... --high_actor_ckpt ... --act_feat_cache pouring_act_feat_hdf5.npz --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor --total_env_steps 500000`)**逐字不动**:

1. `--hiql_value_ckpt outputs_chunk/pouring_value_hdf5.pt` → `outputs_chunk/pouring_value_actfeat_hdf5.pt`
2. `--offline_buffer_cache` → **全新路径**(如 `..._as005_pothiql_actfeat_joint_offcache`)。**必须重建**:势函数 value 变了→离线 reward 被重烤→旧 offcache 签名失效(签名含 potential 身份/维度/stats)。
3. `--output_dir` → `outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_actfeat_joint`
4. `--wandb_name` → 同上后缀

`--act_feat_cache` 已在 rerun0703 里(第 47 行,原供 subgoal),act_feat 势函数复用同一份,无需新增参数。

## 成本 / 依赖 / 风险

- **offcache 重建**:~80min CPU base_policy forward + ~41G 磁盘(potential 变→必重建)。磁盘现余 1.1T,余量充足;plan 仍加"跑前 df 核查"。
- **GPU 授权**:训 value(快)+ smoke + 500k 正式 run 均需用户授权 GPU。
- **顺序**:训 value → smoke(绿)→ 重建 offcache → 起 500k。smoke 未过不进正式 run。
- **回归**:act_feat 势函数已有单测;仅 smoke 抖出代码缺口时才补测试;全量回归保持绿。

## Verification / 成功判据

- 单元:相关 act_feat 势函数单测 + 全量 `pytest` 保持绿(若有增量改动)。
- smoke:离线+在线+eval 端到端无报错,reward 合理。
- A/B:`_pothiql_actfeat` 500k 的 eval success 曲线 vs 对照 eef `rerun0703`(~0.94)。判读关注是否**稳定不塌**(合法性预期保持)且**是否抬升/持平**(信息量收益是经验问题,不预设结论)。

## Open items(默认已定,用户 spec review 可推翻)

- smoke 与正式 run **分两步**(默认;强烈建议,因该路没真跑过)。
- act_feat value `hidden=256`(默认,镜像 eef);欠拟合再议。
- 对照直接**复用现有 `rerun0703`**(不为对照另起同种子重跑;若要严格同种子配对,plan 可加一条重跑 eef,代价翻倍)。
