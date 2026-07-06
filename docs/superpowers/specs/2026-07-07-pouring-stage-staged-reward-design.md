# Pouring stage 检测器 + staged 稠密奖励实验 — 设计

日期:2026-07-07
分支:chunk-residual-validation
参照实验:`pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0703`
(run 脚本 `run_pouring_pothiql_joint_rerun0703.sh`)

## 0. 一句话
给 `TwoArmPouring`(GR1)补一个 **5 段** stage 检测器,并起一个 staged 稠密奖励实验——配置
**逐字对齐上面的 pothiql 参照实验,仅把 reward 从 pothiql(potential/HIQL V)换成 staged**。
与 piece/threading/lifttray 的 `run_*_staged_joint.sh` 同构(它们也是各自 pothiql_joint 配置换 staged)。

## 1. 背景 & 任务结构(已用 1009 条 demo 数据核实)
- pouring 是 `TwoArmPouring`,机器人 `GR1FixedLowerBody`(单人形 + 双灵巧手,`robot.gripper` 是
  dict `{'right','left'}`),底层仍是 robosuite/dexmimicgen env → **stage 机制完全适用**。
- 四个物体:`pad`(绿色固定底座,无 joint)/`cup`(起始装着 ball)/`ball`(球,起始在 cup 内)/`bowl`(碗)。
- 成功条件 `_check_success`(two_arm_pouring.py:363)是复合谓词:
  `ball_in_bowl`(球碰碗) 且 `xy_dist<0.1`(球在碗中心) 且 `bowl_on_pad`(碗碰 pad) 且 `bowl_upright`
  (碗 z 轴近竖直) 且 `xy_dist_to_pad<0.06`(碗在 pad 中心)。
- `datagen_info/subtask_term_signals` 给了任务设计者拆的 4 个里程碑,统计 300 条 demo(采样自 1009 条)
  **first-on 顺序 100% 同序**:
  ```
  cup_grasped(0.26) → ball_in_bowl(0.45) → bowl_grasped(0.78) → bowl_on_pad(0.94)
  ```
  语义(与直觉不同,重点):**先抓起 cup 把球倒进「还在桌上」的碗里 → 再抓起碗搬到 pad 上**
  (不是先摆碗再倒)。
- **持久性**(first-on 之后仍为 1 的帧占比):`ball_in_bowl`=0.99、`bowl_on_pad`=0.98(持久);
  `cup_grasped`=0.94(倒球全程握着)、`bowl_grasped`=0.68(放上 pad 后松手,瞬时)。
- **为什么 5 段最合适**(用 datagen 信号在 300 条 demo 上模拟瞬时→running-max 闩锁后的桶占比):
  - 5 段 `0.26/0.19/0.34/0.17/0.05`,回退率 2.4% — 最均衡,4 里程碑全保留。
  - 4 段(去掉"抓碗搬运")中间段涨到 **0.50**,过肥。
  - 3 段(只用持久接触里程碑)`0.45/0.50/0.05` — 前 45%(接近+抓 cup+抬+倒)全塌进 stage 0,太粗。
  与 threepiece(4 段=3 里程碑)、threading(3 段=2 里程碑)同套路:**起步 + 每里程碑一段,末段=成功**。

## 2. 范围
- 检测器**最小实现**:`pouring_stage` + 注册(`STAGE_DETECTORS` / `NUM_STAGES=5`)+ 契约单测 + 真 env 验证。
- staged 实验 run 脚本:`run_pouring_staged_joint.sh` = 参照脚本仅做 pothiql→staged 的 flag 差异。
- 长跑(500k)由用户在空闲 GPU 启动;本切片交付到 代码 + 全绿单测 + 真 env stage 分布验证(含 GR1 grasp
  可用性)+ smoke + ready 命令 为止。**上 500k 前必须先 smoke 掉真 env sim-replay + stages npz 生成。**

## 3. 明确不做(YAGNI)
- per-物体有状态闩锁检测器(现无状态设计够用;瞬时抓取里程碑靠 wrapper 的 running-max 闩锁兜)。
- stage-conditioned / stage_budget 残差(本切片只做 staged **奖励** + stage_balanced 采样)。
- 改 detector / 新 run 脚本之外的任何原仓库文件;coffee / cansort 的同类改造(另开)。
- 超参调优:bonus/action_scale/joint 等一律沿用参照实验(见 §4.3)。

## 4. 组件改动

### 4.1 stage_detectors.py — 新增 `pouring_stage`(追加,不动现有)
> 注:工作树当前含**未提交**的 piece 5→4 段重构(stage_detectors.py / test_stage_detector.py /
> eval_stage_reach.py / verify_stage_instant.py / 删 verify_stage5_instant.py)。本改动是**纯追加**
> (新函数 + 新 registry 条目 + 新测试),与之正交、可共存。提交边界见 §10。

```python
def pouring_stage(env) -> int:
    """TwoArmPouring 5 段:
    0 起步 / 1 抓起 cup / 2 球倒入碗 / 3 抓起碗搬向 pad / 4 碗放上 pad(成功)。

    顺序由 demo 数据核实(100% 同序):cup_grasped→ball_in_bowl→bowl_grasped→bowl_on_pad。
    顶段用 _check_success(比 datagen bowl_on_pad 信号更可靠,后者仅 84% 成功 demo 触发);
    高段短路优先,闩锁(max-so-far)由 wrapper 负责,这里只判瞬时阶段。
    """
    if env._check_success():                            # 4 成功
        return 4
    ball_in_bowl = env.check_contact(env.bowl, env.ball)
    if ball_in_bowl and _grasped(env, env.bowl):        # 3 球已入碗 + 碗被抓起搬运
        return 3
    if ball_in_bowl:                                    # 2 球已倒入碗
        return 2
    if _grasped(env, env.cup):                          # 1 抓起 cup
        return 1
    return 0
```
- 注册:`STAGE_DETECTORS["TwoArmPouring"] = pouring_stage`;`NUM_STAGES["TwoArmPouring"] = 5`。
- `env.cup/bowl/ball`、`check_contact(a,b)`、`_check_success` 都是 `TwoArmPouring` 现成属性/方法
  (two_arm_pouring.py)。`check_contact(self.bowl, self.ball)` 与 `_check_success` 内 `check_contact`
  逐字同款调法。`env.cup/bowl/ball/_check_success` 在在线 worker(dexmg RobosuiteGymWrapper.env)与
  离线 replay(make_replay_env 的 robosuite env)两条路都是同一个 `TwoArmPouring` 实例,属性一致。
- 复用现有 `_grasped(env, obj)` 辅助(stage_detectors.py:14,已对 dict-gripper 取 values() 兜底)。

### 4.2 ⚠ 唯一真风险:GR1 灵巧手的 grasp 检测(必须先验,再上 500k)
`_grasped`→`env._check_grasp` 只在 **Panda**(threading/threepiece,平行夹爪)验过;pouring 是
**GR1 双灵巧手**。**交付前必做**(§7.2):仿 `verify_stage_instant` + `offline_stage_replay` 跑若干
pouring demo,确认 `set_state` 后 `_grasped(env, env.cup/bowl)` 给出干净的 0→1,且整段 stage 分布呈
0→1→2→3→4、分桶均衡(预期 ~0.26/0.19/0.34/0.17/0.05)。
- **退路(若 GR1 grasp 检测不干净)**:把 cup/bowl 的抓取里程碑换成几何持久代理——「cup/bowl 抬离桌面
  z 超阈值」(与夹爪无关,Panda/GR1 通用,持久态),用 datagen `cup_grasped` 信号标定阈值;最差降级到
  3 段纯接触版(0 起步 / 1 球入碗 / 2 成功,只用 check_contact,零 grasp 依赖)。
- 该退路**不改机制**、只改 detector 内谓词,故不影响 §4.3 脚本与 §5 数据流。

### 4.3 test_stage_detector.py — 新增 pouring 契约测试(追加)
仿现有 `_FakeThreadEnv` 写 `_FakePouringEnv`,暴露 `_check_success()` / `check_contact(a,b)` /
`_check_grasp(gripper, object_geoms)` / `robots`(dict-gripper 变体,复用 `_DictGripperRobot`) /
`cup` / `bowl` / `ball`(带 `contact_geoms`)。用例:
- start(无接触/无抓)→ 0;仅抓 cup → 1;球入碗(check_contact 真)→ 2;球入碗 + 抓碗 → 3;success → 4。
- **高段短路**:success 同时低段满足 → 4;球入碗 + 抓碗但未 success → 3。
- **单调性/分离性**用例(核心意图):模拟"抓 cup(1)→倒入(2)→抓碗(3)"三步,断言瞬时依次给 1、2、3
  (证明 5 段能把三个相位切开,不被压成同一段)。
- 仅抓碗但球未入碗 → 0(stage 3 要求球已入碗,防抓碗动作在倒球前误触发);仅 success 而中间谓词跳过 → 4。
- `NUM_STAGES["TwoArmPouring"] == 5`;`get_stage_detector("TwoArmPouring") is pouring_stage`。
- 既有全部用例保持全绿(piece 4 段现状 + threading + wrapper + potential + lifttray 若已落)。

### 4.4 run_pouring_staged_joint.sh — pothiql→staged 差异
基于 `run_pouring_pothiql_joint_rerun0703.sh` 的**同款 env/python 调用**(residual venv /
MUJOCO_GL=egl / HF_HUB_OFFLINE / base_wandb_id artifact `run_anw5pphu_best:v2`),仅改:
- `--reward_shaping potential` → `--reward_shaping staged`
- **删** `--potential_source hiql`
- **删** `--hiql_value_ckpt outputs_chunk/pouring_value_hdf5.pt`(staged 不需 value.pt)
- **加** `--stage_reward_bonus 1.0`
- **加** `--stage_balanced`
- **加** `--offline_stage_cache outputs_chunk/two_arm_pouring_stages.npz`(不存在→首跑 sim-replay 自动生成)
- 新 `--offline_buffer_cache outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_staged_joint_offcache`
  (staged 签名与 pothiql 不同 → **重建**,勿复用 pothiql 的 offcache)
- 新 `--wandb_name` / `--output_dir`:`pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_joint`
- 其余**全不动**:base_wandb_id / dataset ankile/dexmg-two-arm-pouring / offline_dataset_path
  two_arm_pouring.hdf5 / chunk_length 1 / base_action_mode queue / base_n_action_steps 10 /
  action_scale 0.05 / actor_lr 1e-6 / actor raw / offline_fraction 0.5 / offline_base_mode base_policy /
  demo_bc_coef 0.1 / subgoal_conditioned + gc_value_ckpt + high_actor_ckpt + act_feat_cache /
  subgoal_way_steps 15 / online_finetune_value + online_finetune_high_actor / total_env_steps 500000。
- GPU 用 `$1` 传入(起跑前查空闲卡);gate 复用产物存在性(gc_value / high_actor / act_feat_cache / base artifact)。

## 5. 数据流 & 不变量
```
在线: worker pouring_stage(0..4) → info["stage_id"] → wrapper 闩锁 self._stage/max_in_chunk
      → total_reward += bonus·Δ(staged) → buffer(stage_balanced 采样)→ QAgent.update
离线: --offline_stage_cache 缺失 → offline_stage_replay.precompute_stage_cache(hdf5 不渲染)
      → 逐帧 set_state + pouring_stage → 落盘 two_arm_pouring_stages.npz → stage_balanced 用
```
- staged 复用现有 `shaping_reward(mode="staged")` = `bonus·max(0, end−start)`(chunk_env_wrapper.py:20),
  无新奖励逻辑。reward 吃**闩锁后**的单调 stage(防刷分);obs.stage_id 吃**瞬时**(解耦)。
- NUM_STAGES=5 被 `--stage_balanced`(分桶)消费;`--stage_conditioned/--stage_budget/hiql_potential`
  本实验都不用 → 对它们仅作值域声明。

## 6. 边界条件 / 回归安全
- **护住现有 pouring no-stage 跑法**:注册检测器后 `stage_id` 会由恒 0 变为真值,但
  (a) `--reward_shaping none` 下 `shaping_reward` 恒 0 → 奖励不变;
  (b) `--stage_conditioned` 默认关 → stage_id 不 `append_stage` 进 actor/critic;
  (c) `--stage_balanced` 默认关。∴ 只注册不改既有行为。加一条断言守 (a)。
- 离线 replay 不渲染(has_renderer/offscreen=False, use_camera_obs=False)→ 不需 EGL,纯物理+接触。

## 7. 交付边界 & 验证判据
交付物:
1. 代码(4.1、4.3、4.4)+ 全绿单测。
2. 真 env stage 验证(仿 verify_stage_instant + offline_stage_replay 跑 pouring 若干 demo):
   确认 set_state 后 `_grasped(env, env.cup/bowl)` 与 `check_contact("bowl","ball")` 正确、
   stage 分布呈干净 0→1→2→3→4、分桶均衡。**此步同时验掉 §4.2 的 GR1 grasp 可用性 + stages npz
   自动生成路径能在 GR1+hdf5 上跑通。**
3. `--smoke` 跑通 staged 路径不崩。
4. 一条 ready 500k 命令(`run_pouring_staged_joint.sh <free_gpu>`)。

验证判据(用户跑长跑后看,对标 pothiql 臂 `..._pothiql_joint_rerun0703`):
- critic_loss 有信号 / target_q 出现 stage 结构;
- staged vs pothiql 的 eval success_rate 对比(这次实验的真正目的)。

## 8. 风险 & 缓解
- **GR1 双灵巧手上的 `_check_grasp` 从未验过**(现有 `_grasped` 只在 Panda 验) → §4.2 退路 + §7.2 先 smoke。
- **stages npz sim-replay 在 pouring(GR1)+hdf5 上从未跑过** → §7.2 先 smoke,不直接 500k。
- 首跑慢:staged 新 offcache 重建(base_policy 前向,CPU ~80min/~41G)+ stages npz 首次 replay,一次性。
- shaping 偏置(additive 非 PBS)→ 本就是要和 pothiql(PBS)对比的实验变量,接受。
- bonus 量级 1.0/段:沿用 piece/threading/lifttray,不在本切片调。

## 9. 测试计划(TDD,先写后实现)
- detector 契约(§4.3)先写红 → 实现 `pouring_stage` 转绿。
- 回归:`shaping_reward(mode="none")` 恒 0(守 §6a)。
- 全量 `pytest resfit/rl_finetuning/chunk_residual/tests/` 保持绿(含 piece 4 段现状)。
- 真 env / smoke 属交付物验证(§7),非 CI 单测。

## 10. 工作树 / 提交边界(重要)
- 落码前工作树已有**未提交**的 piece 5→4 段重构(stage_detectors.py / test_stage_detector.py /
  eval_stage_reach.py / verify_stage_instant.py / 删 verify_stage5_instant.py)。**非本任务产生,不得丢弃。**
- 另有 lifttray 的 design + plan doc(docs/superpowers/.../2026-07-07-lifttray-*),lifttray_stage 代码**尚未落**。
  若届时 lifttray 已落码,本 pouring 追加与之正交、可共存。
- 本任务对 stage_detectors.py / test_stage_detector.py 为纯追加。提交时**先与用户确认**提交边界:
  是否把工作树既有的 piece 4 段改动单独先提一个 commit,再叠 pouring;还是一并提交。默认:征得同意后分提。
