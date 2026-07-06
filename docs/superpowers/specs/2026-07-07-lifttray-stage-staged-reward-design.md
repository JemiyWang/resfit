# LiftTray stage 检测器 + staged 稠密奖励实验 — 设计

日期:2026-07-07
分支:chunk-residual-validation
参照实验:`lifttray_actfeat_hdf5_bp_bcfixed01_hiqlv512_sg15_pothiql_joint_rerun0629`
(run 脚本 `run_lifttray_pothiql_joint_rerun0629.sh`)

## 0. 一句话
给 `TwoArmLiftTray` 补一个 4 段 stage 检测器(按"搬上盘方块数"计数),并起一个 staged 稠密奖励
实验——配置**逐字对齐上面的 pothiql 参照实验,仅把 reward 从 pothiql(potential/HIQL V)换成 staged**。
与 piece/threading 的 `run_*_staged_joint.sh` 同构(它们也是各自 pothiql_joint 配置换 staged)。

## 1. 背景 & 任务结构(已用数据核实)
- 已对 1033 条 `two_arm_lift_tray.hdf5` demo 统计确认任务是**三顺序相位**(100% 同序):
  相位1 左臂拿左方块 obj1 放上盘 → 相位2 右臂拿右方块 obj0 放上盘 → 相位3 双臂抬盘(成功)。
  子任务信号 `datagen_info/subtask_term_signals`={obj0_off_ground, obj1_off_ground} 即 dexmimicgen
  自带的相位分界。方块起步在盘外(到 pot 水平距 ~0.43),搬上盘后进入盘内 footprint(~0.09)
  且贴盘面(z 相对 pot 恒 −0.010,p10=p90,近零方差)。
- **为什么里程碑用"方块在盘上"(持久)而不是"方块离地/抓起"(瞬时)**:wrapper 对
  **检测器返回的瞬时 stage 取 max-so-far 闩锁**(chunk_env_wrapper.py:189)。两次拿方块严格先后、
  **不重叠**(每次抓起→放下→释放),所以"当前有几个方块离地"这一瞬时量走 0→1→0→1→0→2,
  经 max 闩锁两次 pick 会被压成同一段 stage 1、只有最后合抬才到 2 —— **分不开两次抓取**。
  而"方块在盘上"是持久态(放好后一直在盘上到抬盘),瞬时计数单调 0→1→2,闩锁后干净地把
  两次搬运切成 stage 1 与 stage 2。若坚持"抓起即计",需给检测器加 per-方块闩锁状态(超范围,见 §3)。

## 2. 范围
- 检测器**最小实现**:`lifttray_stage` + 注册(`STAGE_DETECTORS` / `NUM_STAGES=4`)+ 单测 + 真 env 验证。
- staged 实验 run 脚本:`run_lifttray_staged_joint.sh` = 参照脚本仅做 pothiql→staged 的 flag 差异。
- 长跑(500k)由用户在空闲 GPU 启动;本切片交付到 代码 + 全绿单测 + 真 env stage 分布验证 + smoke
  + ready 命令 为止。**上 500k 前必须先 smoke 掉真 env sim-replay + stages npz 生成。**

## 3. 明确不做(YAGNI)
- "抓起瞬间即计"的 per-方块闩锁有状态检测器(现无状态设计够用;用持久"在盘上"里程碑)。
- stage-conditioned / stage_budget 残差(本切片只做 staged **奖励**)。
- 改 detector 之外的任何原仓库文件;pouring 的同类改造(另开)。
- 超参调优:bonus/action_scale/joint 等一律沿用参照实验(见 §4.3)。

## 4. 组件改动

### 4.1 stage_detectors.py — 新增 `lifttray_stage`(追加,不动现有)
> 注:工作树当前含**未提交**的 piece 5→4 段重构(threepiece_stage / NUM_STAGES 5→4 / 对应测试)。
> 本改动是**纯追加**(新函数 + 新 registry 条目 + 新测试),与之正交、可共存。提交边界见 §10。

```python
def lifttray_stage(env) -> int:
    """TwoArmLiftTray 4 段:0 起步 / 1 一个方块搬上盘 / 2 两个方块都上盘 / 3 抬盘成功。

    里程碑用"方块与盘底(pot_base)接触"(持久态,复用 _check_success 同款谓词),
    按已上盘方块数计数(顺序无关);高段短路优先,闩锁(max-so-far)由 wrapper 负责。
    """
    if env._check_success():                              # 3 抬盘成功
        return 3
    n = int(env.check_contact("pot_base", env.obj0)) + \
        int(env.check_contact("pot_base", env.obj1))
    if n >= 2:                                            # 两块都在盘上
        return 2
    if n >= 1:                                            # 一块在盘上
        return 1
    return 0
```
- 注册:`STAGE_DETECTORS["TwoArmLiftTray"] = lifttray_stage`;`NUM_STAGES["TwoArmLiftTray"] = 4`。
- `env.check_contact("pot_base", env.obj0)` 与 `_check_success`(two_arm_lift_tray.py:374)逐字同款调法,
  签名 `check_contact(geoms_1, geoms_2=None)` 匹配,返回 bool。`env.obj0/obj1/_check_success` 在
  在线 worker(dexmg RobosuiteGymWrapper.env)与离线 replay(make_replay_env 的 robosuite env)两条路
  都是同一个 `TwoArmLiftTray` 实例,属性一致。
- **真·未知(需验证)**:`set_state_from_flattened` + `sim.forward()` 后 `check_contact("pot_base", obj)`
  是否即时正确(接触在 forward 后计算,理论 OK);"方块在盘上"在离线 replay 逐帧是否给出干净 0→1→2→3。

### 4.2 test_stage_detector.py — 新增 lifttray 契约测试(追加)
仿现有 `_FakeThreadEnv` 写 `_FakeLiftTrayEnv`,暴露 `_check_success()` / `check_contact(geom, obj)` /
`obj0` / `obj1`。用例:
- start(无接触)→ 0;仅 obj0 上盘 → 1;仅 obj1 上盘 → 1;两块都上盘 → 2;success → 3。
- 高段短路:success 同时低段满足 → 3;两块上盘但未 success → 2。
- **单调性/分离性**用例(核心意图):模拟"obj1 先上盘(1)→obj0 后上盘(2)"两步,断言瞬时依次给 1、2
  (证明持久里程碑能把两次搬运分成不同段,不被压成同一段)。
- `NUM_STAGES["TwoArmLiftTray"] == 4`;`get_stage_detector("TwoArmLiftTray") is lifttray_stage`。
- 既有全部用例保持全绿(piece 4 段现状 + threading + wrapper + potential)。

### 4.3 run_lifttray_staged_joint.sh — pothiql→staged 差异
基于 `run_lifttray_pothiql_joint_rerun0629.sh` 的**同款 env/python 调用**(residual venv /
MUJOCO_GL=egl / HF_HUB_OFFLINE / LD_LIBRARY_PATH / base_wandb_id artifact 路径),仅改:
- `--reward_shaping potential` → `--reward_shaping staged`
- **删** `--potential_source hiql`
- **删** `--hiql_value_ckpt outputs_chunk/lifttray_value_hdf5.pt`(staged 不需 value.pt)
- **加** `--stage_reward_bonus 1.0`
- **加** `--stage_balanced`
- **加** `--offline_stage_cache outputs_chunk/two_arm_lift_tray_stages.npz`(不存在→首跑 sim-replay 自动生成)
- 新 `--offline_buffer_cache outputs_chunk/lifttray_actfeat_hdf5_bp_hiqlv512_sg15_staged_offcache`
  (staged 签名与 pothiql 不同 → **重建**,勿复用 pothiql 的 offcache)
- 新 `--wandb_name` / `--output_dir`:`lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_staged_joint`
- 其余**全不动**:base_wandb_id / dataset ankile/dexmg-two-arm-lift-tray / offline_dataset_path
  两_arm_lift_tray.hdf5 / chunk_length 1 / base_action_mode queue / base_n_action_steps 10 /
  action_scale 0.05 / actor_lr 1e-6 / actor raw / offline_fraction 0.5 / offline_base_mode base_policy /
  demo_bc_coef 0.1 / subgoal_conditioned + gc_value_ckpt + high_actor_ckpt + act_feat_cache /
  subgoal_way_steps 15 / online_finetune_value + online_finetune_high_actor / total_env_steps 500000。
- GPU 用 `$1` 传入(参照脚本硬编码 5,起跑前查空闲卡另择);gate 复用产物存在性
  (gc_value / high_actor / act_feat_cache / base artifact)。

## 5. 数据流 & 不变量
```
在线: worker lifttray_stage(0..3) → info["stage_id"] → wrapper 闩锁 self._stage/max_in_chunk
      → total_reward += bonus·Δ(staged) → buffer(stage_balanced 采样)→ QAgent.update
离线: --offline_stage_cache 缺失 → offline_stage_replay.make_replay_env(hdf5 不渲染)
      → 逐帧 set_state + lifttray_stage → 落盘 two_arm_lift_tray_stages.npz → stage_balanced 用
```
- staged 复用现有 `shaping_reward(mode="staged")` = `bonus·max(0, end−start)`,无新奖励逻辑。
- NUM_STAGES=4 只被 `--stage_conditioned/--stage_budget/hiql_potential` 消费;本实验都不用 → 仅作值域声明。

## 6. 边界条件 / 回归安全
- **护住现有 lifttray no-stage 跑法**:注册检测器后 `stage_id` 会由恒 0 变为真值,但
  (a) `--reward_shaping none` 下 `shaping_reward` 恒 0 → 奖励不变;
  (b) `--stage_conditioned` 默认关 → stage_id 不 `append_stage` 进 actor/critic;
  (c) `--stage_balanced` 默认关。∴ 只注册不改既有行为。加一条断言守 (a)。
- 离线 replay 不渲染(has_renderer/offscreen=False, use_camera_obs=False)→ 不需 EGL,纯物理+接触。

## 7. 交付边界 & 验证判据
交付物:
1. 代码(4.1–4.3)+ 全绿单测。
2. 真 env stage 验证(仿 verify_stage_instant + offline_stage_replay 跑 lifttray 若干 demo):
   确认 set_state 后 `check_contact("pot_base", obj)` 正确、stage 分布呈干净 0→1→2→3、分桶均衡。
   **此步同时验掉 stages npz 自动生成路径能在 PandaDex+hdf5 上跑通。**
3. `--smoke` 跑通 staged 路径不崩。
4. 一条 ready 500k 命令(`run_lifttray_staged_joint.sh <free_gpu>`)。

验证判据(用户跑长跑后看,对标 pothiql 臂):
- critic_loss 有信号 / target_q 出现 stage 结构;
- staged vs pothiql 的 eval success_rate 对比(这次实验的真正目的)。

## 8. 风险 & 缓解
- **stages npz sim-replay 在 lifttray(PandaDex)+hdf5 上从未跑过** → §7.2 先 smoke,不直接 500k。
- 首跑慢:staged 新 offcache 重建(base_policy 前向,CPU 慢)+ stages npz 首次 replay,一次性。
- shaping 偏置(additive 非 PBS)→ 本就是要和 pothiql(PBS)对比的实验变量,接受。
- bonus 量级 1.0/段:沿用 piece/threading,不在本切片调。

## 9. 测试计划(TDD,先写后实现)
- detector 契约(§4.2)先写红 → 实现 `lifttray_stage` 转绿。
- 回归:`shaping_reward(mode="none")` 恒 0(守 §6a)。
- 全量 `pytest resfit/rl_finetuning/chunk_residual/tests/` 保持绿(含 piece 4 段现状)。
- 真 env / smoke 属交付物验证(§7),非 CI 单测。

## 10. 工作树 / 提交边界(重要)
- 落码前工作树已有**未提交**的 piece 5→4 段重构(stage_detectors.py / test_stage_detector.py /
  eval_stage_reach.py / verify_stage_instant.py / 删 verify_stage5_instant.py)。**非本任务产生,不得丢弃。**
- 本任务对 stage_detectors.py / test_stage_detector.py 为纯追加。提交时**先与用户确认**:
  是否把 piece 4 段改动单独先提一个 commit,再叠 lifttray;还是一并提交。默认:征得同意后分两个 commit。
```
