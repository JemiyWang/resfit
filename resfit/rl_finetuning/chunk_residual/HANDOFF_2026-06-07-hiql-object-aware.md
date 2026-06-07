# HANDOFF 2026-06-07 (③a' object-aware V) — 给 V 加 eef-rel-piece 物体维度

接 `HANDOFF_2026-06-07-hiql.md`(③ HIQL-Φ)。本篇:给 ③a 的 action-free value V 加"双臂 eef 相对两 piece 的位置"(state 18→30 维),缓解 online 偏离 demo 流形时 V 高估。分支 `chunk-residual-validation`。手动复刻 Superpowers 流程(本机没装 superpowers skill),产物在 `docs/superpowers/{specs,plans}/2026-06-07-hiql-value-object-aware*`。

---

## ✅ 完成更新(2026-06-08):Task 8(正确架构)+ Task 9(端到端冒烟)全过

**Task 8 按 §3 重做完毕**(observation.state 保持 18 维,rel_piece 经 info 只喂 Φ)。**Task 9 端到端冒烟 exit 0 通过**。改动均 TDD(先红后绿),**已 commit `a223750`**(12 文件,+565/-47,分支 chunk-residual-validation)。

- **observation.state 始终 18 维**(actor/critic + 部署不变):`dexmg._process_obs`/`_process_obs_for_space_inference` 不再拼 rel;原 `_append_rel_piece` 改为 `_rel_piece_info()`(只算 (12,) 不拼),`step`/`reset` 像 `stage_id` 一样经 `info["rel_piece"]` 透出(eef_piece 模式)。
- **HiqlPotential.phi(state_std, rel_piece_raw=None)**:from_ckpt 读 `state_mode`+`rel_piece_mean/std`;eef_piece 时 `_value_input` 把 raw rel 标准化 `(rel-mean)/std` 后拼成 30 维喂 V(与 `train_hiql_value` 逐位同源);eef 模式忽略 rel(零回归)。**设计决策(偏离原 §4.4)**:online/offline **统一**走 `phi(18维std state, raw rel)`,标准化在 phi 内部做 → 两端数值严格一致(原 §4.4 想让 offline 传 30 维预标准化 state_seq,会和 §4.2 online 口径分叉,故废弃)。
- **online**(`chunk_env_wrapper`):新增 `_extract_rel(info)`(取批后 (num_envs,12) 的 env0)+ `_start_rel_piece` 跨 chunk 携带(镜像 `_start_state_std`);`step` 的 potential 分支 `phi(start_std, start_rel)`/`phi(end_std, rel_next)`。done 时 SAME_STEP autoreset 后 `info` 顶层是 `reset_info`(dexmg.reset 也透 rel),故携带的 rel 与返回的 reset obs 同步——已核 gymnasium 1.1.1 async worker 逻辑。
- **offline**(`offline_hdf5_buffer.transition_rewards/fields` 加 `rel_piece_seq`;`build_offline_buffer` 在 `need_rel`(eef_piece)时每 demo `replay_eef_rel_piece` 出 (T,12) raw 传给 Φ)。buffer 的 observation.state 仍存 18 维 `state_n`。
- **train_chunk_residual**:potential 移到 `create_vectorized_env` **之前**构建 → 按 value.pt 的 `state_mode` 给训练 vec_env 传 `state_mode`(eval_vec 保持 eef、省 sim);加维度命门断言 `potential.model.state_dim == observation.state + (12 if eef_piece)`。
- **🐞 顺手修一个潜在 bug**:`_offline_buffer_signature` 原**不含 Φ 身份** → potential_source=hiql 时换 value.pt(如 18 维↔30 维)会**撞签名错误复用旧 reward 缓存**。已加 hiql 源的 `hiql_value_ckpt/phi_scale/potential_scale/value_state_mode` 进签名(stage 源不加键、**现有 stage 缓存向后兼容**)。⚠️ 副作用:**现有 18 维 hiql 缓存 `offline_buf_piece_hiql` 下次 hiql 启动会失效重建一次**(~14min);stage 缓存 `offline_buf_piece_as0.05` 不受影响。critic_lr 在跑那条不受影响(已 load)。
- **新增/改测试全绿**(84 passed 合并跑):`test_hiql_potential`(+6 eef_piece/签名相关)、`test_chunk_env_wrapper`(+online rel 透传)、`test_dexmg_rel_piece_info`(新,observation.state 不被拼大 + info 透出)、`test_build_offline_buffer_rel`(新,eef_piece replay rel + 签名区分)。
- **端到端冒烟**(gpu7,`outputs_chunk/smoke_objaware`,日志 `/tmp/smoke_objaware.log`):`--potential_source hiql --hiql_value_ckpt three_piece_value_objaware.pt --offline_num_demos 4 --smoke`。打印 `[hiql-phi] ... state_mode=eef_piece scale=4.6093` + `[state-mode] env_state_mode=eef_piece(observation.state 仍 18 维)`;offline 建 937 条(真 rel replay+30 维 V 重算)无错;1 step + eval 渲染跑通;**零 error/assert/traceback,exit 0**。
- **⚠️ 起正式 A/B 仍看 §5**:瓶颈大概率在 critic(非 Φ),起 object-aware 正式实验前先确认 critic_lr 结果。正式跑时 **object-aware 用独立 `--offline_buffer_cache` 目录**(别复用 `offline_buf_piece_hiql`);命令照下方 §6 的 B-run 把 ckpt 换成 `three_piece_value_objaware.pt`、base 用本地 `resfit/out/piecce/best/policy`、hdf5 用 `resfit/dataset/two_arm_three_piece_assembly.hdf5`。

---

## 0. 一句话现状

- **offline 链路 + online 同源命门 + value 训练侧已就位**(Task 1-7 + 4-5,**全部已 commit**:`cb84de6`→`9b0bb11`→`7467eb0`,分支 chunk-residual-validation;EGL 修复 `584a07b`)。
- **但 Task 8(③b 接入)发现一个架构问题必须先改**(见 §3):rel_piece 是特权,**不能进 `observation.state`**(否则 actor/critic 也吃到、违反"脚手架不进房子"),要像 `stage_id` 那样经 `info` 透出、只喂给 V。
- **Task 6 当前实现(把 rel_piece 拼进 observation.state)方向错了,要按 §4 改。**

---

## 1. 目标 + 设计要点

- V 现状只看 18 维 eef 本体(`STATE18_KEYS`,robot0/1 的 eef_pos/quat/gripper),**看不见物体** → online 手到位但件没对上时 V 高估(B run 已证基础版没救起来,见 §5)。
- 加 12 维 = 双臂 eef 相对两 piece 的世界系位置 `[eef0-p1, eef0-p2, eef1-p1, eef1-p2]`,从**仿真特权状态**算(online robosuite obs 没有 piece key,数据集 hdf5 有但 online env 不输出)。
- **特权合法**:V 是 PBS 的 Φ、只训练算 reward、部署丢弃 → 用特权 piece pose 天经地义(同 stage 检测器)。**但前提是只给 V,不给策略**(见 §3)。

---

## 2. 已完成(测试/同源都绿)

| Task | 内容 | 状态 |
|---|---|---|
| 1 | `object_state.py`:eef_rel_piece / read_eef_positions / **compute_eef_rel_piece_from_env** / rel_piece_stats | ✅ commit `cb84de6`+`9b0bb11` |
| 2 | `offline_hdf5_buffer.assemble_state(mode)`:eef(18)/eef_piece(30) | ✅ `cb84de6` |
| 3 | `offline_stage_replay.replay_eef_rel_piece`:replay set_state 从 sim 算 rel_piece | ✅(改用 from_env)`9b0bb11` |
| 6 | `dexmg.RobosuiteGymWrapper(state_mode)` + `_append_rel_piece` | ⚠️ **见 §3,要改** `9b0bb11` |
| 7 | online/offline 同源命门 **PARITY_OK(0.2mm)** | ✅ |
| 4 | `train_hiql_value --state_mode eef_piece`(replay 全 demo + rel_piece stats) | ✅ commit `7467eb0`,冒烟通过(5 demo→state_dim 30) |
| 5 | `hiql_value.save/load_value` 存读 state_mode + rel_piece_mean/std | ✅ commit `7467eb0`,load 验证通过 |

- **同源命门坑(关键,已修)**:offline replay `set_state` 后 `env._get_observations()` 的 robot obs **cache stale**(eef 差 0.24m);必须用 `env._eef0_xpos`/`_eef1_xpos`(sim 实时,== hdf5 eef、差 0.2mm)。两端统一走 `compute_eef_rel_piece_from_env(env)`。
- 全量 30 维 value 训练:`outputs_chunk/three_piece_value_objaware.pt`,2026-06-07 跑(`tee three_piece_value_objaware.log`)。**replay 全 1006 demo 已完成**(`state_mode=eef_piece demos=1006 transitions=238821 state_dim=30`),随后训 5 万步(4 线程 CPU,~2.5min)。**已训完**(2026-06-07):`v_stats={min:0.149, max:1.017, mean:0.448}`(max≫min、有区分度,和 18 维版 {0.173,1.010,0.447} 几乎一致)。30 维 value.pt 就绪,等 Task 8 接入。

---

## 3. ⚠️ 架构问题(Task 8 前必读,必须先改)

**rel_piece 是特权,绝不能进 `observation.state`。** 证据:`train_chunk_residual.py:413` `state_dim = obs0["observation.state"].shape[1]` —— actor/critic 的输入维度直接取自 `observation.state`。Task 6 当前把 rel_piece 拼进 `observation.state`(`dexmg._append_rel_piece`),会让 **actor/critic 也变 30 维、吃到特权 rel_piece** → 策略依赖特权、真机部署拿不到 piece pose → 不可部署,违反"脚手架(Φ)不进房子(策略)"。

正确做法:**observation.state 保持 18 维(actor/critic + 部署),rel_piece 只单独喂给 V(Φ)**,像 `stage_id` 一样经 `info` 透出。

落点(已查清):
- online Φ 在 `chunk_env_wrapper.step` 用 `self.potential.phi(self._start_state_std)`(标准化 observation.state)。
- worker 的 stage 特权经 `dexmg.py step` 里 `info["stage_id"]` 透出(spawn 边界)。rel_piece 照此办理。
- offline Φ 在 `offline_hdf5_buffer.transition_fields` 用 `potential.phi(state_seq)`(L154)。

---

## 4. 待做:Task 8 正确架构 + Task 9

### Task 8(按 §3 重做)
1. **撤销 Task 6 的 observation.state 拼接**:`dexmg.RobosuiteGymWrapper._append_rel_piece` 不再拼进 `observation.state`(保持 18 维)。改为:`dexmg.step` 里算 `rel = compute_eef_rel_piece_from_env(self.env)` 放 `info["rel_piece"]`(像 `stage_id`,只在 eef_piece 模式)。
2. **HiqlPotential 改造**:`from_ckpt` 读 `state_mode` + `rel_piece_mean/std`;`phi(state18_std, rel_piece_raw=None)` 在 eef_piece 时标准化 rel_piece、拼成 30 维再喂 V。
3. **online**:`chunk_env_wrapper.step` 从 `info["rel_piece"]` 拿 raw rel_piece,连同 `_start_state_std`(18) 一起传给 `potential.phi`。
4. **offline**:`transition_fields` 在 eef_piece 时,`state_seq` 用 30 维(18 + 标准化 rel_piece,replay 算;rel_piece 由 `replay_eef_rel_piece` 出)。`state_seq` 只算 Φ、不进 buffer 的 observation.state(那是 18 维)。
5. `train_chunk_residual`:HiqlPotential 按 value.pt 的 state_mode 自动决定 18/30;断言 online/offline 喂 V 的维度一致。
6. 单测:HiqlPotential.phi 在 eef_piece 下拼接正确 + 维度;online(info 透出 rel_piece)/offline(replay)同源。

### Task 9 端到端冒烟
- 用全量 `three_piece_value_objaware.pt`(state_mode=eef_piece)起 ③b `--smoke`(照基础版 hiql smoke 口径),确认:`observation.state` 仍 18 维(actor/critic 不变)、`[hiql-phi] potential on` 打印、offline buffer 用 30 维 V 重算 reward 不报错、跑几步不崩。

### 跑实验(看值不值得)
- 用 `outputs_chunk/plot_hiql_V_curve.py` 思路,跑一条 online **失败** episode 对比 18维V vs 30维V,看"手到位件没对上"段 V 是否不再虚高。
- **但先看 §5**:基础版已证瓶颈在 critic、不在 Φ,object-aware 大概率也救不了 → 起正式实验前先确认 critic 侧实验结果。

---

## 5. 重要背景:B run 结果指向瓶颈不在 Φ(影响要不要继续)

- B run(基础版 18 维 hiql Φ,wandb `nas10_best_pothiql`/`2v1rfnuk`)跑到 100k:10k bump 0.58 → 30k 塌到 0.04 → 后5点滑动均值 ~0.17,stage3 回退率 58%→64%(不降反升)。**和 memory 里 potential/staged/budget/relabel 同款衰减**。
- 坐实:**reward-shaping 模式(换 Φ 形式/信息量)救不了 20k 塌方,根因在 critic 侧(Q 高估)**。
- 已起对照:**critic_lr=3e-5 实验**(`nas10_pothiql_criticlr3e-5`,gpu3)。**用 18 维 value(`three_piece_value.pt`,和 B 同),vs B 唯一差别=critic_lr 从默认 1e-4 降到 3e-5**(控制变量:孤立验证"降 critic_lr 能否救塌方";和 30 维 object-aware 是两条独立线、没混)。当前 10k=0.56(和 B 的 0.58 接近),**关键判读点 20-30k**(B 在那塌到 0.04)——盯 success 是否还塌、`residual_norm` 是否还失控涨。复用 buffer `offline_buf_piece_hiql` 秒级命中。
- **判断**:object-aware V 还是改 Φ,若 critic 是真瓶颈则也白做。后半段(Task 8-9 + 起实验)**优先级取决于 critic_lr 实验结果**。

---

## 6. 关键路径/命令

- 全量训 30 维 value:`conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_value --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 --dataset ankile/dexmg-two-arm-three-piece-assembly --state_mode eef_piece --output outputs_chunk/three_piece_value_objaware.pt`(CPU,`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MUJOCO_GL=egl`;replay 全 demo 十几分钟)。
- 环境/坑见 memory `project_resfit_object_aware_value`、`project_resfit_env_on_mnt`、`project_resfit_egl_render_samecard`、`project_resfit_stage_offline`。
- EGL 修复(本会话):新 run 渲染跟随计算卡需 `deps/robosuite/.../binding_utils.py` 已 patch(放宽 MUJOCO_EGL_DEVICE_ID assert,在 `resfit-egl-patch` 分支 commit `bf5c13ab`)+ dexmg 逻辑号(主仓 `584a07b`)。
