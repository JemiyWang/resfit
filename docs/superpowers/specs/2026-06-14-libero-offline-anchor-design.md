# LIBERO + offline 锚点 接进 chunk_residual 残差RL — 设计

**日期**: 2026-06-14
**状态**: 设计获批,待写实施计划
**前置**: `2026-06-12-libero-residual-pi0-onboarding-design.md`(libero 在线残差已落地,base=pi0_libero)

## 1. 目标

让 `--env_family libero` 也能用 **offline demo 当锚**:打开 `offline_fraction>0`(RLPD 混采)+ `demo_bc_coef>0`(BC 锚)。当前 libero 路 `validate_libero_cfg` 硬禁 `offline_fraction>0`,因为 dexmg 的 offline 灌入管线(`build_offline_buffer`/`read_per_demo_states`)写死了双臂 STATE18、`{name}_image` 的 HDF5 键、且**无条件调 stage 检测器**(libero 无检测器 → crash),无法直接复用。

**范围(第一版)**:单 `(suite, task_id)`;offline demo = **当前训练任务自己的 demo**(按任务语言串自动匹配,不加额外 flag);`offline_base_mode=base_policy`(对每帧 demo 跑 pi0_libero 出 base 动作,锚 = `demo − pi0` 的真残差);不支持 stage/subgoal/reward_shaping(libero 本就不支持)。

**范围外**:改默认值(action_scale 等)是另一件事,等这条通了单独决定。多任务 offline、跨任务锚(`--libero_demo_task_id`)留待后续。

## 2. demo 数据源

**LeRobot `physical-intelligence/libero`**(本机 `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero`,1693 episodes / 40 tasks,即 pi0_libero 训练用的同一份)。每帧 parquet 列:`image`(dict{bytes,path},JPEG,256×256 RGB)、`wrist_image`(同)、`state`(float32[8])、`actions`(float32[7])、`task_index`(int64)、`episode_index` 等。本机**无 raw LIBERO hdf5**,也不需要。

**task→demo 映射**:`benchmark.get_benchmark_dict()[suite]().get_task(task_id).language` 得任务语言串 → 在 `meta/episodes.jsonl` 里筛 `tasks[0] == 语言串` 的 episodes(数据集 task_index 顺序 ≠ benchmark suite/task_id 顺序,**必须按语言串匹配**)。已验证 libero_10 十条语言串都在数据集里;`put both moka pots on the stove` → 29 条 demo / 11808 帧。

## 3. 三个命门(实现必须遵守)

1. **图像朝向(最要命)**:在线 `rollout_libero.py:425` 对 env 渲染做 `[::-1,::-1]` 翻转;openpi 训练侧 `LiberoInputs._parse_image` **不翻**。→ **LeRobot 数据集里的图是预翻正的**。在线 `libero_env._process` 用 `flip_resize_image`(翻转 raw env)得到翻正 84 图。所以 offline reader 读数据集图时**只能 `resize_with_pad`、绝不能再套 `flip_resize_image`**(否则二次翻转,offline 图朝向与在线 obs 相反,锚全废,且喂给 pi0 的朝向也错)。
2. **reward 一致性**:数据集 `rewards/*.npy` 是逐帧**事件 reward**(pick/place 多点给分,单条总分≈4),与在线**稀疏成功 reward(`success=reward==1.0`)语义不一致**。offline 必须用**末帧 `reward=1.0、done=True`,其余 0**(demo 是成功轨迹),与在线一致;**不用** `rewards/*.npy`(混进去会毒化 critic 的单一 Q)。
3. **归一化同源**:offline 的 state/action 必须用**和在线同一套** `ActionScaler`+`StateStandardizer`(`build_libero_scalers` from `stats.json`),否则 offline 与 online 不在同一缩放空间。

## 4. 组件

### 4.1 `resfit/rl_finetuning/chunk_residual/libero_offline.py`(新)

- `resolve_libero_demo_task(suite, task_id, lerobot_root) -> (language: str, task_index: int, episode_parquet_paths: list[str])`
  纯逻辑(只读 meta jsonl + benchmark)。按语言串匹配;找不到任务/无 episode → 明确 `ValueError`。
- `read_libero_demo(parquet_path) -> dict`
  解一条 episode parquet → `{"state": (T,8) f32, "action": (T,7) f32, "agentview": (T,256,256,3) u8, "wrist": (T,256,256,3) u8}`(解 JPEG bytes;`image`→agentview,`wrist_image`→wrist)。
- `build_libero_offline_buffer(offline_rb, *, lerobot_root, suite, task_id, action_scaler, state_standardizer, base_policy, base_mode, base_device, image_size=84, serve_image_size=224, gamma, n_step, num_demos=None) -> None`
  迭代该任务的每条 demo,逐帧组装 transition 灌进 `offline_rb`。
  - **图**:`agentview/wrist` → `resize_with_pad(·, image_size)`(出 HWC uint8)→ transpose 成 **CHW uint8** 存入 buffer(与在线 `add_chunk_transition` 经 `to_uint8` 后存的 CHW uint8 同 layout/同 [0,255] 值域;在线是 `_chw01` 出 CHW float[0,1] 再 `to_uint8`,数值等价)。键名 `observation.images.agentview` / `observation.images.robot0_eye_in_hand`。**无翻转**(只 resize)。
  - **state**:`state_standardizer` 标准化 → `observation.state`(8)。
  - **action**:`action_scaler.scale(demo_action)` → top-level `action`(7,chunk_length=1)。
  - **base_action**:
    - `base_mode="base_policy"`:对每帧用 `build_libero_serve_obs`(serve_image_size=224,**无翻**)过 `base_policy.select_action` → 原始动作 → `action_scaler.scale` → `observation.base_action`(7)。`bc_target = action − base_action`(残差)。需先 `base_policy.reset()` 清队列(逐 demo)。
    - `base_mode="gt"`:`observation.base_action = action`(逐位等价,`bc_target=0`)。第一版默认 base_policy,gt 作为快速/无 serve 的退路保留。
  - **stage_id**:常量 0(`observation.stage_id` (1,) f32)。无 stage 检测、无 sim replay。
  - **next**:下一帧 obs;`done`:仅末帧 True;`reward`:仅末帧 1.0,其余 0.0。
  - top-level:`max_stage=0`、`_priority=10.0`。
  - 产出的 td **必须与在线 `add_chunk_transition` 同构**(obs 键集、action、next/obs/done/reward、max_stage、_priority);单测断言。

### 4.2 `train_chunk_residual.py` 接线

- `validate_libero_cfg`:把 `assert offline_fraction==0`(315 行附近)放开为"允许 `offline_fraction>0`";新增:`offline_fraction>0` 时要求 `base_action_mode=queue`、`chunk_length=1`、`actor=raw`;**仍禁** `subgoal_conditioned`/`stage_conditioned`/`stage_budget`/`reward_shaping not in (None,"none")`/`staged_reward`/`relabel`(第一版)。`demo_bc_coef>0` 时要求 `offline_fraction>0`(沿用现有 606 行约束)。
- offline 构建分支(`if args.offline_fraction > 0.0:` ~715 行):`env_family=="libero"` → 走 libero 缓存/构建路径(下),否则 dexmg 原样不变(零回归)。
- **缓存**:复用现有 `offline_buffer_cache` memmap 机制(`_offline_cache_valid`/`_save_offline_buffer`/`TensorDict.load_memmap`)。新增 libero 版签名 `_libero_offline_signature`(锁:lerobot_root、suite、task_id、language、base_mode、base_policy_type+pi0 连接参数(base_policy 模式)、action_scale、min_range_per_dim、image_size、gamma、n_step、num_demos)。命中则 memmap 直载,免重跑 pi0。
- offline buffer 走与 dexmg 同一个 `_new_offline_rb`(`TensorDictPrioritizedReplayBuffer` + `MultiStepTransform`),保证 n_step/混采行为一致。
- **base_policy 实例**:base_policy 模式构建 offline buffer 时复用已建的 `base_policy`(训练用那个);构建在训练循环前、reset 一次清队列即可。

### 4.3 混采 / BC(无需改)

`offline_fraction` 混采(`concat_mixed_batch`)、`demo_bc_coef` 的 `bc_batch=offline_rb.sample(...)`、`agent.update(..., bc_batch, ref_agent)` 全部复用现有逻辑——只要 offline td 与在线同构即可,无需改训练循环。

## 5. 数据流

```
LeRobot parquet(task8 的 29 条)
  → read_libero_demo 解码(逐帧 state8/action7/agentview256/wrist256)
  →〔base_policy: 每帧 build_libero_serve_obs@224(无翻)→ pi0.select_action → base_action〕
  → ActionScaler.scale(action, base_action) + StateStandardizer(state)   # 与在线同源
  → resize_with_pad(图,84)（无翻）→ CHW uint8
  → 同构 TensorDict(obs{state,base_action,stage_id=0,images}, action, next{obs,done,reward}, max_stage=0,_priority=10)
  → offline_rb(prioritized + MultiStepTransform)  [可 memmap 缓存]
训练: online_rb / offline_rb 按 offline_fraction 混采;demo_bc_coef>0 → bc_batch from offline_rb → actor BC 向 (demo−base) 修正
```

## 6. 错误处理

- 任务语言串匹配不到 / 无 episode → `ValueError`(提示 suite/task_id/语言串)。
- `lerobot_root` 或 meta 缺失 → 明确报错。
- base_policy 模式 serve 不可达 → 复用在线 base_policy 的连接失败路径(同在线)。
- 缓存签名不符 → 重建并覆盖。
- parquet 帧数与 reward/episode meta 不符 → 第一版不用 reward npy,故无此耦合;但 demo 长度 T<2(无 transition)→ 跳过并 warn。

## 7. 测试

**单测(conda env `resfit-libero`,无 serve/sim,纯逻辑+stub)**:
1. `read_libero_demo`:合成迷你 parquet(2–3 帧,JPEG bytes 图)→ 验 `state(T,8)/action(T,7)/agentview(T,256,256,3)/wrist` shape+dtype。
2. `resolve_libero_demo_task`:stub `meta/episodes.jsonl`+benchmark → 语言串匹配到正确 episodes;找不到 → `ValueError`。
3. `build_libero_offline_buffer` `base_mode="gt"` + stub `ActionScaler`/`StateStandardizer`:
   - 验 offline td 与在线 `add_chunk_transition` **同构**(键集/shape/dtype 一致)。
   - reward 末帧=1、其余 0;done 仅末帧;max_stage=0;`observation.stage_id`=0;`_priority`=10。
   - **喂上下非对称的测试图(如上半白下半黑)→ 断言 offline obs 图未被上下翻转**(防误用 flip)。
4. `build_libero_offline_buffer` `base_mode="base_policy"` + **stub base_policy**(`select_action` 返回固定动作):`observation.base_action` 被填、`action` 仍为 demo 动作(隐含 `bc_target=demo−base`)。
5. validate 放开:`offline_fraction>0`+libero+`base_action_mode=queue`+`chunk_length=1`+`actor=raw` 通过;`subgoal_conditioned`/`stage_conditioned`/`reward_shaping=staged` 仍被拒。
6. dexmg 全量回归零回归(residual env)。

**opt-in live smoke(真 serve + 真 parquet,默认 skip)**:task8 建 offline buffer(base_policy,过真 pi0)→ 跑几步 `offline_fraction=0.5`+`demo_bc_coef=0.1`+`--device cuda` 训练 → 验无 NaN、`len(offline_rb)>0`、混采与 BC 路径都跑到。

## 8. 文件清单

- 新增:`resfit/rl_finetuning/chunk_residual/libero_offline.py`
- 新增测试:`resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py`
- 改:`resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(`validate_libero_cfg` 放开 + offline 构建 libero 分支 + `_libero_offline_signature`)
- 复用(不改):`libero_obs.py`(`resize_with_pad`/`build_libero_serve_obs`/`load_libero_norm_stats`)、`offline_stage_replay.py` 的 `_new_offline_rb`/缓存助手、`relabel.py`(BC 采样)、`q_agent.py`(BC update)
