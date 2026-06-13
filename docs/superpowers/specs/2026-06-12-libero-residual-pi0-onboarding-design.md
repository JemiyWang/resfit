# LIBERO 单臂 env 接进 chunk_residual 残差管线(base=pi0_libero)设计

- 日期:2026-06-12
- 状态:设计已定稿,待写实施计划
- 范围:给 resfit chunk_residual 加一条 **LIBERO 单臂分支**(`--env_family libero`,默认 dexmg 零回归),让现成的"pi0 动作当 base + 残差修正"机制在 LIBERO 上跑。**完成标准=代码+回归级**:libero env/shim/接线/单测齐、dexmg 默认路 bit 不变、新 conda env 构建脚本就绪;**live smoke / 真训练留作后续(用户授权 GPU)**。第一版锁**单 `(suite, task_id)`**、`reward_shaping=none`、`offline_fraction=0`、`queue/chunk_length=1`、base=pi0_libero。

## 1. 背景与动机

resfit chunk_residual 已通用实现"pi0 动作当参考 + 残差修正":`build_base_policy`(`train_chunk_residual.py:132`)的 `--base_policy_type pi05` 经 openpi-client websocket 连 pi0 serve,`Pi05PolicyAdapter` 包成 base;`chunk_env_wrapper.py:151` `combined_flat = base_flat + residual_flat` → 反归一化喂 env。**但 env 只有 dexmg(robosuite 1.5.1 双臂),没有 LIBERO**(`create_vectorized_env` 在 `dexmg/environments/dexmg.py`)。

要在 LIBERO 上跑残差 RL,缺两块:(a) LIBERO 单臂 env;(b) base 适配——现有 `Pi05PolicyAdapter` 发 kai0/dexmg 的**嵌套 `images` 方言**,而 pi0_libero serve 的 `LiberoInputs`(`chj/openpi/src/openpi/policies/libero_policy.py:30-81`)吃**扁平** `observation/image|wrist_image|state|prompt`(已核实,风险#1)。残差机制本身不用动。

## 2. 关键约束(命门)

1. **env 不能 websocket 解耦**:sim env 是 RL 要 step 的环境,必须与训练同进程(AsyncVectorEnv worker 由训练进程 spawn)。
2. **robosuite 版本硬冲突**:LIBERO 钉 robosuite 1.4.1(+bddl1.0.1/robomimic0.2.0/mujoco3.2.3),dexmg 用 1.5.1;`residual` 环境还完全没装 libero/bddl/robomimic。→ 采纳**独立 conda env(方案 A)**:新建 LIBERO-residual env(resfit RL 代码 + LIBERO 1.4.1 栈,不含 dexmg)。**前置命门**:这些依赖能否在一个 py 版本共存,计划第一步任务验证;不成立则回退 env-RPC。
3. **默认零影响**:`--env_family` 默认 dexmg,dexmg 全路径 bit 不变。
4. **单臂 / 单 task**:action 7、state 8(axis-angle)、2 路图像;第一版只跑一个 `(suite, task_id)`。

## 3. 设计

### 3.1 `libero_obs.py`(纯逻辑,不 import libero,两环境可单测)
把会踩坑的转换抽成纯函数:
- `flip_resize_image(img) -> (224,224,3) uint8`:`[::-1,::-1]` 翻转 + `resize_with_pad` 到 224(对齐 `rollout_libero.py:425-430`)。
- `assemble_libero_state(obs) -> (8,) float32`:`concat[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]`(对齐 `rollout_libero.py:436-440`)。
- `adapt_4tuple(obs, reward, done, info) -> (obs, reward, terminated, truncated, info)`:`success=(reward==1.0)`、`terminated=success`、`truncated=(done and not success)`(对齐 dexmg `dexmg.py:389-392` 的成功语义)。
- `to_libero_serve_obs(state, base_img, wrist_img, prompt) -> {"observation/image", "observation/wrist_image", "observation/state", "prompt"}`(base shim 与 env 共用,喂 pi0_libero serve 的扁平 schema)。

### 3.2 `libero_env.py`(集成层,惰性 import libero)
- `LiberoGymWrapper`:复刻 dexmg `RobosuiteGymWrapper` 对外契约——obs 键 `observation.state`(8)+ `observation.images.{agentview,robot0_eye_in_hand}`(C,H,W float32 [0,1])、`action_space=Box(-1,1,(7,))`、`reset()->(obs,info)`、`step()->5-tuple`、`render/seed/close/get_wrapper_attr/...`(AsyncVectorEnv worker 需要)。内部用 `from libero.libero.envs import OffScreenRenderEnv`(bddl_file + init_states,照 `rollout_libero.py:100-106,403-411`);**reset 内置 `num_steps_wait≈10` dummy step** 让物体落稳。图像/state 走 `libero_obs` 纯函数。info **不放** stage_id/rel_piece。
- `make_libero_env` + `create_libero_vectorized_env`:镜像 dexmg `make_dexmimicgen_env`/`create_vectorized_env`,**复用** dexmg 的 `VectorizedEnvWrapper` + `AsyncVectorEnv(SAME_STEP, spawn)` + EGL 设备映射(`cuda_to_egl_device_id`)。

### 3.3 `LiberoPi05Adapter`(resfit 侧,子类化 kai0 `Pi05PolicyAdapter`,不动 kai0)
覆写 `_to_openpi_obs`:用 `to_libero_serve_obs` 出扁平 schema(`observation/image|wrist_image|state|prompt`),图像 `resize_with_pad` 224。其余(队列 / `select_action` / `execute_horizon` / 切 `:action_dim`)继承不变。

### 3.4 接线(2 处,flag-gated)
- `train_chunk_residual.py`:加 `--env_family {dexmg,libero}`(默认 `dexmg`)+ 专用 `--libero_suite`(如 `libero_spatial`)`--libero_task_id`(int,默认 0)。libero 分支:`create_libero_vectorized_env(suite, task_id, ...)` + `validate_libero_cfg(args)`(见 §错误处理)。base 起 `LiberoPi05Adapter`(`--pi0_action_dim 7`、`--pi0_prompt`=该 task 的 language instruction,从 `task_suite.get_task(task_id).language` 取)。dexmg 分支(走 `--task`/`--env_family dexmg`)原样不动。
- `load_pi05.py`:`load_pi05_base_policy(..., schema="libero")` 返回 `LiberoPi05Adapter`,否则现有 adapter。

### 3.5 归一化
`--dataset` 指 LIBERO 的 LeRobot stats(本机 `chj/lerobot_cache/physical-intelligence/libero`,实测 state=8/actions=7),`ActionScaler`/`StateStandardizer` 自动按维度建。注意核对 meta.stats 键名(`action` vs `actions`、`observation.state` vs `state`),不一致则加键名兜底。

## 4. 错误处理(fail-fast,默认 dexmg 零影响)
- `validate_libero_cfg(args)`:`--env_family libero` 强制并校验 `base_policy_type=pi05`、`base_action_mode=queue`、`chunk_length=1`、`reward_shaping=none`、`offline_fraction=0`、`pi0_action_dim=7`;**禁开** `--stage_conditioned/--stage_budget/--potential_source hiql/--subgoal_conditioned`(会 assert `task in NUM_STAGES` 或需双臂 rel_piece)。违反报错指明。
- 特权信息绕开:stage 检测器对 LIBERO task 返回 None(`stage_detectors.py:76-78`)→ 不透 stage_id;`NUM_STAGES.get(task,1)` 退化 1 段;rel_piece 仅 eef_piece 模式算,默认 eef 不需要。
- LIBERO env 缺 bddl/init_states/obs 键 → 点名报错;断言 `action_space==7`、pi0 返回 ≥7;图像翻转/resize 必做(纯函数+单测锁)。

## 5. 测试(代码+回归级;不依赖真 libero/GPU)
- **纯逻辑单测(`libero_obs.py`,residual 环境)**:flip_resize_image(翻转+224/uint8)、assemble_libero_state(8 维/切片/axis-angle)、adapt_4tuple(成功→terminated / 超时→truncated)、to_libero_serve_obs(4 扁平键齐、HWC uint8、state8)。
- **`LiberoPi05Adapter` 单测**(stub base,无真 serve):`_to_openpi_obs` 出扁平 schema、图像 224。
- **接线单测**:`--env_family` 默认 dexmg;`validate_libero_cfg` 在违规组合下报错。
- **回归**:现有 dexmg chunk_residual 测试全绿、默认路 bit 不变。
- **live smoke(opt-in,新 env,默认 skip/标 slow)**:真起 LIBERO env + pi0_libero serve 跑 RL loop 几步不崩——`LiberoGymWrapper` 真集成在此验,不进单测套。

## 6. 要改/新增的文件
- 新增 `resfit/rl_finetuning/chunk_residual/libero_obs.py`(纯函数)
- 新增 `resfit/rl_finetuning/chunk_residual/libero_env.py`(LiberoGymWrapper + 工厂)
- 新增 `resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py`(LiberoPi05Adapter)
- 改 `resfit/lerobot/policies/pi05/load_pi05.py`(schema=libero 分发)
- 改 `train_chunk_residual.py`(`--env_family` + libero 分支 + `validate_libero_cfg`)
- 新增 `scripts/setup_libero_residual_env.sh`(新 conda env 构建 + 依赖共存核验)
- 新增 tests:`tests/test_libero_obs.py`、`tests/test_libero_pi05_adapter.py`、`tests/test_env_family_wiring.py`

## 7. 范围边界与前置风险
- **OUT(留后续)**:live smoke / 真训练到收敛(用户 GPU);整 suite 多 task(第一版锁单 `(suite,task_id)`,因 adapter prompt 写死、init_states 按 task);LIBERO 的 offline anchor buffer(`build_offline_buffer` 是 dexmg HDF5+双臂);stage/hiql/subgoal/value;pi0_feat value(另案,搁置)。
- **前置命门(计划第一步)**:新 conda env——resfit RL 依赖 + LIBERO 1.4.1 栈能否一个 py 版本共存。构建脚本验证;**不成立则回退 env-RPC(方案 B,本 spec 不含,另起)**。
- 其余风险:pi0_libero serve 扁平 schema 已核实;翻转+resize 与 rollout_libero 对齐;单 task prompt=指令。
