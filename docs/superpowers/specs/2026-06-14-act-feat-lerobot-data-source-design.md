# act_feat 接 LeRobot 数据源(no-stage,解锁无 raw-hdf5 任务)设计

- 日期:2026-06-14
- 状态:设计已定稿,待用户审阅 → 写实施计划
- 范围:给 act_feat 离线/在线管线加一个 `--data_source lerobot` 数据源,从 **LeRobot 格式数据集**(parquet + 视频)读 demo 帧,免去对 raw dexmg hdf5(带 `states`/`model_file`)的依赖,从而能跑**有 LeRobot 数据 + ACT base 但无 raw hdf5** 的任务(pouring/lifttray)。**act_feat 专用、no-stage**;`eef_piece`/stage 整形不支持(要 raw `states`)。
- 前置:在线 act_feat 子目标注入已落地(见 [[project_resfit_act_feat_online_subgoal]] / `2026-06-13-hiql-subgoal-online-act-feat-design.md`)。本设计只换"离线 demo 帧的来源"。

## 1. 背景与动机

act_feat 管线现在从 **raw dexmg hdf5** 读 demo 帧(`_build_raw_obs_seqs`/`build_offline_buffer` 用 h5py 读 `data/demo_i/obs/<key>`、`states`、`actions`)。其中 `states`(sim qpos)+ `model_file` 只被 **stage replay**(`replay_instant_stages` set_state)和 **eef_piece rel_piece** 用;act_feat 本身只要"每帧 图+本体+动作"。

pouring/lifttray **本地有完整 LeRobot 数据集**(pouring 1009 集/33.8万帧、lifttray 1033 集/51.7万帧,parquet+视频)+ **ACT BC base**(在 `residual-offpolicy-rl/bc_run_*`),但**没有 raw hdf5**。LeRobot 数据有"图+本体+动作",缺 `states`。所以只要 act_feat 走 no-stage(stage_id≡0,不 replay),就能用 LeRobot 数据跑。

**同源前置(Phase 0 已验,灰区接受)**:threading 既有 raw hdf5(无损,=在线渲染)又有 LeRobot 视频,实测同一帧两路 ACT 530 特征 `max|d|=0.05~0.12`(渲染噪声 0.027 的 2~4 倍)。视频压缩有损 → 离线(视频)特征与在线(渲染)特征有 ~0.1 系统性 gap。用户接受此 caveat(只为拿 act_feat vs eef_piece 信号,非发表级严格 A/B)。pouring/lifttray 无 raw hdf5 无法重测,**用 threading 这个 0.12 当同源代理**。

## 2. 关键约束(命门)

1. **默认零影响**:不传 `--data_source lerobot` 时,hdf5 全路径逐位不变。
2. **同源—图像**:LeRobot 图 Phase0 实测 `float32 CHW (3,84,84) [0,1]` → 直接喂 ACT(**不做** hdf5 那套 HWC-uint8→CHW/255)。在线 env 渲染同为 CHW float[0,1]。~0.1 视频-渲染 gap 是已接受 caveat,结果须明标。
3. **同源—proprio**:LeRobot `observation.state` Phase0 实测 = hdf5 raw state(max|d|=0.0000)→ 照样 dataset `StateStandardizer` 标准化 → 与在线 obs.state 一致。
4. **no-stage 成套**:stage_id≡0(离线跳 replay、在线 dexmg 无检测器优雅退化,`dexmg.py:355-359`);`--reward_shaping none`;`--stage_balanced` 退化均匀采样;`--stage_conditioned` 关。
5. **维度/相机自适应**:LeRobot 路直接读 `observation.state`(pouring 36 / lifttray 38 维)+ 用该任务 ACT base 的 `image_features`(各任务相机不同)→ state_dim 自动 = dim_model + 实际本体维。

## 3. 设计

### 3.1 `lerobot_demo_source.py`(新文件,纯读取层)
- `lerobot_episode_count(repo_id, root) -> int`。
- `open_lerobot(repo_id, root) -> ds`:`LeRobotDataset(repo_id, root=root, video_backend="pyav")`。
- `lerobot_episode_frames(ds, ep_idx, image_keys, proprio_key="observation.state") -> dict`:
  用 `ds.episode_data_index["from"/"to"]` 取该集帧范围,`ds[idx]` 解码 → 返回 `{"images": {k: (T,3,84,84) float32[0,1]}, "state": (T,Dp) float32 raw, "actions": (T,Da) float32}`。
  - 图已是 CHW float[0,1](Phase0),原样;若某版本给 HWC/uint8 则归一(防御,见 verify 脚本 `_to_chw01`)。
  - `actions`/`state` 从 `ds.hf_dataset`(parquet)取,无需解码视频。
- 纯读取,不依赖 act_feat/h5py;`LeRobotDataset` 惰性 import。

### 3.2 `read_per_demo_states` act_feat 加 `data_source` 分支(改 `train_hiql_value.py`)
- 新参数 `data_source="hdf5"`、`lerobot_root=None`(+ parser `--data_source {hdf5,lerobot}`、`--lerobot_root`)。
- act_feat build 路:`data_source=="lerobot"` 时,用 `lerobot_episode_frames` 逐集拼 `raw_obs`(images 直接用、state dataset-标准化),替代 `_build_raw_obs_seqs`(hdf5)。其余(extractor.embed_batch → 标准化 → 缓存)不变。
- extractor 用 `data_source` 无关方式建(从 act_base_ckpt 的 ACT,`image_keys` 取该 ACT `image_features`,`proprio_dim` 传**实际本体维**)。
- 缓存命中路不变(读缓存,数据源无关)→ **gc_value/high_actor 只读缓存即自动支持**。

### 3.3 `build_offline_buffer` 加 lerobot/no-stage 分支(改 `offline_stage_replay.py`)
- 新参数 `data_source="hdf5"`、`lerobot_repo_id=None`、`lerobot_root=None`(lerobot 时内部 `open_lerobot` 打开)。
- `data_source=="lerobot"`:
  - 逐集用 `lerobot_episode_frames` 取 images/state/actions(**不开 h5py、不读 `states`/`model_file`、不 replay**)。
  - `instant`(stage_id)≡ 0 全长(no-stage);`transition_fields` 照常(reward 由 success/sparse,stage bonus 在 reward_shaping none 下不生效)。
  - base_action:`base_mode=base_policy` 时 base_policy 逐帧现算(LeRobot 图);`gt` 时 = action。
  - subgoal_z:act_feat 复用 530 缓存 + `subgoal_waypoint`(同 hdf5 路)。
  - state dataset-标准化;图直接用(CHW float01)。
- eef_piece + lerobot **禁止**(assert:lerobot 无 rel)。

### 3.4 接线(改 train_hiql_gc_value / high_actor / train_chunk_residual)
- 三脚本 parser 加 `--data_source {hdf5,lerobot}`(默认 hdf5)+ `--lerobot_root`,透传到 `read_per_demo_states` / `build_offline_buffer` / `setup_act_feat`。
- `train_chunk_residual` lerobot 下:`--offline_dataset_path`(hdf5)/`--offline_stage_cache` 不用;`--reward_shaping none` + no stage;`--dataset` 当 repo_id;build_offline_buffer 走 lerobot。
- 守卫:`--data_source lerobot` 需 `--lerobot_root`(存在)+ `--state_mode act_feat`;否则 ValueError。

### 3.5 不变量
- hdf5 路默认逐位等价;lerobot 路 state_dim = dim_model + 实际本体维;同源 proprio dataset-std;stage_id≡0。

## 4. 测试
- **回归锁**:hdf5 act_feat / eef_piece / eef 全绿,默认 data_source=hdf5 逐位不变。
- **新单测(stub LeRobot ds,不解真视频)**:
  1. `lerobot_episode_frames` 形状/类型:images (T,3,84,84) float01、state (T,Dp)、actions (T,Da);按 episode_data_index 切片正确。
  2. `read_per_demo_states` lerobot 分支(stub ds + stub extractor)→ 正确维度 seqs + dataset-std proprio。
  3. `build_offline_buffer` lerobot 分支(stub ds + stub subgoal)→ 条目含 obs.subgoal、stage_id≡0、不开 h5py。
  4. 守卫:lerobot 缺 lerobot_root / eef_piece+lerobot → ValueError。
- **可跑性 smoke(opt-in,真 LeRobot+ACT base+env,默认 skip)**:pouring 取几集几十步,缓存→gc_value→主训 offline buffer→rollout→eval 全通、V 有限、subgoal z 有限、不崩;打印 ACT base 加载成功。

## 5. 要改/新增文件
- 新增 `resfit/rl_finetuning/chunk_residual/lerobot_demo_source.py`
- 改 `train_hiql_value.py`(read_per_demo_states + parser)
- 改 `offline_stage_replay.py`(build_offline_buffer lerobot/no-stage 分支)
- 改 `train_hiql_gc_value.py` / `train_hiql_high_actor.py` / `train_chunk_residual.py`(parser 透传)
- 新增 tests:`tests/test_lerobot_demo_source.py`、扩 `tests/test_read_per_demo_states_act_feat.py`、`tests/test_build_offline_buffer_rel.py`(lerobot 分支)、`tests/test_act_feat_cli_wiring.py`(守卫)

## 6. 前置依赖与风险
- **视频-渲染 ~0.1 gap(已接受)**:离线视频特征 vs 在线渲染特征系统性偏 ~0.1 → value/subgoal 有轻度退化;结果须明标 caveat。pouring/lifttray 无 raw hdf5 无法重测,用 threading 0.12 当代理。
- **ACT base 能否被 resfit `load_policy` 加载**:pouring/lifttray 的 base 在 residual-offpolicy-rl(lerobot ACT 格式);load_policy 兼容性 smoke 暴露,不成则需转换/适配。
- **LeRobot 视频解码**:torchcodec 全环境坏 → 用 `video_backend="pyav"`(见 [[project_resfit_pouring_traylift_residual]])。
- **维度/相机泛化**:pouring 双手灵巧手(36 维、Inspire 6 关节 action)、lifttray(38 维);extractor/缓存自适应,但首次跑要确认 ACT image_features 与 LeRobot image 键名一致。
- **建缓存慢**:全集帧过冻结 ACT(pouring 33.8万帧 / lifttray 51.7万帧),GPU 上数分钟~十几分钟。
