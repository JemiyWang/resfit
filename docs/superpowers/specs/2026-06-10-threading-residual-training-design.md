# TwoArmThreading 残差 RL 训练接入设计(对齐 three_piece 稳定线)

- 日期:2026-06-10
- 状态:设计已定稿,待落实施计划
- 范围:把已跑通的 three_piece(TwoArmThreePieceAssembly)残差 RL 端到端管线镜像到
  **TwoArmThreading**,对齐到**稳定线**(stage 检测器 + staged/potential 整形 +
  object-aware 30 维 state + `train_hiql_value.py` 产的 `value.pt`),**不碰**在研的
  `gc_value` / `--subgoal_conditioned` / HIQL 分层。

## 1. 背景与目标

项目当前对 three_piece 实现了完整残差 RL 管线:
BC base → offline stage 缓存 → state30(object-aware)缓存 → HIQL value(`value.pt`)
→ 残差 RL 主训(`train_chunk_residual.py`,potential/staged 整形)→ eval。

目标:让 **TwoArmThreading** 走同一套管线。基础设施大部分已就绪(env 已注册、
`ResidualTD3TwoArmThreadingConfig` 已存在、数据集 id 与 BC base wandb id 已登记、
相机/低维 state 键按 env_name 自动路由),**真正要新写的只有任务相关那一层**。

非目标(本设计明确排除):
- `train_hiql_gc_value.py` / `gc_value.pt` / `--subgoal_conditioned` / goal-cond 分层
  (three_piece 自身仍在 Phase2 之前,不在 threading 首版引入)。
- threading 的 dense reward / 更贴任务的几何特征(针尖-环心)进 Φ —— 列为加分项,不进首版。

## 2. threading 环境实情(对照 three_piece)

来源:`deps/dexmimicgen/dexmimicgen/environments/two_arm_threading.py`。

- **物体(2 个)**:`self.needle = NeedleObject(name="needle_obj")`、
  `self.tripod = RingTripodObject(name="tripod_obj")`。env 暴露 `env.needle` / `env.tripod`,
  root body 可读 `env.needle.root_body` / `env.tripod.root_body`(预期
  `"needle_obj_root"` / `"tripod_obj_root"`,以一次性 sim 探针为准)。
- **成功条件 `_check_success()`**:针尖 geom `needle_obj_needle` 世界坐标,与环心
  (`tripod_obj_ring_{i}`,i ∈ [0, `tripod.num_ring_geoms`) 的几何均值)距离 <
  `tripod.ring_size[1]`(环半径)。即「针尖穿入环」。
- **关键差异**:与 three_piece 不同,threading **没有任何 env 内置子阶段谓词**
  (three_piece 有 `_check_first/second_piece_is_assembled`),只有 `_check_success` 一个。
  → stage 检测器必须自行用 grasp 谓词 + success 搭。
- 物体数 = 2,与 three_piece 一致 → object-aware rel_piece 维度不变(12),state 仍 30 维。

## 3. 总体策略

镜像 three_piece 管线,产物按任务命名隔离(`two_arm_threading_*`),**只改 3 个
task-specific 代码点**,其余为"换参数跑一遍"。

```
[前置] 下载 threading hdf5 / lerobot 数据集 / 验证 BC base
         │
         ├─ 改动 A: object_state.py   (object-aware 物体名按任务查表)
         ├─ 改动 B: stage_detectors.py(threading_stage + 注册,3 段)
         └─ 改动 C: dexmg.py 接线     (_rel_piece_info 走按任务查表;很小)
         │
[离线]  precompute_stage_cache → two_arm_threading_stages.npz
        load_or_build_state30  → two_arm_threading_state30.npz
        train_hiql_value       → two_arm_threading_value.pt  (state_mode=eef_piece)
         │
[主训]  train_chunk_residual --task TwoArmThreading ... --potential_source hiql
         │
[验证]  stage 直方图 / object-aware 同源 / smoke / 长跑 eval
```

## 4. 前置:数据与 base(长杆)

现状:three_piece 的 raw hdf5(4.3GB)在 `resfit/dataset/`,lerobot 缓存
`~/.cache/huggingface/lerobot/ankile` 有;**threading 的 raw hdf5 全树缺失**。

- **raw hdf5**:用 `deps/dexmimicgen/scripts/download_hf_dataset.py`(hf_hub_download)
  拉 `ankile/dexmg-two-arm-threading` 的 hdf5 → `resfit/dataset/two_arm_threading.hdf5`。
- **lerobot 数据集**:确认/拉取 `ankile/dexmg-two-arm-threading`(主训取 normalization stats)。
- **BC base**:验证 `dexmg-twoarmthreading-bc/cbv7mqw3` 能从 wandb 下载并 forward 一次。
  **风险分支**:若死链(历史上踩过 wandb base 死链),回退用 `train_bc_dexmg.py` 重训。
- **出口 gate**:hdf5 能开、lerobot 能 load、base policy 能 forward 一帧。

## 5. 改动点 A:`object_state.py`(object-aware 物体名按任务)

现状:模块级常量 `PIECE_ROOT_BODIES = ("piece_1_root", "piece_2_root")` 写死给三件套,
被 online(`dexmg._rel_piece_info`)与 offline(`offline_stage_replay`)共用。

改法(**推荐:仿照已有 `STAGE_DETECTORS`/`NUM_STAGES` 注册表风格,保持代码一致**):

- 新增按任务查表:
  ```python
  OBJECT_ROOT_BODIES = {
      "TwoArmThreePieceAssembly": ("piece_1_root", "piece_2_root"),
      "TwoArmThreading":          ("needle_obj_root", "tripod_obj_root"),  # 以探针确认
  }
  def get_object_bodies(task: str):
      return OBJECT_ROOT_BODIES.get(task)
  ```
- `compute_eef_rel_piece_from_env` / `read_piece_positions` 改为接受调用方传入的
  `piece_root_bodies`(签名已支持),由 `_rel_piece_info` 按 `env_name` 取。
- **维度不变**:threading 也是 2 物体 → `eef_rel_piece` 返回 12 维,state 仍 30。
  `STATE_DIM_BY_MODE`、`assemble_state` 的 `(T,12)` 形状校验**均不动**。
- **同源命门照旧**:eef 用 `_eef0_xpos`/`_eef1_xpos`(set_state 后实时正确),
  **不用** `_get_observations()`(replay 时 stale)。online/offline 共用
  `compute_eef_rel_piece_from_env` 保证严格同源。
- body 字符串以一次性 sim 探针(打印 `env.needle.root_body` / `env.tripod.root_body`
  与 `env.sim` 中可见 body 名)确认后写入表。

## 6. 改动点 B:`stage_detectors.py`(threading 3 段检测器)

新写 `threading_stage(env)` + 注册。**3 段**(用户定稿):

| stage | 含义 | 判定 |
|---|---|---|
| 0 | 起步 | 默认 |
| 1 | 脚架和针都被抓起(双臂协同里程碑) | `_grasped(env, env.needle) and _grasped(env, env.tripod)` |
| 2 | 成功(针穿入环) | `env._check_success()` |

```python
def threading_stage(env) -> int:
    if env._check_success():
        return 2
    if _grasped(env, env.needle) and _grasped(env, env.tripod):
        return 1
    return 0

STAGE_DETECTORS["TwoArmThreading"] = threading_stage
NUM_STAGES["TwoArmThreading"] = 3
```

- 复用现有 `_grasped(env, obj)`(已正确处理双臂 `robot.gripper` 为 dict 取 `.values()` 的坑)。
- **无需阈值 τ**:3 段只靠 grasp + success,比带"近环距离"的方案更省调参。
- 单调闩锁(max-so-far)仍由 env wrapper 负责;检测器只判瞬时阶段(与 three_piece 一致)。

**验证风险(必须用 demo 直方图确认)**:stage 1 要求脚架**被夹爪抓住**。若 threading
demo 里脚架实际是被推/扶而非夹爪 grasp,`_grasped(env.tripod)` 可能很少触发,导致
stage 1 近空、staged 整形退化。**回退**:stage 1 改为只判 `_grasped(env.needle)`。
此判定在 §8 的 stage 直方图 gate 里裁决。

## 7. 改动点 C:`dexmg.py` 接线(很小)

env 已注册(`ENV_ROBOTS["TwoArmThreading"]`、horizon 300、dexmimicgen import 分支均在)。
只需让 `_rel_piece_info`(及其调用的 `compute_eef_rel_piece_from_env`)走 §5 的按任务查表,
不再吃死三件套默认 `PIECE_ROOT_BODIES`。预计 0~几行。stage 经 worker 内
`RobosuiteGymWrapper.step` 算好、`info["stage_id"]` 透出的机制不变。

## 8. 离线管线(机械镜像,只换路径/任务)

1. **stage 缓存**:
   `precompute_stage_cache("resfit/dataset/two_arm_threading.hdf5",
   "outputs_chunk/two_arm_threading_stages.npz")`。
2. **state30 缓存**:
   `load_or_build_state30("resfit/dataset/two_arm_threading.hdf5",
   "ankile/dexmg-two-arm-threading", num_demos=None,
   cache_path="outputs_chunk/two_arm_threading_state30.npz")`。
3. **HIQL value**:
   `train_hiql_value.py --hdf5 resfit/dataset/two_arm_threading.hdf5
   --dataset ankile/dexmg-two-arm-threading --state_mode eef_piece
   --output outputs_chunk/two_arm_threading_value.pt`。

每步出口 gate:
- stage 直方图:每段非空、demo 内近似单调(同时裁决 §6 的 stage 1 回退问题)。
- object-aware 同源:online `info["rel_piece"]` 与 offline replay `rel_piece` 数值对得上。

## 9. 残差 RL 主训 + eval

镜像 three_piece 的 `nas10_best_potstage` / `nas10_best_pothiql_objaware` 配方,只换任务相关项:

```bash
CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual --no-capture-output python -u \
-m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading \
  --base_wandb_id dexmg-twoarmthreading-bc/cbv7mqw3 \
  --dataset ankile/dexmg-two-arm-threading \
  --offline_dataset_path resfit/dataset/two_arm_threading.hdf5 \
  --offline_stage_cache outputs_chunk/two_arm_threading_stages.npz \
  --reward_shaping potential --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/two_arm_threading_value.pt \
  --state_mode eef_piece --stage_balanced --actor raw \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 \
  --wandb_project dexmg-chunk-residual --wandb_name threading_best_pothiql \
  --output_dir outputs_chunk/threading_best_pothiql \
  2>&1 | tee threading_best_pothiql.log
```

- `--potential_source stage`(Φ=整数 stage)为更简单的 fallback,若 HIQL value 不灵先退到它。
- 其余超参沿用 three_piece 现配方,后续再按任务微调。

## 10. 验证 gate(每步卡一道)

1. env smoke:`make TwoArmThreading` 能 reset/step,`info["stage_id"]` 出 0/1/2。
2. stage 直方图(demo replay):三段分布合理、单调;裁决 §6 stage 1 回退。
3. object-aware 同源:online vs offline `rel_piece` 数值一致(沿用现有 verify 思路)。
4. value smoke:`two_arm_threading_value.pt` 维度 = 30(eef_piece),v_stats 合理。
5. 主训 smoke:小步数能起、shaping 数值合理 → 再放长跑。
6. 长跑 eval:success_rate 对照 BC base,确认残差有增益。

## 11. 要改/新增的文件清单

**必改(task-specific)**:
- `resfit/rl_finetuning/chunk_residual/object_state.py` —— §5 物体名按任务查表。
- `resfit/rl_finetuning/chunk_residual/stage_detectors.py` —— §6 `threading_stage` + 注册。
- `resfit/dexmg/environments/dexmg.py` —— §7 `_rel_piece_info` 接线(很小)。

**新增测试**(镜像现有 `tests/test_*`):
- `threading_stage` 单测(构造 mock env 谓词,验 0/1/2 与单调闩锁语义)。
- `object_state` threading 物体名查表 / 维度单测。

**无需改(已通用或已就绪)**:
- `residual_td3.py`(`ResidualTD3TwoArmThreadingConfig` 已存在)、
  相机/低维键自动路由、`offline_hdf5_buffer.STATE_DIM_BY_MODE`、`assemble_state`、
  `train_chunk_residual.py`(参数化即可)、`evaluate_dexmg.py`。

## 12. 风险与开放项

- **BC base 死链** → 可能要 `train_bc_dexmg.py` 重训(§4 风险分支,先验证)。
- **stage 1「脚架被抓起」可能少触发** → §6 回退到只判针 grasp,由 stage 直方图裁决。
- **threading 几何更贴「针尖-环心」** → 首版用 root body 的 eef-rel;若 V 不灵,
  把针尖-环心距离喂进 Φ(加分项,不进首版)。
- **body 字符串** → 以一次性 sim 探针确认后再写入 `OBJECT_ROOT_BODIES`。
