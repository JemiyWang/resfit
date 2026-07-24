# LIBERO-90 四任务 SHORE 训练设计

日期：2026-07-24

## 目标

在 LIBERO-90 的 Task 57、60、63、64 上运行论文中的 SHORE 训练配方。复用已经训练好的
`pi0_libero` 基座策略，但每个新任务都使用自己的 demonstration、`pi0_feat` 缓存、
`gc_value`、`high_actor`、offline replay cache 和在线训练输出。四个在线训练任务分别固定在
GPU2、GPU3、GPU4、GPU5。

本次不修改或覆盖已有的
`run_libero10_task8_pi0feat_bp_bc01_h10.sh`、Task 8 资产、全局
`physical-intelligence/libero` 元数据和其他 GPU 上的进程。

## 已有输入

- 项目：`/mnt/mnt/data/resfit`
- Python（SHORE/LIBERO）：`/mnt/mnt/data/envs/resfit-libero/bin/python`
- Python（离线层级资产）：`/mnt/mnt/data/envs/residual/bin/python`
- Python（OpenPI serve）：`/mnt/mnt/data/chj/openpi/.venv/bin/python`
- π0 checkpoint：
  `/mnt/mnt/data/chj/openpi/checkpoints/pi0_libero/pi0_libero_is-dceq77jzghdxvjj2-devmachine-0_20260523_220232/29999`
- task-local LeRobot 数据：
  `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task<ID>`
- serve：`/mnt/mnt/data/resfit/pi0_serve/serve_with_feat.py`
- 参照配方：`/mnt/mnt/data/resfit/run_libero10_task8_pi0feat_bp_bc01_h10.sh`

## 任务映射

| GPU | Task | Language |
|---:|---:|---|
| 2 | 57 | `pick up the cream cheese and put it in the tray` |
| 3 | 60 | `pick up the black bowl on the left and put it in the tray` |
| 4 | 63 | `stack the left bowl on the right bowl and place them in the tray` |
| 5 | 64 | `stack the right bowl on the left bowl and place them in the tray` |

## 方案选择

采用“任务专属层级资产 + 共享 π0 基座”：

1. 复用 `pi0_libero` checkpoint，保持新任务适应实验的共同起点。
2. 为每个任务从自己的 50 条 demonstration 生成独立 `pi0_feat` 缓存。
3. 用该缓存分别训练任务专属 `gc_value` 和 `high_actor`。
4. 使用 Task 8 已验证的在线残差训练超参数，在 GPU2–5 上一任务一卡运行。

不采用以下方案：

- 直接复用 Task 8 的 `pi0_feat`：episode 数、长度、prompt 和内容均不匹配，无法作为新任务
  offline anchor。
- 直接复用 Task 8 的 `gc_value/high_actor`：其目标与子目标空间来自“把 moka pots 放到炉子上”，
  不代表四个新任务的层级结构。
- 四卡各启一份 π0 服务：每份 JAX 策略占用大量显存，会显著增加 OOM 风险。

## 运行架构

### 1. 共享 π0 feature/base 服务

在 GPU2、端口 8000 启动一份 `pi0_libero` feature server，设置
`XLA_PYTHON_CLIENT_PREALLOCATE=false`，pooling 固定为 `last`。四个任务的请求携带各自 prompt，
因此可以共享同一 checkpoint 和服务。

GPU2 同时承担 Task 57 的在线 SHORE 进程。SHORE 的 PyTorch 残差网络显存占用较小；启动前后都要
检查显存。如果共卡导致 OOM，则保持任务与 GPU 映射不变，改用 GPU2 上的受限 XLA 显存比例，
而不是占用 GPU0、1、6、7 或停止现有进程。

### 2. 任务专属离线资产

每个任务生成：

```text
outputs_chunk/libero90_task<ID>_pi0_feat.npz
outputs_chunk/libero90_task<ID>_pi0_feat_gc_value.pt
outputs_chunk/libero90_task<ID>_pi0_feat_high_actor.pt
```

`pi0_feat` cache builder 的 `--lerobot_root` 指向 task-local 数据根目录，`--language` 使用 benchmark
精确字符串。`gc_value` 和 `high_actor` 沿用 Task 8 的 `pi0_feat` 训练参数和 50k steps。

资产阶段按以下顺序运行：

1. 四个 `pi0_feat` 缓存通过共享 serve 顺序生成，避免并发 websocket 请求导致服务不稳定。
2. 四个 `gc_value → high_actor` 链在各自 GPU 上并行运行；每条链内部严格串行。
3. 每个阶段输出存在且能被 loader 打开后，才进入在线训练。

已有且签名完全匹配的资产可以复用；缺失、损坏或签名不匹配的资产直接报错，不静默沿用。

### 3. 在线 SHORE 训练

每个任务从参照脚本复制以下配方：

- `actor=raw`
- `chunk_length=1`
- `action_scale=0.05`
- `pi0_execute_horizon=10`
- `offline_fraction=0.5`
- `demo_bc_coef=0.1`
- `subgoal_conditioned`
- `subgoal_way_steps=25`
- `renorm_subgoal`
- `total_env_steps=500000`
- `learning_starts=10000`
- `utd=4`
- `seed=0`

必须改变的 task-local 参数：

- `--libero_suite libero_90`
- `--libero_task_id <ID>`
- `--libero_stats_json .../converted/libero_90/task<ID>/meta/stats.json`
- `--pi0_feat_cache`、`--gc_value_ckpt`、`--high_actor_ckpt`
- 独立 `--offline_buffer_cache`
- 独立 `--output_dir`
- 独立 `--wandb_name`
- GPU 与 EGL device

四个任务共用 `127.0.0.1:8000`。为避免四个进程同时首次构建 offline base-action cache 压垮
serve，launcher 按 Task 57、60、63、64 顺序启动，并等待前一个任务日志确认 offline cache
已经写入或进入 online loop 后再启动下一个。

## 脚本边界

新增两个小脚本，避免复制四份几乎相同的文件：

1. `run_libero90_shore_task.sh`
   - 参数：task ID、GPU ID。
   - 解析固定任务语言和路径。
   - 运行单任务资产链或在线训练。
   - 拒绝未知 task ID、缺失数据和不匹配的 GPU 映射。

2. `launch_libero90_shore_4gpu.sh`
   - 做全局 preflight。
   - 在独立 tmux session 中启动/检查 serve。
   - 顺序生成 feature cache。
   - 并行生成四任务的 value/high-actor 资产。
   - 按 offline cache 构建状态依次启动四个在线训练 pane/session。
   - 将日志写入 `logs/libero90_shore/`。

顶层 launcher 可通过 `PHASE=assets|train|all` 重入；默认 `all`。所有输出路径固定且互不覆盖。

## 失败处理和可恢复性

- 端口 8000 已被非目标进程占用：停止并报告，不结束该进程。
- GPU2–5 有未知计算进程：停止并报告，不抢占。
- serve 未在限定时间内监听：保留日志，终止本次新建 serve，不启动资产/训练。
- 缓存或 checkpoint 构建失败：该任务不进入下一阶段，其他已完成资产保留。
- 在线训练退出：保留 output、日志和 tmux 状态，不删除 offline cache。
- launcher 重跑时复用验证通过的产物，避免重复进行数十分钟的 feature extraction。

## 验证

启动前：

- `bash -n` 检查两个脚本。
- 验证四份 LeRobot 数据各有 50 episodes。
- 验证原始 Task 8 脚本和全局元数据哈希未改变。
- 验证 checkpoint、Python 环境、LIBERO BDDL 和端口状态。

资产完成后：

- 用 `load_pi0_feat_cache` 检查每个任务 50 条序列、prompt、pooling、serve checkpoint ID。
- 用 `load_gc_value`、`load_high_actor` 检查 checkpoint 可加载且签名相互一致。

在线启动后：

- 四个训练 PID 分别绑定 GPU2、3、4、5。
- 日志中确认正确的 suite/task/language、offline transition 数量和 online loop 启动。
- 检查没有 traceback、CUDA OOM、端口错误或 task-language mismatch。

## 完成标准

四个 tmux 训练任务处于运行状态，分别使用 task-local 数据和 task-specific hierarchy assets；
共享 `pi0_libero` serve 正常响应；日志与输出路径独立；GPU0、1、6、7 的既有进程未被停止或修改。
