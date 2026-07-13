# 在新机器上复现 resfit 残差实验（five-task 迁移方案）

> 目标：在另一台电脑上从零把 `three_piece / threading / pouring / lifttray / can_sort`
> 五个任务的残差 RL 实验跑起来。
> 覆盖 5 个任务的 chunk_residual（`train_chunk_residual`）主实验。

要跑起来需要**三类东西**，缺一不可：

| 类别 | 来源 | 能否下载 |
|------|------|----------|
| **① 代码** | git clone（JemiyWang/resfit 的 `chunk-residual-validation`） | ✅ git |
| **② Python 环境 + GPU/渲染** | 在新机器**重建**（最费劲，HF 下不来） | ❌ 必须自建 |
| **③ 数据 + 权重资产** | 官方数据集(HF) + 你自己的 HF 私有库 | ✅ 下载 |

> **省心铁律**：如果新机器能用**完全相同的绝对路径**
> （`/mnt/mnt/data/resfit`、`/mnt/mnt/data/envs/residual`、
> `/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/...`），
> 那几乎所有 run 脚本都**不用改**。路径不同才需要改脚本（见第 6 节）。

---

## 1. 拉代码

```bash
# 上游 = JemiyWang（数据/代码主线在这）
git clone git@github.com:JemiyWang/resfit.git   # 或走你配的 SSH 隧道别名
cd resfit
git checkout chunk-residual-validation          # 最全的分支
```
> `resfit/dataset/`、`resfit/out/`、`outputs_chunk/`、`deps/`、`wandb/`、`*.log`
> 都在 `.gitignore` 里，**不会随 clone 下来**——它们靠第 2、5、6 节补齐。

---

## 2. 重建 Python 环境（最关键，别忽略）

目标环境规格（源机器实测）：
- conda env 名 `residual`，**Python 3.10.20**
- **torch 2.7.1+cu128**，**robosuite 1.5.1（源码安装，在 `deps/robosuite`）**
- lerobot 依赖：`resfit/lerobot/lerobot_requirements.txt`
- torchcodec GPU 解码路：`resfit/lerobot/shell/torchcodec_env.sh`

### 2.1 推荐做法：使用仓库内环境快照（最可靠）

仓库已经带了源机器导出的环境清单：
- `env/residual_pip_freeze.txt`
- `env/residual_env.yml`

这两个文件已去掉本机 conda `prefix:`，并移除了私有 `openpi-client` editable 依赖（只影响 LIBERO/pi0，不影响这里的 DexMG five-task chunk 实验）。clone 到新机器后直接用即可。

### 2.2 在新机器建环境
```bash
conda create -n residual python=3.10 -y
conda activate residual
pip install -r env/residual_pip_freeze.txt
# lerobot 侧依赖：
pip install -r resfit/lerobot/lerobot_requirements.txt
```

或直接用 conda yml：
```bash
conda env create -f env/residual_env.yml
conda activate residual
```

### 2.3 robosuite 从源码装（版本冲突，必须独立处理）
`deps/` 被 gitignore，`deps/robosuite` 是源码。两种办法：
- **拷 `deps/` 目录过来**（最省事），或
- 重新 clone robosuite 1.5.1 到 `deps/robosuite` 后 `pip install -e deps/robosuite`。

装完跑一次 robosuite 宏配置：
```bash
python deps/robosuite/robosuite/scripts/setup_macros.py
```

### 2.4 torchcodec GPU 解码（pouring/lifttray/can_sort 的 LeRobot 路要用）
按 `resfit/lerobot/shell/torchcodec_env.sh` 配好
（nvidia-npp + `LD_LIBRARY_PATH` shim，约 8x，避免 pyav 慢路）。
用到 LeRobot 数据源的任务在跑之前 `source` 这个脚本。

---

## 3. GPU / 渲染（MuJoCo EGL）

每个 run 脚本都需要能在新机器 GPU 上离屏渲染：
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# 多卡/渲染漏卡时,显式钉物理卡号:
export MUJOCO_EGL_DEVICE_ID=<物理卡号>
```
> **坑**：EGL 渲染可能漏到别的物理卡；单卡掩码下渲染要传逻辑号
> `env_id % N`。渲染进程别误当残留 kill。先跑第 9 节的 smoke 验证渲染 OK 再上量。

---

## 4. 登录（HF + wandb）

```bash
huggingface-cli login      # 用你的 JEMIY 账户(下私有资产 + 官方数据集要它)
wandb login                # 训练要写 wandb
# 下载走镜像可加速:
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 5. 官方数据集（DexMimicGen 演示 hdf5）

每个任务需要一个 hdf5 放到 `resfit/dataset/`。**精确的 hdf5 文件名 + HF dataset id
以各任务 run 脚本里的 `--offline_dataset_path` / `--dataset` 为准**，一键抠出来：
```bash
grep -hoE '\-\-(offline_dataset_path|dataset) +[^ ]+' run_*.sh | sort -u
```
对应关系（HF 上 ankile 转托的 DexMimicGen 官方数据）：

| 任务 | HF dataset id | 本地放置路径（示例） |
|------|---------------|----------------------|
| three_piece | `ankile/dexmg-two-arm-three-piece-assembly` | `resfit/dataset/two_arm_three_piece_assembly.hdf5` |
| threading | `ankile/dexmg-two-arm-threading` | `resfit/dataset/two_arm_threading.hdf5` |
| pouring | `ankile/dexmg-two-arm-pouring` | `resfit/dataset/two_arm_pouring.hdf5` |
| lifttray | `ankile/dexmg-two-arm-lift-tray` | `resfit/dataset/two_arm_lift_tray.hdf5` |
| can_sort | `ankile/dexmg-two-arm-can-sort`（注意连字符） | `resfit/dataset/...can_sort....hdf5` |

> hdf5 下到 `resfit/dataset/` 下、文件名对上脚本里的 `--offline_dataset_path` 即可。

---

## 6. 你的私有资产（基座 + 权重 + 缓存）—— 放在【魔搭 ModelScope】

已上传到魔搭私有库 **`jemiywang/residual-artifacts`**
（改用魔搭是因为本机传 HF 上传被限速到 ~30kB/s，魔搭域内 ~11MB/s，快数百倍）。
在**克隆好的 repo 根目录**执行：
```bash
pip install modelscope
modelscope login --token <你的魔搭Token>        # 私有库必须先登录
modelscope download --model jemiywang/residual-artifacts --local_dir .
```
下载内容会自动落位：
- `resfit/out/{piecce,threading,can_sort}/best/` ← three_piece/threading/can_sort 基座 ✅
- `outputs_chunk/*_stages*.npz`、`*_gc_value_*.pt`、`*_high_actor_*.pt`、`*_act_feat*.npz` ✅

### ⚠️ 两个外部基座（pouring / lifttray）要手动归位
它们在库里放在 `bases/` 下，而脚本引用的是**绝对路径**。二选一：

**A. 复刻绝对路径（推荐，脚本免改）**
```bash
ART=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts
mkdir -p "$ART"
cp -r bases/pouring_run_anw5pphu_best_v2  "$ART/run_anw5pphu_best:v2"
cp -r bases/lifttray_run_e0o0sckj_best_v4 "$ART/run_e0o0sckj_best:v4"
```
**B. 或改这两个 run 脚本里的 `BASE=` / `--base_wandb_id` 指到 `bases/...`。**

---

## 7. 路径修正（仅当新机器路径与本机不同）

若没用相同绝对路径，改这几处：
- 每个 run 脚本头部：`PYTHONPATH=`、`cd `、`PY=`（指向新的 repo / residual python）。
- pouring/lifttray 的 `BASE`（见第 6 节）。
- 一键检查所有写死路径：
  ```bash
  grep -nE '/mnt/mnt/data' run_*.sh | grep -vE '^\s*#'
  ```

---

## 8. 跑实验（offcache 首次自动重建）

各任务主实验的启动脚本与用法（**offcache 26G+ 首次跑自动建，别提前拷，留够磁盘**）：

| 任务 | 启动脚本 | 用法示例 |
|------|----------|----------|
| three_piece | `run_three_piece_staged_joint_rerun.sh` | `setsid bash run_three_piece_staged_joint_rerun.sh > three_piece.log 2>&1 &` |
| threading | `run_threading_staged_joint_seed.sh` | `setsid bash run_threading_staged_joint_seed.sh <gpu> <seed> > threading_s<seed>.log 2>&1 &` |
| pouring | `run_pouring_pothiql_joint_rerun0703.sh` | `setsid bash run_pouring_pothiql_joint_rerun0703.sh <gpu> > pouring.log 2>&1 &` |
| lifttray | `run_lifttray_staged_joint.sh` | `setsid bash run_lifttray_staged_joint.sh <gpu> > lifttray.log 2>&1 &` |
| can_sort | （无 run 脚本，默认配方，需手写命令，见下） | — |

> 脚本启动时会 `gate()` 校验 `gc_value / high_actor / act_feat` 存在——
> 第 6 节资产下齐后就能过；缺则 FATAL 退出，按提示补文件。

**can_sort 特别说明**：无 run 脚本、无 stage 缓存、gc_value 只有**默认配方**
（`cansort_gc_value_actfeat_hdf5.pt`，非 hiqlv512）。复现要参照其它任务的命令手写，
关键差异：不带 `--reward_shaping staged / --stage_balanced / --offline_stage_cache`，
gc_value/high_actor 用 `cansort_*_actfeat_hdf5.pt`（无 hiqlv512 后缀）。

---

## 9. 每任务精确文件对照表（复现清单）

| 任务 | 基座 | stage 缓存 | gc_value | high_actor | act_feat |
|------|------|-----------|----------|-----------|----------|
| three_piece | `resfit/out/piecce/best/` | `three_piece_stages_4stage.npz` | `three_piece_gc_value_actfeat_hiqlv512.pt` | `three_piece_high_actor_actfeat_hiqlv512.pt` | `three_piece_act_feat.npz` |
| threading | `resfit/out/threading/best/` | `two_arm_threading_stages.npz` | `two_arm_threading_gc_value_actfeat_hiqlv512.pt` | `two_arm_threading_high_actor_actfeat_hiqlv512_sg15.pt` | `two_arm_threading_act_feat.npz` |
| pouring | `.../artifacts/run_anw5pphu_best:v2` | `two_arm_pouring_stages.npz` | `pouring_gc_value_actfeat_hdf5_hiqlv512.pt` | `pouring_high_actor_actfeat_hdf5_hiqlv512.pt` | `pouring_act_feat_hdf5.npz` |
| lifttray | `.../artifacts/run_e0o0sckj_best:v4` | `two_arm_lift_tray_stages.npz` | `lifttray_gc_value_actfeat_hdf5_hiqlv512.pt` | `lifttray_high_actor_actfeat_hdf5_hiqlv512.pt` | `lifttray_act_feat_hdf5.npz` |
| can_sort | `resfit/out/can_sort/best/` | （无） | `cansort_gc_value_actfeat_hdf5.pt` | `cansort_high_actor_actfeat_hdf5.pt` | `cansort_act_feat_hdf5.npz` |

> `outputs_chunk/` 下的 `.npz/.pt`；基座见"基座"列。offcache 不在表内（自动重建）。

---

## 10. 常见坑（来自历史经验，务必留意）

- **磁盘满静默死**：offcache 每个 26G+，多实验并行极易撑爆盘。跑前确认剩余空间，
  挂内存/磁盘监控；磁盘满会导致进程静默死，排查时先看 `df -h`。
- **offcache 复用/重建**：`action_scale` 进缓存签名——改了动作幅度必重建；
  缓存签名用 ckpt 路径，改内容但同路径会**静默复用旧 cache**。
- **base_policy 建 offcache 很慢**：`--offline_base_mode base_policy` 首次要跑
  大量 base forward（CPU 上可达 ~80min、几十 G），耐心等，别当卡死。
- **EGL 漏卡**：见第 3 节；渲染可能落到别的物理卡，先 smoke 验证。
- **robosuite 版本冲突**：必须独立 conda env（第 2.3 节）。
- **torchcodec**：LeRobot 数据路解码走 `torchcodec_env.sh` 的 GPU 路，否则 pyav 慢路成瓶颈。
- **一致性**：gc_value/high_actor 与 act_feat/基座同源——若在新机器重建 act_feat，
  必须用**同一个基座**，否则特征对不上、权重失效。

---

## 附：最小 smoke 验证（先证明环境/渲染通，再上量）

以 threading 为例，先只跑很短步数确认能起来（在脚本里临时把
`--total_env_steps` 调小，或加 `--smoke`），观察：
1. 能加载基座 / gc_value / high_actor / act_feat（gate 通过）；
2. 环境能渲染（无 EGL 报错）；
3. offcache 开始构建、磁盘在涨；
4. wandb 有 run 写入。
四项都 OK 再放全量 500k。
