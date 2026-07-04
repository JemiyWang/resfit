# Pouring ActFeat pothiql A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 pouring 的 pothiql 势函数换一个新训的单状态 act_feat value(图像+本体),其余对齐 `rerun0703`,跑一个只隔离"势函数 V 输入信息量"的 A/B。

**Architecture:** act_feat 单状态势函数机制早已实现+单测(见 spec `2026-07-04-pouring-actfeat-pothiql-ab-design.md` 与 `2026-06-28-actfeat-hiql-potential-design.md`)。本计划是"rollout/执行":①训 `pouring_value_actfeat_hdf5.pt` → ②端到端 smoke(这条路没真跑过)→ ③A/B 正式 run(复制 rerun0703 只改 4 处、offcache 重建)。不改 pothiql 分支逻辑。

**Tech Stack:** conda env `residual`;`train_hiql_value.py`(单状态 value 训练)、`train_chunk_residual.py`(残差 RL 主训)、dexmg TwoArmPouring 环境(MuJoCo EGL)、冻结 ACT base policy 出 act_feat 特征。

## Global Constraints

- 只做 **pouring**(TwoArmPouring);不碰 lifttray/three_piece/threading/potsubgoal。
- 除势函数 value 的输入表征外,训练配置**逐字对齐** `pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0703`。
- BASE(ACT base policy)= `/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2`。
- act_feat 缓存 = `outputs_chunk/pouring_act_feat_hdf5.npz`(签名:act_ckpt_id=BASE、pooling=mean、proprio_key=observation.state、image_keys=[agentview, robot0_eye_in_left_hand, robot0_eye_in_right_hand]、dim=548、1009 集)。
- 数据集 id = `ankile/dexmg-two-arm-pouring`;源 hdf5 = `resfit/dataset/two_arm_pouring.hdf5`。
- value 配方镜像 eef `pouring_value_hdf5.pt`:hidden=256,gamma/expectile/ema/lr/steps 用 `train_hiql_value` 默认(0.99/0.7/0.005/3e-4/50000)。
- **零回归**:任何代码改动只允许"加在现有 act_feat 分派之后",对 eef/eef_piece pothiql、potsubgoal、全量 `pytest` 不变绿。
- **GPU 授权**:Task 2/Task 3 需在 GPU 上跑,启动前必须由用户授权;`.pt` 大产物落 `outputs_chunk/`(gitignored),git 只提交脚本。
- 所有命令工作目录 = `/mnt/mnt/data/resfit`;环境变量 `PYTHONPATH=/mnt/mnt/data/resfit`,`HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1`,GPU 任务加 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_EGL_DEVICE_ID=<物理卡号>`。

---

### Task 1: 训练单状态 act_feat value(CPU,无需 GPU 授权)

**Files:**
- Create: `run_pouring_value_actfeat.sh`(训练脚本,提交 git)
- Produce: `outputs_chunk/pouring_value_actfeat_hdf5.pt`(gitignored 产物)

**Interfaces:**
- Consumes: 现有 `outputs_chunk/pouring_act_feat_hdf5.npz`(缓存命中→跳过 embed);`resfit/dataset/two_arm_pouring.hdf5`;BASE。
- Produces: `outputs_chunk/pouring_value_actfeat_hdf5.pt`,`state_mode=act_feat`,`state_dim=548`,`hidden=256`,含 `act_feat_signature`(= 缓存签名)供 Task 2/3 在线一致性校验。

- [ ] **Step 1: 前置核查(缓存存在 + 磁盘余量)**

Run:
```bash
cd /mnt/mnt/data/resfit
ls -la outputs_chunk/pouring_act_feat_hdf5.npz resfit/dataset/two_arm_pouring.hdf5
ls -la "/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2/policy/model.safetensors"
df -h /mnt/mnt/data | tail -1
```
Expected: 三个文件都存在;磁盘 Avail ≥ 50G(value 本身 ~11M,充裕)。

- [ ] **Step 2: 写训练脚本**

Create `run_pouring_value_actfeat.sh`:
```bash
#!/usr/bin/env bash
# 训练单状态 act_feat value(pouring),喂给 pothiql 当势函数 Φ(s)=V(image_feat+proprio)。
# 复用现有 act_feat 缓存(命中即跳过 embed);配方镜像 eef pouring_value_hdf5.pt(hidden256/默认超参)。
set -eu
export PATH="/root/miniconda3/bin:$PATH"
export PYTHONPATH=/mnt/mnt/data/resfit
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
cd /mnt/mnt/data/resfit
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
conda run -n residual --no-capture-output python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --hdf5 resfit/dataset/two_arm_pouring.hdf5 \
  --dataset ankile/dexmg-two-arm-pouring \
  --state_mode act_feat \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --act_base_ckpt "$BASE" \
  --value_hidden 256 \
  --output outputs_chunk/pouring_value_actfeat_hdf5.pt
```
（`--act_base_ckpt` 在缓存命中时不触发加载 ACT,只作 ckpt_id;pooling/proprio_key/gamma/expectile/steps 全走默认,与缓存签名及 eef value 配方一致。）

- [ ] **Step 3: 跑训练**

Run:
```bash
cd /mnt/mnt/data/resfit && bash run_pouring_value_actfeat.sh 2>&1 | tail -25
```
Expected: 打印 `[read_per_demo_states] act_feat 缓存命中 ...`、`[hiql_value] state_mode=act_feat demos=1009 transitions=... state_dim=548`、`[hiql_value] saved outputs_chunk/pouring_value_actfeat_hdf5.pt; state_mode=act_feat v_stats={...}`,无异常。

- [ ] **Step 4: 验证产物(这是本任务的"测试")**

Run:
```bash
cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python - <<'PY'
import torch, json
c=torch.load('outputs_chunk/pouring_value_actfeat_hdf5.pt', map_location='cpu', weights_only=False)
assert c['state_mode']=='act_feat', c['state_mode']
assert c['state_dim']==548, c['state_dim']
assert c['hidden']==256, c['hidden']
vs=c['v_stats']; assert all(map(lambda x: x==x, vs.values())), vs   # 非 NaN
sig=c.get('act_feat_signature') or {}
assert sig.get('pooling')=='mean' and sig.get('proprio_key')=='observation.state', sig
assert 'run_anw5pphu_best' in str(sig.get('act_ckpt_id')), sig.get('act_ckpt_id')
print('OK act_feat value:', c['state_dim'], 'v_stats=', vs)
print('signature ckpt:', sig.get('act_ckpt_id'))
PY
```
Expected: 打印 `OK act_feat value: 548 v_stats=...`;所有断言通过。

- [ ] **Step 5: 提交训练脚本**

Run:
```bash
cd /mnt/mnt/data/resfit
git add run_pouring_value_actfeat.sh
git commit -m "feat: pouring single-state act_feat value training script (pothiql potential)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(不 add `.pt`——`outputs_chunk/` 为 gitignored 产物。)

---

### Task 2: 端到端 smoke(GPU;这条路从没真跑过)

**Files:**
- Create: `run_pouring_pothiql_actfeat_smoke.sh`(smoke 脚本,提交 git)

**Interfaces:**
- Consumes: Task 1 的 `pouring_value_actfeat_hdf5.pt`;现有 `pouring_gc_value_actfeat_hdf5_hiqlv512.pt`、`pouring_high_actor_actfeat_hdf5_hiqlv512.pt`、`pouring_act_feat_hdf5.npz`、BASE、源 hdf5。
- Produces: 确认 act_feat pothiql 在线+离线+eval 端到端跑通(或一处零回归修复)。

- [ ] **Step 1: 写 smoke 脚本(对齐 rerun0703 关键参数,但只跑 40 步/4 demo/wandb 关)**

Create `run_pouring_pothiql_actfeat_smoke.sh`:
```bash
#!/usr/bin/env bash
# act_feat pothiql 端到端 smoke:验证势函数换成 act_feat value 后离线烤 reward+在线提特征+eval 全链跑通。
# 用法: bash run_pouring_pothiql_actfeat_smoke.sh <gpu>
set -eu
GPU=${1:-0}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
export PYTHONPATH=/mnt/mnt/data/resfit
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
cd /mnt/mnt/data/resfit
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
conda run -n residual --no-capture-output python -u \
  -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping potential --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/pouring_value_actfeat_hdf5.pt \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_num_demos 4 \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 40 --learning_starts 10 --eval_every_env_steps 30 \
  --eval_num_envs 2 --eval_num_episodes 2 --utd 1 \
  --wandb_mode disabled --output_dir /tmp/pouring_pothiql_actfeat_smoke_out
```
(相对 rerun0703 仅:value→act_feat、加 `--offline_num_demos 4`、总步/eval/utd 缩小、wandb disabled、/tmp 输出、无 offcache。)

- [ ] **Step 2: 跑 smoke**

Run（GPU 授权后;选一张空闲卡 `<gpu>`,物理卡号见 nvidia-smi):
```bash
cd /mnt/mnt/data/resfit && bash run_pouring_pothiql_actfeat_smoke.sh <gpu> 2>&1 | tee /tmp/pouring_actfeat_smoke.log | tail -40
```
Expected(全链通过的判据):
- 打印 `[hiql-phi] potential on; ckpt=outputs_chunk/pouring_value_actfeat_hdf5.pt state_mode=act_feat scale=...`
- 打印 `[reward-shaping] mode=potential ...` 与 offline 灌装成功(`[offline] 灌装 N 条 ...`,N>0)
- 在线 40 步无异常,出现 `[env_steps ...] eval success_rate=...`
- 结尾无 Python traceback。

- [ ] **Step 3: 断言 smoke 关键行(自动化判据)**

Run:
```bash
grep -qE "potential on; ckpt=outputs_chunk/pouring_value_actfeat_hdf5.pt state_mode=act_feat" /tmp/pouring_actfeat_smoke.log \
 && grep -qE "\[offline\] 灌装 [1-9]" /tmp/pouring_actfeat_smoke.log \
 && grep -qE "eval success_rate=" /tmp/pouring_actfeat_smoke.log \
 && ! grep -qiE "Traceback|Error:|assert" /tmp/pouring_actfeat_smoke.log \
 && echo "SMOKE PASS" || echo "SMOKE FAIL — 见 /tmp/pouring_actfeat_smoke.log"
```
Expected: `SMOKE PASS`。

- [ ] **Step 4:【仅当 SMOKE FAIL】零回归 TDD 修复**

若 smoke 报错(例:act_feat 势函数在线/离线某处缺口),进入 systematic-debugging + TDD:
1. 定位报错文件/行,读错误堆栈。
2. 在对应模块 `tests/` 下先写一个复现该缺口的**失败**单测(用 fake act_feat value/encoder,勿依赖 GPU/env),`pytest <新测> -v` 确认 FAIL。
3. 在**现有 act_feat 分派之后**做最小增量修复(不动 eef/eef_piece/potsubgoal 路径)。
4. `pytest <新测> -v` 确认 PASS。
5. 全量回归:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`,必须全绿(零回归)。
6. 重跑 Step 2–3 直到 `SMOKE PASS`。
7. 提交:`git add <改动文件> <新测>; git commit -m "fix: <缺口简述> in act_feat pothiql path (zero-regression)"`。

若 SMOKE 一次通过,本 Step 跳过。

- [ ] **Step 5: 提交 smoke 脚本**

Run:
```bash
cd /mnt/mnt/data/resfit
git add run_pouring_pothiql_actfeat_smoke.sh
git commit -m "test: pouring act_feat pothiql end-to-end smoke script

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: A/B 正式 run 脚本 + 启动(GPU;offcache 重建;长任务)

**Files:**
- Create: `run_pouring_pothiql_actfeat_joint.sh`(正式 run 脚本,提交 git)
- Produce: offcache `outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_actfeat_joint_offcache`(运行时首建);run 产物目录同名。

**Interfaces:**
- Consumes: Task 1 value + Task 2 已验证的代码路径 + rerun0703 的全部现有产物(gc_value/high_actor/act_feat cache/BASE)。
- Produces: 500k A/B run,eval 曲线对照现有 eef `pouring_pothiql_joint_rerun0703`(~0.94)。

- [ ] **Step 1: 由 rerun0703 复制并只改 4 处**

Create `run_pouring_pothiql_actfeat_joint.sh` —— 逐字复制 `run_pouring_pothiql_joint_rerun0703.sh`,仅改:
1. `OFFCACHE=` → `outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_actfeat_joint_offcache`（新路径→必重建;**删掉**"只读复用"注释与 `gate "$OFFCACHE/buffer_meta.json"` 那行,因首建时不存在)
2. `--hiql_value_ckpt outputs_chunk/pouring_value_hdf5.pt` → `--hiql_value_ckpt outputs_chunk/pouring_value_actfeat_hdf5.pt`
3. `--wandb_name ..._pothiql_joint_rerun0703` → `..._pothiql_actfeat_joint`
4. `--output_dir outputs_chunk/..._pothiql_joint_rerun0703` → `outputs_chunk/..._pothiql_actfeat_joint`

其余行(task/base/dataset/offline_dataset_path/chunk1/queue/base_n10/as0.05/lr1e-6/raw/reward_shaping potential/potential_source hiql/offline_fraction0.5/base_policy/bc0.1/subgoal + gc_value + high_actor + act_feat_cache/way_steps15/joint online_finetune/total 500000/wandb_project)**逐字不动**。同时把 `gate "outputs_chunk/pouring_value_hdf5.pt"` 改成 `gate "outputs_chunk/pouring_value_actfeat_hdf5.pt"`。

- [ ] **Step 2: 静态核对(diff 只应有预期几处)**

Run:
```bash
cd /mnt/mnt/data/resfit
diff run_pouring_pothiql_joint_rerun0703.sh run_pouring_pothiql_actfeat_joint.sh
```
Expected:改动**只**落在这些行——(a)`OFFCACHE=` 路径 + 其"只读复用"注释 + 删掉的 `gate "$OFFCACHE/buffer_meta.json"` 行;(b)`gate` 的 value 路径 `pouring_value_hdf5.pt`→`pouring_value_actfeat_hdf5.pt`;(c)`--hiql_value_ckpt` 值;(d)`--wandb_name`;(e)`--output_dir`;(f)注释/echo 里的 run 名。**逐行确认所有 `--` 训练超参(chunk/queue/base_n10/as0.05/lr1e-6/raw/reward_shaping/potential_source/offline_fraction/base_policy/bc0.1/subgoal/gc_value/high_actor/act_feat_cache/way_steps/joint/total 500000)完全没变。**

- [ ] **Step 3: 提交正式 run 脚本(启动前先入库)**

Run:
```bash
cd /mnt/mnt/data/resfit
git add run_pouring_pothiql_actfeat_joint.sh
git commit -m "feat: pouring act_feat pothiql joint A/B run script (potential value eef->act_feat)

Aligned byte-for-byte with pouring_pothiql_joint_rerun0703 except potential
value input (eef 36d -> act_feat 548d) + fresh offcache + run naming.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4:【GPU 授权 + 磁盘核查后启动】**

先核查磁盘(offcache 首建 ~41G + base_policy CPU forward ~80min):
```bash
df -h /mnt/mnt/data | tail -1   # Avail 应 ≥ 80G
```
用户授权 GPU 后启动(挑空闲卡 `<gpu>`,物理卡号对齐):
```bash
cd /mnt/mnt/data/resfit
setsid bash run_pouring_pothiql_actfeat_joint.sh <gpu> \
  > pouring_pothiql_actfeat_joint.log 2>&1 < /dev/null &
```
Expected(启动里程碑,看日志前若干分钟):`[hiql-phi] potential on ... state_mode=act_feat`、`[offline] 已建 N 条并落盘 ...`(首建 offcache,~80min)→ 进入训练循环、周期性 `[env_steps ...] eval success_rate=...`。

- [ ] **Step 5: 收敛判读(A/B)**

跑满 500k(数天)后,提取 eval 曲线与对照对比:
```bash
cd /mnt/mnt/data/resfit
for f in pouring_pothiql_joint_rerun0703.log pouring_pothiql_actfeat_joint.log; do
  echo "== $f =="; grep -oE "\[env_steps [0-9]+\] eval success_rate=[0-9.]+ \(best [0-9.]+\)" "$f" | tail -6
done
```
判据:act_feat 版应**稳定不塌**(合法性预期保持);成功率相对 eef ~0.94 **是否抬升/持平**为经验结论,不预设。把结论回写记忆(potsubgoal 根因记忆的关联项)。

---

## Notes / 依赖与风险(执行时留意)

- **同源命门**:Task 1 的 value 训在 `pouring_act_feat_hdf5.npz` 上,在线 `PotentialActFeatureEncoder` 复现同一 BASE/image_keys/pooling 特征;由 `_validate_actfeat_potential_cache` 签名校验强制。若 Task 2 报"签名不符",八成是 value 与 cache 的 signature 没对上——回 Task 1 确认没误传 `--act_image_keys`/`--pooling`(应全默认走缓存签名)。
- **offcache 必重建**:势函数 value 变了→离线 reward 变→旧 `_pothiql_joint_offcache` 对本 run 无效;用全新路径(Step 1 已处理)。
- **GPU 卡漏渲染铁律**:dexmg/EGL 下 `MUJOCO_EGL_DEVICE_ID` 必须=物理卡号,渲染漏到别卡的进程别当残留 kill(见记忆 egl_render_samecard)。
- **无断点续训**:若中途中断,按 shutdown_restart_recipe——mv 旧 output_dir/.log 到 .bak 再重启,盯"命中缓存 vs 重建"+磁盘。
