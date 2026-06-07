# HANDOFF 2026-06-07 (HIQL-Φ) — ③ 已实现,待跑 A/B(可在另一台机器接手)

接 `HANDOFF_2026-06-07.md`(② relabeling 的 A/B)。本篇是给"换一台机器把 ③ HIQL-Φ 的 A/B 跑起来"用的自包含说明。

---

## 0. 一句话现状

- ① stage_budget / ② relay relabeling:已实现+push(各自 handoff)。
- ③ **HIQL-Φ**(用学出来的平滑 value 当 PBS 势函数 Φ,替代整数阶梯 stage Φ)**代码已实现+push**:
  - ③a 离线训 value(`train_hiql_value.py` -> 冻结 value.pt)。
  - ③b 把 value 当 Φ 接进 reward shaping 的 online+offline 两端(`--potential_source hiql`)。
- 三模块全部 default-off、关掉逐位等价 baseline。
- **重要**:③b 的端到端 `main()` hiql 路径还没真跑过(开发时无 value.pt/GPU,靠静态走查+单测);**所以第 3 步先 --smoke 冒烟**,确认链路通再正式 A/B。

---

## 1. 先做:同步代码 + 环境自检

```bash
cd <repo根>            # 原机器 /data2/RL/residual-offpolicy-rl
git fetch origin && git checkout chunk-residual-validation && git pull origin chunk-residual-validation
# 需含 commit f360d0f(③b train 接线)
```
conda env `residual`;**所有命令从仓库根跑**;训练/eval 要 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`。

自检单测(CPU,~30s):
```bash
conda run -n residual python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py \
  resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k "not monotone" -q
```
预期全绿(hiql_value 16 + hiql_potential 10,monotone 慢测跳过)。

---

## 2. ③ 是什么(已实现,无需改代码)

- 现状 `--reward_shaping potential`(PBS, Ng 1999)的 Φ=**整数 latch stage**(阶梯,stage 内部无梯度)。
- ③ 用 action-free IQL expectile value `V(s)`(③a 学的,平滑的"离成功还有多近")当 Φ:stage 内部也有连续梯度。
- PBS 理论:Φ 任意(只要固定)都不改最优策略 -> 换平滑 V 安全(最坏=没增益)。**硬约束**:online 与 offline 两端必须用同一个 Φ(同 V + 同缩放),代码已保证(train 构一个 HiqlPotential 传两端)。
- Φ 尺度:`V*scale`,`auto_scale=(num_stages-1)/(v_max-v_min)`(来自 value.pt 里的 v_stats,让 V 动态范围≈现状 stage Φ 的 [0,4]),`--phi_scale` 默认 1.0 微调。

---

## 3. 前置 + 三步走

### 3.0 前置(同 ②)
- base 用 best:`--base_wandb_id dexmg-threepiece-bc/7zklm69g`。
- 数据文件在:`deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5`(4.1G)、
  `outputs_chunk/three_piece_stages.npz`(stage 缓存)。

### 3.1 第一步:训 value.pt(③a,几分钟,CPU/GPU 都行)
```bash
cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --output outputs_chunk/three_piece_value.pt \
  2>&1 | tee three_piece_value.log
```
- 默认 50000 步 / expectile 0.7 / gamma 0.99;打印 `demos=... transitions=... state_dim=18` 和末尾 `v_stats={min,max,mean}`(max>min 即有区分度)。
- **dataset 必须和后面 RL 的 `--dataset` 一致**(state 标准化同源,否则 V 分布对不上)。
- 可选扫:`--expectile 0.9`(更保守)。

### 3.2 第二步:冒烟(确认 main() hiql 路径端到端通,关键)
在 B run 命令(见 3.3)基础上加 `--smoke`(wandb disabled),先确认:
- 打印 `[hiql-phi] potential on; ckpt=... scale=...`;
- offline buffer 构建完成(用 V 重算 reward,不报错);
- 跑起若干训练 step 不报错。
确认无误后 Ctrl-C,转正式 A/B。(这是 ③b 端到端第一次真跑,务必先冒烟。)

### 3.3 第三步:A/B(强 base = nas10 那套;两臂都用 --reward_shaping potential)
**关键区别**:hiql Φ 只在 `potential` 模式有意义(断言),所以 ③ 两臂都用 `--reward_shaping potential`(不是 ①② 的 staged)。A=stage 源,B=hiql 源,唯一差别是 `--potential_source`。

**A run(potential + 整数 stage Φ,③ 的 baseline):**
```bash
CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual --no-capture-output python -u \
  -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly \
  --base_wandb_id dexmg-threepiece-bc/7zklm69g \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --reward_shaping potential --potential_source stage --stage_reward_bonus 1.0 \
  --action_scale 0.05 --actor_lr 1e-6 --stage_balanced --actor raw \
  --offline_dataset_path deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/three_piece_stages.npz \
  --wandb_project dexmg-chunk-residual --wandb_name nas10_best_potstage \
  --output_dir outputs_chunk/nas10_best_potstage \
  2>&1 | tee nas10_best_potstage.log
```

**B run(potential + HIQL V Φ):** 在 A 的基础上把 `--potential_source stage` 换成下面三行(并改 name/output/log):
```bash
  --potential_source hiql --hiql_value_ckpt outputs_chunk/three_piece_value.pt --phi_scale 1.0 \
  --wandb_project dexmg-chunk-residual --wandb_name nas10_best_pothiql \
  --output_dir outputs_chunk/nas10_best_pothiql \
  2>&1 | tee nas10_best_pothiql.log
```
两臂各占一张卡并行最好。可选扫 `--phi_scale`(0.5 / 1.0 / 2.0)看幅度敏感性。

---

## 4. 看什么 / 判据

- 主指标:`eval success_rate` 后 5 点滑动均值 + per-stage 到达率(`eval_stage_reach.py`),重点 **stage3 回退率**、reach3->reach4。③ 的赌注是"stage 内部有平滑梯度"能不能帮到精插阶段。
- 旁证:B run 启动打印的 `[hiql-phi] potential on; scale=...`(确认 V 接上了);wandb reward 曲线。
- PBS 安全性:Φ 不准也不改最优策略,最坏=B 和 A 持平(没增益)。若 B 明显低于 A,优先怀疑 scale 过大(扫小 phi_scale)或 value 质量(lowdim 不含物体 pose,见下)。
- 判读规矩:eval 方差极大,别被单点带,看后 5 点滑动均值。

---

## 5. 注意事项

- ③ 两臂都用 `--reward_shaping potential`(不是 staged);A 与 B 唯一差别是 `--potential_source`。
- B run 前必须先有 `value.pt`(3.1)且 **先 --smoke 冒烟**(3.2)。
- `value.pt` 的 `--dataset` 与 RL 的 `--dataset` 必须一致(标准化同源)。
- 别加 `--stage_conditioned`(已证伪);③ 先单开,别和 `--stage_budget`/`--relabel` 同开(先各自验证)。
- **已知局限**:value 用 lowdim 18 维 eef state(不含物体 pose),在成功流形上准、online 偏离时可能高估 -> 若 ③ 增益不明显,这是首要怀疑点;后续可把 ③a 的 value 升级为观测(ViT)输入,③b 接口不变。

---

## 6. 三模块全景 + 后续
- ① budget / ② relabeling / ③ HIQL-Φ 代码全齐(default-off,可单开,关掉逐位等价)。
- 各自 A/B:① HANDOFF_2026-06-06.md;② HANDOFF_2026-06-07.md;③ 本篇。
- 都跑出曲线后一起判读,再决定哪个值得叠加/深挖。
- memory:`project_stage_budget_and_3module_plan`(诊断 + 三模块路线 + 实现细节)。
