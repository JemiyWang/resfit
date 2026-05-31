# chunk_residual — Handoff(2026-05-31,代码完成 / 待 GPU 跑)

## 0. 一句话现状
**在 dexmg 上验证 kai0/rlt 的 `residual_flow` actor"有没有戏"的 chunk 级残差 RL harness 已实现完毕**(纯新增、零改动原仓库),在分支 **`chunk-residual-validation`** 上,**13 个 CPU 单元测试全绿**,**尚未跑过 GPU**。下一步是手动跑三个里程碑(M0.3 训 AE → M1.3 chunk-raw 短训 → M2.3 flow vs raw 的 A/B)。

---

## 1. 目标 & 核心思路
快速验证:把 ResFiT 的单步残差 RL 抬到 **chunk 级**,移植 rlt 的 `residual_flow` actor(经动作自编码器 AE 的 latent flow + decoder 差分锚定产生残差),与最朴素的 `raw_residual`(直接 MLP)做**干净 A/B**,看 residual_flow 能否稳定训练并追平/超过 raw。

**为什么必须 chunk 级**:rlt 文档 §17——residual_flow 只在 chunk 级残差上划算(单步退化)。ACT 基座本就以 20 步 chunk 开环执行(`n_action_steps=20`),RL 决策点天然对齐到 chunk 边界。

**核心原则**:冻结一切,只换 actor。env / critic / buffer / TD3 更新 / dexmg 任务 / eval 全部复用 ResFiT;只动两样——残差粒度(单步→20 步 chunk)和 actor 参数化。

---

## 2. 文件位置速查
- **新代码目录**(本工作全部产物):`resfit/rl_finetuning/chunk_residual/`
  - `action_autoencoder.py` — PyTorch 版 ActionAutoencoder(时序卷积 AE)+ `action_autoencoder_loss`
  - `train_action_ae.py` — AE 离线预训练脚本(M0)
  - `chunk_act_base.py` — `get_action_chunk()`:不改 ACT,一次性取整段 chunk
  - `chunk_env_wrapper.py` — `ChunkResidualEnvWrapper`:替代 BasePolicyVecEnvWrapper,chunk 级
  - `residual_flow_actor.py` — PyTorch 版 `ResidualFlowActor`(返回 TruncatedNormal,zero-init,冻结 AE)
  - `train_chunk_residual.py` — 训练编排(`--actor raw|flow`)
  - `tests/` — 4 个测试文件(13 个用例)
- **设计 spec**:`docs/superpowers/specs/2026-05-31-residual-flow-dexmg-validation-design.md`
- **实现计划**:`docs/superpowers/plans/2026-05-31-residual-flow-dexmg-validation.md`
- **项目 memory**:`/home/ubuntu/.claude/projects/-data2/memory/project_residual_flow_dexmg.md`
- 母项目环境/复现要点见仓库根 `HANDOFF.md` 与 memory `project_resfit_repro.md`。

---

## 3. git 状态
- 分支 `chunk-residual-validation`(从 main 拉出),**8 个提交,只新增 `chunk_residual/` 下文件,未碰任何现有文件,未合并未推送**。main 上你的复现工作未提交改动原样保留。
- 提交链(新→旧):
  ```
  2f6b227 fix: AE ckpt 对 weights_only 安全(stats 转 list)+ ae_config 补全
  43afffb feat: 注入 residual_flow actor(--actor flow)+ 注入单元测试
  b477a75 feat: PyTorch residual_flow actor 移植 + 单元测试
  4e322ae fix: M1.2 review 修复(num_envs/checkpoint/清理)
  f80ae2c feat: chunk 级训练编排(--actor raw)
  e93edb5 feat: chunk 级 env wrapper + ACT 整段 chunk helper
  9e38288 feat: AE 离线预训练脚本
  b94ee4c feat: PyTorch ActionAutoencoder 移植 + 单元测试
  ```
- 跑前先 `git checkout chunk-residual-validation`(editable 安装,新文件在原仓库路径下才能 import)。

---

## 4. 环境 & 运行约定
- conda env **`residual`**;从仓库根目录跑;eval 渲染需 **`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`**;torch 2.7.1。
- 单元测试(纯 CPU,无需数据/GPU):
  ```bash
  cd /data2/RL/residual-offpolicy-rl
  conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -v   # 13 passed
  ```

---

## 5. 待跑的三个里程碑(手动,需空闲 GPU)

### M0.3 — 训冻 AE
```bash
cd /data2/RL/residual-offpolicy-rl
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_action_ae \
    --dataset ankile/dexmg-two-arm-box-cleanup \
    --chunk_length 20 --action_scale 0.2 --min_range_per_dim 0.1 \
    --steps 20000 --batch_size 256 \
    --out resfit/rl_finetuning/chunk_residual/ckpt/ae_boxcleanup.pt
```
go/no-go:末尾 `[VAL] recon_l1` 应 < ~0.05(归一化到 [-1,1] 后)。不达标就加 `--steps`/`--hidden_dim` 重训。`--action_scale` 必须与 RL 端一致(0.2)。

### M1.3 — chunk-raw 短训(验 chunk 骨架,与 flow 无关)
```bash
CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    --actor raw --total_env_steps 100000 2>&1 | tee chunk_raw_boxcleanup.log
```
go/no-go:能稳定训不崩,eval success_rate 从基座(BoxCleanup base ~0.1)不显著下降、最好上升。**先冒烟**:加 `--smoke --eval_num_envs 2 --eval_num_episodes 2` 验维度通。

### M2.3 — flow vs raw A/B(真正的验证)
```bash
# A:chunk-raw
... train_chunk_residual --actor raw  --seed 0 --total_env_steps 300000 | tee chunk_raw_ab.log
# B:residual_flow(同种子同配置,只换 actor + 给 AE)
... train_chunk_residual --actor flow --ae_ckpt resfit/rl_finetuning/chunk_residual/ckpt/ae_boxcleanup.pt \
    --seed 0 --total_env_steps 300000 | tee chunk_flow_ab.log
```
判据("有没有戏"):① flow 靠 zero-init 起步=基座、不塌到 ~0.1 以下(硬门槛);② success_rate 追平/超过 raw(追平但更平滑/更省样本也算)。负结果同样有价值=单任务 sim 没戏。
- best 模型存到 `outputs_chunk/best.pt`(success_rate 创新高时)。
- 看曲线:`grep "eval success_rate" *_ab.log`。

---

## 6. 关键设计事实(实现/调试必读,均已在代码里核实)
- 维度:TwoArmBoxCleanup `D=24`、chunk `L=20`、展平 `FLAT=480`;state 38;3 相机 84×84。
- **取整段 chunk**(`chunk_act_base.py`,不改 ACT):`normalize_inputs → 重组 observation.images → model(batch)[0] → unnormalize_outputs`,在 `eval()+no_grad` 下。
- **两套归一化别混**:ACT 内部动作用 MEAN_STD;RL 残差空间用 `ActionScaler`(min-max→[-1,1])。chunk 从 ACT 拿到是原始尺度,wrapper 再过 `action_scaler.scale` 进 [-1,1]。AE 也训在 ActionScaler 空间。
- **base 相加约定**:`agent.act` 返回**纯残差**;wrapper 执行时 `clamp(base+residual,-1,1)`;buffer 存 **combined**(`info["scaled_action"]`,480 维);critic Q 吃 combined。
- **复用靠"构造+注入,不改源码"**:`QAgent(action_dim=480, residual_actor=True)` 自动得到 chunk 级 critic+actor;raw 用现成 Actor;flow 在 `_maybe_inject_flow_actor` 里把 `agent.actor/actor_target` 换成 `ResidualFlowActor`、`actor_opt` 重建为只含 velocity 网络参数(冻结 AE 不进优化器)。
- **residual_flow 机制**:`z_ref=sg(encode(a_ref))` → `a_ref_hat=sg(decode(z_ref))` → 速度网络(zero-init 末层)出 `latent_delta=tanh(v)·0.05` → `z_corr=z_ref+latent_delta` → `decoded_delta=smooth_clip(decode(z_corr)-a_ref_hat, action_delta_clip)` → 返回 `decoded_delta`。zero-init ⇒ 起步残差=0=基座;梯度只经 `decode(z_corr)` 回到 velocity(AE 冻结)。
- **公平 A/B 关键**:flow 的 `action_delta_clip` = raw 的 `action_scale` = 0.2(残差幅度上限对齐),只比"残差怎么产生"。
- `num_envs==1`(训练);`offline_fraction=0`(纯 online,免建 chunk 离线 buffer)。

---

## 7. Handoff 注意点(已知、不阻塞、对 A/B 公平)
1. **eval 多环境会截断整段 chunk**:wrapper 遇任一 env done 就 break,8 环境 eval 下其它 env 的 chunk 被提前切。success_rate 仍有效、对 raw/flow 一致(A/B 公平);求 faithful 的 chunk 级 eval 加 `--eval_num_envs 1`(慢但准)。
2. **env_steps 近似计数**:每 chunk 一律计 `chunk_length`(=20)步,早停略高估;A/B 两边一致,不影响结论,只是 x 轴与单步基线对比时略偏。
3. **AE ckpt 已修 weights_only**:stats 存为纯 list,兼容 torch 2.7.1 默认 `weights_only=True`。若你手改保存格式,注意别塞 numpy。
4. eval 的 Q 绘图已 `save_q_plots=False` 规避;`run_dexmg_evaluation` 的 Q 维度=480 与 critic 一致,无碍。

---

## 8. 下一步可选扩展(超出当前范围)
- 有戏 → 扩到 CanSort(第二个双臂任务)、补多种子。
- 奖励/折扣的 chunk 内约定目前是"chunk 内求和 + 每步 gamma"近似(稀疏成功奖励下≈did-succeed);若要更严谨可改 chunk 内折扣和 + 每 chunk gamma^L,两边保持一致即可。
- `z_rl` 当前恒置 0(确定性 TD3);未来若要 flow 采样可启用(`residual_flow_actor.py` 已留槽)。
