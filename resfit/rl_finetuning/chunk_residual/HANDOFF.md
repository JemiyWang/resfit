# chunk_residual — Handoff(2026-05-31 晚,M0.3 AE 已训 / M1-M2 待跑)

## 0. 一句话现状
**在 dexmg 上验证 kai0/rlt 的 `residual_flow` actor"有没有戏"的 chunk 级残差 RL harness**(纯新增、零改动原仓库),分支 **`chunk-residual-validation`**,13 个 CPU 单测全绿。本轮新增了**任务化启动脚本**、修了一个 **AE 视频解码崩溃 bug**,并跑完了 **3 个新任务的 AE(M0.3)**:

| AE ckpt | recon_l1(门槛~0.05) | 备注 |
|---------|--------------------|------|
| `ckpt/ae_drawer.pt`    | **0.0267** ✅ | 好 |
| `ckpt/ae_threading.pt` | **0.0436** ✅ | 达标 |
| `ckpt/ae_piece.pt`     | **0.0515** ⚠️ | 边缘略超,见 §7.5——可接受或加 steps 重训 |
| boxcleanup / coffee / cansort / transport | — | 尚未训 |

下一步:**M1.3 chunk-raw 短训 → M2.3 flow vs raw A/B**。**阻塞点**:drawer/piece/threading/transport 的 **ACT 基座 BC 还没训好**(`residual_td3.py` 里 `base_wandb_id` 仍是 `TODO_FILL_BC_RUN_ID`),没基座就起不了 chunk RL;只有 **boxcleanup** 基座现成(但它的 AE 还没训)。

---

## 1. 目标 & 核心思路
把 ResFiT 的单步残差 RL 抬到 **chunk 级**(20 步),移植 rlt 的 `residual_flow` actor(经动作自编码器 AE 的 latent flow + decoder 差分锚定产生残差),与最朴素的 `raw_residual`(直接 MLP)做**干净 A/B**,看 residual_flow 能否稳定训练并追平/超过 raw。
**核心原则**:冻结一切(ACT 基座 + flow 的 AE),只换 actor;env/critic/buffer/TD3/eval 全复用 ResFiT,只动残差粒度(单步→20 步)和 actor 参数化。
**为什么 chunk 级**:rlt §17——residual_flow 只在 chunk 级划算(单步退化);ACT 本就 20 步开环执行,RL 决策点天然对齐 chunk 边界。

---

## 2. 文件位置速查
- 新代码目录:`resfit/rl_finetuning/chunk_residual/`
  - `action_autoencoder.py` — ActionAutoencoder(时序卷积 AE)+ loss
  - `train_action_ae.py` — AE 离线预训练(M0);**本轮修复:不再解码视频**(见 §6/§7.1)
  - `chunk_act_base.py` — `get_action_chunk()`:不改 ACT,一次性取整段 chunk
  - `chunk_env_wrapper.py` — `ChunkResidualEnvWrapper`:chunk 级 env wrapper
  - `residual_flow_actor.py` — `ResidualFlowActor`(返回 TruncatedNormal,zero-init,冻结 AE)
  - `train_chunk_residual.py` — 训练编排(`--actor raw|flow`)
  - **`train_ae.sh`** ★新 — M0.3 启动脚本,任务化(见 §4/§5)
  - **`run_chunk_residual.sh`** ★新 — M1.3/M2.3 启动脚本,任务化 + tmux
  - **`verify_ae_no_video.py`** ★新 — 验证 AE 数据集不解码视频(可读过坏 episode)
  - `ckpt/` ★新 — AE 产物(ae_drawer/threading/piece.pt 已生成)
  - `tests/` — 4 个测试文件(13 用例,纯 CPU)
- 设计/计划:`docs/superpowers/specs/2026-05-31-residual-flow-dexmg-validation-design.md`、`docs/.../plans/...`
- 项目 memory:`project_residual_flow_dexmg`、`project_resfit_two_new_tasks`(新任务+坏帧坑)、`project_resfit_repro`

---

## 3. git 状态(⚠️ 一堆未提交)
分支 `chunk-residual-validation`,9 个提交(最新 `061ea84 docs: 添加 HANDOFF`)。**本轮改动尚未提交**:
- `M resfit/rl_finetuning/chunk_residual/train_action_ae.py` — AE 视频解码修复
- `?? .../chunk_residual/train_ae.sh`、`run_chunk_residual.sh`、`verify_ae_no_video.py`、`ckpt/`
- 本 HANDOFF.md 是已跟踪文件(此次为更新)
跑前先 `git checkout chunk-residual-validation`(editable 安装,新文件需在原仓库路径下才能 import)。
> 注:仓库根目录另有大量母项目的未跟踪文件(各任务 BC/RL 日志、run_* 目录、根 `HANDOFF.md`),与本 chunk_residual 工作无关。

---

## 4. 环境 & 运行约定
- conda env **`residual`**;**从仓库根目录跑**;torch 2.7.1。
- AE 训练(M0):纯离线、**不需要 EGL**(本轮修复后连视频都不解码)。
- chunk RL(M1/M2):eval 要渲染,需 **`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`**(脚本已内置)。
- 单元测试:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -v`(13 passed)。
- **两个启动脚本都任务化**,task 简称:`boxcleanup|coffee|cansort|drawer|piece|threading|transport`,也可直接传完整 HF dataset id。

---

## 5. 里程碑命令(用启动脚本)

### M0.3 — 训冻 AE（✅ drawer/threading/piece 已完成）
```bash
bash resfit/rl_finetuning/chunk_residual/train_ae.sh <task> <gpu>
#   自动:dataset→ckpt/ae_<task>.pt,日志 chunk_ae_<task>_train.log,前台跑
```
- go/no-go:末尾 `[VAL] recon_l1 < ~0.05`。**只在最后第 20000 步存盘,中途无 checkpoint**(中断=全废,见 §7.3)。
- 想后台/防断线:`tmux new -s chunk_ae_<task> "bash .../train_ae.sh <task> <gpu>"`。
- 待补:**boxcleanup**(M1/M2 首选任务,基座现成,优先训它)。

### M1.3 — chunk-raw 短训(验 chunk 骨架)
```bash
bash resfit/rl_finetuning/chunk_residual/run_chunk_residual.sh raw <task> <gpu> --total_env_steps 100000
# 先冒烟: ... raw <task> <gpu> --smoke --eval_num_envs 2 --eval_num_episodes 2
```
go/no-go:稳定训不崩,eval success_rate 不显著低于基座。

### M2.3 — flow vs raw A/B(真正验证)
```bash
# 同 task 同 seed 同步数,只差 actor;flow 自动补 ckpt/ae_<task>.pt
bash .../run_chunk_residual.sh raw  <task> <gpu> --seed 0 --total_env_steps 300000
bash .../run_chunk_residual.sh flow <task> <gpu> --seed 0 --total_env_steps 300000
```
- 脚本按 task 自动配 `--task`(env 名)/`--dataset`/`--ae_ckpt`/`--base_wandb_id`;**基座 wandb_id 是占位时直接报错**,提示先训 BC 或用 `--base_wandb_id` 覆盖。
- session/日志按 `chunk_<task>_<actor>` 命名,可同机并跑;best 存 `outputs_chunk/best.pt`。
- 判据:① flow 靠 zero-init 起步=基座、不塌(硬门槛);② success_rate 追平/超过 raw。负结果也有价值。
- **当前只有 boxcleanup 基座可跑**;其余任务等 BC(`bash resfit/lerobot/shell/run_bc.sh <task> <gpu>`)出 >0% `_best` 并把 wandb_id 填进 `residual_td3.py` 后才能跑。

---

## 6. 关键设计事实(实现/调试必读,均已核实)
- 维度:TwoArm 双臂 `D=24`、chunk `L=20`、展平 `FLAT=480`;3 相机 84×84。
- **取整段 chunk**(`chunk_act_base.py`,不改 ACT):`normalize_inputs → 重组 images → model(batch)[0] → unnormalize_outputs`,`eval()+no_grad`。
- **两套归一化别混**:ACT 内部用 MEAN_STD;RL 残差空间用 `ActionScaler`(min-max→[-1,1])。chunk 从 ACT 拿到是原始尺度,wrapper 再 `scale` 进 [-1,1];AE 也训在 ActionScaler 空间。
- **base 相加约定**:`agent.act` 返回**纯残差**;wrapper 执行 `clamp(base+residual,-1,1)` 后 unscale 开环逐步执行;buffer 存 **combined**(`info["scaled_action"]`,480);critic Q 吃 combined。
- **复用靠"构造+注入,不改源码"**:`QAgent(action_dim=480, residual_actor=True)` 得 chunk 级 critic+actor;flow 在 `_maybe_inject_flow_actor` 换 `agent.actor/actor_target`、`actor_opt` 重建为只含 velocity 网络(AE 冻结不进优化器)。
- **residual_flow 机制**:`z_ref=sg(encode(a_ref))` → `a_ref_hat=sg(decode(z_ref))` → velocity(zero-init)出 `latent_delta=tanh(v)·0.05` → `z_corr=z_ref+latent_delta` → `decoded_delta=smooth_clip(decode(z_corr)-a_ref_hat, action_delta_clip)`。zero-init ⇒ 起步残差=0=基座;梯度只经 `decode(z_corr)` 回 velocity。
- **公平 A/B 命门**:flow 的 `action_delta_clip` = raw 的 `action_scale` = 0.2。
- `num_envs==1` 训练;`offline_fraction=0`(纯 online)。

---

## 7. Handoff 注意点(坑 & 近似)
1. **★AE 视频解码崩溃(本轮修复)**:dexmg 数据集有坏视频帧(drawer ep505、piece {540,709,825,862} 等,torchcodec `Could not push packet to decoder`)。AE 本只用动作,但 `LeRobotDataset.__getitem__` 只要 `meta.video_keys` 非空就给每样本解码全部相机视频 → 撞坏帧崩(drawer)或 worker 满 CPU 拖慢(threading)。**修法**:`build_chunk_dataset` 构造后清空 video 类特征 → `video_keys` 空 → 完全跳过视频解码(既绕坏帧又大幅加速,对所有任务通用)。`verify_ae_no_video.py` 已验证可读过 drawer ep505 不崩。
2. **AE 训练只最后存盘**:`train_action_ae.py` 跑满 20000 步后才 `[VAL]` + `torch.save` 一次,中途无 checkpoint。中断=磁盘无产物、全废。需要的话可自行加"每 N 步存一次"(AE 很快、通常不必)。
3. **piece AE 边缘(0.0515)**:略超 ~0.05 软门槛。可先用;若要 <0.05,`bash train_ae.sh piece <gpu>` 改裸命令加 `--steps 40000` 或 `--hidden_dim 384` 重训。drawer(0.027)/threading(0.044)无虞。
4. **eval 多环境会截断整段 chunk**:wrapper 遇任一 env done 就 break,8 env eval 下其它 env 的 chunk 提前切;success_rate 仍有效、对 raw/flow 一致(A/B 公平)。求 faithful 加 `--eval_num_envs 1`(慢但准)。
5. **env_steps 近似计数**:每 chunk 一律计 `chunk_length`(=20),早停略高估;A/B 两边一致,不影响结论。
6. **AE ckpt 已对 weights_only 安全**(stats 存纯 list);flow 注入端 `torch.load(weights_only=True)` 直接读。
7. eval 的 Q 绘图已 `save_q_plots=False` 规避。

---

## 8. 下一步 & 当前快照
- **优先**:训 **boxcleanup AE**(`bash train_ae.sh boxcleanup <gpu>`)→ 跑 boxcleanup 的 M1.3 冒烟+短训 → M2.3 A/B(基座现成,是验证 flow"有没有戏"的最快路径)。
- 其余任务(drawer/piece/threading/transport):需先把各自 BC 训出 >0% `_best`、wandb_id 填进 `residual_td3.py`,才能跑 chunk RL;AE 已就绪(piece 边缘待定)。
- 奖励/折扣 chunk 内为"求和 + 每步 gamma"近似(稀疏成功奖励下≈did-succeed),两边一致即可。
- `z_rl` 当前恒 0(确定性 TD3);未来 flow 采样可启用(`residual_flow_actor.py` 已留槽)。
- **快照(写时)**:3 个 AE 已训完存盘,无 AE 在跑;GPU 0/4/5 空。本轮改动未提交(见 §3)。
