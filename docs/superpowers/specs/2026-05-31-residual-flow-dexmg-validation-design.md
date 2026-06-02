# residual_flow 在 dexmg 上的快速验证 — 设计文档

- 日期:2026-05-31
- 目标读者:diffusion/flow matching 初学者(直觉优先,公式用纯文本)
- 状态:设计已与用户逐节确认,待写实现计划

---

## 1. 目标(scope)

**快速验证 kai0/rlt 的 `residual_flow` actor 在 dexmg(dexmimic)仿真任务上"跑得通、有没有戏",而不是产出论文级 A/B 曲线。**

判据后置在第 7 节。核心是:在一个干净的对照实验里,看 `residual_flow` 这种"经动作自编码器(AE)latent flow 产生残差"的参数化,能不能稳定训练、并追平或超过最朴素的 `raw_residual`(直接 MLP 出残差)。

非目标:多任务扫、多种子统计、复现论文图、Coffee 任务(基座 0% 起不了 RL)。

---

## 2. 背景与已核实事实(实现者必读)

### 2.1 两个项目
- **kai0/rlt**(JAX/Flax):chunk 级残差 RL,基座是 VLA(pi0.5)。`residual_flow` 是其新颖 actor。
- **ResFiT / residual-offpolicy-rl**(PyTorch):单步残差 RL(TD3 + RLPD recipe),基座是 ACT。dexmg 任务、env、critic、buffer、eval 都现成且能跑。

### 2.2 ACT 基座怎么生成动作(已核到代码)
- ACT 是 **CVAE + Transformer**,**不是扩散/flow 模型**。
- 推理时 VAE latent 设为全 0(`resfit/lerobot/policies/act/modeling_act.py:542`),**一次前向输出整段 chunk**(`:599`),确定性。没有加噪、没有迭代去噪、没有 ODE 积分。
- 本 benchmark 配置:`chunk_size=50, n_action_steps=20, temporal_ensemble_coeff=None`(默认值见 `configuration_act.py`,被 `train_bc_dexmg.py:478-479` 覆盖为 20)。
- 走 select_action 的队列分支(`modeling_act.py:202-219`):每次预测 50 步、保留前 20 步入队,**每个 env step 弹 1 步执行,20 步走完才重新预测**。即每个 chunk 的前 20 步全部被开环执行。

含义:ACT 确定性 → 同观测下参考动作 `a_ref` 稳定,这对 `residual_flow` 的锚定(见 4.2)是**有利条件**;且 ACT 本来就以 20 步 chunk 开环执行,RL 决策点天然可对齐到 chunk 边界。

### 2.3 ResFiT 现状(单步残差)
- wrapper `BasePolicyVecEnvWrapper`:每 env step 取 1 个 base 动作 + 1 个残差,`combined = base + residual`(`residual_env_wrapper.py:136`),step 一次环境。
- 训练写死 **`num_envs == 1`**(`train_residual_td3.py:313`,"because of how n_step is implemented")。
- `QAgent`(`off_policy/rl/q_agent.py`)内部按 `action_dim` 建 critic+actor(`:79-86`),TD3 更新对动作维无所谓。约定:**actor 输出残差,QAgent/wrapper 再加 base_action**。
- critic / actor 都按 `action_dim` 参数化(`critic.py:251`、`actor.py:69`)。
- buffer 是 torchrl `TensorDictPrioritizedReplayBuffer` + `MultiStepTransform(n_steps, gamma)`。
- 任务配置 `config/residual_td3.py`:双臂 `action_scale=0.2`、`n_step=3~5`、`UTD=4~8`、buffer 200k~300k、`actor_lr≈5e-6`、`critic_lr=1e-4`、`tau=0.005`。基座 wandb_id 见 `HANDOFF.md §4`。

### 2.4 residual_flow 的真实机制(已核到代码,kai0/rlt)
两个文件:
- `models_jax/action_autoencoder.py` — 时序卷积 AE,**单独离线预训练后冻结**。
- `models_jax/residual_flow_actor.py` — actor,内部用冻结 AE。

AE:`encode`= Dense→relu→Conv1d×2(kernel=3,SAME,LayerNorm)→沿时间 mean-pool→Dense 到 latent(默认 64)→LayerNorm;`decode`= Dense(chunk_len×hidden)→reshape→Conv1d×2→Dense(action_dim)。损失 = L1 重构 + 0.1×速度损失(相邻步差分 L1)。

actor `__call__(z_rl, s_p, ref_action, action_ae_params)` 七步:
1. `z_ref = stopgrad(encode(a_ref))`
2. `a_ref_hat = stopgrad(decode(z_ref))`
3. 速度网络:`concat[z_ref, z_rl, LayerNorm(s_p), a_ref]` → silu MLP(num_layers=3,hidden=512,LayerNorm) → `velocity_out`(Dense 到 latent_dim,**zero-init**,`velocity_init_scale=0.0`)
4. `latent_delta = tanh(velocity) * latent_delta_scale`(默认 0.05);`z_corr = z_ref + latent_delta`
5. `a_corr_hat = decode(z_corr)`
6. `decoded_delta = smooth_clip(a_corr_hat - a_ref_hat, action_delta_clip)`(默认 0.05;smooth_clip(x,L)=L·tanh(x/L))
7. `action = a_ref + decoded_delta`

两个关键设计:
- **zero-init**:velocity 输出层初值 0 → 起步残差=0 → action=a_ref(精确等于基座)→ RL 从基座成功率起步,不崩。
- **decoder 差分锚定**:残差 = 两次解码之差,AE 重构误差在 `a_ref + (decode(z_corr) − decode(z_ref))` 中自动抵消;修正始终落在"合法动作流形"附近。

设计约束(rlt 文档 §17):`residual_flow` 只对 **chunk 级**残差划算;单步残差下退化(多了 AE 误差+延迟,无收益)。→ 所以验证它必须把 ResFiT 抬到 chunk 级。

---

## 3. 总体原则与里程碑

**核心原则:冻结一切,只换 actor。** env / critic 结构 / buffer 机制 / 训练循环 / dexmg 任务 / eval 全部沿用 ResFiT;只动两样:**残差粒度(单步 → 20 步 chunk)** 和 **actor 参数化**。这样 chunk-raw vs residual_flow 的对比干净。

> 记号约定:`D` = 每步动作维(env 动作空间维度);chunk 长度 `L = 20`(= ACT 的 `n_action_steps`,即实际开环执行的步数,非预测的 50);展平后的 chunk 动作维 = `L×D`,全文记作 `20×D`。

**不动原仓库代码**:所有新文件放进项目内独立新目录 `resfit/rl_finetuning/chunk_residual/`(`import` 正常,不修改任何现有文件);复用现有类靠"用新 `action_dim` 构造 + 把 residual_flow actor 注入 QAgent"实现,不改 `q_agent.py` 等源码一行。

里程碑(每个有明确 go/no-go):

| 里程碑 | 内容 | go/no-go |
|---|---|---|
| **M0** | PyTorch 版 ActionAutoencoder + 离线预训练(冻结) | held-out 重构 L1 足够小;`decode(encode(a))≈a` |
| **M1** | chunk 级骨架 + 现成 raw_residual actor(零新 ML) | 能稳定训、成功率从基座 ~0.1 往上爬(至少不崩) |
| **M2** | 注入 PyTorch 版 residual_flow actor,做干净 A/B | ① 稳定训不塌;② 成功率追平/超过 chunk-raw |

任务:第一个用 **TwoArmBoxCleanup**(README/HANDOFF 命令最全、RL config 现成、基座 ~0.1 有头room)。CanSort 作备选第二个。Coffee 跳过。

---

## 4. 组件设计

### 4.1 复用 vs 新写

| 组件 | 处理 |
|---|---|
| `create_vectorized_env`、`ACTPolicy`、`ActionScaler`、`StateStandardizer`、checkpoint 工具 | import 直接用 |
| `Critic` / `QAgent` | import,用 `action_dim = 20×D` 构造;TD3 更新喂 chunk transition |
| replay buffer(torchrl + MultiStepTransform) | 新脚本里用同样构造器建,动作 shape 改 20×D,n_step 按 chunk |
| chunk 级 env wrapper | 🆕 新写 |
| chunk 级训练脚本(编排循环) | 🆕 新写(重写 train_residual_td3 主循环) |
| PyTorch ActionAutoencoder + 预训练脚本 | 🆕 新写(M0) |
| PyTorch residual_flow actor | 🆕 新写(M2) |

新目录建议布局:
```
resfit/rl_finetuning/chunk_residual/
  __init__.py
  chunk_env_wrapper.py        # 4.3
  action_autoencoder.py       # 4.4(M0)
  train_action_ae.py          # 4.4(M0 预训练脚本)
  residual_flow_actor.py      # 4.2(M2)
  train_chunk_residual.py     # 4.5(M1/M2 训练编排)
```

### 4.2 chunk 级 residual_flow actor(PyTorch,M2)

- **接口对齐**:`forward(obs, std) → 残差(20×D)`,与现成 `Actor` 完全一致,以便注入 QAgent。返回的是残差(decoded_delta),QAgent 再加 base。
- 输入来源:`a_ref = obs["observation.base_action"]`(20×D);条件 `s_p = concat(obs["feat"], obs["observation.state"])`;`z_rl = zeros`(确定性 TD3,探索交给 QAgent 在动作空间加的高斯噪声,与 chunk-raw 一致)。
- 内部七步严格照搬 2.4(zero-init、tanh·scale、decode 差分、smooth_clip)。
- 冻结 AE 作为子模块(`requires_grad=False`);梯度只流到速度网络。`deepcopy` 出 actor_target 时 AE 一起拷、仍冻结。注入后 `actor_opt` 只收速度网络参数。
- 工程注意:JAX Conv 是 channels-last,PyTorch `Conv1d` channels-first,中间转置。

### 4.3 chunk 级 env wrapper(新写)

替代 `BasePolicyVecEnvWrapper`,**单环境**(沿用 num_envs==1,免去批量 ragged 处理):
- `reset()`:重置 env + ACT;直接调 ACT 底层 `model` 一次拿整段 20 步 `a_ref`(复用 ACT 自带输入归一化,不改 ACT 代码),放进 `obs["observation.base_action"]`,返回。
- `step(residual_chunk 20×D)`:`combined = a_ref + residual_chunk` → `action_scaler.unscale` → 在底层 env 上**开环走完 20 步**;累积奖励;中途 terminate 则提前结束并标 done;为 chunk 末观测取新 `a_ref`,返回 `(aug_obs, 累积reward, terminated, truncated, info)`,`info` 带 combined chunk 供入 buffer。
- 必须保证"加残差的 chunk = 实际执行的 chunk"(绕开 ACT 内部队列的二次预测)。M1 阶段验证。

### 4.4 PyTorch ActionAutoencoder + 预训练(M0)

- 结构/损失照搬 2.4。
- 数据:dexmg BC 演示数据切出的 20 步动作 chunk,**用 RL 同一个 ActionScaler 归一化**(AE 必须与残差同处一个归一化动作空间)。
- 训完冻结,产出权重供 M2 加载。

### 4.5 chunk 级训练脚本(编排,M1/M2 共用)

重写 `train_residual_td3.py` 主循环(放新目录):
- build env = `create_vectorized_env` + 新 chunk wrapper。
- build agent = `QAgent(action_dim=20×D, residual_actor=True)`;**M2 额外**:把 `agent.actor`/`agent.actor_target` 替换为 residual_flow actor,重建 `actor_opt`。
- build buffer = torchrl 同构造器,动作 20×D。
- 循环:reset → 每 chunk:`agent.act` 出 20×D 残差 → wrapper 开环 20 步、累积奖励、返回 chunk 末观测 → 存 chunk transition → `agent.update` × UTD。
- eval:复用现成 eval 逻辑,但策略经**同一个 chunk wrapper**(chunk 级评估)。
- 通过开关 `--actor {raw,flow}` 在 M1/M2 间切换,其余配置共享。

---

## 5. 公平 A/B 的关键

- **对齐残差幅度**:chunk-raw 残差上限 = `action_scale`(双臂 0.2);residual_flow 上限 = `action_delta_clip`。把两者设相等(都 0.2),A/B 只比"残差怎么产生",不掺权限差异。`latent_delta_scale` 是内部旋钮,调到 decoded delta 够得着 clip。
- critic 统一用最简单的 **MSE** 版(去掉 C51/HL-Gauss 混淆项),两边一致。
- 同任务、同 ACT 基座、同 buffer/n_step/UTD/lr/种子;**只换 `agent.actor`**。
- **奖励/折扣的 chunk 内约定**(求和 vs 折扣和、每 chunk 的 γ^L)是旋钮,但 chunk-raw 与 residual_flow 必须用同一套。

---

## 6. 实验协议

- 各跑一条(先 1 种子看趋势,有戏再补种子),eval 成功率 vs 环境步数打到 wandb(复用现成记录)。
- 监控:`ae_ref_recon_l1`(AE 在实测 chunk 上的重构误差)、latent_delta/decoded_delta 幅度、Q 是否发散、成功率曲线。

---

## 7. 成功判据("有没有戏")

- **硬门槛**:residual_flow 稳定训练,不塌到基座成功率以下(zero-init 保障起步=基座)。
- **信号**:成功率追平或超过 chunk-raw;"追平但更平滑/更省样本"也算有戏。
- **负结果也有价值**:干净 A/B 下 residual_flow 赢不了 chunk-raw → "单任务 sim 上没戏"的明确结论,避免后续无谓投入。

---

## 8. 风险与缓解

1. **AE 离分布**:AE 在演示 chunk 上训,ACT 实测可能吐出偏离演示流形的 chunk → 解码失真。缓解:decoder 差分锚定让重构误差抵消 + smooth_clip 兜底;RL 时盯 `ae_ref_recon_l1`。
2. **表达力受限(核心科学问题)**:residual_flow 只能往 AE decoder 的像里修,chunk-raw 的 MLP 无此约束。这种"在合法流形上修正"到底是帮助(更协调/更省样本)还是拖累(够不着最优修正)——正是本实验要回答的。
3. **吞吐**:num_envs=1 且每决策走 20 步,采样偏慢;快速验证可缩短 horizon、少评几次。
4. **奖励/折扣约定**跨 A/B 必须一致(见第 5 节)。
5. **接口对齐风险**:residual_flow actor 必须严格匹配 `Actor.forward(obs, std)` 的调用约定(返回残差、QAgent 加 base、target 用 deepcopy)。M2 注入后先做单元检查:初始 forward 残差≈0(action≈base)、梯度只到速度网络。

---

## 9. YAGNI / 明确不做

- 不做多任务扫、不做多种子统计、不复现论文图。
- 不用 C51/HL-Gauss critic(验证期只用 MSE)。
- 不做 temporal ensembling。
- 不碰 Coffee 任务。
- 不做 JAX↔PyTorch 数值对拍(可选,非必须)。

---

## 10. 开放旋钮(实现时定,A/B 两边保持一致)

- chunk 内奖励聚合方式与每 chunk 折扣。
- `latent_delta_scale` 初值与是否需要调。
- AE 的 latent_dim / hidden / conv_layers(先沿用 rlt 默认 64/256/2)。
- horizon、eval 频率、总步数(快速验证可比官方双臂的 500k 小)。
