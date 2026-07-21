# HiRes-RL ↔ RISE Dynamics Model 接入设计（实现依据）

> 日期：2026-07-19 起草，**2026-07-21 大更新**（集成完成、S0/S1 通过）。
> 定位：**本文是接入层的唯一实现依据。**
> 配套：`docs/superpowers/specs/…-design.md`（决策）、`…/plans/….md`（TDD 计划与测试代码）。
> 所有 `file:line` 引用均已在代码中逐行核对。

> ## 📍 当前状态（2026-07-21）
> **三样冻结组件 + LTX 基座全部就位，S0/S1 端到端通过，wm_bridge 前置代码建成。**
> - **依赖**：D（`RISE_Hi/ckpt/step_18000`）+ block V（`outputs_chunk/block_value_pi0feat.pt`，
>   pi0_feat/ψ，success_signed ±1，成/失败末态完全分离）+ kai0（`data2/kai0/checkpoints/49999`，
>   `pi05_block_awbc`）+ LTX 基座（`data2/dyanmic_model/checkpoints/{tokenizer,text_encoder,vae}`）。
> - **★ 跨环境架构（关键，初稿没有）**：D 在 `rise_venv`（diffusers），wm_bridge/V/训练在
>   residual env，kai0 在 openpi venv —— 一个进程装不下多套 torch/diffusers。故 **D 也包成
>   websocket serve**（`d_serve.py`，与 kai0 serve 同款），residual 侧 `WmServeClient`
>   （`wm_client.py`）鸭子类型 `wm.infer(...)`。三 serve 各占一卡、websocket 解耦。见 §6.10。
> - **V 定为 pi0_feat（吃 kai0 ψ），不是 act_feat**（block 无可用 ACT encoder）；同源锚 =
>   value.pt 的 `pi0_feat_signature.serve_ckpt_id`（不是 act_weight_sha）。
> - **S0 通过**：动作可控性像素差 0.13/0.10/0.18（>>1e-3 约 100 倍），WM 明确动作敏感。
> - **S1 通过**：全 `ImaginationVecEnv` 端到端（真 D-serve+kai0+V），Φ₀0.28、每 chunk 1 次
>   serve（窗口缓存）、reward 有限、2 段截断对。
> - **infer 实测 ~0.5s/次（10 步去噪）→ 500k 步长训 ≈ <2h**（初稿估 5.5h 偏保守）。
> - **剩**：Task10 builder 组装（把上述组件接进 launcher 的 3 假符号）+ 正式起点采样
>   （teleavatar EpisodeSource）+ 实验矩阵 + 长训。**无实机 eval 步骤**（见 §10）。

---

## 0. 逻辑流：HiRes-RL 在世界模型里是怎么跑起来的

### 0.1 三个冻结组件（离线准备，都不在训练中更新）

| 组件 | 来源 | 在回路里干什么 |
|---|---|---|
| **D** dynamics model | RISE 在三域混合质量数据上微调 | 吃「4 帧历史 + 25 动作 token」，吐 25 帧预测 |
| **π_base** kai0/pi05 | 专家数据训练，起 websocket serve | 吃想象帧，吐 50 步基座动作 **+ `prefix_feat`（即 ψ）** |
| **V** HIQL value | 成功集 ∪ 失败集，成功末 +1／失败末 −1 | 吃 `标准化(ψ ⊕ proprio)`，吐标量势 Φ |

三者的耦合点是 **ψ**：它是 π_base 推理的副产物，同时是 V 的输入。
故 **π_base 的权重必须与训 V 时冻结的那份逐位相同**（§7.3 同源校验）。

### 0.2 一个想象段的完整生命周期

一段 = **2 个 chunk**；一个 chunk = **50 个动作步 = 1.667 s 墙钟时间**。

```
reset()
 ├─ 从真实数据采起点：4 帧 × 3 视角窗口 W₀ + 该帧 proprio p₀ + caption（常量）
 ├─ π_base(W₀) ──► 50 步基座动作 A₀ ，副产物 ψ₀
 ├─ Φ_prev ← V(ψ₀ ⊕ p₀)
 └─ 返回 obs₀ ＝ { 图像(W₀ 末帧→84×84), observation.state=p₀, 原生帧旁路 }

     ┌───────────────────── 残差 TD3 的一个 transition ─────────────────────┐
     │                                                                      │
step() × 50   ← ChunkResidualEnvWrapper 逐时间步调用（见 §9 修正一）
 ├─ 第 1..49 次：把动作攒进缓冲区，返回「缓存 obs / reward=0 / done=False」
 │              （wrapper 对中间步的返回值本就不消费，§9 修正一有逐行证明）
 └─ 第 50 次：点火
      ├─ 50 个物理动作 ──每 2 个抽 1──► 25 步 ──min-max 归一化──► (1,25,30) act_tokens
      ├─ D.infer(W₀, act_tokens) ──► (3,3,29,192,256) ──切掉前 4 帧历史──► 25 预测帧
      ├─ W₁ ← 预测段末 4 帧
      ├─ p₁ ← 本段第 50 个动作（绝对关节，零跟踪误差假设，§6.3）
      ├─ π_base(W₁) ──► 下一段基座动作 A₁，副产物 ψ₁      ← 同一次调用，缓存复用
      ├─ Φ_next ← V(ψ₁ ⊕ p₁)
      ├─ reward ← γ·Φ_next − Φ_prev          ← PBRS 端点差（§5）
      ├─ Φ_prev ← Φ_next
      └─ terminated=False（恒定）， truncated=(seg_step ≥ 2)
     │                                                                      │
     └──────────────────────────────────────────────────────────────────────┘

seg_step 达到 2 → truncated=True → wrapper autoreset → 回到 reset()
```

### 0.3 残差在哪里介入

`ChunkResidualEnvWrapper`（既有代码，**零改动**）负责：

```
残差网络输出 Δ (50×16 展平)
        ↓
combined = clamp(base_action_normalized + Δ, −1, 1)     # chunk_env_wrapper.py:169
        ↓
env_chunk = ActionScaler.unscale(combined)              # :172  → 物理空间
        ↓
for t in range(50): vec_env.step(env_chunk[:, t])       # :181-182 ← 我们在这里攒批
```

所以**残差修正的是基座动作**，而想象空间只负责回答"照这 50 个动作走下去，世界会变成什么样、势函数升还是降"。

### 0.4 学习信号的来路

```
成功/失败语义  ──只在离线训 V 时进入──►  V 的 ±1 终端奖励
                                            ↓
                                        Φ = V(ψ ⊕ p)
                                            ↓
想象回路内不做任何成败判定  ─────────►  reward = γΦ' − Φ
                                            ↓
                                    残差 TD3 的 critic
```

**想象回路里没有成功检测器、没有阈值、没有分类器、没有终止判定。**
这是 RISE 的明确设计选择，不是妥协（§3）。

### 0.5 吞吐与规模

| 量 | 值 |
|---|---|
| WM batch 维 | **1**（被 RL 侧写死，§9 修正二） |
| 每 chunk 的 D.infer 调用 | 1 次（≈2 s） |
| 每 chunk 的 kai0 serve 调用 | **1 次**（窗口缓存，ψ 与基座动作共用，§6.5） |
| 每 chunk 覆盖的 env_steps | 50 |
| 500k env_steps 的耗时 | ≈ 5.5 小时 |
| 产出的 transition 数 | 10 000 条 |

---

## 1. 事实核验：WM 的真实 I/O 契约

写任何代码前，这张表是唯一可信的接口定义（均已在代码中核对）。

### 1.1 输入 / 输出张量

| 张量 | 形状 | 说明 |
|---|---|---|
| `obs`（入） | `(3, 3, 4, 192, 256)` bf16, `[-1,1]` | B=1 时即 `(B*3, 3, 4, H, W)`；3 视角折进 batch 维；`t=4` 历史帧 |
| `act_tokens`（入） | `(1, 25, 30)` bf16, `[-1,1]` | 25 步 × 30 特征槽（从 16 维零填充） |
| `preds['video']`（出） | `(3, 3, 29, 192, 256)` `[-1,1]` | `29 = 4 历史 + 25 预测` |

- 调用入口：`RISE_Hi/policy_and_value/policy_online/rlinf/models/embodiment/modules/dynamics_model.py:173` `infer()`
- 形状硬闸：`dynamics_model.py:230`，`act_tokens.shape[1:] != (25, 30)` 直接 raise
- 解包：取 `video[:, :, 4:]` 为预测段

> `infer()` 的 docstring 写 `act_tokens shape=(bs,25,14)`，**与代码里的断言矛盾**；
> 以断言 `(25, 30)` 为准，docstring 是陈旧的。

### 1.2 机器人状态：两头都没有

**输入侧无状态。**
- `dynamics/dynamics_model/data/data_finetune.py:474`：`sample = dict(video=..., caption=..., action_tokens=...)`，无 `state` 键
- `runner/finetune_trainer.py:653`：状态分支被 `if getattr(self.args, "add_state", False)` 门控，而 `finetune.yaml` 里 `add_state: False`
- `infer()` 的 `normed_state` 参数存在，但只流向 `history_action_state`，后者**仅在 `if return_action:` 分支内被消费**（`action_encoder_control.py:710`），该分支在 RL 路径上恒为 False → **传了也不起作用**

**输出侧无状态。** `forward()` 只返回 `final_output['video']`，无 state head、无 reward head、无 done head。

**所以 proprio 必须由接入模块自己维护**（§6.3）。

> ⚠️ **原实现的空间不一致（我们必须修正）**：RISE 的 `next_state` 取自 `action_pred`
> （**归一化模型空间**，`openpi_action_model.py:791`），而喂 WM 的 `wm_act_tokens` 取自
> `actions_decoded`（**物理空间**）。混用会静默错位。
> **本模块统一在物理空间维护 proprio，只在打包 `act_tokens` 时才归一化。**

### 1.3 动作契约

- 表示：**绝对关节位置**，`action_type: absolute`, `action_space: joint`
- 维度：**显式钉死 16**（Piper 双臂）。上游在 14/16/30/32 之间漂移（§8 坑 7），必须加断言
- 归一化：min-max 到 `[-1,1]`，**统计量由本模块自带**，不依赖上游 config（§8 坑 1）
- 时间对齐：策略出 50 个动作 @30Hz → 每 2 个抽 1 → 25 个 token；WM 出 25 帧 @15Hz。
  **两者覆盖同一段 1.667 s 墙钟时间**

> 该推导**只对 30fps 的域成立**。paper 域成功集是 20fps（50 动作 = 2.5 s ≠ 1.667 s），
> 接该域前必须重推。**首选域定为 block**，正是因为它 30fps 原生 16 维、无格式转换史。

---

## 2. 硬约束与接入形态

### 2.1 硬约束

**WM 侧（`RISE_Hi/`）与 RL 侧（`resfit/rl_finetuning/` 既有文件）严格 0 行改动。**

唯一例外是 §7 的 V 训练改动（`hiql_value.py` / `train_hiql_value.py`）——那是独立的离线
管线步骤，不属于接入模块，用户已明确授权。

### 2.2 落地方式：launcher

在 `sys.modules` 预置假模块拦截 3 个符号，再用 `runpy` 以 `__main__` 跑原 trainer。
仓库先例：`run_td3_meta_only_wrapper.py`。

```
resfit/rl_finetuning/wm_bridge/          ← 全部新文件
├── __init__.py
├── launch_imagination.py    入口：预置假模块 + runpy
├── builder.py               参数解析 + 三个假符号的组装
├── contract.py              结构断言（零改动路线唯一的兜底）
├── imagination_env.py       ★ 核心：攒 50 步 → 点火 WM → PBRS reward
├── wm_driver.py             动作打包/帧解包/归一化
├── state_tracker.py         proprio 外推（物理空间）
├── scorers.py               Φ 打分器
├── base_bridge.py           kai0 shim + 窗口缓存
├── init_states.py           想象起点采样
├── lerobot_source.py        从 LeRobot 读原生帧
├── psi_fingerprint.py       kai0 权重指纹（同源校验）
├── wm_loader.py             装载 D
├── fake_eval.py             接管 checkpoint 存盘
└── s0_contract_check.py     S0 契约固化 + 动作可控性检查
```

### 2.3 三个注入点

| # | 目标符号 | 为什么必须注入 |
|---|---|---|
| 1 | `resfit.dexmg.environments.dexmg.create_vectorized_env` | env 工厂。函数内 lazy import（`train_chunk_residual.py:840`），预置假模块后连 robosuite 都不用装 |
| 2 | `resfit.rl_finetuning.utils.evaluate_dexmg.run_dexmg_evaluation` | `train_chunk_residual.py:1315` 是**全脚本唯一**的 `save_checkpoint`，且在 eval 块内。关掉 eval 等于整个 run 不存盘 |
| 3 | `resfit.lerobot.policies.pi05.load_pi05_base_policy` | kai0 adapter 只有 `select_action`（步级），replan 路需要 `get_action_chunk`（`chunk_env_wrapper.py:87` 优先用它）；且既有 schema 是 libero（单臂 8 维）/dexmg，实机是三相机 16 维 |

> ★ **绝不能 patch `build_base_policy`。** 它定义在 `train_chunk_residual.py` **自身**，
> 而 launcher 用 `runpy.run_module(..., run_name="__main__")` 执行该模块——runpy 会创建
> **新的模块对象**，对已 import 版本的 patch 作用不到 `__main__` 实例上。
> 必须往下钻一层，patch 它在 `:173` 处 lazy import 的 `load_pi05_base_policy`。
>
> 注入点 1、2 无此问题：它们都是函数内从**独立模块** lazy import 的。
>
> 实现细节：预置 `sys.modules['a.b.c']` 时，同时对父包 `a.b` `setattr(c, mod)`。

### 2.4 为什么接缝这么窄

`ChunkResidualEnvWrapper`（`chunk_env_wrapper.py:54`）对 `vec_env` 只用到：

```python
vec_env.action_space.shape[-1]   # :82
vec_env.num_envs                 # :83（getattr，缺省 1）
vec_env.reset(**kwargs)          # :145 → (obs_dict, info)
vec_env.step(action)             # :182 → (obs, reward, term, trunc, info)
vec_env.render()                 # :246（可 raise NotImplementedError）
vec_env.close()                  # :249
__getattr__ 委托到 vec_env       # :251（兜住其余未知属性）
```

`stage_purity_summary`（`:233`）定义在 wrapper 自身；我们的 info 不带 `stage_id`
→ `_stage_total_steps` 恒 0 → 返回 `"no stage steps"`，安全，无需处理。

---

## 3. 成功/失败判定：想象回路内不判定

### 3.1 官方 RISE 的三层结构

必须分清，否则容易误以为 "RISE 有成功检测器"：

| 层 | 是否判定成功 | 怎么做 |
|---|---|---|
| **想象回路内** | **完全不判定** | 无 done、无终止、无吸收态、无成功声明。定长 H 步，唯一停止条件是**递归 ≤2 次**的硬上限 |
| **Value 离线训练** | 间接 | episode 级 success/failure 标签只进 TD loss 的 `r_t` |
| **最终评测** | 人工 | 20 次 autonomous trials，10 分制子目标 rubric |

原文依据（Sec. III-C 末句）：

> "RISE **avoids explicitly simulating terminal states to obtain rewards**, yet produces
> chunk-wise advantage for proposed actions directly."

理由在 Intro：判定终局成功需要 WM 模拟完整任务执行，**"beyond the reliable horizon of most
generative world models"**。

**我们逐字沿用这一层设计**：`terminated` 恒 False，`truncated` 纯粹是递归上限，
reward 里没有任何终端奖励。审稿人质疑"你怎么在想象里判成功"时，答案是"我们和 RISE 一样
不判，因为那超出生成式 WM 的可靠视界"——这是引用得到的立场，不是辩解。

### 3.2 明确否掉的三个方案

| 方案 | 否掉理由 |
|---|---|
| progress-V 过阈值当成功 | 阈值是内生的、agent 可操纵 → TD3 会主动优化去顶阈值，正是论文 §4.4 机制(ii) 的 farming；且假终止污染 critic bootstrap |
| 另训二分类成功判别器 | 多一个会漂移的模型；想象帧上的判别器面临 OOD；**RISE 没有，会被问"为什么你需要而 RISE 不需要"** |
| 沿用真实轨迹剩余长度当终止 | 想象态一旦偏离起点轨迹，那个终止时刻就没有语义了 |

### 3.3 优势估计器的正确位置

你们的优势估计器（kai0 训，依赖人工 `meta/stage_annotations/`）**不进想象回路**。
进回路会同时破坏三件事：引入 farming 门、打断 telescoping、与 supp 附录"无额外 reward
model"冲突。

它的正当用途有两个（**互斥，只能选一个**）：

- **S1.5 的独立第三方标尺**：验证 `Φ(ψ̂_k)` 与 `Φ(ψ_k)` 的一致性。价值在于监督来源独立
  （它学人工 stage 标注，V 学自举 TD 目标）；用 V 验 V 是循环论证
- **给 rollout 逐集打成功/失败标签**：供 §7 的 V 训练用

★ 一旦用它打的标签去训 V，V 就继承了它的误差，再拿它验 V 又变回循环论证。
**用户已人工区分成功集与失败集，故打标不需要它 → 保留给 S1.5 当标尺。**

第三个用途与前两者都不冲突：**想象 rollout 的有效性闸**——判定轨迹跑到流形外就丢弃该段。
注意这是**丢样本**不是**给奖励**，不构成 agent 可优化的信号。

### 3.4 真实成功率只在实机测

20 次 autonomous trials + 子目标 rubric。**想象空间不产出任何被报告的指标**，
避免"用世界模型自己给自己打分"的质疑。

---

## 4. 截断即终止：为什么这是对的

代码事实：
- `train_chunk_residual.py:1200`：`done = terminated | truncated`
- `q_agent.py:432`：`discount = gamma * (1 - done)`

→ 截断会把 bootstrap 清零。这**恰好正确**，不是需要绕过的问题：

想象段是一个**语义完整的有限视界 episode**，其回报本身就是势函数增益的累积；
RISE 本身也**从不 bootstrap 超出想象视界**。所以 `done = terminated | truncated`
的合并与我们的语义完全一致，**零改动**。

**代价（写进 limitation）**：信用分配视界 = 2 × 1.667 s ≈ **3.3 s**。
超出 3.3 s 的长程信用完全由 V 的 progress 信号承担。这其实是正确的分工
（V 被训来编码长程进度，critic 只需管局部修正），但必须明说。

---

## 5. Reward：PBRS 端点差

### 5.1 公式

```
r_chunk = γ · Φ(ψ̂_末) − Φ(ψ_起)
```

其中 `Φ(ψ, p) = V(标准化(ψ ⊕ p))`。

### 5.2 为什么不用 RISE Eq.(2) 的均值

RISE 原文 Eq.(2)（逐字）：

```
A(o_t, a_t, ℓ) = ( (1/H) Σ_{k=1..H} V(ô_{t+k}, ℓ) ) − V(o_t, ℓ)
```

初稿选了它，理由是"逐字对齐 RISE，且就是势函数差分"。**后半句不成立**：
跨 chunk 的均值形式不 telescoping。

推翻它的是团队自己的实证（`project_resfit_potsubgoal_collapse_rootcause`）：
势函数不 telescoping 会产生 farming 伪奖励（实测 0 vs +0.13），正是 potsubgoal 塌方的根因。
沿用 Eq.(2) 等于重蹈已知覆辙。

**工程上还便宜 25 倍**：`prefix_feat` 没有纯特征端点，每个 ψ 都要跑一次**完整 pi05
扩散推理**。均值形式要给 25 个预测帧各打一次分 = 25 次 serve 调用；端点差只需起止两个 ψ，
而这两个**恰好都是基座调用的副产物**（§6.5 窗口缓存）→ 每 chunk 仅 1 次。

**代价**：偏离 RISE Eq.(2)，须在论文中说明理由（telescoping / policy-invariance）。

### 5.3 ★ `truncated` 时绝不可把 Φ 置零

仓库既有的 `potential_shaping`（`chunk_env_wrapper.py:32`）约定 done 时 Φ(s')=0——
那是为**真终止态**设计的。想象段的 `truncated` 是人为视界切断，不是真终止。照抄会毁掉学习信号：

```
不置零： r₀ + γr₁ = γΦ₁ − Φ₀ + γ²Φ₂ − γΦ₁ = γ²Φ₂ − Φ₀    ✓ 干净相消
置 零 ： r₀ + γr₁ = γΦ₁ − Φ₀ − γΦ₁       = −Φ₀           ✗ 常数，与策略无关
```

置零后回报是常数，残差学不到任何东西。**这条与仓库既有约定相反，是最容易被"好心对齐"
改坏的地方**，必须加断言 + 回归测试锁死。

### 5.4 ★ 必须关掉 wrapper 自己的 shaping

`chunk_env_wrapper.py:202-213` 会在我们返回的 reward 之上再叠一层：

```python
if self.potential is None:
    total_reward += shaping_reward(...)        # mode="none" 时返回 0.0（:35-36）
else:
    total_reward += potential_shaping(...)     # potential 非 None 时走这条
```

所以运行时**必须** `--reward_shaping none` 且**不传** `--potential_source`，
否则就是 double-shaping。`contract.py` 须硬断言。

---

## 6. 组件设计

### 6.1 `imagination_env.py` —— 核心

```python
class ImaginationVecEnv:
    def __init__(self, *, wm, base, scorer, sampler, normalizer,
                 gamma=0.995, max_segments=2, num_denois_steps=10):
        self.num_envs = 1
        self.action_space = _ActionSpace(16)
        ...

    def reset(self, **kwargs):
        st = self.sampler.sample()
        self._window = st.obs_window            # (3,3,4,192,256) [-1,1]
        self._caption = st.caption
        self._tracker = ProprioTracker(st.proprio)
        self._act_buf, self._seg_step = [], 0
        self._token += 1
        self.base.reset()
        obs = self._obs_from_window()
        _, psi = self.base.query(obs)
        self._phi_prev = self.scorer.phi(psi, self._tracker.proprio)
        self._obs_cache = obs
        return obs, {}

    def step(self, action):
        a = <转 np.float32, reshape(-1)[:16], 断言形状>
        self._act_buf.append(a)
        if len(self._act_buf) < 50:
            return self._obs_cache, torch.zeros(1), \
                   torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}

        actions = np.stack(self._act_buf); self._act_buf = []
        act_tokens = pack_act_tokens(actions, self.normalizer)
        out = self.wm.infer(obs=torch.from_numpy(self._window).to(torch.bfloat16),
                            act_tokens=act_tokens, prompt=self._caption,
                            num_denois_steps=self.num_denois_steps)
        <坑6：infer 后显式 self.wm.transformer.eval()>

        video = out["video"];  video = video[0] if video.ndim == 6 else video
        pred = split_predicted_frames(video.float())        # (3,3,25,192,256)

        self._tracker.advance(actions)                      # proprio ← actions[-1]
        self._window = build_obs_window(pred, range(21, 25)).numpy()   # 末 4 帧
        self._token += 1

        obs = self._obs_from_window()
        _, psi = self.base.query(obs)                       # 与下游取动作共用缓存
        phi_next = self.scorer.phi(psi, self._tracker.proprio)

        reward = self.gamma * phi_next - self._phi_prev     # ★ 不因 truncated 置零
        self._phi_prev = phi_next

        self._seg_step += 1
        self._obs_cache = obs
        return (obs, torch.tensor([reward]),
                torch.tensor([False]), torch.tensor([self._seg_step >= self.max_segments]),
                {})

    def render(self): raise NotImplementedError("想象空间不支持 render")
    def close(self):  return None
```

### 6.2 观测字典契约

| key | 来源 | 形状 / 类型 |
|---|---|---|
| `image_keys`（取自基座 `config.image_features`，`train_chunk_residual.py:796`） | WM 预测帧 192×256 → resize | `[1,3,84,84]` float **[0,1]** |
| `observation.state` | state_tracker 外推，**raw 物理空间** | `[1,16]` |
| `observation.base_action` | wrapper `_augment` 自加（`:113`） | — |
| `observation.stage_id` | wrapper 自加，恒 0 | — |
| `_wm_native_frames` | 原生 192×256 三视角 | 旁路键，供注入点 3 |
| `_wm_window_token` | 单调递增 int | 旁路键，base_bridge 的缓存键 |

- **旁路键安全**：buffer 只存 `set(image_keys) | set(lowdim_keys)`（`:95`），多余键被忽略
- **`observation.state` 必须给 raw**，wrapper 自己标准化（`:115`）
- **图像给 float[0,1]**：`to_uint8`（`:81-85`）按 `clamp(0,1)*255` 处理，而 WM 出 `[-1,1]`，
  由 `wm_driver` 负责转换

**★ 84×84 是强制的，不是取舍**：`min_vit.py:38` `PatchEmbed2.num_patch = 81` 是写死常量，
`:103` `pos_embed = nn.Parameter(torch.zeros(1, 81, embed_dim))`。喂其他尺寸会在加位置编码时
形状不匹配当场崩。`VitEncoder`（`encoder.py:13`）虽收 `obs_shape` 但并不传给 `MinVit`。

### 6.3 `state_tracker.py`

```python
class ProprioTracker:
    def __init__(self, init_proprio, action_dim=16): ...
    @property
    def proprio(self) -> np.ndarray: ...          # 返回副本
    def advance(self, action_chunk_physical):      # (L,16) → 取 [-1]
        self._p = arr[-1].copy(); return self.proprio
```

成立前提是 `action_type: absolute` + `action_space: joint`。
**代价是假设零跟踪误差**——接触、柔顺、碰撞导致的实际偏差全部被忽略。
这是 WM 无状态头的直接后果，不是我们的选择，写进 limitation。

### 6.4 `wm_driver.py`

常量：`ACTION_DIM=16`、`CHUNK_LENGTH=50`、`WM_TOKEN_STEPS=25`、`WM_TOKEN_SLOTS=30`、
`WM_ACTION_INTERVAL=2`、`N_PREVIOUS=4`、`FRAME_H,W=192,256`、`AGENT_IMG=84`、
`CAMERA_KEYS=("observation.images.top_head","...hand_left","...hand_right")`（顺序即视角顺序，不可改）

```python
class ActionNormalizer:                 # min-max → [-1,1]，逐维；统计量本模块自持
    def normalize(a):   return 2*(a - min)/(max - min) - 1
    def denormalize(a): return (a + 1)*0.5*(max - min) + min

def pack_act_tokens(actions_physical, normalizer) -> Tensor:   # (50,16) → (1,25,30) bf16
    sampled = actions[::2]              # 25 步
    tokens = zeros((25, 30)); tokens[:, :16] = normalizer.normalize(sampled)
    return from_numpy(tokens).unsqueeze(0).to(bfloat16)

def split_predicted_frames(video):      # (V,C,29,H,W) → (V,C,25,H,W)，切掉前 4 帧
def frames_to_obs_images(frames, t):    # → {cam: [1,3,84,84]}，[-1,1]→[0,1]→resize
def build_obs_window(frames, idx):      # → (V,C,4,H,W)
```

### 6.5 `base_bridge.py` —— kai0 shim + 窗口缓存

```python
class Kai0ImaginationBase:
    def query(self, raw_obs) -> (actions (50,16), psi):
        token = raw_obs["_wm_window_token"]
        if token == self._cache_token: return self._cache      # ★ 缓存命中
        result = self.client.infer(self._serve_obs(raw_obs))
        psi = result.get("prefix_feat")
        if psi is None: raise RuntimeError("kai0 serve 未透出 prefix_feat")
        ...
    def get_action_chunk(self, raw_obs, chunk_length) -> Tensor [1,50,16]:
        assert chunk_length == 50
        return from_numpy(self.query(raw_obs)[0]).unsqueeze(0)
    def reset(self): pass
    config.image_features = {cam: None for cam in CAMERA_KEYS}
```

**★ 窗口缓存是 1 次 vs 2 次 serve 调用的关键。** 同一个窗口会被请求两次：
① `ImaginationVecEnv` 要 ψ 算 reward；② wrapper 在 `chunk_env_wrapper.py:219` 调
`_base_chunk_flat` 要基座动作。两次针对同一窗口，缓存后每 chunk 仅 1 次调用，
**ψ 完全是基座调用的副产物**。

**★ serve obs schema（07-21 实测锁定）**：`_serve_obs` 复用 `build_teleavatar_serve_obs`，
产出 block kai0 的**嵌套** schema `{"state", "images":{裸cam键: CHW uint8@224}, "prompt"}`
（对齐 `piper_deploy.py`，建 ψ 缓存时验过、S1 再验）。初稿设想的扁平 `observation/*` schema
是猜的、已废弃。值域：`_wm_native_frames` 是 [-1,1]（imagination_env 统一），先转 [0,1]
再进 `_to_hwc_uint8`（它按 [0,1] 处理 float）。

**★ 分辨率**：kai0 客户端侧 `resize_with_pad` 到 224（与部署 `piper_deploy.py` 同款）。
想象里 WM 只出 192×256（拿不到 480×848 原生），故想象与实机部署间有分辨率落差——这是
ψ̂ OOD 的一个具体机制，写进 limitation（§11）。

### 6.6 `scorers.py`

```python
class Scorer(Protocol):
    def phi(self, psi, proprio) -> float: ...

class Kai0HiqlScorer:                    # 唯一可报告的实现
    def phi(self, psi, proprio):
        state = concat([psi, proprio])
        assert state.shape[0] == self.state_dim     # == value.pt 的 state_dim
        z = (state - self.mean) / self.std
        with torch.no_grad(): return float(self.model(from_numpy(z)).reshape(-1)[0])

    @classmethod
    def from_value_ckpt(cls, path, device="cpu"):
        model, info = load_value(path, map_location=device)   # hiql_value.py:99
        return cls(model, mean=info["mean"], std=info["std"],
                   expected_psi_anchor=(info.get("pi0_feat_signature") or {}).get("serve_ckpt_id"))

class DummyScorer:  phi(...) -> 0.0      # 仅单元测试；真实训练须 --allow_dummy_scorer
```

**`RiseFrameScorer` 只能当调试工具，不可作为可报告的臂**：supp 附录已写死
"无 stage 特权、无额外 reward model / frame scorer"，而它正是一个额外的 frame scorer。

安全线：shaping 只用**单状态 V**；gc value 的 `V(s,z)` **绝不进 reward**。

### 6.7 `init_states.py`

```python
BLOCK_CAPTION = "build block"
BLOCK_DATASETS = ()                      # ★ 无默认路径

class InitStateSampler:
    def sample(self) -> InitState:       # obs_window (3,3,4,192,256) + proprio (16,) + caption
        src = random.choice(self.sources)
        end = randint(N_PREVIOUS - 1, src.n_frames)     # 末帧索引 ≥ 3，否则窗口越界
        window = stack([src.read(t) for t in range(end-3, end+1)], axis=2)
```

**★ caption 钉死为常量，禁止从数据集读。** WM 微调时 block 域 caption 已统一为
`"build block"`（`RISE_Hi/temp/三域数据构建方案.md` §3.4），而数据集里可能是
`"build blocks"`（复数）。WM 靠 T5 编 caption 做条件，喂错一个字符即偏离训练分布。

**★ 无默认数据路径。** 用户实际使用的数据不在本机；本机同名数据集**不是**用户的数据。
路径由 `--init_state_dataset`（可重复）显式传入，缺参硬失败。

**★ 成功集与失败集都要采。** 残差的主战场是"基座跑偏"的状态；只用专家集会让它从没见过
需要救场的局面。失败集是基座实跑轨迹，与部署分布天然对齐。

### 6.8 `fake_eval.py`

```python
def make_imagination_evaluator(output_dir, config):
    def run_dexmg_evaluation(*, env=None, agent=None, num_episodes=None,
                             device=None, global_step=0, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        save_checkpoint(agent, join(output_dir, "imagination_last.pt"),
                        global_step=global_step, config=config, success_rate=0.0)
        return {"eval/success_rate": 0.0}
    return run_dexmg_evaluation
```

**★ 返回恒定 0.0，不返回递增哨兵去骗过 `:1312` 的 `sr > best_sr` 比较**——那是把存盘
寄托在一个假指标上。存盘由本模块显式负责，trainer 的 best 逻辑自然失效。

### 6.9 `contract.py` —— 8 条断言

零改动路线唯一的兜底。monkeypatch 最大的风险是上游改了符号而我们**静默失配**。

1. `create_vectorized_env` 存在且签名含 `num_envs` / `device`
2. `run_dexmg_evaluation` 存在
3. **`ChunkResidualEnvWrapper.step` 仍是逐时间步循环**（攒批机制的前提，用
   `inspect.getsource` 检 `"for t in range(self.chunk_length)"` 与
   `"self.vec_env.step(env_chunk[:, t])"`）
4. `PatchEmbed2.num_patch == 81`
5. `--reward_shaping none` 且未传 `--potential_source`（防 double-shaping）
6. `--chunk_length 50`
7. **ψ 同源**：`value.pt` 记的 sha == kai0 本地 ckpt 指纹
8. serve 确实透出 `prefix_feat`

> 第 7 条的处理对齐仓库先例 `assert_act_base_samesource`（`act_feature.py:149-153`）：
> **指纹缺失 = 无法验证，不等于已知异源** → warn 并要求先过 S1.5，**不 raise**；
> 只有"指纹都在且不等"才 raise。
>
> 指纹须对**本地 checkpoint 目录**算（`psi_fingerprint.kai0_ckpt_fingerprint`），
> 不能问运行中的 serve——kai0 在 websocket 后面，客户端拿不到 `state_dict()`。

---

## 7. V 训练侧的必要改动

> 改 `hiql_value.py` / `train_hiql_value.py`。**不属于接入模块**，用户已授权。
> **所有既有仿真 run 的 V 必须逐位不变**，由回归测试证明。

### 7.1 现有实现不区分成功与失败

```python
# hiql_value.py:53-56  build_transitions
d = np.zeros(T - 1); d[-1] = 1.0        # 每条 demo 末尾都置 1
# hiql_value.py:139    train_value
reward = done                            # r = 1 if done else 0
# hiql_value.py:23     discounted_target
y = reward + gamma * (1 - done) * next_v
```

即 **V ≈ γ^(距轨迹终点步数)**，纯"到轨迹末尾的进度"势函数。
`train_hiql_value.py` 全文无任何成功/失败过滤。

**★ 反直觉的边界**：在**纯专家数据**上这是**正确的**——轨迹终点就是成功。既有仿真实验
（three_piece / pouring / lifttray 那批）全是专家 demo，所以这个问题一直没暴露。
**只有喂混合质量数据才中招**：失败轨迹的终止态同样被赋予 V≈1，PBRS 会奖励残差
**驱向失败终止态**。

### 7.2 修法：数据集层面的 ±1 终端奖励

用户的数据**成功集与失败集完全分开**（已人工区分），故标签落在数据集层面。

```python
# hiql_value.py 新增（既有 build_transitions 一行不动）
def build_transitions_with_rewards(state_seqs, success_flags):
    """→ (s, s_next, done, reward)。成功轨迹末 +1、失败轨迹末 −1、中间 0。"""
    assert len(success_flags) == len(state_seqs)
    for seq, ok in zip(state_seqs, success_flags):
        if T < 2: continue                          # ★ 与 build_transitions 逐位一致
        d = zeros(T-1); d[-1] = 1.0
        r = zeros(T-1); r[-1] = 1.0 if ok else -1.0
    ...

# hiql_value.py 改 train_value（其余一行不动）
def train_value(s, s_next, done, *, reward=None, ...):
    reward = done if reward is None else reward     # 默认逐位等价
```

CLI（遵循仓库"新旗标默认等价、显式翻转"惯例）：

- `--terminal_reward_mode {legacy, success_signed}`，**默认 `legacy`**
- `--success_dataset` / `--failure_dataset`（可重复），`success_signed` 下各至少一个

**★ 禁止按目录名推断标签。** RISE 的约定（`custom_lerobot_dataset.py:508`：
`'infer' in repo_id or 'fail' in repo_id`）在本项目数据上会**静默失效**——`rollout_*`
这类名字两个关键词都不含，会被全部误判为成功，方向正好反。

**★ 一次性构造 transitions 与 rewards。** `build_transitions` 跳过 T<2 的 demo；
若 `success_flags` 单独构造而不跟着跳，标签会整体错位一格，**且不报错**。

### 7.3 ψ 同源

V 训练时编 ψ 的 kai0 权重，必须与在线 serve 的那份**逐位相同**。

**"用了冻结的 kai0" ≠ "用了这一份冻结的 kai0"**。冻结只说明训练中没更新它，
不说明它是哪个 checkpoint。训 V 用什么数据不决定该配哪个 encoder——决定的是当时冻的是哪份权重。

核实路径：① `load_value(path)` 的 `info["pi0_feat_signature"]["serve_ckpt_id"]` 与在线 serve 标签比对（已落地，S1 验过 dim2064/anchor 对）；
② 没记则看训 V 时的 ψ 缓存还在不在，用现在的 kai0 重编几帧比对数值；③ 都没有则只能靠
S1.5 带着风险跑，或重训 V（这次记上指纹）。

**训 V 时 expert 与 rollout 必须一起喂**：只用 rollout 会是 2:117 的极端失衡，
V 退化为"一切皆失败"。合并后约 203 : 117，比例健康，无需额外重加权。

---

## 8. 已知坑清单（RISE_Hi 侧，在 `wm_bridge` 里绕过，不改上游）

均已在代码中核实：

1. **`rl_release.yaml:197` 指向的 `configs/ltx_model/infer.yaml` 没有 `min_val`/`max_val` 键**
   → `openpi_action_model.py:576` 会 `AttributeError`。只有 `configs/ltx_model/box_1.yaml:71,73`
   定义了。**我们模块自带归一化统计量，不依赖它。**
2. **`wm_utils.py:193` 设备不匹配**：用 CPU `FloatTensor` 减可能在 CUDA 上的 `actions`。
   **我们自己实现打包，不调这个函数。**
3. **`wm_utils.py:129` 少了函数调用**：`forward_inputs['base_0_rgb'].view` 赋的是**绑定方法**
   而非张量。只在 `use_his_obs=False` 分支，被 `rl_release.yaml` 的 `use_his_obs: True` 掩盖。
4. **`dynamics_model.py:227`**：`torch.rand(1, 25, 30, self.device, ...)` 把 device 当成了
   位置尺寸参数 → `act_tokens=None` 的兜底路径是坏的。**我们永远显式传 `act_tokens`。**
5. **`dynamics_model.py:44` 相机名不一致**：`front_color/left_color/right_color` vs 数据集的
   `top_head/hand_left/hand_right`。只影响 `image_root` 路径。**接数据当天须核对真实键名。**
6. **`custom_pipeline.py:997`**：`pipe.infer` 退出时调 `self.transformer.train()`，即使在
   `@torch.no_grad()` 下也会把模块留在 train 模式。**每次 infer 后我们显式 `.eval()`。**
7. **动作维度漂移**：norm 常量 14 维 / `norm.py` 改成 16 / 当前数据 16 / 配置
   `action_in_channels: 14`（死参数）/ `act_in: Linear(16, 4096)` / RL 配置 `action_dim: 32` /
   reward model state 32 维零。**显式钉死 16，加断言。**
8. **`guidance_scale=1.0`** → `do_classifier_free_guidance` 为 False
   （`custom_pipeline.py:522` 要求 `> 1.0`），CFG 分支在 RL 路径上是死代码。
9. **`n_chunk>1` 不是隐空间递归**：它把像素解码后重新编码（`custom_pipeline.py:960-985`），
   每段一次完整 VAE 往返。我们的 ≤2 段递归走同一路径，成本要计入。
10. **文档矛盾**：`temp/三域数据构建方案.md` §9 称 `train_mode: 'video_only'` 会冻结动作条件。
    **这是错的**，`命令文档.md` §9.2 已更正。`finetune_trainer.py:427` 只冻结名字含 `'action_'`
    的参数，而动作条件模块叫 `act_vit_in`/`act_in`，**仍可训练**且 `forward()` 无条件应用它们。

> 第 10 条对我们是好消息：**shipped 配置下 WM 确实是动作条件的**。但条件方式是
> "动作 token 加到 T5 文本嵌入上"（`action_encoder_control.py:646-659`），这是相对**弱**的
> 耦合机制——**动作敏感度是个经验问题，S0 的可控性检查必须做，不能假设。**

---

## 9. 本版对初稿的 4 处修正

### 修正一：`step()` 收到的是单步动作，不是 action chunk

初稿 §4.1 假设 `step(action_chunk)` 一次收到整段动作直接进 WM。**错。**

`chunk_env_wrapper.py:181-182` 是逐时间步循环，`chunk_length=50` 时 `vec_env.step()`
被调用 **50 次、每次 1 个 16 维动作**。

**解法：攒 50 次再点火。** 前 49 次返回缓存 obs / reward=0 / False。
中间步在语义上确实是空转，逐行证明：

| wrapper 行 | 中间步的命运 |
|---|---|
| `:182` `raw_obs` 每轮覆盖 | 只有最后一轮进 `:219` / `:222` |
| `:186` `last_info` 每轮覆盖 | 同上 |
| `:183` `total_reward` 累加 | 前 49 步加 0，不影响和 |
| `:184-185` `term/trunc` OR 累积 | 前 49 步 False，不影响 |
| `:198-199` 提前 break | 只在 term/trunc 为真时触发，我们只在第 50 次置位 |

### 修正二：批量想象做不成，B 恒为 1

初稿 §4.1 称 `num_envs = B`、B 从 8 起步、B=32 时约 16 transition/s。**做不到。**

- `train_chunk_residual.py:841` 训练 env 写死 `num_envs=1`
- 更硬的约束：入 buffer 时 `obs[k][0]` / `done[0]` / `reward[0]` / `combined_action[0]`
  （`:96-105`）——**只存 env 0**，并行流会被直接丢弃

放开需改这些索引，违反零改动约束。吞吐重算：1 次 WM 调用 ≈ 2 s，覆盖 50 个 env_step
（`:1228` `env_steps += args.chunk_length`）→ 500k env_steps ≈ **5.5 小时**，可接受。

### 修正三：残差粒度不是被 WM 强制的

初稿 §4.4 断言"残差决策粒度从 1 步变 50 步……是 WM 接口强制的"。
**因修正一的攒批机制，该断言不成立**——`chunk_length=1` 同样可接 WM。

两条路都通。本设计选 `chunk_length=50` + `replan`，理由是 **MDP 精确性**：
`cl=1` 下 WM 要集齐 50 个动作才出帧，中间 49 步只能发陈帧，等于在 MDP 里系统性地
告诉 agent"你还在原地"而世界已经动了。`cl=50` 下观测、动作、奖励、下一观测一一对应。

**代价**：残差从"每步微调"变为"每段偏置"，与仿真主表（`cl=1`）**不可直接比较，论文须单列**。

### 修正四：reward 用 PBRS 端点差，不用 RISE Eq.(2) 均值

见 §5.2。

---

## 10. 分阶段落地（无实机 eval 阶段）

**本方案不含实机 eval 步骤。** 结果口径 = 想象里 500k 长训 + 每 50k 步存权重并 rollout 10 集、
用优势估计器给每集打**进度势 proxy 值**（只存原始值，不判成败；阈值离线自定，见 §6.7）。

| 阶段 | 目标 | 验收 | 前置 | 状态 |
|---|---|---|---|---|
| **S0** | 契约固化：真实起点 → `infer()` → 看预测帧 | 肉眼合理；**动作可控性**：同起点喂不同动作画面须显著分化 | D | ✅ **07-21 通过**（像素差 0.13/0.10/0.18，>>1e-3 约 100 倍）|
| **S1** | `ImaginationVecEnv` 端到端 reset→step×50→step×50 | 时序/形状/proprio 正确；reward 有限；2 段截断 | D+V+kai0 | ✅ **07-21 通过**（Φ₀0.28、1serve/chunk、reward 有限、截断对）|
| **S1.5** | ψ 一致性验收 | `Φ(ψ̂_k)`（想象帧）vs `Φ(ψ_k)`（真实帧）随 k 一致性；优势估计器当独立标尺 | D+V+kai0 | 待跑 |
| **S2** | 残差 TD3 **500k 长训 + 50k/10集优势估计器 proxy 监控** | 训练不发散；critic loss 收敛；残差不撞 `action_scale` 上限；`imagined_adv_eval.jsonl` 逐 eval 点累积 | D+V+kai0 | 待跑 |

**结果产出（S2 的 jsonl）**：用户离线画 ① 收敛曲线（adv 均值±方差 vs step）② proxy 成功率曲线
（过 θ 比例 vs step，θ 用 block_success/fail 真实末态标定、看 SHORE-RL 结果前固定）。
★ 这些是 **imagined proxy**，图/caption 须标 `imagined proxy, not real-robot success`（§6.7）。

**★ S0 不能跳过，且不通过就全线停止**（已通过）。依据见 §8 第 10 条：WM 动作条件是"动作 token
加到 T5 文本嵌入上"，耦合弱；若同起点喂不同动作画面几乎不变，残差改变不了想象未来，前提不成立。
判据：三个平均像素差都须显著 > `1e-3`。

---

## 11. Limitation（写进论文）

1. **零跟踪误差假设**：proprio 由绝对关节动作外推，接触/柔顺/碰撞偏差被忽略（§6.3）
2. **信用分配视界 3.3 s**：想象段 2 chunk 封闭，超出部分完全由 V 的 progress 信号承担（§4）
3. **残差粒度 50 步**：与仿真主表不可直接比较，须单列（§9 修正三）
4. **想象空间不产出被报告指标**：避免"用世界模型自己给自己打分"（§3.4）
5. **B=1**：零改动约束的成本，WM 算力未充分利用（§9 修正二）
6. **偏离 RISE Eq.(2)**：reward 用 PBRS 端点差而非 chunk 均值，须说明理由（§5.2）
7. **ψ̂ 的 OOD 风险**：V 在真实帧的 ψ 上训，想象里喂预测帧的 ψ̂。S1.5 的一致性曲线是唯一的
   量化验收；若随 k 迅速发散，说明 D 的预测质量不足以支撑 Φ，整条路线需重新评估

---

## 12. 运行方式

```bash
python -m resfit.rl_finetuning.wm_bridge.launch_imagination \
    --wm_ckpt <D 目录> \
    --value_ckpt <value.pt> \
    --kai0_ckpt <kai0 权重目录> \
    --action_norm_json <block 域动作 min/max 统计量> \
    --init_state_dataset <成功集> --init_state_dataset <失败集> \
    --base_policy_type pi05 --pi0_host <host> --pi0_port <port> \
    --pi0_action_dim 16 \
    --chunk_length 50 --base_action_mode replan \
    --reward_shaping none \
    <其余原有 trainer 参数>
```

**归一化统计量必须实算**：默认的 ±1 只是让单元测试跑通的占位，真实训练前不替换的话
`act_tokens` 会全部饱和在 ±1。

---

## 13. 接数据当天必须核对的三项

核对结果只改 `wm_bridge/` 内的文件，**不改上游**：

1. **`DynamicsModel` 构造签名**（`wm_loader.py`）—— 见 `dynamics_model.py` 的 `__init__`
2. **数据集真实相机键名**（`lerobot_source.py` 的 camera map）—— §8 坑 5，
   `dynamics_model.py:44` 与数据集用的是两套命名
3. **kai0 serve 的 obs schema**（`base_bridge._serve_obs`）—— 实机三相机 16 维，
   与既有 libero（单臂 8 维）/ dexmg schema 都不同

**今天下午需从另一台服务器传入三样**：D 权重、block 数据集、kai0 权重（专家版，
且须与训 V 时冻结的那份同源）。
