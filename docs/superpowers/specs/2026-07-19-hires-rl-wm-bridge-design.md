# HiRes-RL ↔ RISE Dynamics Model 接入模块 · 设计

日期：2026-07-19
前置：`temp/hires_rl_wm_bridge_design.md`（同日凌晨草案）、`temp/rise_integration_design.md` §14–§16
范围：只写接入层。D 的微调、V 的训练、三结构实验规划均不在本文。

## 0. 本文与凌晨草案的关系

`temp/hires_rl_wm_bridge_design.md` 是接入层的第一版草案。本文在其基础上，经过对 RL 侧代码的
逐行核实，**修正了草案的 3 处事实错误**，并敲定了草案 §7 遗留的待确认项。

草案中关于 WM 侧 I/O 契约（§1）、成败判定（§2）、RISE_Hi 坑清单（§6）的结论**经抽查属实，继续有效**，
本文不重复，只引用。

---

## 1. 硬约束（用户 2026-07-19 拍板）

**WM 侧与 RL 侧代码严格 0 行改动。** 接入模块是纯新增的独立包，两侧仓库保持干净。

这条约束否掉了草案 §3.2 推荐的"一条约 5 行的 `elif` 缝"。落地方式改为 launcher：
启动时在 `sys.modules` 预置假模块拦截若干符号，再用 `runpy` 以 `__main__` 方式运行原 trainer。
仓库已有同款先例：`run_td3_meta_only_wrapper.py`。

---

## 2. 对草案的 3 处修正

### 修正一：`step()` 收到的是单步动作，不是 action chunk

草案 §4.1 假设 `ImaginationVecEnv.step(action_chunk)` 一次收到整段动作、直接进 WM。**这是错的。**

`chunk_env_wrapper.py:181-182` 是逐时间步循环：

```python
for t in range(self.chunk_length):
    raw_obs, reward, term, trunc, info = self.vec_env.step(env_chunk[:, t])
```

`chunk_length=50` 时 `vec_env.step()` 被调用 **50 次，每次 1 个 16 维动作**。

**解法：`ImaginationVecEnv` 内部攒动作，第 50 次调用才点火 WM。**
前 49 次返回缓存 obs、`reward=0`、`term=trunc=False`；第 50 次打包 25 token → `infer()` → 打分 → 返回真 obs 与 advantage。

这不是将就，中间步在语义上确实是空转，已逐行确认：

| wrapper 行 | 中间步的命运 |
|---|---|
| `:182` `raw_obs` 每轮覆盖 | 只有最后一轮进 `:219` / `:222` |
| `:186` `last_info` 每轮覆盖 | 同上 |
| `:183` `total_reward` 累加 | 前 49 步加 0，不影响和 |
| `:184-185` `term/trunc` OR 累积 | 前 49 步 False，不影响 |
| `:198-199` 提前 break | 只在 term/trunc 为真时触发，我们只在第 50 次置位 |

### 修正二：批量想象做不成，B 恒为 1

草案 §4.1 称"`num_envs = B` 直接给 WM 当 batch 维，B 从 8 起步，B=32 时约 16 transition/s"。**做不到。**

- `train_chunk_residual.py:841` 训练 env 写死 `num_envs=1`
- 更硬的约束：入 buffer 时 `obs[k][0]` / `done[0]` / `reward[0]` / `combined_action[0]`
  （`train_chunk_residual.py:96-105`）——**只存 env 0**，并行流会被直接丢弃

放开需改这些索引，即改 RL 代码，违反 §1。**故 B=1，WM 每次 `infer` 的 batch 维为 1。**

吞吐重算：1 次 WM 调用 ≈ 2s，覆盖 50 个 env_step（`:1228` `env_steps += args.chunk_length`）。
500k env_steps = 1 万 chunk ≈ **5.5 小时**。用户已确认可接受。
代价：H100 上以 batch=1 跑 WM，GPU 利用率低。这是零改动约束的直接成本。

### 修正三：残差粒度不是被 WM 强制的

草案 §4.4 断言"残差决策粒度从 1 步变 50 步……这不是可调参数，是 WM 接口强制的"。
**因修正一的攒批机制，该断言不再成立**——`chunk_length=1` 同样可接 WM。

两条路都通，本设计选 `chunk_length=50` + `replan`，理由见 §4。

### 修正四：reward 用 PBRS 端点差，不用 RISE Eq.(2) 均值

草案 §2.2 选定 `A = (1/H)Σ_k V(ô_{t+k}) − V(o_t)`，理由是逐字对齐 RISE、且"就是势函数差分"。
**后半句不成立**：跨 chunk 的均值形式不 telescoping。

推翻它的是团队自己的实证——`project_resfit_potsubgoal_collapse_rootcause`：势函数不 telescoping
会产生 farming 伪奖励（实测 0 vs +0.13），这正是 potsubgoal 塌方的根因。沿用 Eq.(2) 等于
重蹈已知覆辙。

改用 PBRS 端点差：`r_chunk = γ·Φ(ψ̂_末) − Φ(ψ_起)`。同时它与 supp 附录已写死的
"想象中的奖励 = 自导出单状态 V(s) 势函数"逐字一致，且开销低 25 倍（§6.5）。

**代价**：偏离 RISE Eq.(2)，须在论文中说明理由（telescoping / policy-invariance）。

---

## 3. 本轮敲定的决策

| # | 决策 | 取值 | 依据 |
|---|---|---|---|
| D1 | 接入形态 | 严格零改动 launcher | 用户硬约束 §1 |
| D2 | 首选域 | **block** | 30fps 原生 16 维、无格式转换史；paper 成功集 20fps 会打乱草案 §1.3 的时间对齐推导；cup 数据不在本机 |
| D3 | 起点分布 | expert 201 ∪ rollout 119 **混采** | 残差主战场是"基座跑偏"的状态；rollout 集是基座实跑轨迹，与部署分布天然对齐 |
| D4 | 图像分辨率 | **强制 84×84** | 非取舍，见 §5.3 |
| D5 | chunk 粒度 | `chunk_length=50` + `replan` | MDP 精确性，见 §4 |
| D6 | 可报告的 Scorer | **`Kai0HiqlScorer`（自导出 Φ）唯一** | `RiseFrameScorer` 与 supp 附录冲突，见下 |
| D7 | reward 形式 | **PBRS 端点差** `γΦ(ψ̂_末) − Φ(ψ_起)` | §2 修正四 |
| D8 | V 的终端奖励 | **成功末端 +1 / 失败末端 −1**（数据集层面打标） | §12 —— 现有实现不区分成败，必须修 |

**★ D2 的理由修正（2026-07-19）**：最初选 block 的理由之一是"本机数据齐全"，
**该理由作废** —— 用户实际使用的数据不在本机。block 仍然是首选，但依据只剩两条：
30fps 原生 16 维、无格式转换史（paper 域 20fps 会打乱 §1.3 的时间对齐推导）。
连带后果：`init_states.BLOCK_DATASETS` **不可写死路径**，须参数化传入。

**关于 `RiseFrameScorer`**：supp 附录（`paper/aaai2026-unified-supp.tex`，2026-07-19）已写死
"想象中的奖励 = 自导出单状态 V(s) 势函数，**无 stage 特权、无额外 reward model / frame scorer**"。
而 hybrid 臂恰恰就是一个额外的 frame scorer。故它**只能作为调试工具**（S1 阶段验证想象闭环），
**不可作为可报告的臂**，否则附录的协议描述即为不实。

**前提（用户 2026-07-19 确认）**：V 用离线数据训练，视为已就绪。故 S2 / S3 不再阻塞。
`DummyScorer` 降级为测试桩（§8），不再是首版实现。

---

## 4. chunk 粒度：为什么选 50 + replan

`chunk_length=1` + `queue` 的诱惑在于与仿真主实验的金标准配置逐字一致，论文可直接并排。
但它有一个结构性代价：

WM 必须集齐 50 个动作才能出帧，所以中间 49 步的图像只能发陈帧。这等于在 MDP 里
系统性地告诉 agent"你还在原地"，而世界已经动了。proprio 可以逐步外推不受影响，图像不能。

`chunk_length=50` + `replan` 下，观测、动作、奖励、下一观测一一对应，无虚构中间态：

```
obs ──► 残差出 50 个动作 ──► WM 一次 ──► 25 帧 ──► advantage
 ▲                                                    │
 └──────────── next_obs ◄── 预测段末 4 帧 ◄───────────┘
```

代价（写进 limitation）：残差从"每步微调"变为"每段偏置"，表达能力下降；探索噪声的时间
相关性改变；**与仿真主表不可直接比较，论文里必须单列**。

---

## 5. 架构

### 5.1 模块布局

```
resfit/rl_finetuning/wm_bridge/          ← 全部新文件，两仓 0 行改动
├── __init__.py
├── launch_imagination.py    入口：预置 3 个假模块 + runpy 原 trainer
├── contract.py              ★ 结构断言：被 patch 的目标符号/前提仍成立
├── imagination_env.py       ★ 唯一对外接口，攒 50 步 → 点火 WM
├── wm_driver.py             张量打包/解包 + 归一化 + 调 infer
├── state_tracker.py         proprio 外推（物理空间）
├── scorers.py               Scorer 协议 + DummyScorer
├── base_bridge.py           kai0 serve shim，提供 get_action_chunk
├── init_states.py           采想象起点，caption 钉死常量
└── fake_eval.py             接管 save_checkpoint
```

### 5.2 三个注入点

| # | 目标符号 | 为什么必须注入 |
|---|---|---|
| 1 | `resfit.dexmg.environments.dexmg.create_vectorized_env` | env 工厂。函数内 lazy import（`:840`），预置假模块后连 robosuite 都不用装 |
| 2 | `resfit.rl_finetuning.utils.evaluate_dexmg.run_dexmg_evaluation` | `:1315` 是全脚本**唯一**的 `save_checkpoint`，且在 eval 块内。关掉 eval 等于整个 run 不存盘 |
| 3 | `resfit.lerobot.policies.pi05.load_pi05_base_policy` | kai0 adapter 只有 `select_action`（步级），replan 路需要 `get_action_chunk`（`chunk_env_wrapper.py:87` 优先用它）；且现有 schema 是 libero（单臂 8 维）/ dexmg，实机是三相机 16 维 |

> ★ **不能 patch `build_base_policy` 本身。** 它定义在 `train_chunk_residual.py` 自身，而 launcher
> 用 `runpy.run_module(..., run_name="__main__")` 执行该模块——runpy 会创建**新的模块对象**，
> 对已 import 版本的 patch 作用不到 `__main__` 实例上。必须往下钻一层，patch 它在 `:173`
> 处 lazy import 的 `load_pi05_base_policy`（独立模块，预置有效）。
>
> 注入点 1、2 无此问题：`create_vectorized_env`（`:840`）与 `run_dexmg_evaluation`（`:1302`）
> 都是函数内从**独立模块** lazy import 的。
>
> 实现细节：预置 `sys.modules['a.b.c']` 时，同时对父包 `a.b` `setattr(c)`，避免个别
> import 形式绕过 `sys.modules` 查找。

`stage_purity_summary`（`chunk_env_wrapper.py:233`）定义在 wrapper 自身，我们的 info 不带
`stage_id` → `_stage_total_steps` 恒 0 → 返回 `"no stage steps"`，安全，无需处理。
`__getattr__`（`:251`）委托到 vec_env，兜住其余未知属性。

### 5.3 观测字典契约

| key | 来源 | 形状 / 类型 |
|---|---|---|
| `image_keys`（取自基座 `config.image_features`，`:796`） | WM 预测帧 192×256 → resize | `[1,3,84,84]` float[0,1] |
| `observation.state` | state_tracker 外推，**raw 物理空间** | `[1,16]` |
| `observation.base_action` | wrapper `_augment` 自加（`:113`） | — |
| `observation.stage_id` | wrapper 自加，恒 0 | — |
| `_wm_native_frames` | 原生 192×256 三视角 | 旁路键，供注入点 3 |

旁路键安全：buffer 只存 `set(image_keys) | set(lowdim_keys)`（`:95`），多余键被忽略。
`observation.state` 必须给 raw，wrapper 自己标准化（`:115`）。
图像给 float[0,1]：`to_uint8`（`:81-85`）按 `clamp(0,1)*255` 处理，而 WM 出的是 `[-1,1]`，
`wm_driver` 负责 `[-1,1] → [0,1]` 转换。

**84×84 是强制的，不是取舍**：`min_vit.py:38` `PatchEmbed2.num_patch = 81` 是写死常量，
`:103` `pos_embed = nn.Parameter(torch.zeros(1, 81, embed_dim))`。喂其他尺寸会在加位置编码时
形状不匹配当场崩。`VitEncoder`（`encoder.py:13`）虽收 `obs_shape` 但并不传给 `MinVit`。

---

## 6. 组件设计

### 6.1 `imagination_env.py`

```
reset():
  1. init_states 采 1 个起点：4 帧 × 3 视角窗口 + 该帧 proprio + caption 常量
  2. state_tracker 置为真实 proprio
  3. base_bridge 对起始窗口调 1 次 kai0 → 缓存 50 动作 + ψ_0；Φ_prev = Φ(ψ_0)
  4. seg_step = 0, act_buf = []
  返回 obs_dict

step(action):                      # 单个 16 维物理动作
  act_buf.append(action)
  if len(act_buf) < 50:
      返回 缓存obs, 0.0, False, False, {}        # 空转，见 §2 修正一
  # 第 50 次：点火
  1. wm_driver.pack(act_buf) → (1,25,30) act_tokens
  2. dynamics_model.infer(obs_window, act_tokens) → (3,3,29,192,256)
  3. 取 [:,:,:,4:] → 25 预测帧
  4. state_tracker: proprio ← act_buf[-1]（绝对关节，物理空间）
  5. obs_window ← 预测段末 4 帧；act_buf 清空
  6. base_bridge 对新窗口调 1 次 kai0 → 缓存 50 动作 + ψ̂_末
     reward = gamma * Φ(ψ̂_末) − Φ_prev      # PBRS 端点差
     Φ_prev = Φ(ψ̂_末)
  7. seg_step += 1
     terminated = False（恒定）
     truncated  = (seg_step >= 2)      # RISE 递归上限
  返回 obs, reward, terminated, truncated, info
```

★ **`truncated` 时绝不可把 Φ 置零。** 仓库既有的 `potential_shaping`（`chunk_env_wrapper.py:32`）
约定 done 时 Φ(s')=0，那是为真终止态设计的；想象段的 `truncated` 是人为视界切断，不是真终止。
照抄会毁掉学习信号：

```
不置零：r_0 + γr_1 = γΦ(ψ_1) − Φ(ψ_0) + γ²Φ(ψ_2) − γΦ(ψ_1) = γ²Φ(ψ_2) − Φ(ψ_0)   ✓ 干净相消
置 零：r_0 + γr_1 = γΦ(ψ_1) − Φ(ψ_0) − γΦ(ψ_1)              = −Φ(ψ_0)            ✗ 常数
```

置零后回报与策略无关，残差学不到任何东西。这条与仓库既有约定相反，**是最容易被"好心对齐"
改坏的地方**，`contract.py` 须加断言。

截断即终止是**语义正确**的，不是将就：想象段是一个有限视界的封闭 episode，
其回报本身就是 RISE advantage 的累积，RISE 也从不 bootstrap 超出想象视界。
故 `done = terminated | truncated`（`:1200`）与 `discount = gamma*(1-done)`（`q_agent.py:432`）
的合并恰好正确，零改动。代价是信用分配视界 = 2 × 1.667s ≈ 3.3s（写进 limitation）。

### 6.2 `wm_driver.py`

打包：50 个物理动作 → 每 2 个抽 1 → 25 → 用**本模块自带的 block 域 min-max 统计量**
归一化到 `[-1,1]` → 16 维零填充至 30 槽 → `(1,25,30)` bf16。

绕坑（均不改上游）：

- 坑 1（`rl_release.yaml:197` 指向的 `infer.yaml` 缺 `min_val`/`max_val`）→ 自带统计量，不依赖它
- 坑 2（`wm_utils.py:193` CPU/CUDA 设备不匹配）→ 自己实现打包，不调该函数
- 坑 4（`dynamics_model.py:227` `torch.rand(1,25,30,self.device,...)` 把 device 当尺寸传）→ 永远显式传 `act_tokens`
- 坑 6（`custom_pipeline.py:997` 退出时 `.train()`）→ 每次 `infer` 后显式 `.eval()`
- 坑 7（动作维度在 14/16/30/32 之间漂移）→ 显式钉死 16，加断言

### 6.3 `state_tracker.py`

proprio **全程在物理空间**维护。这修正了 RISE 原实现的空间不一致：
`next_state` 取自 `action_pred`（归一化空间）而喂 WM 的 token 取自 `actions_decoded`（物理空间），
混用会静默错位。

replan 下每段结束：`proprio ← act_buf[-1]`（第 50 个绝对关节动作）。
成立前提是 `action_type: absolute` + `action_space: joint`，
**代价是假设零跟踪误差**——接触、柔顺、碰撞导致的实际偏差全部被忽略。
这是 WM 无状态头的直接后果，不是我们的选择，写进 limitation。

### 6.4 `scorers.py`

PBRS 端点差只需要"给一个 ψ 打一个 Φ"，接口因此比草案简单：

```python
class Scorer(Protocol):
    def phi(self, psi) -> float: ...     # 单个 ψ → 标量势
```

**`Kai0HiqlScorer`（唯一可报告实现）**：`ψ → hiql_value.ValueMLP → Φ`。ψ 由 base_bridge
从 kai0 serve 的 `prefix_feat` 取得（§6.5），本模块不自己编码。

**同源命门（必须断言）**：V 离线训练时用来编 ψ 的 kai0 权重，与在线 base serve 的权重
**必须同源**。仓库已有先例——`hiql_value.py:74` 的 `act_weight_sha` / `act_feat_signature`
与 `train_chunk_residual.py:886` 的 `assert_act_base_samesource`。本模块须比照实现：
value.pt 里记 ψ 编码器权重 sha，启动时与 serve 上报的权重 sha 比对，不符即拒绝启动。
否则 Φ 会被喂进一个它没见过的特征空间，且**不会报错，只会静默给出垃圾势**。

**前提**：kai0 serve 必须以透出 `prefix_feat` 的方式启动。`libero_pi05_adapter.py:54`
是 `result.get("prefix_feat")`，serve 不透特征时返回 None——须在 `contract.py` 里断言非 None。

**测试桩 `DummyScorer`**：返回 0.0，仅供单元测试。launcher 须拒绝在真实训练中使用它，
除非显式传 `--allow_dummy_scorer`。理由：reward 恒 0 的 TD3 会安静跑完全程并产出看起来
正常的 loss 曲线，这种失败模式最难发现。

**调试用 `RiseFrameScorer`**：复用 RISE `reward_model.py:383 predict_reward()`，
仅用于 S1 阶段验证想象闭环是否通。**不可作为可报告的臂**（§3）。

安全线（沿用草案 §4.3）：shaping 只用单状态 V；gc value 的 `V(s,z)` 绝不进 reward。

### 6.5 `base_bridge.py`

替换 `load_pi05_base_policy`（不是 `build_base_policy`，理由见 §5.2）返回一个 shim，
包住 kai0 serve 的 websocket client。`build_base_policy` 照常构造 `BasePolicyConfig`
（host / port / prompt / action_dim / execute_horizon / image_key_map / kai0_paths 全部可由
CLI 配），原样传给我们的假 `load_pi05_base_policy`：

- `get_action_chunk(raw_obs, 50)` → 从 `_wm_native_frames` 取原生帧构造 serve obs → 调 kai0 一次 → 返回 `[1,50,16]` 物理动作
- `config.image_features` → 三相机 key
- `reset()` → no-op

**★ 按窗口做结果缓存，这是 1 次 vs 2 次 serve 调用的关键。** 同一个观测窗口会被请求两次：
一次是 `ImaginationVecEnv` 自己要 ψ 来算 reward（§6.1 step 第 6 步），一次是 wrapper
在 `chunk_env_wrapper.py:219` 调 `_base_chunk_flat(raw_obs)` 要基座动作。两次针对的是
**同一窗口**，故以窗口指纹为键缓存 `(actions, psi)`，第二次直接命中。

于是 **ψ 完全是基座调用的副产品，每 chunk 只需 1 次 serve 调用**。
对比 RISE Eq.(2) 均值形式需要给 25 个预测帧各打一次分，而 `prefix_feat` 只能由**完整的
pi05 推理**（含扩散去噪）产出、无纯特征端点——那是 25 倍开销。这是 §2 修正四选 PBRS 的
工程理由。

用原生帧而非 84×84：实机部署时 kai0 吃的是原生分辨率，想象空间应与部署一致。
若喂 84×84 上采样，基座会被人为削弱，反而**抬高**残差的相对增益——这是论文的效度威胁。

### 6.6 `init_states.py`

从 `--init_state_dataset`（可重复）传入的数据集混采，取 4 帧窗口 + 该帧 proprio。

**★ 无默认路径。** 用户实际使用的 block 数据不在本机；本机 `lingyu_datasets/` 下的
同名数据集**不是**用户的数据。任何硬编码默认都会静默读到错的数据集，故缺参即硬失败。
成功集与失败集**都要传**——残差的主战场是"基座跑偏"的状态（D3）。

**caption 钉死为常量 `"build block"`，不从数据集读。** 本机 expert 集是 `"build blocks"`（复数）、
rollout 集 caption 亦不一致；而 WM 微调时 block 域 caption 已统一为 `"build block"`
（`RISE_Hi/temp/三域数据构建方案.md` §3.4）。WM 靠 T5 编 caption 做条件，喂错即偏离训练分布。

### 6.7 `fake_eval.py`

自己调 `save_checkpoint` 存 `imagination_last.pt`，返回 `{"eval/success_rate": 0.0}`。

**不返回递增哨兵去骗过 `:1312` 的 `sr > best_sr` 比较**——那是把存盘寄托在一个假指标上。
返回恒定 0.0 让 trainer 的 best 逻辑自然失效，存盘由本模块显式负责。

想象空间不产生任何被报告的指标。真实成功率只在实机 eval 测（20 trials + 子目标 rubric）。

### 6.8 `contract.py`

启动时断言，不符**当场硬失败**（零改动路线唯一的兜底，防上游改动导致静默失配）：

1. `create_vectorized_env` 符号存在且签名含 `num_envs` / `device`
2. `run_dexmg_evaluation` 符号存在且返回被以 `m["eval/success_rate"]` 消费
3. `chunk_env_wrapper.step` 仍是逐时间步循环（攒批机制的前提）
4. `PatchEmbed2.num_patch == 81`（84×84 前提）
5. **`--reward_shaping none` 且未传 `--potential_source`** ——否则 wrapper 会在我们的 PBRS
   reward 之上再叠一层自己的 shaping（`chunk_env_wrapper.py:202-213`），即 double-shaping。
   `mode == "none"` 时 `shaping_reward` 返回 0.0（`:35-36`），`potential is None` 时走该分支
6. **`truncated` 时 Φ 未被置零**（§6.1）
7. **ψ 同源**：value.pt 记录的 ψ 编码器权重 sha == serve 上报的权重 sha
8. **serve 确实透出 `prefix_feat`**（非 None）

---

## 7. 错误处理

| 情形 | 处理 |
|---|---|
| 被 patch 的符号不存在/签名变了 | `contract.py` 启动即 raise，不进训练 |
| WM `infer` 形状不符 | 上游 `dynamics_model.py:230` 已有硬闸，直接向上抛 |
| DummyScorer 且未加 `--allow_dummy_scorer` | launcher 拒绝启动 |
| kai0 serve 断连 | 向上抛。**不做静默重试**——静默重试会掩盖 serve 侧的真实故障 |
| 动作维度 ≠ 16 | 断言失败 |
| serve 不透 `prefix_feat`（返回 None） | `contract.py` 启动即 raise |
| ψ 编码器权重与 V 训练时不同源 | `contract.py` 启动即 raise（静默失配代价最高） |
| `--reward_shaping` 非 none | `contract.py` 启动即 raise（防 double-shaping） |

---

## 8. 测试

全部不需要 GPU、不需要 D，用桩 WM：

1. 攒批时序：50 次 `step()` 中前 49 次返回零奖励与缓存 obs，第 50 次点火且只点火一次
2. 2 段截断：第 2 个 chunk 末 `truncated=True`，之前恒 False
3. obs 契约：键齐全、形状 `[1,3,84,84]` / `[1,16]`、图像值域 [0,1]、`observation.state` 未标准化
4. proprio 外推：`proprio == act_buf[-1]`，且全程在物理空间
5. 归一化往返：物理 → 归一 → 反归一 与原值一致
6. 契约断言：符号缺失/改名/`--reward_shaping` 非 none/ψ 不同源时确实抛错
7. caption 常量：不随数据集 caption 变化
8. **PBRS 相消**：桩 Φ 下，两段回报 `r_0 + γ·r_1` 恰等于 `γ²Φ(ψ_2) − Φ(ψ_0)`
9. **`truncated` 不置零 Φ**：回归测试锁死这条与仓库既有约定相反的行为（§6.1）
10. **serve 单次调用**：一个 chunk 内 kai0 只被调用 1 次（窗口缓存命中，§6.5）

回归：现有 `tests/test_env_family_wiring.py` 等须全绿，证明 dexmg / libero 两条既有路径逐位未变。

---

## 9. 分阶段落地

| 阶段 | 目标 | 验收 | 前置 |
|---|---|---|---|
| S0 | 契约固化：真实起点 → `infer()` → 存 25 帧 | 肉眼确认预测帧合理；**动作可控性**：同起点喂不同动作，画面须分化 | D |
| S1 | `ImaginationVecEnv` 跑通 reset→step×50→step×50→reset | 时序、形状、proprio 正确；`RiseFrameScorer` 仅用于此阶段验闭环 | D |
| S1.5 | ψ 一致性验收 | `Φ(ψ̂_k)`（想象帧）vs `Φ(ψ_k)`（真实帧）随 k 的一致性曲线——**同时验收 D 的质量与 kai0 在想象帧上的重编码是否 OOD** | D + V |
| S2 | 残差 TD3（`Kai0HiqlScorer` + PBRS） | 训练不发散；critic loss 收敛；残差幅度不撞 `action_scale` 上限 | D + V |
| S3 | 实机 eval | 20 trials × rubric | S2 |

**S0 的动作可控性检查不能跳过。** 草案 §6 第 10 条指出，shipped 配置下 WM 确实是动作条件的
（`train_mode: 'video_only'` 只冻结名字含 `'action_'` 的参数，而动作条件模块叫 `act_vit_in`/`act_in`），
但条件方式是"动作 token 加到 T5 文本嵌入上"，这是相对**弱**的耦合。动作敏感度是经验问题，
不能假设——若 WM 对动作不敏感，整条想象 RL 路线的前提就不成立。

---

## 10. Limitation（写进论文）

1. **零跟踪误差假设**：proprio 由绝对关节动作外推，接触/柔顺/碰撞偏差被忽略（§6.3）
2. **信用分配视界 3.3s**：想象段 2 chunk 封闭，超出部分完全由 V 的 progress 信号承担（§6.1）
3. **残差粒度 50 步**：与仿真主表不可直接比较，须单列（§4）
4. **想象空间不产出被报告指标**：避免"用世界模型自己给自己打分"（§6.7）
5. **B=1**：零改动约束的成本，WM 算力未充分利用（§2 修正二）
6. **偏离 RISE Eq.(2)**：reward 用 PBRS 端点差而非 chunk 均值。须说明理由——telescoping /
   policy-invariance，以及团队自己的 potsubgoal 塌方实证（§2 修正四）
7. **ψ̂ 的 OOD 风险**：V 在真实帧的 ψ 上训练，想象里喂的是 WM 预测帧的 ψ̂。S1.5 的一致性
   曲线是唯一的量化验收；若该曲线随 k 迅速发散，说明 D 的预测质量不足以支撑 Φ，
   整条路线需重新评估

---

## 11. 遗留

- D 权重 2026-07-19 下午从另一台服务器拷入，路径待定
- cup 域数据不在本机，若要扩到 cup 需一并拷
- paper 域 20fps，接入前须重推时间对齐（50 动作 @20Hz = 2.5s ≠ WM 25 帧 @15Hz = 1.667s）
- `value_failure_reward` 取 −1.0（论文）还是 −0.6（RISE 开源 config 默认）：属 V 训练参数，
  训 V 时再定，建议取 −1.0 并记录偏差
- `chunk_length=50` 这个新 setting 在论文里怎么摆：有结果后再定
- **V 训练侧需交付给本模块的两样东西**：① `value.pt` 内记录 ψ 编码器权重 sha（供 §6.4 同源断言）；
  ② kai0 serve 的启动方式须透出 `prefix_feat`。这两项若缺，本模块启动即拒绝运行
- supp 附录目前写"无额外 reward model / frame scorer"，与本设计一致；若日后决定报告
  hybrid 臂，附录该句必须同步改

---

## 12. V 训练侧的必要改动（不属于 wm_bridge，但 wm_bridge 依赖它）

> **本节改的是 `hiql_value.py` / `train_hiql_value.py`，属 RL 侧代码。**
> §1 的"0 行改动"约束是针对**接入模块**的；训 V 是独立的离线管线步骤，用户 2026-07-19
> 明确授权改动。

### 12.1 现有实现不区分成功与失败（必须修）

逐行核实：

```python
# hiql_value.py:53-56  build_transitions
d = np.zeros(T - 1); d[-1] = 1.0        # 每条 demo 末尾都置 1
# hiql_value.py:139    train_value
reward = done                            # r = 1 if done else 0
# hiql_value.py:23     discounted_target
y = reward + gamma * (1 - done) * next_v
```

即 **V ≈ γ^(距轨迹终点步数)**，一个纯"到轨迹末尾的进度"势函数。
`train_hiql_value.py` 全文无任何成功/失败过滤。

**后果**：在混合质量数据上训，失败轨迹的终止态同样被赋予 V≈1。想象里 PBRS 会奖励残差
**驱向失败终止态**——势函数在教它把事情搞砸。这与 RISE 的 value（成功末端 +1、失败末端 −1）
根本不同，不能声称"follow RISE"。

### 12.2 修法：数据集层面的 ±1 终端奖励

用户的数据**成功集与失败集完全分开**（已人工区分），故标签落在数据集层面，
不需要逐集标注文件。

**★ 禁止按目录名推断标签。** RISE 的约定（`custom_lerobot_dataset.py:508`：
`'infer' in repo_id or 'fail' in repo_id`）在本项目数据上会**静默失效**——
`rollout_*` 这类名字两个关键词都不含，会被全部误判为成功。标签只能显式传入。

**★ 一次性构造 transitions 与 rewards，不可分两次。** `build_transitions` 会跳过 T<2 的
demo；若 `success_flags` 单独构造而不跟着跳，标签会整体错位一格，**且不报错**。

```python
# hiql_value.py 新增(既有 build_transitions 保持不动,零风险)
def build_transitions_with_rewards(state_seqs, success_flags):
    """→ (s, s_next, done, reward)。跳过 T<2 的逻辑与 build_transitions 逐位一致,
    success_flags 在同一次遍历里消费,对齐由构造保证。
    reward: 成功轨迹末 +1,失败轨迹末 −1,中间 0。done 语义不变(轨迹末=1)。
    """

# hiql_value.py 改 train_value 签名(默认逐位等价)
def train_value(s, s_next, done, *, reward=None, ...):
    reward = done if reward is None else reward
```

CLI（遵循仓库"新旗标默认等价、显式翻转"惯例）：

- `--terminal_reward_mode {legacy, success_signed}`，**默认 `legacy`**
- `--success_dataset` / `--failure_dataset`（可重复），`success_signed` 下两者至少各一，
  缺失即硬失败，不猜

如此所有既有仿真 run 的 V（three_piece / pouring / lifttray 那一批）**逐位不变**，
由回归测试证明；只有实机这条新路显式翻转。

### 12.3 样本比例

用户数据约 203 个成功终端 : 117 个失败终端，比例健康，无需额外重加权。
**但 expert 与 rollout 必须一起喂**——只用 rollout 会是 2:117 的极端失衡，V 退化为"一切皆失败"。
