# 设计：cl=1 step 级 stage-conditioned residual

日期：2026-06-04
分支：chunk-residual-validation
关联文档：`/data2/kai0/rlt/residual_flow_design_and_training.md` §18.1 / §22

## 1. 背景与动机

chunk_residual 线（dexmg / robosuite, TwoArmThreePieceAssembly）的 per-stage 诊断
（`eval_stage_reach.py`）显示：

```
residual_norm 0 -> 1.5 时, success(reach4) 选择性崩到 0, 而 reach1-3 基本保留
```

即一个**全局单一**的残差映射，必须同时应付"粗接近(可大胆纠)"和"精插入(几乎不能碰)"，
只能折中，而折中出的修正在精插入阶段(stage3->4)就是毒药。

cl=1 queue 探针补充：把 chunk_length 降到 1（每步闭环，base 走 ACT 原生 queue，
action_scale=0.1 / actor_lr=1e-6 对齐原版）后，critic 被 potential 奖励救活、success 未塌，
但零初始化 + actor_lr=1e-6 下 residual_norm 长期约 0，残差"还没长出来"。两条证据合起来支持：
给残差**按阶段特化**的能力。

本设计把 §22.2 的 stage-conditioned residual 落到 cl=1 step 级。

## 2. 目标 / 非目标

目标：
- 让 actor 与 critic "知道当前阶段 stage_id"，从而学出分段特化的局部纠错（粗接近多纠、精插入近乎不动）。
- 全部以 config 开关控制，**默认关闭**：关闭时代码行为与现状逐字节相同（保留 baseline = A/B 的 A）。

非目标（本次不做）：
- 不动 chunk 级 / flow actor（§17：cl=1 step 级 16/24D 残差用 raw MLP 足够）。
- 不做 stage budget 的缩放逻辑（只留 config 占位，§22.6 先验证后扩展）。
- 不做 obs->stage 预测器去特权化（§22.6 第二步，依赖第一步结论）。
- 不跑训练 / A/B（实现 + 单测为本次范围，A/B 由用户后续手动 GPU 跑）。

## 3. 现状（已核对）

- cl=1 queue 模式用 raw Actor：`resfit/rl_finetuning/off_policy/rl/actor.py`，`residual_actor=True`。
- `observation.stage_id` 已存在 obs dict 与 replay buffer：
  - 由特权检测器 `chunk_residual/stage_detectors.py::threepiece_stage` 算出（int 0-4，5 段）。
  - `chunk_env_wrapper.py::_augment` 在 train 与 eval 都拼进 obs，key=`observation.stage_id`，
    shape [B,1]，float，用 instant 值 `_stage_now`（可回退，但范围仍 0-4）。
  - buffer 的 obs dict 里存有该字段；另有 `max_stage`（latched）供 stage-balanced sampler。
- **关键缺口**：stage_id 从未喂给 actor / critic 的 forward（grep 0 命中）。本设计即补这一环。
- 任务段数：`stage_detectors.py::NUM_STAGES["TwoArmThreePieceAssembly"] = 5`。

## 4. 设计决策（已与用户确认）

1. stage 编码 = **one-hot（num_stages 维）**。离散阶段最干净，不强加 0-4 大小顺序，
   也避免单个 float 淹没在约 600 维特征向量里。
2. 条件付加在 **actor + critic 两边**。critic 不知 stage 会对"同样视觉 obs"跨阶段平均 Q，
   留着 critic 侧的"全局单一函数"病。
3. 两个**独立开关**，默认关：
   - `--stage_conditioned`（默认 False）：喂 stage one-hot 进 actor/critic。
   - `--stage_budget_mode`（默认 "none"）：按阶段缩放残差上限。**第一版只占位，不实现逻辑**。

可拆出的对照（实现后由用户跑）：
```
A baseline  : stage_conditioned=off, budget=none   现状全局单一残差
B           : stage_conditioned=on,  budget=none   纯 conditioning（本次主验证目标）
C（后续）    : off, budget=on                       纯硬卡精插入幅度
D（后续）    : on,  budget=on                       两者合一
```

## 5. 改动清单

### 5.1 stage 编码 helper（新增）
- 函数 `stage_onehot(stage_id, num_stages) -> [B, num_stages]`：
  取 obs 的 `observation.stage_id`（float [B,1]）-> `.long().clamp(0, num_stages-1).squeeze(-1)`
  -> `F.one_hot(..., num_classes=num_stages)` -> float。
- 位置：放 chunk_residual 包内一个小工具模块（与现有约定一致），actor/critic 调用侧引用。
- `num_stages` 参数化，不写死 5；由任务的 `NUM_STAGES` 推出。

### 5.2 Actor（`off_policy/rl/actor.py`）
- 构造增加 `stage_conditioned: bool = False`、`num_stages: int = 0`；
  当 `stage_conditioned=True` 时 `prop_dim += num_stages`。
- `forward`：当开启时，把 `stage_onehot(obs, num_stages)` append 进 `all_input`
  （位置在 state / base_action 之后），再 `torch.cat`。
- 关闭时分支与现状逐值相同（回归保护，见测试 2）。

### 5.3 Critic（`off_policy/rl/critic.py` + `q_agent.py`）
- **不改 critic.forward 签名**。改为在 QAgent 把 stage one-hot 拼进 `prop` 再传 critic。
- critic 构造时 `prop_dim += num_stages`（当开启）。
- QAgent 封装 `_prep_critic_prop(obs)`：统一返回拼好 stage 的 prop，
  在所有调 critic 处使用（当前 Q、target Q、actor loss 里的 Q），避免漏拼导致维度/语义不一致。
- target critic 同样走 `_prep_critic_prop`。

### 5.4 stage budget（占位）
- `train_chunk_residual.py` 加 `--stage_budget_mode`（默认 "none"），第一版仅解析、不接逻辑。
- 设计意图（后续 C）：`a_exec = base + stage_scale(stage_id) * actor_output`，
  精插入阶段给最小 scale；train 与 exec 须一致（否则 critic 看到的与执行不一致）。本次不实现。

### 5.5 config（`train_chunk_residual.py`）
- `--stage_conditioned`（store_true，默认 False）。
- `--stage_budget_mode`（默认 "none"，仅占位）。
- `num_stages` 由任务推出，不需手填；传入 Actor/Critic/QAgent 构造。

## 6. 测试（pytest，沿用 `chunk_residual/tests/` 约定）

1. `stage_onehot` 编码正确性：0..4 映射到对应 one-hot；越界值被 clamp；shape/dtype 正确。
2. Actor 回归保护：`stage_conditioned=False` 时 forward 输出与改前**逐值相同**
   （证明默认零改变）。
3. Actor 条件生效：`stage_conditioned=True` 时 forward 不报错、输出 shape 正确，
   且喂不同 stage_id -> 输出不同（证明 stage 真的"效"到）。
4. Critic 条件生效：开启时 Q forward 通、维度正确，不同 stage -> Q 不同。
5. QAgent 烟雾测试：开启 conditioning 跑一步 actor/critic update，不出现 NaN，
   stage 维度全程一致（actor 输入、critic prop、target critic prop 对齐）。

## 7. A/B 验证（实现后，非本次编码范围）

- A = `--stage_conditioned` 关，B = 开。同超参。
- 指标：`eval_stage_reach` 的 per-stage 到达率，重点 reach3->reach4(精插入) 与最终 success。
- 判据：B 是否在不牺牲早期 reach 的前提下把 stage3->4 救起来。
- 先决条件（§22.4）：残差得真长出来。actor_lr 1e-6 可能两边 residual_norm 都约 0，
  比不出名堂；跑 A/B 前确认 residual_norm 不是 0（必要时 actor_lr 调 5e-6 或给足步数）。

## 8. 风险与处理

- **stage one-hot 被淹没**：one-hot 已比单 float 强；若仍弱，可在拼接前对 stage 段单独 LayerNorm 或后续换 embedding（非本次）。
- **维度漏对齐**：critic 多处调用，靠 `_prep_critic_prop` 单一入口 + 测试 5 兜底。
- **instant stage 回退**：`_stage_now` 可回退，但范围仍 0-4，one-hot 合法；属既有语义，不在本设计改。
- **特权口径**：stage 来自特权检测器，stage-conditioned 后残差成"特权策略"。
  本项目全程仿真、eval 也是 robosuite，端到端可用；去特权化是 §22.6 第二步，本次不做。
```
