# 设计：stage_budget（逐阶段残差幅度预算）

日期：2026-06-06
分支：chunk-residual-validation
关联文档：`/data2/kai0/rlt/residual_flow_design_and_training.md` §18.3 / §22.5
关联：本模块是"三个 default-off 模块"计划的 ①（② RPL relay relabeling、③ HIQL-Φ 各自另开 spec）

## 1. 背景与动机

chunk_residual 线（dexmg / robosuite, TwoArmThreePieceAssembly, NUM_STAGES=5, stage 0-4）
在 2026-06-06 对三条在跑实验（GPU0/1/2）的对照分析得到的事实：

```
run            base(env1)   近期滑动均值   best     变量(相对基准)
stageON  GPU2    0.30          ~0.12       0.30     + --stage_conditioned   <- 崩(净负)
OFFLINE  GPU0    0.22          ~0.21       0.34       (基准: 无 stage)        <- 中性
nas10    GPU1    0.40          ~0.36       0.54     + --base_n_action_steps 10 <- 中性
```

三个 run 的内部 diag 几乎相同：residual_norm 各阶段都 ~0.16（残差都长出来了，不是约 0）；
critic 的 target_q 各阶段都是 stage1 虚高 ~1.4 / stage2 约 0 / stage3 ~0.9（违反"越接近完成
value 越高"的单调性 = critic 估歪指纹）；stage3 回退率 60-72%（失败位点）。

两条关键结论：
1. 残差在任何配置下至多中性；stage_conditioning（把 stage one-hot 拼进 actor/critic 输入）
   是唯一把它从中性推到净负的因素 —— 它给了网络"按阶段作恶"的表达力，并被用来在 stage3 作恶。
2. 真正抬成功率的失败位点是 stage3->4 精插入。一个全局单一残差映射必须同时应付"粗接近
   (可大胆纠)"和"精插入(几乎不能碰)"，折中出的修正在精插入阶段就是毒药。

stage_budget 的思路（§18.3 stage-dependent residual budget）：不给网络更多 stage 信息，而是
**机械地按 stage 缩放残差输出包络**，在 stage3/4 把残差幅度卡小。这与刚证伪的 stage_conditioning
有本质分界：stage_conditioning 改网络输入（给表达力），stage_budget 只改输出幅度（网络保持
stage-agnostic）。

## 2. 目标 / 非目标

目标：
- 让残差的输出包络（均值 + 探索噪声）随当前 stage_id 缩放，在精插入阶段近乎不动。
- 以 CLI flag 控制，**默认关闭**：关闭时代码行为与现状逐字节相同（保留 baseline = A/B 的 A）。
- 单点改动、改面最小，复用现有 stage_id 透出与 QAgent 接线。

非目标（本模块不做）：
- 不做 stage_conditioning（不把 stage 喂进网络输入）。
- 不支持 `--actor flow`（flow actor 有自带 action_delta_clip，延后；第一版仅 raw）。
- 不动 critic、不动 reward、不动 buffer。
- 不实现 RPL relabeling / HIQL-Φ（各自另开 spec）。

## 3. 关键设计事实（代码已核实）

- 残差幅度定点：`off_policy/rl/actor.py:183` `scaled_mu = mu * self.cfg.action_scale`
  （mu 是 tanh 输出 ∈ [-1,1]）。随后 `action_dist = TruncatedNormal(scaled_mu, std)`（line 186）。
- stage_id 已在 `obs["observation.stage_id"]`：rollout=瞬时（wrapper `_stage_now`）、
  update=batch obs、target=next_obs，三处都现成。`actor.py:175` 的 stage_conditioned 分支已在读它。
- `TruncatedNormal`(`common_utils/utils.py:154`) 的 scale 参数可直接接受张量（只有 float 才转），
  故 per-sample 的 std 张量可行。
- `QAgent.act`（q_agent.py:268）把整个 obs（含 stage_id）传进 actor.forward，rollout 路径拿得到 stage。
- `QAgent.__init__` 已透传 `residual_actor / stage_conditioned / num_stages` 给 Actor 与
  actor_target（actor_target 靠 deepcopy 继承），新增 `stage_budget` 顺同一路。

## 4. 设计

### 4.1 budget 的施加（单点）

在 `Actor.forward` 内，`scaled_mu = mu * action_scale` 之后：

```
if self.stage_budget is not None:
    sid = obs["observation.stage_id"]                                  # [B,1] float
    factor = stage_budget_factor(sid, self._stage_budget, self.num_stages)  # [B,1]，纯函数
    factor = factor.to(feat.device)
    scaled_mu = scaled_mu * factor
    std = (std if torch.is_tensor(std) else torch.full_like(scaled_mu, std)) * factor
action_dist = utils.TruncatedNormal(scaled_mu, std)
```

其中 `stage_budget_factor(stage_id, budget_tensor, num_stages)` 是抽出的纯函数（gather + clamp，
见 §5 测试 1 / §7），施加点只调用它，便于单测与复用。

因为 actor / actor_target / rollout / eval / actor-loss 全走这条 forward，**一处改、五处一致**，
不需别处接线。mean 与 std 都按 budget 缩（已选定：stage3 例 budget=0.1 -> mean 缩到 ~0.005，
若不缩 std 则 rollout 被 ~0.05 噪声主导、budget 失效）。

### 4.2 接线（顺 stage_conditioned 老路）

- `Actor.__init__` 加 `stage_budget: list[float] | None = None`。非 None 时存成 buffer 张量
  `self._stage_budget`（`register_buffer`，长度 num_stages，随模块迁移 device），`self.stage_budget`
  记原 list（None 表关）。
- `QAgent.__init__` 加 `stage_budget` 形参，透传给构造 Actor 的两处（actor 与 actor_target）。
- `train_chunk_residual.main` 解析 `--stage_budget` 并传给 QAgent。

### 4.3 CLI

```
--stage_budget "1,1,1,0.3,0.1"   # 逗号 list，长度必须 == num_stages（three-piece=5）；不传=关
```

解析：split(",") -> [float]；断言 `len == num_stages`；断言 `--actor raw`（第一版作用域）。
废弃现占位 `--stage_budget_mode`（choices=["none"]，本无逻辑），由 `--stage_budget` 取代。

### 4.4 默认关 = baseline 逐位等价

不传 `--stage_budget` -> `stage_budget=None` -> forward 走原路径（`scaled_mu=mu*action_scale`，
std 为传入标量，`TruncatedNormal(scaled_mu, std)` 不变）-> 与现状一字节不差。这是仓库纪律，
也是单测必钉的回归点。

### 4.5 作用域与正交性（第一版）

- 仅 `--actor raw`（加 assert）；flow 延后。
- 与 stage_conditioned 正交，可共存（独立形参）；但 three-piece 上 conditioned 已弃，典型用法是
  budget 单开。
- 与 reward_shaping / offline_fraction / stage_balanced 全部正交。

### 4.6 特权口径（注脚）

budget 用 stage_id（特权检测器读 sim）缩输出，与 §22.3 同口径：仿真训练/eval（robosuite）零成本，
要部署再补 obs->stage 预测器。关键区别：**budget 不把 stage 喂进网络**，网络保持 stage-agnostic，
只有输出包络被 stage 塑形。

## 5. 测试方案（TDD 先行）

纯函数 + 单点 forward 改动，易单测：

1. `stage_budget_factor(stage_id_tensor, budget_list)` 纯函数：gather + clamp；混合 stage 的
   batch 每行乘子正确；越界 stage_id 被 clamp。
2. forward 关（stage_budget=None）：输出 dist 的 loc/scale 与改前逐位一致（baseline 等价回归测）。
3. forward 开：给定已知 stage_id 的 batch，dist.loc == mu·action_scale·factor、
   dist.scale == std·factor（逐行）。
4. QAgent 集成：带 stage_budget 构造（actor 与 actor_target 都拿到 buffer），act() 能跑出动作；
   高缩放阶段（如 stage3 budget=0.1）的残差 norm 统计上显著小于 budget=1 的阶段。
5. CLI 解析：`"1,1,1,0.3,0.1"` -> [1,1,1,0.3,0.1]；长度 != num_stages 报错；`--actor flow` + 
   `--stage_budget` 报错。

## 6. A/B 验证方案（实现完成后，plan/run 阶段；不属本实现）

- 配置：用更强 base（nas10 那套：`--base_n_action_steps 10` + `--offline_fraction 0.5` +
  `--reward_shaping staged` + `--base_action_mode queue` + `--chunk_length 1`），其余对齐。
- A = `--stage_budget` 不传（baseline）；B = `--stage_budget "1,1,1,0.3,0.1"`。
- 主指标：`eval_stage_reach` 的 per-stage 到达率，重点 reach3->reach4 与最终 success。
- 旁证：stage-diag 里 stage3/4 的 residual_norm 应明显下降（确认机制生效）。
- 判据：B 是否在不牺牲早期 reach 的前提下，把 stage3->4 救起来。

## 7. 文件改动清单

- `resfit/rl_finetuning/off_policy/rl/actor.py` —— Actor 加 `stage_budget` 形参 + forward 内施加；
  新增纯函数 `stage_budget_factor`（可单测，放本文件或 stage_utils.py）。
- `resfit/rl_finetuning/off_policy/rl/q_agent.py` —— QAgent 加 `stage_budget` 形参并透传两处 Actor。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` —— 加 `--stage_budget` 解析 +
  断言 + 传 QAgent；移除占位 `--stage_budget_mode`。
- 测试：`off_policy/rl/` 下新增 stage_budget 单测文件（或并入现有 actor 测试）。

## 8. 风险与缓解

- mean+std 都缩可能让 stage3 探索过少、学不动 -> 这正是意图（精插入近乎不动）；若过度，可调 budget
  值（如 0.3 而非 0.1），list 可扫。
- budget 用瞬时 stage_id（rollout）vs 闩锁（reward 用闩锁），二者解耦是现状设计（handoff §2，避免
  stage 桶被回退样本污染），budget 用瞬时与 actor 输入语义一致，无需改。
- 越界/缺 stage_id：clamp 兜底；非 three-piece 任务 num_stages=1 时 budget list 长度 1，等价全局缩放。
