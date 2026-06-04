# 设计文档：残差 RL 基座策略支持 pi05（ACT / pi05 可切换）

- 日期：2026-06-04
- 目标仓：`/data2/RL/residual-offpolicy-rl`
- 关联仓：`/data2/kai0`（openpi / pi05 / Pi05PolicyAdapter 所在）
- 场景：robomimic 仿真（沿用现框架的 robomimic 路线）

## 1. 背景与目标

`residual-offpolicy-rl` 当前用 ACT 作为冻结基座策略，在 robomimic 上做残差离线 RL：
最终动作 = 基座动作 + 残差动作（相加），基座全程冻结、不参与梯度。

目标：把基座从 ACT 扩展为可选 pi05（Physical Intelligence π0.5，flow-matching VLA），
**ACT 与 pi05 并存，用配置开关切换**，不破坏现有 ACT 实验与可比性。

非目标（本期不做）：
- 不做真机 / 非 robomimic 仿真（dexmg 等）。
- 不把 adapter 重写为正规 lerobot PreTrainedPolicy 子类（方案 C，本期不选）。
- 不复用 kai0 的 `dsrl_pi05` 框架（方案 B，本期不选）。

## 2. 关键决策

- **集成路线 = 方案 A**：在 `residual-offpolicy-rl` 内部把基座由 ACT 换/选为 pi05，
  保留它现有的 robomimic 仿真环境、评估流程、RL 算法栈（pixel-SAC / TD3 残差）。
- **ACT 不删除**：通过配置开关 `base_policy.type` 在 `"act"` 与 `"pi05"` 间切换，默认 `"act"`，
  保证旧实验行为字节级不变。
- **基座模型选型 = pi05**（接口与 pi0 几乎一致；pi05 语言条件更强、单任务微调更易）。
- **基座需先微调**：当前没有 robomimic 上微调好的 pi05 checkpoint，微调是本期工作量大头。

## 3. 现状盘点（复用基础）

| 部件 | 现状 | 缺口 |
|---|---|---|
| `Pi05PolicyAdapter`（`/data2/kai0/resfit_pi05/pi05_policy_adapter.py`） | 已实现 `select_action / reset / config.image_features`，正好对上残差框架对 base_policy 的接口 | 视角名写死为 aloha 风格（`top_head/hand_left/hand_right`），需映射到 robomimic 视角；可能缺 `normalize_inputs/unnormalize_outputs/model`（见 §6 待核实） |
| robomimic → LeRobot 转换 | 两份现成脚本：`resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`、`/data2/kai0/train_deploy_alignment/data_augment/convert_h5_lerobot.py` | 需调视角名映射、action 维度、prompt |
| openpi 微调脚本 | `train.py` 通用、现成 | openpi 无 robomimic 的 data config 条目，需新增 |
| RL 训练栈（Actor/Critic/QAgent/buffer/残差组合/归一化器） | 完全不用动 | — |

结论：整体约 80% 是"拼接 + 对齐"，非从零开发。

## 4. 框架对基座的接口契约

残差框架（在不含本设计的待核实点前提下）只通过以下接口认基座：

- `base_policy.select_action(obs_dict) -> Tensor[B, action_dim]`：每个 env step 调一次，
  返回原始尺度单步动作（基座内部维护 chunk 队列，队列空时才真前向）。
- `base_policy.reset(env_ids=None)`：重置内部 chunk 队列。
- `base_policy.config.image_features`：取图像 key 列表。

`Pi05PolicyAdapter` 三者均已具备 → "开关"本质 = 初始化时让框架拿到一个"长得像 ACTPolicy 的对象"，
之后框架分辨不出区别。

## 5. 设计详述

### 5.1 基座选择开关（改动点，方案 A）

只在两处硬编码 ACT 的地方插分发层，RL 循环不碰：

1. **配置开关**：base policy 配置新增 `type: "act" | "pi05"`（默认 `"act"`）。
   pi05 额外字段：`config_name`（如 `pi05_robomimic_xxx`）、`checkpoint_dir`、`prompt`、
   `action_dim`、`execute_horizon`、视角映射表（见 5.3）。

2. **加载分发**：`resfit/lerobot/utils/load_policy.py` 的 `load_policy()` 按 `type` 分发：
   - `"act"` → 原 `ACTPolicy.from_pretrained`（不改）。
   - `"pi05"` → `Pi05PolicyAdapter.from_checkpoint(...)`。

3. **actor 类型判断分发**：`scripts/train_residual_td3.py` 现用 `isinstance(base_cfg, ACTConfig)`
   定 `actor_name`，扩展为同时认 pi05 配置类型，设到对应 residual actor 名。

### 5.2 pi05 微调链路（工作量大头）

1. robomimic HDF5 → LeRobot 数据集（用现成转换脚本）。
2. 新增 `Pi05RobomimicDataConfig`（基于 openpi 的 `SimpleDataConfig`），挂视角/动作/prompt transforms。
3. 计算归一化统计（`compute_norm_states`）+ 注册 `TrainConfig(name="pi05_robomimic_xxx", model=Pi0Config(pi05=True), data=...)`。
4. 跑 `train.py pi05_robomimic_xxx` 产出 checkpoint。
5. **基座质量 gate**：接 RL 前，先用 `Pi05PolicyAdapter` 单独在 robomimic eval 该 checkpoint 成功率，
   达可用阈值才往下（参考 Coffee "基座全 0%" 教训，避免在塌掉的基座上做 RL）。

### 5.3 接口对齐（adapter 小改造）

1. **视角映射**：把 adapter 写死的 aloha 视角名改为**从 base policy 配置读取映射表**，
   robomimic 的 `agentview / robot0_eye_in_hand` 对到 pi05 三槽位；缺的 wrist 槽位用零图 +
   `image_mask=False` 占位（pi05 支持图像 mask）。
2. **补缺失接口（若 §6 核实需要）**：给 adapter 补 `normalize_inputs/unnormalize_outputs/model`
   （openpi 内部已做归一化，多为空实现或薄封装）。
3. **action 维度一致**：adapter 输出维度 = robomimic action 维度 = 残差叠加维度，三者对齐
   （单臂 OSC 一般 7 维；adapter `action_dim` 截取，pi05 内部 32 维 padding）。

## 6. 调用基座的代码点（已核实，2026-06-04）

核实结论：框架有两条调用基座的路径——

- **step 级**：`resfit/rl_finetuning/wrappers/residual_env_wrapper.py`（`BasePolicyVecEnvWrapper`），
  仅用 `select_action()`（:102/:149）、`reset()`（:98/:156）、`config.image_features`（:63）。
  `Pi05PolicyAdapter` 已全部具备 → 几乎零改（只需视角映射可配）。
- **chunk 级**：`resfit/rl_finetuning/chunk_residual/chunk_act_base.py` 的 `get_action_chunk()`，
  用 `normalize_inputs / config.image_features / model / unnormalize_outputs`，被 `chunk_env_wrapper.py` 调用。
  `Pi05PolicyAdapter` 缺 `normalize_inputs/unnormalize_outputs/model`。

**本期决策：走 step 级路径**（`BasePolicyVecEnvWrapper`）。因此：
- 不修改 `chunk_act_base.py` / chunk 路径；
- adapter **无需**补 `normalize_inputs/unnormalize_outputs/model`；
- §5.3 第 2 点（补缺失接口）本期不做。

后续若要 chunk 级，再单独立项给 adapter 加原生 `get_action_chunk` 并让框架委托。

## 7. 测试策略（TDD）

- **单测**：视角映射（robomimic obs → 正确 pi05 obs + mask）、action 维度截断、
  chunk 队列弹出与 `reset`、`config.image_features` 正确。
- **集成 smoke test**：mock 一个 pi05 policy（`infer` 返回固定 chunk），
  跑通"加载分发 → wrapper 调 select_action → 残差组合"，确认开关切 pi05 不报错。
- **回归保护**：`type="act"` 行为与改造前完全一致。

## 8. 分阶段落地（每阶段带验证 gate）

- **Phase 0**：数据转换 + 视角/动作/prompt 对齐，产出 LeRobot 数据集。Gate：能 load、维度对。
- **Phase 1**：微调 pi05 + 单独 eval 基座成功率。Gate：成功率达可用阈值，否则停下调数据/prompt。
- **Phase 2**：接口开关改造 + 测试。Gate：ACT 回归不变 + pi05 smoke 通过。
- **Phase 3**：pi05 基座 + 残差 RL 端到端。Gate：训练不崩、成功率有提升趋势。

建议第一版只挑一个简单单臂 robomimic 任务（如 Lift / Can）跑通全链路，验证 domain gap 可控后再扩。

## 9. 风险

- **算力**：pi05 约 3B 参数 VLA，微调需较大 GPU。
- **domain gap**：pi05 为真机/aloha 大图设计，在 robomimic 仿真小图上微调效果有不确定性 → 小步先验证。
- **基座质量门槛**：残差 RL 强依赖基座本身可用；Phase 1 gate 不过则不应进入 Phase 3。
- **接口对接点未完全确认**：见 §6，writing-plans 开头先消除。

## 10. 可配置项（默认值）

- 目标任务：可配置，第一版默认单臂任务（Lift 或 Can）。
- 视角映射表、action 维度、prompt：随任务配置。
- `base_policy.type` 默认 `"act"`。
