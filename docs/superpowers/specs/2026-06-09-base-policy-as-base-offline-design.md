# base-policy-as-base 离线 buffer 设计(flag 新模式,默认关)

> 状态:已与用户确认(2026-06-09)。下一步:writing-plans → subagent-driven-development。

## 背景与动机

resfit chunk 残差 RL 的离线 demo buffer 由 `chunk_residual/offline_stage_replay.py::build_offline_buffer`
构建。当前实现是 **GT-as-base**:每条 transition 的 `observation.base_action` 与 `action` **都**填成
缩放后的 GT 专家动作(`offline_stage_replay.py:237/240/255`),`build_offline_buffer` 从不调用 base policy。

这造成两条线同时被毒化:

1. **BC 线**:`bc_target = action − base_action`(`q_agent.py:24/504`)。GT-as-base 下
   `= scale(GT) − scale(GT) = 0` → BC 把残差摁向 0,等价"在 demo 状态照搬 base 输出",
   既没锚向专家、又把 demo 里最有价值的"专家 vs base 差"信息算成 0。
2. **critic 线**:offline transition 经 RLPD 混采(`train_chunk_residual.py:630-632`)进 critic TD 更新。
   GT-as-base 让 critic 在 demo 上看到 `(s, a_GT)` 配高 reward/高 Q;在线却是 `(s, base+小残差)` 低 Q
   —— 两条脱节动作流形 → critic 外推高估 → ~20k 塌方(见 memory `project_resfit_gt_as_base_bug`)。

**目标**:新增 base-policy-as-base 模式——离线每帧用冻结 base policy 现算 `base_action`(`action` 仍存 GT),
使 `bc_target = scale(GT) − scale(base(s))`(真·专家修正量,combined 锚向专家),且 critic 的 offline 锚
与在线同一 base 流形对齐。这正是原版 `use_base_policy_for_base_actions=True` 的默认语义。

## 范围

改动**全部局部化**在:离线 buffer 构建(`offline_stage_replay.py`)+ 一个 CLI flag + 缓存签名
(`train_chunk_residual.py`)。**不动**:BC loss(`q_agent.py`)、critic 混采、在线 rollout、
subgoal 接线、eval 路径。flag=`gt` 时与现状**逐位等价**。

## 设计

### 1. 新 CLI flag

`train_chunk_residual.py` 新增:

```
--offline_base_mode {gt, base_policy}   default="gt"
    gt(默认,逐位等价):offline base_action = scale(GT)(GT-as-base,残差目标 0)。
    base_policy:offline base_action = scale(base_policy.select_action(s_demo))(action 仍 GT,
                bc_target=scale(GT)−scale(base),combined 锚向专家;critic offline 锚对齐在线流形)。
```

在 `build_offline_buffer(...)` 调用处(`train_chunk_residual.py:553`)把 `base_policy` 与
`args.offline_base_mode` 透传进去(`base_policy` 已在更早处由 `build_base_policy` 加载)。

### 2. build_offline_buffer 改动(核心)

签名新增 `base_policy=None, base_mode="gt"`。

- **`base_mode=="gt"`**:与当前代码**逐字节相同**(`observation.base_action = act_n[t]`)。
- **`base_mode=="base_policy"`**:对每条 demo
  1. `base_policy.reset()`(清 ACT action queue,对齐在线 `reset`,`chunk_env_wrapper.py:131`);
  2. 按帧 `t=0..T-1` **顺序**构造 raw_obs_t(原始未标准化 state18 + 该帧图像,batch=1、上 device),
     调 `base_policy.select_action(raw_obs_t)` 取原始 base 动作,`action_scaler.scale(...)` → `base_n`(T,D);
     **必须顺序**——base policy 有状态(queue 每 `n_action_steps` 才重规划、其余弹队列,见
     `chunk_env_wrapper.py:94-97`);
  3. 存储:`curr["observation.base_action"]=base_n[t]`、`nxt["observation.base_action"]=base_n[t+1]`;
     `"action"=act_n[t]=scale(GT)` **不变**。

`base_n` 与现有 `act_n`/`state_n`/`subgoal_z` 等长、对齐同一帧索引。其余 transition 字段
(reward/done/stage/subgoal/图像)与现状完全一致。

### 3. obs 格式契约(plan 首个勘察 step)

离线拼的 raw_obs 必须与**在线 env 喂给 `select_action` 的格式逐项一致**:同样的 image key 集合、
图像 layout/dtype、**原始(未标准化)** `observation.state`、batch 维、device。

实现时复用现有在线/eval 的 obs 构造路径(prefer 复用 helper,不另写一套)。plan 第一个 step
专门勘察并锁定该格式,避免格式错配静默产出错的 base_action。

### 4. 缓存签名与向后兼容

`_offline_buffer_signature`(`train_chunk_residual.py:167`)**条件性**加键,照搬现有
`potential_source=="hiql"` 的写法(`:194`):

- `base_mode=="gt"`:**不加**任何新键 → 现有 gt 缓存签名不变、零重建、历史实验逐位不变。
- `base_mode=="base_policy"`:加 `offline_base_mode="base_policy"` + base 身份
  (`base_wandb_id` 的 abspath,或 pi05 的 host/port/kai0 标识)→ 新缓存独立重建;
  换 base 也使签名变、强制重建(base_action 依赖 base 权重)。

### 5. 约束与边界

- `base_mode=="base_policy"` 断言 queue:`chunk_length==1` 且 `base_action_mode=="queue"`
  (与在线 queue 语义匹配;replan 模式当前不支持,留 YAGNI)。
- 接口对任何暴露 `select_action`/`reset` 的 base 通用(本地 ACT 便宜;pi0 走 websocket 对
  ~23.8 万帧昂贵——构建前 `print` 一条 warn,但不拦,`--offline_buffer_cache` 使其一次性)。

## 测试

1. **逐位等价(回归)**:`base_mode="gt"` 路径不变 → 现有 offline buffer 单测 + 全量回归
   (基线 119 passed)仍全绿;baseline(flag 不传/=gt)逐位等价。
2. **新单测 — base_policy 模式正确锚**:用假 base policy(`select_action` 返回已知常量 raw 动作、
   `reset` 可计数)在 tiny demo 上建 buffer,断言:
   - `observation.base_action == scale(fake)`(**非** GT);
   - `action == scale(GT)`(不变);
   - 隐含 `bc_target = scale(GT) − scale(fake) ≠ 0`。
   对照 `gt` 模式:`base_action == action == scale(GT)`、`bc_target == 0`。
3. **queue/顺序语义**:断言每 demo `base_policy.reset()` 恰调用一次、`select_action` 按帧顺序
   调用 T 次(spy/mock 计数)。
4. **签名**:断言 `gt` 模式签名 dict **无** `offline_base_mode` 键(缓存向后兼容);
   `base_policy` 模式**有**该键 + base 身份键;换 base 身份使签名变。

## 不改动项(显式声明)

`q_agent.py` 的 BC loss 与 `bc_target`、critic 混采、在线 rollout、`hiql_subgoal.py`/subgoal 接线、
`evaluate_dexmg.py` eval 路径——**全部不动**。本修复只换"offline buffer 里 base_action 这一栏的来源"。

## 落地后的使用(超出本 spec 范围,供参考)

改完后,用 `--offline_base_mode base_policy --demo_bc_coef <coef> --offline_buffer_cache <新缓存>`
起一个新臂(或重启 B-staged),即得"BC 锚向专家"的版本;与现有 gt 缓存的臂可直接 A/B。
具体跑法(重启 B-staged vs 另起平行臂、系数取值)在实现完成后单独定。
