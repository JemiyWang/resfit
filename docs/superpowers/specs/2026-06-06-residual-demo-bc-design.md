# 设计：residual actor 的 demo-BC 消费端(模块 ②a)

日期：2026-06-06
分支：chunk-residual-validation
关联文档：`/data2/kai0/rlt/residual_flow_design_and_training.md` §11 / §17 / §18.6 / §21.4(RPL DAPG)
关联：三模块 default-off 计划的 ②a。② RPL relay relabeling 拆两步:
  - **②a(本 spec)= 残差 BC 消费端** —— 实现残差 actor 的 BC loss,用现成 offline GT demo 当 bc_batch 验证。
  - ②b(后续另开 spec)= 在线 relay relabeling 生产端(失败轨迹到达 stage 的产出段 harvest 进 relabel buffer 当 bc_batch)。
  前置:模块 ①(stage_budget)已实现(commit eef98a6/df83c4e/53811a4)。

## 1. 背景与动机

2026-06-06 对 GPU0/1/2 三条 three-piece cl=1 实验对照(见 memory `project_stage_budget_and_3module_plan`):
残差在任何配置下至多中性、critic 估歪(stage2 Q≈0)。RPL/§18.5 的对策是从"部分成功"里学:失败轨迹到达
stage K 的那段对"到达 stage K"是成功的,重标成正样本喂学习器。三模块决策里定了 relabeling **喂 actor-AWBC
侧**(喂单任务 critic 会教它"到 stage3=满分",加重高估)。

但探查发现:actor 侧的 BC 机制虽存在(`update_actor_rft` + `_compute_actor_bc_loss` + DAPG 动态系数),
**对残差 actor 是被硬挡的**:`_compute_actor_bc_loss` 第一行 `assert not self.residual_actor, "Not implemented"`;
且 chunk_residual 里 `update(..., bc_batch=None)` 从没启用过。所以"喂 actor"不是开个开关,需要先把**残差 BC
消费端**造出来。本 spec(②a)就造它,并用现成 offline GT demo 当 bc_batch 验证它不砸——②b 的在线 relabel
段将复用同一消费端。

## 2. 目标 / 非目标

目标:
- 给残差 actor 实现 BC loss:解开 `assert not residual_actor`,target = 采取动作 − 基座动作(残差目标)。
- 在 chunk_residual 训练里接上 bc_batch(取自现成 offline_rb)+ `--demo_bc_coef` 旋钮(均匀 BC)。
- 默认关(demo_bc_coef=0)逐位等价现状。

非目标(本模块不做):
- 不做在线 relabel harvest(那是 ②b)。
- 不开 DAPG 动态系数(bc_loss_dynamic 保持 False;动态路径用 `q_value_for_policy`/ref_agent,对残差 actor 需额外适配,留后续)。
- 不支持 `--actor flow`(第一版仅 raw)。
- 不动 critic、不动 reward、不动 buffer 结构。

## 3. 关键设计事实(代码已核实)

- `_compute_actor_bc_loss`(`q_agent.py:473`):`assert not self.residual_actor`;`pred = _act_default(eval_mode=False,
  stddev=0)`;`loss = mse(pred, batch["action"]).sum(1).mean(0)`。残差 actor 的 `_act_default` 返回的是**残差**
  (chunk_residual 主循环 `agent.act` 返回残差 [1,480],env wrapper 再加 base)。
- `update_actor_rft`(`q_agent.py:531`):算正常 RL actor loss + `bc_loss`;`loss = actor_loss_total +
  (bc_loss_coef * ratio * bc_loss).mean()`;`bc_loss_dynamic=False` 时 `ratio=1`(动态块 566-589 整段跳过,
  不碰 `q_value_for_policy`/ref_agent)。`assert actor_loss_total.size()==bc_loss.size()`:两者都是标量,bc_batch
  尺寸无需匹配 RL batch。
- `update`(`q_agent.py:617`):`bc_batch is None` -> `update_actor`(现状);否则 `update_actor_rft`,且
  `assert ref_agent is not None`。
- offline_rb(`train_chunk_residual.py:287`)在 `offline_fraction>0` 时存在,`.sample()` 给的 TensorDict 结构
  = obs(图/state/base_action/stage_id)+ action + next,正好可当 bc_batch(`_compute_actor_bc_loss` 会自己
  `_encode` 出 feat)。offline demo 是 GT-as-base:`obs.base_action == action`(scaled GT)->残差 target=0。

## 4. 设计

### 4.1 残差 BC loss(落点 `q_agent.py:_compute_actor_bc_loss`)

- 删 `assert not self.residual_actor`。
- target 抽纯函数 `bc_target(action, base_action, residual_actor)`:残差时 `action - base_action`,否则 `action`。
- 改后:

```python
def _compute_actor_bc_loss(self, batch, *, backprop_encoder):
    obs = batch["obs"]
    assert "feat" not in obs, "safety check"
    obs["feat"] = self._encode(obs, augment=True)
    if not backprop_encoder:
        obs["feat"] = obs["feat"].detach()
    pred_action = self._act_default(obs=obs, eval_mode=False, stddev=0, clip=None, use_target=False)
    target = bc_target(batch["action"], obs["observation.base_action"], self.residual_actor)
    loss = nn.functional.mse_loss(pred_action, target, reduction="none").sum(1).mean(0)
    return loss
```

`bc_target` 放 `q_agent.py` 模块顶部(与 `_compute_actor_bc_loss` 同文件;纯函数,可 light 单测):

```python
def bc_target(action, base_action, residual_actor):
    """残差 actor 的 BC 目标 = 采取动作 − 基座动作;非残差 actor = 采取动作(行为不变)。"""
    return action - base_action if residual_actor else action
```

语义:offline GT demo(base_action==action)-> target=0 -> 把残差在 demo 流形上锚向 0;②b 在线 relabel 段 ->
target=实际采取的残差。同公式两用。

### 4.2 接线(`train_chunk_residual.py`)

- 新 CLI:`--demo_bc_coef`(float,默认 0.0;>0 开启 demo-BC,值即 BC 权重)。
- 建 agent 前:`cfg.agent.bc_loss_coef = args.demo_bc_coef`;`cfg.agent.bc_loss_dynamic = False`。
- 解析与依赖断言:`demo_bc_coef>0` 时 assert `offline_fraction>0`(bc_batch 取自 offline_rb)且 assert `--actor raw`。
- 主循环 update 调用改为:

```python
    bc_batch = None
    if args.demo_bc_coef > 0 and update_actor and offline_rb is not None:
        bc_batch = offline_rb.sample(args.batch_size).to(args.device, non_blocking=True)
    m_upd = agent.update(batch, args.stddev, update_actor,
                         bc_batch=bc_batch, ref_agent=(agent if bc_batch is not None else None))
```

(ref_agent 传 agent 满足 update 的非空断言;bc_loss_dynamic=False 时它不被用到。)

### 4.3 默认关 = baseline 逐位等价

`--demo_bc_coef 0`(默认)-> bc_batch 恒 None -> `update(bc_batch=None)` -> 走 `update_actor`(非 rft)->
与现状一字节不差。单测必钉。

### 4.4 作用域 / 正交性

- 仅 `--actor raw`(加 assert);flow 延后。
- bc_loss_dynamic 保持 False。
- 与 ① stage_budget 正交(budget 缩输出包络;demo-BC 拉残差向 target)。注意叠加时:stage_budget 开则 pred 是
  budget 缩后的残差,BC 把"缩后残差"拉向 target;对 demo target=0 无碍。与 critic / reward / buffer 无关。

### 4.5 特权口径

②a 用的是 GT demo(无特权 stage 依赖,demo 本就是数据集)。②b 的在线 relabel 才会用特权 stage 判定"到达
stage K"(与 reward shaping 同口径,仿真零成本)。本 spec 不涉及。

## 5. 测试方案(TDD)

- `bc_target` 纯函数(light):残差时 == action−base_action、非残差时 == action;shape 不变。
- manual 集成(VitEncoder,CPU):QAgent(residual_actor=True) 带 bc_batch 跑 `update(bc_batch=...)`:
  actor_loss / bc_loss 有限、metrics 含 `rft/bc_loss`;对 demo(action==base_action)BC target≈0。
- manual 回归:`residual_actor=False` 的 QAgent 走 `_compute_actor_bc_loss` 仍用 target=action(不破坏原非残差路径)。
- manual 默认关:`update(bc_batch=None)` 仍走 update_actor(无 rft/bc_loss metric)。
- CLI(light):`--demo_bc_coef` 默认 0.0、解析 float;`demo_bc_coef>0 + offline_fraction=0` 报错;
  `demo_bc_coef>0 + --actor flow` 报错。

## 6. A/B 验证方案(实现后,plan/run 阶段;不属本实现)

- 配置:nas10 强 base + best 基座(见 HANDOFF_2026-06-06 §3-4),`--offline_fraction 0.5` 必开(bc_batch 需 offline_rb)。
- A = `--demo_bc_coef 0`(baseline);B = `--demo_bc_coef`(扫 0.01 / 0.1)。
- 主指标:eval success / `eval_stage_reach` per-stage;理想是 demo 锚让残差更受控、stage3 回退降、不掉早期 reach。
- 旁证:wandb `rft/bc_loss`(应非零且收敛)。
- 定位:本发主要确认"消费端造好且 demo-BC 不砸";真正治稀疏靠 ②b(在线 relabel)。

## 7. 文件改动清单

- `resfit/rl_finetuning/off_policy/rl/q_agent.py` —— 模块顶部加纯函数 `bc_target`;`_compute_actor_bc_loss` 解 assert + 用 `bc_target`。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` —— `--demo_bc_coef` + config 设置 + 依赖断言 + update 调用接 bc_batch。
- 测试:`chunk_residual/tests/test_demo_bc.py`(新建:bc_target light + manual 集成/回归/默认关 + CLI)。

## 8. 风险与缓解

- demo-BC 把残差拉向 0,可能压制 RL 学习 -> 这是温和 anchor,靠 `--demo_bc_coef` 调强弱(扫小值);本就是
  default-off,baseline 不受影响。
- bc_batch 取自 offline_rb,要求 offline_fraction>0:已加 assert,提示清晰。
- 残差 actor 的 `_act_default` 返回残差(非 combined):已核实(主循环用法 + env wrapper 加 base 的约定);
  BC target 用 action−base_action 与之一致。
- 与 stage_budget 叠加:budget 缩后的残差被 BC 拉向 target,对 demo(target=0)无碍;若将来 ②b relabel
  + budget 同开,target 是"缩后实际残差",仍自洽(都在缩后空间)。
