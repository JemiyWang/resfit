# LIBERO eval 路在线子目标注入 设计

> 状态:待实现(brainstorm 已定稿)。范围聚焦、单一实现计划可覆盖。

## 背景与问题

pi0_feat 在 LIBERO 上的残差 RL **训练**管线已完备:离线 buffer 存 `observation.subgoal`、在线训练采集 `compute_online_subgoal`(`train_chunk_residual.py:914/921`)均已落地并跑过 smoke。但带 `--subgoal_conditioned` 的 LIBERO 跑**一进 eval 就崩**:

```
File ".../libero_eval.py", line 46, in run_libero_evaluation
    actions = agent.act(obs, eval_mode=True, ...)
File ".../off_policy/rl/actor.py", line 190
    all_input.append(obs["observation.subgoal"].to(feat.device))
KeyError: 'observation.subgoal'
```

根因(4 条证据):

1. **`actor.py:189-190`**:`if self.subgoal_conditioned: all_input.append(obs["observation.subgoal"]...)` —— 只要 actor 是 subgoal-conditioned,**训练和 eval 无差别**地要这个 key,无 train/eval 门控。
2. **`libero_eval.py` 全 git 历史只有 1 个 commit**(`29e4d7e`,2026-06-14),创建时就**故意**不带 subgoal —— commit message 原话 "without ... the dexmg-only Q-plots/video/**subgoal**"。
3. **时间线**:`run_libero_evaluation` 是 6-14 建的;pi0_feat 在线子目标(`compute_online_subgoal`/`last_prefix_feat`)是 6-15 才接进**训练**的。eval 评测器早于 subgoal 功能一天诞生,后续接 subgoal 时只改了训练采集路,**没回头补 eval 路**。
4. **eval 在 `env_steps >= next_eval`(`next_eval` 初值 0)几乎立刻触发** → subgoal-conditioned 的 LIBERO 跑必然一进 eval 就 KeyError。

对照:dexmg 那条 eval 路是**有**子目标注入的 —— `run_dexmg_evaluation(...)` 收 `subgoal=/base_policy=`,内部 `_eval_inject_subgoal(subgoal, obs, base_policy, rel_raw)`(`evaluate_dexmg.py:20`)在每次 `agent.act` 前给 `obs`、`env.step` 后给 `next_obs` 注入。LIBERO 这条是缺失项。

## 目标

让 `run_libero_evaluation` 在 `--subgoal_conditioned` 下与 dexmg eval 行为对称地注入 `observation.subgoal`,使 eval 跨过 KeyError;`subgoal=None` 时与现状**逐字节一致**(零回归)。

## 范围

**做**:① `run_libero_evaluation` 加 `subgoal=/base_policy=` 参数 + libero 本地注入小函数;② `train_chunk_residual.py` 的 libero eval 调用处传 `subgoal`/`base_policy`;③ 单元 + 回归测试;④ 集成 smoke 验证(人工,不入 pytest)。

**不做**:dexmg eval 路的任何改动;`_eval_inject_subgoal` 抽共享模块(本次按"libero 本地小函数"落地,避免 `libero_eval.py` import `evaluate_dexmg` 而被 robosuite-1.5 拖崩);非 pi0_feat 的 LIBERO 子目标(eef_piece 需 mujoco replay、act_feat 未在 libero 子目标 eval 上接,均留后续)。

## 架构

改 2 个文件。

### 1. `resfit/rl_finetuning/chunk_residual/libero_eval.py`

新增 libero 本地注入函数(**不** import `evaluate_dexmg`,保持本模块 import 干净):

```python
def _inject_subgoal(subgoal, obs, base_policy):
    """LIBERO eval 子目标注入。对称 evaluate_dexmg._eval_inject_subgoal,
    但 LIBERO 无 rel_piece,且当前只支持 pi0_feat(其它 state_mode fail-fast)。"""
    if subgoal.state_mode == "pi0_feat":
        return subgoal.subgoal_online(obs, prefix_feat=base_policy.last_prefix_feat())
    raise NotImplementedError(
        f"LIBERO eval 子目标注入目前只支持 state_mode=pi0_feat,得到 {subgoal.state_mode!r}")
```

`run_libero_evaluation` 加两个**关键字、默认 None** 的参数,并在两处注入:

```python
def run_libero_evaluation(*, env, agent, num_episodes, device="cpu",
                          subgoal=None, base_policy=None) -> dict:
    if subgoal is not None and base_policy is None:
        raise ValueError("subgoal!=None 时必须传 base_policy(读 last_prefix_feat)")
    ...
    obs, _ = env.reset()
    if subgoal is not None:
        obs["observation.subgoal"] = _inject_subgoal(subgoal, obs, base_policy)   # 首帧
    while done_episodes < num_episodes:
        with torch.no_grad():
            actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
        next_obs, reward, terminated, truncated, _info = env.step(actions)
        if subgoal is not None:
            next_obs["observation.subgoal"] = _inject_subgoal(subgoal, next_obs, base_policy)  # 续帧
        ...
        obs = next_obs
```

其余 rollout/统计逻辑**完全不变**。`subgoal=None` 时不进任何注入分支 → 与现状逐字节一致。

### 2. `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(约 991-994 行)

libero 分支的 eval 调用补传 2 参:

```python
if args.env_family == "libero":
    from resfit.rl_finetuning.chunk_residual.libero_eval import run_libero_evaluation
    m = run_libero_evaluation(env=eval_env, agent=agent,
                              num_episodes=args.eval_num_episodes, device=args.device,
                              subgoal=(subgoal if args.subgoal_conditioned else None),
                              base_policy=eval_base_policy)
```

## ⚠️ 核心正确性命门:传 `eval_base_policy` 而非 `base_policy`

LIBERO 用 `--base_action_mode queue`,`train_chunk_residual.py:651` 给 eval 单独建了 `eval_base_policy` 实例(queue 有状态,eval 多 env 与训练单 env 共享会互踩),`eval_env` 用的就是它(663 行)。`last_prefix_feat()` 返回"最近一次 base 推理"的 prefix 特征 —— 只有读 **eval_env 真正在跑的那个实例**,特征才对应当前 eval 的 obs;若误传训练用的 `base_policy`,读到的是张冠李戴的陈旧特征,子目标 z 全错且**静默**(不报错,只是评测数字失真)。故调用处**必须** `base_policy=eval_base_policy`。

(旁注:dexmg 路 `evaluate_dexmg` 调用传的是训练 `base_policy`,在 queue+pi0_feat 下存在同类陈旧风险;但 pi0_feat 不在 dexmg 上跑,不在本次范围,仅记录。)

## 数据流 / 时序

注入用的 `last_prefix_feat` 仅在 base queue 重规划那帧更新(queue 模式 base 每 `pi0_execute_horizon` 步重推一次),续帧复用上次特征。这与**训练采集路**(`train:914` 注入 `obs`、`921` 注入 `next_obs`)、**dexmg eval** 完全同款时序。设计**不另造语义**,靠"与训练同源"保证 eval 用的 z 口径和离线/在线训练一致 —— 这正是 pi0_feat 子目标 quality gate 成立的前提。

首帧(reset 后)注入依赖 `eval_base_policy.last_prefix_feat()` 已被 reset 时的 base 重规划填充(queue 模式 reset 即 prime 队列 → 触发一次 base infer)。与训练循环 `obs, reset_info = env.reset()` 后紧接 `914` 行注入是同一前提;集成 smoke 覆盖此。

## 错误处理(fail-fast)

- `subgoal!=None` 且 `base_policy is None` → `ValueError`(早抛,别退化成晦涩 AttributeError)。
- `subgoal.state_mode != "pi0_feat"` → `NotImplementedError`(LIBERO 当前只支持 pi0_feat 子目标)。

## 测试

**单元(入 pytest,`tests/test_libero_eval.py`,现有 5 个测试)**:
- `subgoal!=None`:stub `agent`/`env`/`subgoal`/`base_policy`,跑 1-2 episode,断言**每次** `agent.act` 收到的 obs 含 `observation.subgoal`(形状 = `rep_dim`)。用一个记录入参的 stub agent 捕获 obs。
- `base_policy=None` 而 `subgoal!=None` → `pytest.raises(ValueError)`。
- `state_mode != "pi0_feat"` → `pytest.raises(NotImplementedError)`。
- **回归**:`subgoal=None` 路径与现状一致 —— 现有 5 个 libero_eval 单测全绿、且 obs 不含 `observation.subgoal`。

**集成(人工,不入 pytest)**:重跑 task3 那条 subgoal-conditioned smoke(serve 已在、三件套已就绪、offcache 可秒复用),确认这次**跨过 eval**(不再 KeyError)、注入的 z 有限、`best.pt` 落盘、打印 `eval success_rate`。

## 成功标准

1. 新单元测试全过;现有 5 个 libero_eval 单测零回归;聚焦回归(libero_eval + train_chunk_residual 相关)全绿。
2. `subgoal=None` 路径逐字节等价(diff 仅新增分支,旧路径不变)。
3. 集成 smoke 跨过 eval,无 `KeyError: 'observation.subgoal'`,z 有限。
4. 跨过 eval 即解锁:去掉 `--smoke`、加 `--total_env_steps`、复用 offcache 的**正式** LIBERO pi0_feat 残差训练。

## 相关

- 训练采集路对称参照:`train_chunk_residual.py:176 compute_online_subgoal`、`914/921` 注入。
- dexmg eval 参照:`resfit/rl_finetuning/utils/evaluate_dexmg.py:20 _eval_inject_subgoal`。
- 缺口由来:`libero_eval.py` commit `29e4d7e`(早于 pi0_feat 子目标一天)。
