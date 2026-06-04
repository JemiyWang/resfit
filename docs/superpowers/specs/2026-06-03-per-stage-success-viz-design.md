# Per-stage 成功率可视化 + eval 阶段到达率探测

日期: 2026-06-03
分支: chunk-residual-validation
关联: HANDOFF_2026-06-03.md(stage-purity 诊断)、outputs_chunk/plot_stage_diag.py

## 背景与动机

`outputs_chunk/plot_stage_diag.py` 当前画 1x3:target_q / residual_norm / 整体
eval success_rate。诉求是在最右区域additionally展示"每个阶段的成功率"。

调研结论(数据现状):

- eval(`resfit/rl_finetuning/utils/evaluate_dexmg.py`)只算并返回整体
  `eval/success_rate` 和 `eval/mean_return`,**不按 stage 拆**。
- 唯一按 stage 拆的是训练 rollout 的 `[stage-purity]` 行(回退/掉件率),
  语义是"训练时各阶段稳定度",不是 eval 成功率。
- 训练只存 `best.pt`(无周期 checkpoint),故对已跑完的 run 重评只能得到
  **单一时间点快照**,不是随 env_steps 的曲线。

因此"每阶段成功率"拆成两种互补语义,分开呈现,避免混淆。

## 目标

1. 把 `plot_stage_diag.py` 改成 2x2,新增两块每阶段信息。
2. 给 eval 加"阶段到达率"(stage reach)探测,additive 不动现有指标。
3. 提供重评脚本,对 best.pt 产出 reach 快照,喂给绘图。

非目标(本轮):实际跑重评(需 GPU,由用户执行);改训练逻辑/奖励/critic。

## Deliverable 1 — plot_stage_diag.py 改 2x2

布局:

| 位置 | 内容 | 数据来源 |
|------|------|----------|
| (0,0) | target_q per stage(不变) | `[stage-diag]` |
| (0,1) | residual_norm per stage(不变) | `[stage-diag]` |
| (1,0) | 整体 success_rate(黑)+ 每阶段"未回退率"曲线 vs env_steps | `[stage-purity]` |
| (1,1) | best.pt 每阶段到达率(柱状图) | sidecar `<log>_reach.json` |

(1,0) 细节:

- 解析 `[stage-purity]` 行内每个 `stageK:a/b=P%`(a=regress 数, b=该桶总数)。
- 画"未回退率 = 1 - a/b"(越高越稳),与整体 success_rate 同 [0,1] 轴叠加。
- 按 `cur_step` 配对(与 stage-diag/eval 同步,沿用现有 cur_step 机制)。
- 边界:b==0 的桶 → 该点 NaN 跳过;stage0 通常恒 ~100%。

(1,1) 细节:

- 读 log 同名 sidecar `<log>_reach.json`(D2 重评脚本产出),结构
  `{"step": <int>, "n_episodes": <int>, "reach": {"1": f, ..., "K": f}}`
  (key 为 stage 序号 1..K,与 `stage_reach_rates` 返回域一致)。
- 文件不存在 → 画占位文字 "run re-eval to populate",不报错、不中断其余三图。

解析逻辑(stage-purity 行、reach json)抽成纯函数放新模块
`resfit/rl_finetuning/chunk_residual/stage_log_parse.py`(plot 脚本在 outputs_chunk/
非 package,故纯函数下沉到 package 才能被 pytest 导入)。

## Deliverable 2 — 独立 reeval 脚本(num_envs=1)

### 关键约束(调研发现,修订原方案)

`ChunkResidualEnvWrapper` 的 stage 闩锁是**单标量、只读 env0**
(`chunk_env_wrapper.py:150` `s = int(info["stage_id"][0])`,`self._stage` 标量,
:170 任一 env done 即归零)。因此**多环境(eval 用 8)下 stage 归属是错的**——
这正是 `eval_stage_reach.py` docstring 说的"多环境 stage 追踪坑",它用 `num_envs=1` 绕开。

结论:**不插桩共享的 `evaluate_dexmg.py`(8 env 会算错),改为独立 num_envs=1 脚本。**
原方案"改 evaluate_dexmg + train 打印 [stage-reach]"作废(8 env 下错;且训练 eval 也是 8 env)。

### 复用现成件(无需重写聚合)

- `stage_reach.py::stage_reach_rates(episode_max_stages, num_stages)` —— 已存在且有测试
  (`tests/test_stage_reach.py`),直接复用。
- `stage_detectors.py::NUM_STAGES[task]` —— 阶段数。
- wrapper 已透出 `info["max_stage_in_chunk"]`(:178)。
- `eval_stage_reach.py::run_for_norm` —— 单环境逐 episode 记最高 stage 的成熟范式
  (此脚本注入随机残差;reeval 改成用训练好的 agent 动作驱动)。

### reeval 脚本(放 chunk_residual/reeval_stage_reach.py,不放根目录)

- `torch.load(run_dir/best.pt, weights_only=False)`;`ckpt["config"]` 是训练时存的
  argparse Namespace(`save_checkpoint(config=args)`),含 task/dataset/action_scale/
  chunk_length/base_wandb_id/base_action_mode 等;`ckpt["global_step"]` 给 step。
- 据此重建:ActionScaler/StateStandardizer、base_policy(`build_base_policy`)、
  `num_envs=1` 的 `ChunkResidualEnvWrapper`(reward_shaping_mode="none")、
  `QAgent`(`agent.load_state_dict(ckpt["agent_state_dict"])`,`agent.train(False)`)。
- 循环:`agent.act(obs, eval_mode=True)` → `env.step` → `fold_episode_max_stage` 累计;
  done 时记录、`env.reset()`(同 eval_stage_reach),跑满 n_episodes。
- `stage_reach_rates` 汇总 → 写 `<log>_reach.json`:
  `{"step": int, "n_episodes": int, "reach": {"1": f, ..., "K": f}}`。
- 默认目标 run:`cl1_queue_potential`、`cl1_queue_staged_as0.05`(命令行参数化)。
- 自洽校验:reach[top_stage] ≈ 该 run 的 success_rate。
- 本轮只交付脚本,不实际运行(GPU/conda env residual/MUJOCO_GL=egl 由用户执行)。

### fold_episode_max_stage(纯函数,可单测)

抽到新模块 `stage_log_parse.py`(与 plot 解析共址):
`fold_episode_max_stage(prev_max, max_stage_in_chunk, reward, top_stage) -> int`
= `max(prev_max, max_stage_in_chunk)`,若 `reward>=1.0` 则再 `max(.., top_stage)`。
封装"取最高 + 成功置顶阶段"逻辑,reeval 循环调用。

## 测试(TDD)

新模块 `stage_log_parse.py` 的三个纯函数,单测放
`resfit/rl_finetuning/chunk_residual/tests/test_stage_log_parse.py`:

- `parse_stage_purity_line(line)`:真实行解析、b==0 桶跳过、非 purity 行返回 {}。
- `load_reach_sidecar(path)`:存在(正确还原 step/reach)、缺失返回 None。
- `fold_episode_max_stage(...)`:取最高、成功置顶、reward<1 不动。

`stage_reach_rates` 已有 `tests/test_stage_reach.py`,不重复。
plot 渲染与 reeval 编排不单测:plot 渲染由"在真实 log 上跑出 png"验证(无需 GPU);
reeval 编排由用户在 GPU 上 `--smoke`/实跑验证(本轮只交付)。

## 风险 / 注意

- stage-purity 是训练 rollout 量,(1,0) 必须明确标注"未回退率/training",勿误读成 eval 成功率。
- reach 快照只是 best.pt 单点,(1,1) 标题须标 step,勿误读成最终/平均。
- 改 evaluate_dexmg.py 是公共 eval 函数,additive 改动须保证不破坏其它调用方
  (train_residual_td3 等)——只增 key、stage_id 缺失时静默跳过。
