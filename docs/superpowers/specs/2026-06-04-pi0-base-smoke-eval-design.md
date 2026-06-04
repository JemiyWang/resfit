# 设计文档：pi0 基座跨进程链路冒烟 eval（eval_pi05_base.py 第一版）

- 日期：2026-06-04
- 仓库：residual-offpolicy-rl（分支 `chunk-residual-validation`）
- 关联：
  - 主设计 spec：`docs/superpowers/specs/2026-06-04-pi05-base-policy-switch-design.md`
  - 主实现 plan：`docs/superpowers/plans/2026-06-04-pi05-base-policy-switch.md`（本脚本是 Task 10 的第一版）
  - 交接：`docs/superpowers/2026-06-04-pi05-base-policy-switch-HANDOFF.md`

---

## 1. 目的与范围

验证新增的跨进程基座链路在真实任务上跑通：

```
serve(官方 pi0_aloha_sim) -> websocket -> Pi05PolicyAdapter -> three-piece dexmg env
```

这版只做**冒烟**（管线健全性），不做成功率评估。脚本即 plan Task 10 的 `eval_pi05_base.py`
第一版，后续在同一文件上长成正式基座 gate（加成功率统计 / 阈值判定）。

**为什么先冒烟**：websocket 跨进程是本期最新、最高风险的部分。先用零微调的官方 ckpt 把链路跑通，
再投入数据对齐 + pi0 微调。官方 aloha 基座在 three-piece 上成功率必然约 0，看了误导，故本版不看成功率。

**起步选型（已定）**：任务 three-piece（`ankile/dexmg-two-arm-three-piece-assembly`，本地已缓存 84x84）；
基座 pi0（冒烟用官方 `pi0_aloha_sim`）；分辨率走甲（直接用 84，openpi 内部 ResizeImages 放大到 224）。

## 2. 关键事实（已核实，2026-06-04）

- three-piece：action 14 维（双臂各 7）、state 18 维、3 视角
  （`observation.images.agentview` / `robot0_eye_in_hand` / `robot1_eye_in_hand`，均 84x84）。
- 三视角正好对 pi0 三槽位 base / left_wrist / right_wrist，不用补零图。
- pi0_aloha_sim 也是双臂 14 维 delta eef -> 冒烟时维度与控制方式正好对得上（语义不对无所谓）。
- 分辨率链路：env 渲染 84 -> ACT/残差网络吃 84；pi0 链路 env 给 84 -> server 端 openpi
  ResizeImages(224,224) 放大到 224 喂模型。adapter 不做 resize，发 84 原图即可。

## 3. 架构（三个可独立测的单元）

1. `check_action(arr, action_dim) -> None`（纯函数，raise on bad）
   - 断言：最后一维 == `action_dim`(14)、浮点、无 NaN/Inf、数值在合理界（|x| <= 阈值，默认 5，
     防对 delta action 误杀）。坏输入 raise，带清晰信息。

2. `run_smoke(env, base_policy, n_episodes, max_steps) -> SmokeReport`（循环骨架，env / base_policy 注入）
   - 每 episode：`env.reset()` + `base_policy.reset()`；逐 step
     `base_policy.select_action(obs) -> [B,14]` -> `check_action` -> `env.step(base_action)`（纯基座，无残差）。
   - 收集诊断：每步推理耗时、action min/max、episode 步数、异常定位。
   - 不判成功率。

3. `main()`（真环境装配，手动集成 smoke）
   - argparse：`--host` / `--port` / `--n_episodes` / `--max_steps`。
   - 从 `ResidualTD3ThreePieceAssemblyConfig` build 真 env（复用 train_residual_td3.py 的 env 创建段）。
   - 构造 `BasePolicyConfig(type="pi05", host, port, action_dim=14, image_key_map=三视角, prompt, execute_horizon)`。
   - `load_pi05_base_policy(cfg, device)` 连 server -> 调 `run_smoke` -> 打印报告。

## 4. 数据流

```
env.reset() -> obs dict
  -> base_policy.select_action(obs) -> action[B,14]
  -> check_action(action, 14)
  -> env.step(action)          # 纯基座，无残差叠加
  -> 循环至 episode 结束 / max_steps
```

## 5. 错误处理

- 连不上 server：捕获连接异常，明确提示“先按 plan Task 9b 在 GPU 端起 pi0 server”。
- 断言失败：报第几 episode / 第几 step / 哪条断言挂，附 action 值。
- `env.step` 异常：捕获并定位到 step，原样抛出栈。

## 6. 测试策略（TDD）

- 单测 `tests/rl_finetuning/test_eval_pi05_base.py`（快、无 GPU / 无 server / 无 mujoco）：
  - `check_action`：合法 / NaN / 错维度 / 超界，各一例。
  - `run_smoke`：喂 fake env（reset/step 返回固定 obs，N 步后 done）+ fake base_policy
    （select_action 返回固定 [B,14] chunk），验证跑满 N episode、诊断字段齐、坏 action 触发 raise。
- 手动集成 smoke：`main()` 连真 server + 真 env，GPU 上手动跑，不进 CI。

## 7. 文件放置

- 脚本：`resfit/rl_finetuning/scripts/eval_pi05_base.py`
- 测试：`tests/rl_finetuning/test_eval_pi05_base.py`
（跟现有布局走，不放仓库根目录。）

## 8. 实现第一步必须先核实的依赖

在写 `main()` 前先读代码确认，否则 image_key_map / env 装配会写错：

1. env wrapper 喂给 `base_policy.select_action` 的 obs 里 image key 的实际命名
   （LeRobot 式 `observation.images.xxx`，还是 robosuite raw 如 `agentview_image`）——
   决定 `image_key_map` 的 key 写法。
2. three-piece env builder 在 `train_residual_td3.py` 里的入口 / 函数签名。
3. `pi0_aloha_sim` 输出 action 维度 >= 14（预期 14，满足 adapter 截取）。

## 9. YAGNI（本版明确不做，留 Task 10）

- 成功率统计与 gate 阈值判定。
- 录视频 / 可视化。
- 多任务 / 多分辨率。
- pi05（非 pi0）路径——本版只跑 pi0 官方 ckpt。
