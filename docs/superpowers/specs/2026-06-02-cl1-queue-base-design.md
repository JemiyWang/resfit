# 设计:chunk_residual 的 queue 基座模式(cl=1 复刻原版 step 级残差)

日期:2026-06-02
分支:chunk-residual-validation
状态:已批准设计,待写实现计划

## 1. 背景与动机

chunk_residual 当前在 chunk_length=20 下早期塌(见 `HANDOFF_2026-06-02_staged-reward.md`)。
per-stage 诊断锁定根因:残差漂离 identity + **chunk-20 开环**把残差误差放大成基座破坏,
尤其摧毁最终精插入(stage3→4)。与奖励信号无关。

诊断结论指向一个干净的对照实验:**回到 step 级(每步闭环修正)看能否复活**。但有两个前提:
1. 原版单步残差(`train_residual_td3.py` + `residual_env_wrapper.py`)当年死在**稀疏奖励**
   (critic≡0),不是 step 修正本身不行。现在有了 staged 稠密奖励(已证明能救活 critic),
   "step 级 + staged 奖励"是一个**从没真跑过的新组合**。
2. chunk_env_wrapper 在 cl=1 时**不等于**原版:原版基座走 ACT 原生 50 步 action queue
   (开环执行 chunk、每 50 步重规划);chunk wrapper 的 `get_action_chunk` 每步重跑模型只取
   第一步(receding-horizon,horizon=1)。两者基座轨迹不同,后者可能 OOD/更抖。

本设计:让 chunk_residual 在 cl=1 时**复刻原版基座行为**(走 queue),同时保留 chunk 管线已有的
staged 奖励 / stage 机制,从而得到"step 级 + staged 奖励"的最干净版本。

## 2. 目标 / 非目标

目标:
- cl=1 + queue 基座 + staged 奖励,可一键跑起,作为 step 级诊断基线。
- 基座行为在 cl=1 下与原版 `residual_env_wrapper.py` **完全等同**(含不 clamp combined 动作)。
- cl>1 的现状(replan 路径 + 全部现有测试)零影响。

非目标:
- 不改 cl>1 的基座(范围仅 cl=1;queue 模式 assert chunk_length==1)。
- 不引入 offline demo buffer(保持纯 online;offline 是单独议题)。
- 不做 chunk_length 全扫描的解耦基座版本(YAGNI,本轮只要 cl=1 对照点)。

## 3. 方案(路子 A:base_action_mode 参数)

### 3.1 改 `chunk_env_wrapper.py`
- 构造新增 `base_action_mode: str = "replan"`。
  - `mode == "queue"` 时 assert `chunk_length == 1`(queue 每次只吐一个动作)。
- `_base_chunk_flat`:
  - `replan`(默认):保持原 `get_action_chunk(base_policy, obs, chunk_length)` 路径,**完全不变**。
  - `queue`:调 `self.base_policy.select_action(raw_obs)` 拿 [B, D] → `action_scaler.scale` →
    reshape [B, D](= L*D,L=1)。走 ACT 原生 queue。
- `step` 内 combined 动作:`queue` 模式**不 clamp**(`combined = base + residual`),与原版一致;
  `replan` 模式保持现有 `clamp(base+residual, -1, 1)` 不变。
  - 实现上以 mode 分支,避免影响 replan 路径。
- 其余不动:done 时 `base_policy.reset()` 已清 queue;staged 奖励、stage_id、stage-balanced、
  buffer 字段(含 max_stage)照旧。

### 3.2 改 `train_chunk_residual.py`
- 新增 `--base_action_mode {replan, queue}`(默认 `replan`)。
- 透传给训练 env 与 eval env(两者同 mode,保持一致)。
- **eval 用独立 base_policy 实例**(queue 模式必须):queue 有状态(per-env action queue),
  训练 env(num_envs=1)与 eval env(num_envs=8)共享同一 base_policy 会互相踩 queue
  (eval 跑完留下的 queue 污染训练的开环 chunk 连续性,且 `_ensure_action_queues` 在 1↔8 间
  trim/grow)。对齐原版 `train_residual_td3.py` L265-270:再 `build_base_policy` 一个
  `eval_base_policy` 给 eval_env。
  - replan 路径无状态,共享本无害;为最小改动,**仅在 `base_action_mode=="queue"` 时**额外建
    eval 实例,replan 仍共享(行为/显存不变)。

### 3.3 运行配置
```
--chunk_length 1 --base_action_mode queue --reward_shaping staged --stage_balanced
```
外加 handoff 踩坑清单:各 run 独立 `--output_dir`;`conda run --no-capture-output` + `PYTHONUNBUFFERED=1`;
哨兵盯真 python PID。

## 4. 与原版的差异清单(改完后仍存在的,均已确认接受)
- staged 奖励**开着** —— 本次目的,故意不同(原版无 shaping)。
- `stage_id` 进 obs、`max_stage` 进 buffer:多出来但**不喂策略网络**(actor 输入仅 state+base_action),
  仅供 stage-balanced 采样 / 诊断,无害。
- 无 offline demo buffer:原版有,本次范围外。
- clamp:queue 模式下**已去掉**,与原版一致(本设计的决定)。

## 5. 测试(TDD)
新增 queue 模式单测(用带 `select_action` + 内部 queue 的 fake 基座):
1. cl=1 queue 模式下,wrapper 取基座动作走的是 `select_action`,而非 model 重跑(用 spy/计数验证)。
2. 连续两步从**同一次规划**的 queue 顺序取动作(第 2 步不触发重规划),证明是开环 chunk 执行。
3. done 之后 queue 被清空并重填(下一 episode 从新规划起步)。
4. `base_action_mode="queue"` 且 `chunk_length>1` 触发 assert。
5. queue 模式 combined 动作**不被 clamp**(给一个会超出 [-1,1] 的 base+residual,验证透传)。
现有 replan 路径全部测试保持绿。

## 6. 验收标准
- 全部单测通过(现有 + 新增 queue 测试)。
- cl=1 + queue + staged 能正常启动并产出 eval 曲线(冒烟:`--smoke`)。
- 行为核对:queue 模式下基座动作序列与直接调原版 `select_action` 一致(同 seed/同 obs 流)。
