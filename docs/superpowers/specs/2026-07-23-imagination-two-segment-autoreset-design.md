# Imagination 两段视界与 SAME_STEP Autoreset 设计

日期：2026-07-23  
范围：仅修复 `resfit.rl_finetuning.wm_bridge` 的 imagined episode 生命周期；不改变
chunk-level residual、PBRS reward、TD3 算法或世界模型接口。

## 1. 背景与根因

当前 `ImaginationVecEnv` 在每次世界模型点火后执行：

```text
seg_step += 1
truncated = (seg_step >= max_segments)
```

配置 `max_segments=2` 时，第二段开始返回 `truncated=True`，但该环境不会自动
`reset()`。外层 `ChunkResidualEnvWrapper` 只重置 base policy 与 stage，并假设底层
vector env 已经 autoreset；trainer 则无条件执行 `obs = next_obs`。

因此当前实际序列是：

```text
WM1: real s0 -> imagined s1, truncated=False
WM2: s1 -> imagined s2, truncated=True
WM3: s2 -> imagined s3, truncated=True
WM4: s3 -> imagined s4, truncated=True
...
```

`max_segments` 只改变 transition 标签，没有限制世界模型递归长度，导致生成图像、
proprio、base action 与 residual action 的闭环误差持续累积。

## 2. 选定行为

采用保守的两段有限视界，语义对齐原始 RISE：

```text
真实起点A
  -> WM1 -> imagined s1
  -> WM2 -> imagined s2
  -> 保存第二条有限视界 transition
  -> SAME_STEP autoreset 到真实起点B
  -> WM计数从0重新开始
```

硬约束：

1. 每个真实起点最多触发 `max_segments` 次世界模型推理；当前为2。
2. 第二段的 PBRS reward 必须使用 imagined 终点 `s2` 计算，不能用 reset 后的新起点。
3. 第二段保持 `truncated=True`，沿用当前 `done = terminated | truncated`，不 bootstrap。
4. 返回给外层 wrapper 的 observation 是 reset 后的新真实起点，满足其 SAME_STEP
   autoreset 假设。
5. 世界模型预测帧收集发生在 reset 前，评估仍能收齐两段共50张预测帧。
6. 下一次外层 `step()` 必须是新 episode 的第1段，返回 `truncated=False`。

## 3. 实现边界

修改 `wm_bridge/imagination_env.py`：

1. 第50个底层动作触发世界模型，构造 imagined 终点 observation。
2. 使用 imagined 终点查询 kai0 特征并计算：

   ```text
   chunk_reward = gamma * Phi(imagined_endpoint) - Phi(previous_endpoint)
   ```

3. 增加 `seg_step` 并判断是否达到 `max_segments`。
4. 未达到边界时，保存 imagined 终点为当前状态并正常返回。
5. 达到边界时：
   - 保留 imagined 终点供 reward 与诊断使用；
   - 调用 `reset()`，重新采样真实窗口、proprio 与初始势函数；
   - 返回 reset observation、原 chunk reward、`terminated=False`、
     `truncated=True`；
   - 在 info 中提供 imagined 终点作为 `final_observation`，避免边界信息丢失。

不修改：

- `ChunkResidualEnvWrapper` 的逐步执行逻辑；
- trainer 的 `done` 与 TD target 计算；
- `max_segments` 默认值2；
- `n_step=1`；
- PBRS 截断端点不置零的约定；
- 原始 RISE 仓库。

## 4. Replay 与 TD 语义

第一条 transition：

```text
s0 -> s1
truncated=False
target = reward0 + gamma * Q_target(s1, next_action1)
```

第二条 transition：

```text
s1 -> reset_observation
truncated=True
target = reward1
```

第二条的 `next_obs` 是下一真实 episode 的起点，但不会进入 target Q，因为当前保守方案
对 `truncated` 不 bootstrap。真正的 imagined 终点 `s2` 保存在
`info["final_observation"]`，并已用于计算 `reward1`。

未来若评估 bootstrap-on-truncation，必须单独设计 replay 对 final observation 的存储与
bootstrap mask；本次不加入该开关，避免把未训练的边界 Q 外推引入第一版修复。

## 5. 测试设计

新增或扩展回归测试，锁定以下行为：

1. 内层环境的截断序列：

   ```text
   False, True, False, True
   ```

2. 对应世界模型调用序列为：

   ```text
   WM1, WM2, reset, WM1, WM2, reset
   ```

3. 第二段结束后：
   - `seg_step == 0`；
   - sampler调用次数增加；
   - 返回 observation 的 window token 属于新真实起点；
   - `info["final_observation"]` 属于第二段 imagined 终点。

4. 第二段 reward 仍为：

   ```text
   gamma * Phi(s2) - Phi(s1)
   ```

   reset 起点的势函数不得污染该 reward。

5. 经 `ChunkResidualEnvWrapper` 调用四次时，每次外层调用只触发一次世界模型，
   截断序列仍为 `False, True, False, True`。

6. 既有 window、PBRS telescoping、预测帧收集和 base query 缓存测试继续通过。

## 6. 验收标准

- 连续执行任意数量的训练 chunk，不存在超过两段的 generated-to-generated 链。
- 每两次世界模型调用至少发生一次真实数据采样。
- 第二段之后的第一条新 transition 重新从真实起点开始。
- 核心 wm_bridge 与 chunk wrapper 回归测试全部通过。
- 不启动真实训练、不修改用户已有训练产物或无关工作树文件。
