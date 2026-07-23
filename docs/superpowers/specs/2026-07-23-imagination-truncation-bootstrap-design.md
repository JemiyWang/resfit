# Imagination 截断边界 Bootstrap 设计

日期：2026-07-23

状态：已确认，待实现

## 1. 背景

RISE_Hi 当前将一次世界模型推理视为一条 chunk transition。每个 chunk 包含 50 个实际动作，世界模型最多连续推理两段，以限制生成误差累积。

第二段结束时，`ImaginationVecEnv` 会设置 `truncated=True` 并执行 SAME_STEP autoreset：环境把第二段的终点状态 `s2` 放进 `info["final_observation"]`，同时把 reset 后的新真实状态 `B` 作为 `step()` 的返回 observation。

当前训练代码把 `terminated | truncated` 直接写成 replay buffer 的 `done`，并把返回的 observation `B` 写成 `next_obs`。因此第二条 imagined transition 实际为：

```text
(s1, action_chunk2, reward2, B, done=True)
```

这会产生两个问题：

1. 第二段虽然生成了有效终点 `s2`，TD3 却不从 `s2` bootstrap。
2. replay 中的 `next_obs` 是下一条 imagined episode 的 reset 状态 `B`，不是当前动作真正到达的 `s2`。

这里的 `truncated=True` 仅表示“停止继续使用世界模型向前生成”，不表示任务在 `s2` 真实终止。因此，它应结束 imagined episode 的控制流程，但不应让 TD target 终止。

本设计取代
`2026-07-23-imagination-two-segment-autoreset-design.md`
中“第二段 truncation 不 bootstrap”的 TD 边界语义；两段推理上限和 SAME_STEP autoreset 行为保持不变。

## 2. 目标

第二段世界模型推理结束后：

- 不再从 `s2` 继续进行第三次世界模型推理；
- 环境立即 reset，并让外层控制循环从真实状态 `B` 开始下一条 imagined episode；
- replay buffer 保存当前动作真正到达的终点 `s2`；
- replay 中该 transition 的 `done=False`，使 TD3 从 `s2` bootstrap；
- `s2` 中包含完整且一致的 `base_action`、标准化 state 和 stage 信息；
- 不额外调用一次 kai0 / π0.5 服务来获得 `s2` 的基础动作。

## 3. 核心语义：控制边界与价值边界分离

一次 `step()` 同时产生两个不同用途的 next observation：

```text
世界模型第二段：
    s1 --combined_chunk2--> s2
                               |
                               +-- replay_next_obs = s2
                               |
                               +-- 达到两段上限，立即 reset
                                         |
                                         +-- control_next_obs = B
```

对应的两个结束标志为：

```text
episode_done   = terminated OR truncated
replay_done    = terminated
```

以上 `replay_done` 规则只应用于带有明确 imagination bootstrap 标记的 truncation。普通真实环境的 terminated/truncated 处理保持原样。

第二条 replay item 应为：

```text
obs       = s1
action    = combined_chunk2
reward    = reward2
next_obs  = s2
done      = False
```

其中：

```text
reward2 = imagination_gamma * Phi(s2) - Phi(s1)
```

TD3 现有 target 计算无需修改。由于 replay 中 `done=False`，`MultiStepTransform(n_steps=1)` 会产生 `nonterminal=1`，最终 target 为：

```text
next_residual = actor_target(s2)
next_combined = clamp(base_action(s2) + next_residual)

target2 =
    reward2
    + gamma * min(
        Q1_target(s2, next_combined),
        Q2_target(s2, next_combined)
      )
```

世界模型是否继续 rollout 与 critic 是否从边界状态 bootstrap 是两个独立问题。两段上限用于控制模型误差，不应被解释成 MDP terminal。

## 4. 数据流设计

### 4.1 `ImaginationVecEnv`

第二段世界模型推理得到 `endpoint_obs=s2` 后，环境已经需要查询 base policy 来计算 potential scorer 使用的 `psi`。这次查询同时返回 `s2` 的基础动作 chunk。

环境应在 reset 之前保存：

```text
info["final_observation"]       = raw s2
info["_wm_final_base_chunk"]    = base_chunk(s2)
info["bootstrap_on_truncation"] = True
```

随后保持现有 SAME_STEP autoreset：

```text
returned observation = reset 后的真实状态 B
terminated           = False
truncated            = True
```

`_wm_final_base_chunk` 使用物理动作空间中的原始 chunk。它由 wrapper 统一执行 action scaling，避免环境层与训练层使用不同的缩放逻辑。

因为 `base_chunk(s2)` 来自计算 `Phi(s2)` 时已经发生的查询，所以该设计不增加一次 kai0 / π0.5 serve。

### 4.2 `ChunkResidualVecWrapper`

wrapper 收到 `bootstrap_on_truncation=True` 时，在清理当前 episode 的 stage 状态之前，将 raw `s2` 转成训练所需的完整 augmented observation：

1. 读取 `_wm_final_base_chunk`；
2. 使用现有 `action_scaler` 缩放并展平基础动作；
3. 使用当前 episode 的 stage 信息调用现有 observation augmentation；
4. 将 augmented `s2` 写回 `info["final_observation"]`。

wrapper 仍然对 reset 状态 `B` 查询并附加它自己的 base action，作为 `step()` 返回的 control observation。这样：

```text
info["final_observation"] = augmented s2
step() 返回 observation   = augmented B
```

两者不会混淆，且 actor target 在 `s2` 上能够读取与 `s2` 对应的 base action。

### 4.3 训练循环

训练循环增加一个纯数据选择 helper，将环境返回值拆成控制语义和 replay 语义：

```text
resolve_replay_transition(
    control_next_obs,
    terminated,
    truncated,
    info,
) -> (replay_next_obs, replay_done)
```

规则如下：

```text
若 bootstrap_on_truncation=True：
    replay_next_obs = info["final_observation"]
    replay_done     = terminated

否则：
    replay_next_obs = control_next_obs
    replay_done     = terminated OR truncated
```

主循环使用：

```text
episode_done = terminated OR truncated
```

处理 imagined episode 统计、harvester flush、finetuner episode end 等控制边界；仅在写 replay buffer 时使用 `replay_next_obs` 和 `replay_done`。循环下一轮的 `obs` 始终设为 `control_next_obs`，即 reset 后的 `B`。

### 4.4 QAgent 与 replay transform

`QAgent` 不需要修改。它当前已经按以下方式构造 target：

```text
effective_discount = gamma * nonterminal
target = reward + effective_discount * target_q
```

只要第二段 transition 在 replay 中保存 `done=False`，现有 replay transform 就会生成 `nonterminal=1`，自然启用 `s2` bootstrap。

## 5. 契约与错误处理

为避免静默训练错误，以下情况直接报错：

- `bootstrap_on_truncation=True`，但缺少 `final_observation`；
- `bootstrap_on_truncation=True`，但缺少 `_wm_final_base_chunk`；
- 同一 transition 同时出现 `terminated=True` 和 imagination bootstrap 标记；
- `max_segments <= 0`；
- imagination 模式启用 `subgoal_conditioned=True`。

最后一项暂时禁止，是因为 reset 后状态 `B` 与终点状态 `s2` 可能需要不同 subgoal。当前设计只保证 `s2` 的 base action、state 和 stage augmentation 正确；在补齐终点 subgoal 传递协议前，禁止该组合比静默使用错误 subgoal 更安全。

## 6. 兼容性范围

保持不变：

- 世界模型最多连续推理两段；
- 第二段之后 SAME_STEP autoreset；
- 每段 50 个实际动作；
- PBRS reward 及其 telescoping 关系；
- 普通真实环境的 replay terminal 语义；
- TD3 critic/actor 的 target 计算代码；
- 第一段 imagined transition 的写入方式。

发生变化：

- 第二段 truncation 的 replay `next_obs` 从 reset 状态 `B` 改为终点 `s2`；
- 第二段 truncation 的 replay `done` 从 `True` 改为 `False`；
- truncation 仍然结束当前 imagined episode，但不再终止 TD bootstrap。

## 7. 测试与验收

测试必须使用内容可区分的 `s2` 和 `B`，不能只依赖 window token，以确保数据内容没有串线。

需要覆盖：

1. 第二段结束时，control next observation 是 reset 状态 `B`；
2. 同一 step 的 replay next observation 是 imagined 终点 `s2`；
3. `episode_done=True`，同时 `replay_done=False`；
4. `s2` 的 augmented observation 使用保存的 `base_chunk(s2)`；
5. 处理 truncation 不会额外增加 kai0 / π0.5 查询次数；
6. replay transform 对第二段 transition 生成 `nonterminal=1`；
7. TD target 包含 `gamma * Q_target(s2, next_combined)`；
8. truncation 序列保持 `False, True, False, True`，证明两段后已 reset；
9. 普通环境的 terminated 和 truncated replay 行为保持不变；
10. 现有 50 帧动作收集、PBRS telescoping 和 autoreset 测试继续通过；
11. 缺少终点 observation/base chunk、非法 terminated 标记、非法 `max_segments` 和不支持的 subgoal 配置都会快速失败。

## 8. 非目标

本次修改不包含：

- 增加世界模型连续推理段数；
- 改变世界模型自身的生成方式；
- 修改 TD3 网络结构或 loss；
- 为 imagination 模式实现 subgoal-conditioned 终点协议；
- 把 SAME_STEP autoreset 改成显式 NEXT_STEP reset；
- 让策略在 `s2` 上实际执行第三段世界模型 rollout。
