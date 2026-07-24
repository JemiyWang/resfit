# Block Mixed-Replay 固定 BC 权重设计

## 目标

将 `launch_block_imagination.sh` 启动的 Block mixed-replay 实验从
“BC 权重由 0.1 线性下降到 0.01”改为“BC 权重全程固定为 0.1”。

## 方案

启动参数保留：

```text
--demo_bc_coef 0.1
```

删除：

```text
--bc_coef_final 0.01
```

`train_chunk_residual.py` 已将 `bc_coef_final` 的默认值定义为 `None`。
未提供该参数时，训练循环不会调用线性调度函数，Agent 初始化时写入的
`bc_loss_coef=0.1` 将在整个训练过程中保持不变。

## 影响边界

- 只修改 `launch_block_imagination.sh`。
- 不修改 BC loss 的计算方法。
- 不修改 Online/Offline 混采比例和 batch size。
- 不修改 Trainer 的全局默认行为，因此不影响其他启动脚本。
- 不改变 Offline endpoint/replay cache 的内容或签名。
- 之后使用该启动脚本新启动或恢复的训练采用固定 BC 权重；已经退出的旧进程不会被自动重启。

## 验证

1. 检查启动脚本中保留 `--demo_bc_coef 0.1`，且不存在 `--bc_coef_final`。
2. 使用 `bash -n` 验证脚本语法。
3. 通过 Trainer 参数解析验证 `demo_bc_coef=0.1`、`bc_coef_final=None`。

