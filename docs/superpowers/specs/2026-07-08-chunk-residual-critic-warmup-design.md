# chunk_residual critic warmup(对齐 offpolicy)设计

日期:2026-07-08
分支:chunk-residual-validation
状态:设计已确认,待写实现 plan

## 背景与目标

`train_chunk_residual.py`(chunk_residual 线)当前的 warmup 只是"前 `learning_starts`(默认 10000)个环境步纯收集数据、不做任何梯度更新",填完 online buffer 后**直接进入 actor+critic 联合训练**。

参考线 offpolicy(`residual-offpolicy-rl/…/train_residual_td3.py`)在填完 buffer 之后、放开 actor 之前,多了一个**独立的 critic-only warmup 阶段**(`_run_critic_warmup`,`critic_warmup_steps=10000`):在**静态** buffer 上连跑 1 万次"只更 critic"的梯度更新,让 critic 先在固定数据分布上收敛,再让 actor 去利用它(缓解 actor 过早利用未收敛 critic 的高估→塌方)。

**目标**:给 chunk_residual 加上等价的 critic-only warmup 阶段,做成默认关闭的开关,不影响任何现有实验/脚本。

## 选定方案:A(忠实复刻,静态 buffer 上的独立 critic-only 段)

已排除方案 B(在主循环里对"进入学习后的前 N 步"强制 `update_actor=False`,边采集边更 critic)。B 的问题:① 还在 step 昂贵的环境 → 慢;② offpolicy 的 1 万是**梯度步**不是环境步,B 会把两者搅在一起,语义不齐;③ buffer 非静态,稀释了 critic warmup 的本意。A 反而更快(纯 GPU 梯度、不 step 环境)且逐行对齐 offpolicy。

## 设计细节

### 1. 新增 flag
`train_chunk_residual.py` argparse 增加:
```python
p.add_argument("--critic_warmup_steps", type=int, default=0)
```
- 默认 `0` = 现在行为(不做 critic warmup),现有脚本/在跑实验逐位不变。
- 对齐版跑法传 `--critic_warmup_steps 10000`(与 offpolicy 一致)。

### 2. 触发位置(一次性)
在主循环里,当 `env_steps` **首次达到 `learning_starts`** 时、在任何"放开 actor"的联合更新之前,插入 critic warmup 段;用 `did_critic_warmup` 布尔标志保证整个训练只跑一次。放在现有更新门控 `if env_steps >= args.learning_starts and len(online_rb) > online_batch_size:` 块内、`for i in range(args.utd)` 联合更新循环**之前**。

### 3. warmup 段行为(逐行对齐 offpolicy `_run_critic_warmup`)
```python
if args.critic_warmup_steps > 0 and not did_critic_warmup:
    for _ in range(args.critic_warmup_steps):
        # 与主循环完全相同的采样:stage_balanced 或 uniform 取 online + concat_mixed_batch 混 offline
        if args.stage_balanced:
            online_batch = sample_stage_balanced(online_rb, online_batch_size, generator=sample_gen)
        else:
            online_batch = online_rb.sample(online_batch_size)
        online_batch = online_batch.to(args.device, non_blocking=True)
        if offline_rb is not None:
            offline_batch = offline_rb.sample(offline_batch_size).to(args.device, non_blocking=True)
            batch = concat_mixed_batch(online_batch, offline_batch)
        else:
            batch = online_batch
        agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=None)  # 只更 critic
    did_critic_warmup = True
```
不变量:
- **期间不 step 环境**——纯梯度循环,在 `env_steps==learning_starts` 时刻的 buffer 上跑。
- **actor 冻结**:`update_actor=False`。
- **HIQL value / high_actor 自动冻结**:它们由 `finetuner.on_step()`/`on_episode_end()`(绑环境步)驱动,此段不 step 环境 → 不被调用 → 真"只训 critic"。**无需额外冻结代码**,但实现时要 assert/验证确未触碰 finetuner。
- **无 bc loss**:bc 是 actor 侧(`update_actor=False` 时 chunk 主循环本就不喂 `bc_batch`)。
- **`stddev=0.0`**:target smoothing 用 0.0,对齐 offpolicy(offpolicy critic warmup 亦用 0.0;与 chunk 主循环的 0.05 不同,这是有意的对齐选择)。

### 4. warmup 段之后
主循环从 `env_steps==learning_starts` 正常继续,`for i in range(args.utd)` 联合更新按原有 `update_actor = ((i+1) % args.utd == 0)` policy-delay 放开 actor。critic warmup 只是"第一次进入学习"前的一次性插入,之后逻辑完全不变。

### 5. 日志
- 每 ~1000 个 critic warmup step 打一行进度(含最近 critic_loss),照 offpolicy 风格。
- 可选:wandb 记 critic warmup 期的 critic_loss(用负的/独立 step 轴或 tag,避免和主训练 step 轴冲突)。实现时最简做法=只打印到 stdout,wandb 记录为可选增强。

### 6. 复用与依赖
- 复用已有:`sample_stage_balanced`、`concat_mixed_batch`、`agent.update`、`online_rb`/`offline_rb`/`online_batch_size`/`offline_batch_size`/`sample_gen`。
- 不新增模块、不改 `agent.update` 签名(它已支持 `update_actor=False`,主循环 line 1248 就是这么调的)。
- 仅改 `train_chunk_residual.py` 一个文件。

## 测试

- **冒烟(真跑)**:`--critic_warmup_steps 50 --smoke`,验证:① warmup 段确实执行了 50 次 critic 更新;② critic_loss 有在动;③ **段前后 actor(policy 网络)参数逐位不变**(证明真没训 actor);④ **段前后 finetuner 的 value/high_actor 参数逐位不变**(证明只训 critic);⑤ 之后正常进联合训练不报错。
- **回归**:`--critic_warmup_steps 0`(默认)时,起手若干步的行为与改动前一致(agent/optimizer 状态推进路径不变)。
- 断言用 `torch` 参数快照对比(clone 关键网络的 state_dict,warmup 前后比对)。

## 不在范围内(YAGNI)

- 不改 offpolicy 仓。
- 不给 critic warmup 加 target/schedule/PER 特殊逻辑(除非主循环本就有,直接复用)。
- 不改 offline/online 缓存机制。
- 不动现有 as005/as01/staged/pothiql 实验的默认;对齐实验作为**新的一臂**、显式传 flag 时才启用。

## 影响面

- 单文件改动(`train_chunk_residual.py`);默认 0 时零行为变化。
- 4 个在跑实验不受影响(进程已加载旧代码;且默认 0)。
- 新对齐实验需重新起、并新建自己的 offcache/online buffer(与现有 run 独立)。
