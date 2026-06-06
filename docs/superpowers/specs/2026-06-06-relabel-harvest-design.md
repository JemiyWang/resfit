# 设计：在线 relay relabeling 生产端(模块②b)

日期：2026-06-06
分支：chunk-residual-validation
关联文档：`/data2/kai0/rlt/residual_flow_design_and_training.md` §18.5 / §21.2② / §21.4(RPL IRIL)
关联：三模块 default-off 计划的 ②b。② RPL relay relabeling 的第二步(②a 残差 BC 消费端已实现,
  commit db557a1/90f40d0/4828844)。本 spec 造"生产端":harvest 在线产出段当正样本,喂 ②a 消费端。

## 1. 背景与动机

RPL/§18.5:长程稀疏奖励下,episode 失败但走到了 stage2/3,这段对"到达 stageK"是成功的,重标成正样本
就能从部分成功里学。三模块决策定了 relabeling **喂 actor-AWBC 侧**(②a),**不喂单任务 critic**(会教它
"到 stage3=满分"加重高估)。②a 已把残差 BC 消费端造好(`--demo_bc_coef`,bc_batch 从 offline GT demo 采,
BC target=action−base_action)。但 ②a 只做 demo 正则,**不含任何 relabeling**。

本 spec(②b)造生产端:在训练循环里 harvest 在线失败-但-推进过-stage 的轨迹的"产出前缀"段
(起点→最后一次 stage 推进,丢掉之后折腾失败的尾巴)的 (obs, 实际采取的动作),进一个 relabel buffer;
bc_batch 改为 relabel+demo 50/50 混采,喂 ②a 消费端。这才是 RPL relabeling 治稀疏的本体。

## 2. 目标 / 非目标

目标:
- 训练循环里按 episode harvest"产出前缀"段(布置 `RelabelHarvester`),进 relabel buffer。
- bc_batch 在 `--relabel` 开时 = relabel + offline demo 50/50 混采(复用 ②a 的 offline_rb 当 demo 半边)。
- 以 CLI flag 控制,**默认 `--relabel` 关 => 逐位等价 ②a/baseline**。

非目标(本模块不做):
- 不喂 critic、不动 reward/done/online_rb(RL critic 用的);relabel 只进 bc_batch(actor 侧)。
- 不做 advantage 加权(②a 均匀 BC;按阶段加权留后续)。
- 不支持 `--actor flow`(随 ②a,仅 raw)。
- 不改 ②a 的 BC loss / bc_target / `--demo_bc_coef` 语义。

## 3. 关键设计事实(代码已核实)

- 主循环(整合后 `train_chunk_residual.py`):`while env_steps<=total`(462)-> `env.step`(465)->
  `done=terminated|truncated`(466)-> `add_chunk_transition(...)`(467)-> update 循环(474)-> `agent.update(..., bc_batch=...)`(489)。
- 每个 chunk transition 可拿:`obs`(当前观测,含 observation.base_action)、`info["scaled_action"]`(= combined
  动作,base+残差;②a 的 BC target=action−base_action=实际残差)、`info["max_stage_in_chunk"]`(latch,
  episode 内 running-max,非降)、`done`。
- relabel buffer 当 bc_batch:消费端 `_compute_actor_bc_loss` 只读 `batch["obs"]`(需含 base_action)+ `batch["action"]`。
  故 relabel 条目 = `{obs:{图keys, state, base_action, stage_id}, action}`(与 offline_rb.sample 同构)。
- ②a:`--demo_bc_coef>0` 时 bc_batch=`offline_rb.sample(batch_size)`;`offline_rb` 在 `offline_fraction>0` 时建(L397)。

## 4. 设计

### 4.1 选段:产出前缀(纯函数 `productive_prefix_len`)

`stage_seq` = 一条 episode 的逐 chunk max_stage(latch,非降)。"最后一次推进"= max 首次出现处。

```python
def productive_prefix_len(stage_seq, min_stage=1):
    """产出前缀长度:起点→最后一次 stage 推进(含),丢掉之后无推进的尾巴。
    stage_seq 是 episode 逐 chunk 的 latch(非降)。max < min_stage(没推进够)返回 0(不 harvest)。
    """
    if not stage_seq:
        return 0
    top = max(stage_seq)
    if top < min_stage:
        return 0
    return stage_seq.index(top) + 1      # latch 非降 -> max 首次出现 = 最后一次推进
```

例:[0,1,1,2,2]->4(尾 [2] drop);[0,0,0]->0;[1,2,3,4]->4(全长);成功轨迹(走到顶)->全长。

### 4.2 Harvester(`chunk_residual/relabel.py`)

```python
class RelabelHarvester:
    """攒当前 episode 的 (entry, max_stage);done 时按 productive_prefix_len 返回产出前缀 entry,清空。
    entry 是调用方给的不透明对象(本项目=每 chunk 的 bc td)。"""
    def __init__(self, min_stage=1): ...
    def add(self, entry, max_stage): self._entries.append(entry); self._stages.append(int(max_stage))
    def flush(self):
        n = productive_prefix_len(self._stages, self._min_stage)
        out = self._entries[:n]
        self._entries, self._stages = [], []
        return out
```

### 4.3 Relabel buffer

`TensorDictReplayBuffer`(`LazyTensorStorage(max_size=args.relabel_buffer_size, device="cpu")`,FIFO,
无 priority/multistep——BC 单步不需要),`batch_size=relabel_half`。条目 = bc td `{obs:{...}, action}`
(用与 `add_chunk_transition` 同构的打包,但只留 obs+action,不要 next/n-step)。

### 4.4 Mixed bc_batch(改 ②a 的采样处)

`--relabel` 开 + `demo_bc_coef>0` 时,update_actor 步:

```python
relabel_half = args.batch_size // 2
if relabel_rb is not None and len(relabel_rb) >= relabel_half:
    # relabel 条目只有 {obs, action};offline 还含 next/_priority/_weight。
    # concat_mixed_batch 取两者公共 key 再 cat(bc 消费端只读 obs+action,公共 key 足够)。
    bc = concat_mixed_batch(relabel_rb.sample(relabel_half),
                            offline_rb.sample(args.batch_size - relabel_half))
else:                                   # 冷启动:relabel 不够半批 -> 整批回退 demo
    bc = offline_rb.sample(args.batch_size)
bc_batch = bc.to(args.device, non_blocking=True)
```

`--relabel` 关 -> 走 ②a 原逻辑(纯 offline demo)。BC 权重仍是 `--demo_bc_coef`(共用)。
`concat_mixed_batch`(`offline_hdf5_buffer.py` 已存在)取公共 key cat,正好容忍 relabel/offline 的 key 差异。

### 4.5 训练循环接线

- step 后(`add_chunk_transition` 之后):`--relabel` 开则 `harvester.add(bc_entry, info["max_stage_in_chunk"])`
  (bc_entry = 该 chunk 的 `{obs, action}` td,CPU);`done` 则 `for e in harvester.flush(): relabel_rb.add(e)`。
- update 处:按 §4.4 选 bc_batch。

### 4.6 CLI + 依赖 + 默认关等价

- 新 flag:`--relabel`(store_true,默认关)、`--relabel_buffer_size`(int,默认 50_000)、
  `--relabel_min_stage`(int,默认 1)。
- 依赖断言:`--relabel` 需 `--demo_bc_coef>0`(BC 权重)且 `--offline_fraction>0`(demo 半边)且 `--actor raw`。
- 默认关:不传 `--relabel` -> 不建 harvester/relabel_rb、bc_batch 走 ②a 路径 -> 逐位等价 ②a(及 baseline,
  当 demo_bc_coef 也为 0)。

### 4.7 正交性

与 ① stage_budget、critic、reward、online_rb(RL critic 用)全正交;relabel 只进 actor 侧 bc_batch,
不碰 critic 的 reward/done(避开"喂单任务 critic 加重高估"红线)。harvest 用特权 stage(latch),与 reward
shaping 同口径,仿真零成本。

## 5. 测试方案(TDD)

- `productive_prefix_len` 纯函数(light):[0,1,1,2,2]->4、[0,0,0]->0、[1,2,3,4]->4、空->0、min_stage 过滤
  (max< min_stage ->0)。
- `RelabelHarvester` 纯逻辑(light):add 多步 + flush 返回正确前缀 + flush 后清空 + min_stage 过滤 + 无推进返回空。
- manual 集成(CPU):喂几条假 episode -> relabel_rb 有样本;混采 bc_batch 一半 relabel 一半 demo、shape 对;
  relabel_rb 不足半批时整批回退 demo;且带 relabel 的 `agent.update(bc_batch=...)` 跑通(rft/bc_loss 有限)。
- CLI(light):三个 flag 默认值;`--relabel` 缺依赖(demo_bc_coef=0 / offline_fraction=0)报错。
- 默认关(light/grep):不传 --relabel 时 update 的 bc_batch 走 ②a 路径(无 harvester/relabel_rb 引用执行)。

## 6. A/B 验证方案(实现后,plan/run 阶段;不属本实现)

- 配置:nas10 强 base + best 基座,`--demo_bc_coef 0.1 --offline_fraction 0.5`。
- A = `--relabel` 关(= ②a demo-only);B = `--relabel` 开(relabel+demo 50/50)。
- 主指标:eval success / `eval_stage_reach`,重点 stage3 回退率、reach3->reach4。
- 旁证:relabel_rb 大小增长曲线、wandb `rft/bc_loss`。
- 这才是 RPL relabeling 治稀疏的真正验证(②a 只是消费端;部分成功的学习靠本模块)。

## 7. 文件改动清单

- `resfit/rl_finetuning/chunk_residual/relabel.py`(新)—— `productive_prefix_len` + `RelabelHarvester`。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` —— 3 个 flag;建 relabel_rb + harvester(--relabel 时);
  循环 harvest 接线;update 处混采 bc_batch;依赖断言。
- 测试:`chunk_residual/tests/test_relabel.py`(新:纯函数 + harvester light + manual 集成 + CLI)。

## 8. 风险与缓解

- 冷启动 relabel_rb 空:§4.4 回退整批 demo,BC 不断。
- 产出前缀仍含"接近成功但最终失败"的次优残差:这是 RPL 的已知取舍(从部分成功学);min_stage 调高可只收走得远的。
- relabel 条目存 obs(含图)较占内存:relabel_buffer_size 默认 50k、CPU storage;可调小。
- harvest 用 latch(非降)-> `productive_prefix_len` 用 max 首次出现定位最后推进,依赖 latch 单调(已由
  chunk_env_wrapper `self._stage=max(...)` 保证)。
- 与 ②a 叠加正确:--relabel 关时严格走 ②a;开时 BC target 仍 action−base_action(relabel 条目的 action=
  实际 combined,target=实际残差),语义自洽。
