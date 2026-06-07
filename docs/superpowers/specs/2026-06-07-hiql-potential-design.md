# 设计：HIQL value 当 PBS 势函数 Φ 接进 reward shaping(模块 ③b)

日期：2026-06-07
分支：chunk-residual-validation
关联文档：`/data2/kai0/rlt/residual_flow_design_and_training.md` §21;Ng et al. 1999(PBS)。
关联：三模块 default-off 计划的 ③(HIQL-Φ)第二步。
  前置:**③a 已实现**(spec/plan 2026-06-07-hiql-value;产出冻结 `value.pt`,含 `ValueMLP` 权重 + v_stats)。
  ③a 学了 value,本 spec(③b)把它当 PBS 势函数 Φ 接进 online + offline 两端的 reward shaping。

## 1. 背景与动机

现状 `--reward_shaping potential`(PBS, Ng 1999)的势函数 **Φ = 整数 latch stage**(`chunk_env_wrapper.py:23-40 shaping_reward`
的 potential 分支 `float(int(stage))`):阶梯状,只在跨 stage 给一次脉冲,stage 内部无梯度。③a 学好的
`V(s)`(action-free IQL expectile,平滑的"离成功还有多近")可当更细腻的 Φ:stage 内部也有连续梯度,缓解稀疏。

PBS 理论:Φ 任意(只要固定)都不改最优策略,所以换平滑 V 当 Φ 是安全的(最坏=没增益)。但有一条硬约束:
**online 与 offline 两端必须用同一个 Φ**(同一 V + 同一缩放),否则两端 shaped reward 口径不一,Q-target 被扭曲。

## 2. 目标 / 非目标

目标:
- 新增 `--potential_source {stage,hiql}`(默认 stage)。`hiql` 时,PBS 的 Φ 用 ③a 的 `V(state)*scale`,
  **online(chunk_env_wrapper)+ offline(build_offline_buffer)两端同步**用同一个 value + 同一缩放。
- Φ 尺度:用 ③a 的 `v_stats` 自动算 `auto_scale=(num_stages-1)/(v_max-v_min)`(让 V 动态范围 ≈ 现状 stage Φ 的
  [0,num_stages-1],`--stage_reward_bonus` 不用重调)+ `--phi_scale`(默认 1.0)微调。**只缩放不平移**。
- 默认 `--potential_source stage` -> 两端逐位走现状路径 = baseline 等价。

非目标(本模块不做):
- 不训 value(③a 已做);value 冻结、不在线更新。
- 不动 staged / none 模式;不动 critic / actor / buffer 结构 / done / base reward。
- 不做分层 / goal-conditioned policy。
- 不支持 `--actor flow` 之外的特殊路径变化(本模块与 actor 类型正交,不加额外限制)。

## 3. 关键设计事实(代码已核实)

- `shaping_reward(start_stage, end_stage, *, mode, bonus, gamma, done)`(`chunk_env_wrapper.py:23-40`):
  potential 分支 `phi_start=float(int(start_stage)); phi_next=0.0 if done else float(int(end_stage));
  return bonus*(gamma*phi_next - phi_start)`。**online 与 offline 都调它**(根基)。
- **坑1(online)**:`step()`(`chunk_env_wrapper.py:119-179`)只存 chunk 起点的 `start_stage=self._stage`(int),
  **没存起点的 state**。shaping_reward 调用在 164-166,此处 `raw_obs`=chunk 最后一步原始 obs(可标准化得 end state);
  起点 state 需 wrapper 额外维护。obs.state 标准化在 `_augment`(105):`self.state_standardizer.standardize(...)`。
- **坑2(offline)**:`build_offline_buffer`(`offline_stage_replay.py:158-216`)里 `transition_rewards`(经
  `transition_fields` 181-182)在读标准化 `state_n`(184-186,(T,18))**之前**调。按 V(state) 算 reward 需把读
  state_n 提前并传入。`transition_rewards`(`offline_hdf5_buffer.py:111-127`)逐 transition 调 shaping_reward。
- reward 进 buffer:online `add_chunk_transition` 的 `next.reward`(train_chunk_residual.py:71);
  offline `next.reward`(offline_stage_replay.py:209)。
- CLI 传递:`--reward_shaping/--stage_reward_bonus/--gamma` 经 `resolve_shaping_mode`(train:343)传
  `ChunkResidualEnvWrapper`(346-350)与 `build_offline_buffer`(454-459)。新 flag 在这两处接。
- `ValueMLP` + `load_value`(③a `hiql_value.py`):`load_value(path)->(model, info)`,`info["v_stats"]={min,max,mean}`。
  V 在标准化 state 上训练,两端喂的 obs.state 已用同一 `StateStandardizer` 标准化 -> 与 ③a 同分布。

## 4. 设计

### 4.1 新模块 `chunk_residual/hiql_potential.py`

```python
def potential_shaping(phi_start, phi_next, *, bonus, gamma, done):
    """通用 PBS 整形:F = bonus*(gamma*phi_next - phi_start),done 时 phi_next=0。
    phi 可为 int stage 或 V(state)*scale。纯函数,可单测。"""
    pn = 0.0 if done else float(phi_next)
    return bonus * (gamma * pn - float(phi_start))


class HiqlPotential:
    """加载 ③a 的冻结 value,提供把标准化 state 映射到 PBS 势函数值的 phi()。"""
    def __init__(self, model, scale, device="cpu"):
        # 存 self.model(.eval() + 所有参数 requires_grad_(False))、self.scale、self.device

    @classmethod
    def from_ckpt(cls, path, *, num_stages, phi_scale=1.0, device="cpu"):
        model, info = load_value(path, map_location=device)
        vmin, vmax = info["v_stats"]["min"], info["v_stats"]["max"]
        auto_scale = (num_stages - 1) / max(vmax - vmin, 1e-6)   # V 动态范围 -> [0,num_stages-1]
        return cls(model, scale=auto_scale * phi_scale, device=device)

    @torch.no_grad()
    def phi(self, state_std):
        """state_std: [B,18] 已标准化 -> [B] 势函数值 = V(state)*scale(只缩放,不平移)。"""
        return self.model(state_std.to(self.device)).squeeze(-1) * self.scale
```

- `auto_scale` 让 V 的动态范围匹配现状整数 stage Φ;`--phi_scale` 额外乘子。**不平移**(PBS 下平移常数 c
  会引入每步 (gamma-1)c 的存活项,非完全无害),允许 phi 略负。
- model 冻结(requires_grad False)+ eval + no_grad。

### 4.2 `shaping_reward` 重构(potential 分支复用 potential_shaping)

把 `chunk_env_wrapper.shaping_reward` 的 potential 分支改为调 `potential_shaping(int(start_stage),
int(end_stage), bonus=bonus, gamma=gamma, done=done)` —— 数值与现状逐位相同(int stage 当 phi)。
staged / none 分支不变。**这步是纯重构,任何模式都逐位等价**(单测钉)。

### 4.3 online 接线(chunk_env_wrapper.py)

- `__init__` 加 `potential=None`(`HiqlPotential` 或 None)。
- 维护 `self._start_state_std`:`reset()` 里设为初始 obs 的标准化 state;`step()` 末尾更新为本 chunk 结束的标准化
  state(下个 chunk 的起点)。
- `step()` 算 shaping 时:
  - `potential is None`(source=stage)-> 现状 `shaping_reward(start_stage, max_in_chunk, ...)`。
  - 否则(source=hiql)-> `phi_start=self.potential.phi(self._start_state_std)`、
    `phi_next=self.potential.phi(end_state_std)`(end_state_std = 标准化的本 chunk 末 obs.state),
    `total_reward += potential_shaping(phi_start, phi_next, bonus=..., gamma=..., done=chunk_done)`。
  - mode 非 potential 时(staged/none)potential 不参与(hiql 只在 potential 模式有意义,见 §4.6 断言)。

### 4.4 offline 接线(offline_stage_replay.py + offline_hdf5_buffer.py)

- `build_offline_buffer` 加 `potential=None` 参数;把读 `state_n`(184-186)提到算 reward 之前。
- `transition_rewards`(及 `transition_fields`)加可选 `potential=None` + `state_seq=None`:
  - `potential is None` -> 现状(latch stage 逐 transition 调 shaping_reward)。
  - 否则 -> `v = potential.phi(state_seq)`([T] 张量);逐 transition `done=success and t==T-2`,
    `base=float(done)`,`shaped=potential_shaping(v[t], v[t+1], bonus=bonus, gamma=gamma, done=done)`,
    `reward[t]=base+shaped`。
- 两端用**同一个 HiqlPotential 实例/同一 scale**(train_chunk_residual 里构一次,分别传 wrapper 与 build_offline_buffer)。

### 4.5 CLI(train_chunk_residual.py)

- 新 flag:`--potential_source {stage,hiql}` 默认 `stage`;`--hiql_value_ckpt`(str,默认 None);
  `--phi_scale`(float,默认 1.0)。
- source=hiql 时构 `potential = HiqlPotential.from_ckpt(args.hiql_value_ckpt, num_stages=num_stages,
  phi_scale=args.phi_scale, device=args.device)`,传 wrapper(346-350)与 build_offline_buffer(454-459)。
  source=stage 时 potential=None(两端走现状)。

### 4.6 依赖断言 + default-off 等价

- 断言:`--potential_source hiql` 需 `--reward_shaping potential`(hiql Φ 只在 PBS 模式有意义)且
  `--hiql_value_ckpt` 非空且文件存在。
- 默认 `--potential_source stage` -> potential=None -> online/offline 两端走现状 shaping_reward(stage)路径
  -> **逐位等价 baseline**(§4.2 的重构也保证 stage 路径数值不变)。单测必钉。

### 4.7 正交性

- 与 ① stage_budget(actor 输出包络)、② relabeling(actor BC)正交:③b 只改 PBS 的 Φ 来源(reward 端)。
- 与 critic/actor/done/base reward/buffer 结构无关;value 冻结,不进任何优化器。
- Φ 用标准化 obs.state(非特权 stage),与 stage detector 解耦(stage 仍用于 stage-balanced 采样等,不受影响)。

## 5. 测试方案(TDD)

- `potential_shaping` 纯函数(light):done 时只剩 `-bonus*phi_start`;非 done = `bonus*(gamma*phi_next-phi_start)`;
  传 int 与 float 行为一致。
- `shaping_reward` 重构等价(light):potential 模式下,重构后 `shaping_reward(s0,s1,...)` 与旧值逐位相同
  (几组 stage/done 组合);staged/none 不变。
- `HiqlPotential.from_ckpt` + `phi`(light,用假 value.pt:小 ValueMLP + 注入 v_stats):auto_scale 计算正确
  (=(num_stages-1)/(vmax-vmin));`phi(state)` == `V(state)*auto_scale*phi_scale`;形状 [B]。
- online wrapper(manual/集成,CPU 假 env):source=hiql 时 reward 用 V(state)、与手算 potential_shaping 一致;
  `_start_state_std` 跨 step 正确推进;source=stage 时 reward 与现状逐位相同。
- offline `transition_rewards`(light):source=hiql 用 V(state_seq) 逐 transition,done 末步 phi_next=0,
  与手算一致;source=None 与现状逐位相同。
- default-off 等价(light/grep):不传 `--potential_source`(=stage)时两端不构 potential、走现状路径。
- CLI(light):三个 flag 默认值;`hiql` 缺 `--reward_shaping potential` / 缺 ckpt 报错。

## 6. A/B 验证方案(实现后,run 阶段;不属本实现)

- 前置:先用 ③a 训出 `value.pt`(`train_hiql_value.py`)。
- 配置:nas10 强 base + best,`--reward_shaping potential`。A=`--potential_source stage`(整数 stage Φ);
  B=`+ --potential_source hiql --hiql_value_ckpt value.pt`(扫 `--phi_scale` 1.0 附近)。
- 主指标:eval success / per-stage 到达率,重点 stage 内部 shaping 是否变密、stage3 回退是否降。
- 旁证:wandb reward 曲线、shaping 项幅度;PBS 安全性 -> 最坏中性。

## 7. 文件改动清单

- `resfit/rl_finetuning/chunk_residual/hiql_potential.py`(新)—— `potential_shaping` + `HiqlPotential`。
- `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py` —— `shaping_reward` potential 分支复用
  `potential_shaping`;`__init__` 加 `potential`;`_start_state_std` 维护;`step` 的 hiql 分支。
- `resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py` —— `transition_rewards`/`transition_fields`
  加 `potential`+`state_seq`。
- `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py` —— `build_offline_buffer` 加 `potential`,
  调顺序调整(先 state_n 后 reward),传入。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` —— 3 个 flag + 构 HiqlPotential + 传两端 + 断言。
- 测试:`resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`(新)+ 现有 reward shaping 测试的回归。

## 8. 风险与缓解

- **两端 Φ 不一致**(最大风险):online 与 offline 必须用同一 HiqlPotential(同 model + 同 scale)。设计上
  train_chunk_residual 构一次传两端;single source of truth。单测/集成核对两端同 scale。
- value 仅在成功流形准,online 偏离时 V 高估 -> PBS 安全性兜底(不改最优策略,最坏中性)。
- 缩放不当 -> shaping 幅度异常:auto_scale 让默认幅度 ≈ 现状;`--phi_scale` 兜底微调。
- online 多存一个 state 张量(_start_state_std):内存极小(18 维),无虑。
- 重构 `shaping_reward` 不能破坏 staged/none 与现状 stage potential:重构等价单测必钉(§5)。
