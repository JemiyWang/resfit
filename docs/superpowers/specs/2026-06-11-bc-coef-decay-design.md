# BC 系数线性衰减(demo-BC coef decay)设计

**日期:** 2026-06-11
**分支:** chunk-residual-validation
**状态:** 已批准设计,待写 plan

## 1. 背景与动机

残差 actor 的 demo-BC(模块②a)用一个**固定**系数 `--demo_bc_coef` 把残差动作锚向专家。
关键一行 `resfit/rl_finetuning/off_policy/rl/q_agent.py:605`:

```
loss = actor_loss_total + (self.cfg.bc_loss_coef * ratio * bc_loss).mean()
```

`bc_loss_coef` 在建 agent 时由 `train_chunk_residual.py:475` 一次性写死(`cfg.agent.bc_loss_coef = args.demo_bc_coef`),整个训练不变。

**动机:** 训练早期残差策略很差,BC 锚专家防止 critic 高估塌方;到后期希望残差自行探索、甚至超越专家,固定的强 BC 会一直把残差往专家拽、压住上限。所以让系数随训练步数**线性衰减**——早期强约束、后期放手。

## 2. 需求(用户已拍板)

- 衰减**形状**:线性 linear。
- 衰减**区间**:整个训练(`env_steps` 0 → `total_env_steps`)。
- 衰减**终值**:降到一个可配置的小 floor(用户可传 0 即降到 0)。
- **总开关 + 终值合一**:新增 `--bc_coef_final`(float,默认 `None`);`None`=不衰减(逐位等价现状),传值 v=线性从 `demo_bc_coef` 降到 v。
- **校验:** 传了 `--bc_coef_final` 但 `--demo_bc_coef<=0` 报错;要求 `0 <= bc_coef_final <= demo_bc_coef`。
- **可观测:** wandb 记录当前系数 `rft/bc_coef_cur`。

## 3. 设计

### 3.1 纯函数(新建 `resfit/rl_finetuning/chunk_residual/bc_schedule.py`)

无重依赖、可独立单测:

```python
def linear_bc_coef(env_steps, *, c0, c_final, total_steps):
    """BC 系数线性衰减:env_steps 0→total_steps 时 c0→c_final,区间外 clip。

    total_steps<=0 时退化为 c_final(防除零)。progress clip 到 [0,1],
    故 env_steps<0 返回 c0、env_steps>=total_steps 返回 c_final。
    """
    if total_steps <= 0:
        return c_final
    progress = min(max(env_steps / total_steps, 0.0), 1.0)
    return c0 + (c_final - c0) * progress
```

### 3.2 接入 `train_chunk_residual.py`

1. **argparse**(`--demo_bc_coef` 定义附近,317 行后):
   ```python
   p.add_argument("--bc_coef_final", type=float, default=None,
                  help="demo-BC 系数线性衰减的终值(floor);不传=固定 demo_bc_coef(逐位等价)。"
                       "传值 v 则 bc_loss_coef 从 demo_bc_coef 线性降到 v(区间 0→total_env_steps)。"
                       "需 demo_bc_coef>0 且 0<=v<=demo_bc_coef")
   ```

2. **校验**(`demo_bc_coef>0` 的 assert 块附近,477-480 行):
   ```python
   if args.bc_coef_final is not None:
       assert args.demo_bc_coef > 0, "--bc_coef_final 需 --demo_bc_coef>0(BC 未开则衰减无意义)"
       assert 0.0 <= args.bc_coef_final <= args.demo_bc_coef, \
           f"--bc_coef_final 需在 [0, demo_bc_coef={args.demo_bc_coef}] 内,得到 {args.bc_coef_final}"
   ```

3. **训练循环 actor-update 分支**(682 行 `if args.demo_bc_coef > 0 and update_actor ...` 内,
   在取 bc_batch / 调 agent.update 之前):
   ```python
   if args.bc_coef_final is not None:
       agent.cfg.bc_loss_coef = linear_bc_coef(
           env_steps, c0=args.demo_bc_coef,
           c_final=args.bc_coef_final, total_steps=args.total_env_steps)
   ```
   `None` 时不写,保持 475 行设的固定值(逐位等价)。BC 是否启用仍由 `demo_bc_coef>0`(初值)gate,不变。

4. **wandb log**(已有 `build_train_log_dict(...)` 的 wandb.log 处,705 行附近):
   把 `agent.cfg.bc_loss_coef` 当前值并入日志键 `rft/bc_coef_cur`(衰减关时即为固定值,曲线为水平线,无副作用)。

5. **import**:文件顶部加 `from resfit.rl_finetuning.chunk_residual.bc_schedule import linear_bc_coef`。

### 3.3 向后兼容

- 默认 `--bc_coef_final=None` → 不进衰减分支 → 逐位等价现状。**正在跑的 4 个 run 不受影响**(它们没这个 flag)。
- 不动 agent 签名 / cfg 结构 / q_agent.py;只在循环外层运行时改 `agent.cfg.bc_loss_coef` 这一可变属性。

## 4. 测试(新建 `tests/test_bc_schedule.py`)

纯函数 `linear_bc_coef`,无 lerobot 依赖,但仍按项目惯例用 `conda run -n residual` 跑全量:

- `t=0` → `c0`
- `t=total_steps` → `c_final`
- `t=total_steps/2` → `(c0+c_final)/2`(中点)
- `t>total_steps` → `c_final`(上界 clip)
- `t<0` → `c0`(下界 clip)
- `total_steps=0` → `c_final`(防除零)
- `c_final==c0` → 任意 t 恒等 `c0`(衰减幅度为 0 的边界)
- 一个具体数值例:`c0=0.1, c_final=0.01, total=500_000, t=250_000` → `0.055`

## 5. 文件清单

- **新建** `resfit/rl_finetuning/chunk_residual/bc_schedule.py`(纯函数)
- **新建** `resfit/rl_finetuning/chunk_residual/tests/test_bc_schedule.py`
- **改** `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(flag + 校验 + 循环调用 + wandb log + import)

## 6. 部署:hiql3 run 怎么用

跑 hiql3 的 residual run 时,在原命令基础上加 `--bc_coef_final <v>`
(hiql3 的 `demo_bc_coef` 已是 0.1,floor 跑前用户定,如 `0.0` 或 `0.01`)。
其余配置不变。

## 7. 非目标(YAGNI)

- 不做 cosine / exponential 形状(用户选了 linear)。
- 不做自定义起止步窗口(区间固定为整个训练 0→total_env_steps)。
- 不动 relabel / DAPG dynamic BC(`bc_loss_dynamic`)路径。
- 不改 RLPD 主线(`train_rlpd_dexmg.py`)的同名系数。
