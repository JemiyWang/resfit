# 设计：HIQL 三处对齐(done-mask · 优势 min→mean · goal 表征 concat→goal-only)

- 日期：2026-06-11
- 分支：`chunk-residual-validation`
- 文件：`resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`、`hiql_high_actor.py`、`train_hiql_gc_value.py`、`train_hiql_high_actor.py`
- 参考实现 HIQL：`/mnt/mnt/data/wjm/residual/HIQL/`(`src/agents/hiql.py`、`src/gc_dataset.py`、`src/special_networks.py`)
- 关联文档：
  - `docs/superpowers/specs/2026-06-10-hiql-alignment-deltas-design.md`(§7 记的"已知遗留 HIQL 差异",本 spec 收其中三条)
  - `docs/superpowers/specs/2026-06-10-hiql-value-loss-align-design.md`(已对齐 value 训练的双 critic + expectile)
- memory：`project_resfit_hiql_value_loss_align`、`project_resfit_hiql_alignment_deltas`、`project_resfit_hiql_defaults_aligned`、`project_resfit_hiql_hierarchy_progress`

## 1. 背景与动机

resfit 分层路(gc_value + high_actor)逐行对照参考 HIQL 后，曾收敛出三处仍未对齐的差异(alignment-deltas spec §7 / value-loss-align spec §2 记录)：

1. **done-mask**：value 的 TD mask，resfit `(1−success)·(1−done)` 多了一个 `(1−done)`；HIQL 只 `1−success`。
2. **高层优势的双 critic 聚合**：resfit high_actor `adv=min(vw)−min(vs)`(min)；HIQL `(v1+v2)/2`(mean)。
3. **goal 表征**：resfit goal 编码器吃 `concat[g,s]`(concat-φ)；HIQL 默认只吃 `g`(goal-only，`rep_type='state'`)。

前几轮已把 `goal_future_mode`/`use_layer_norm`/`value_loss_mode`/`target_mode`/`high_p_randomgoal` 默认翻向 HIQL(见 `project_resfit_hiql_defaults_aligned`)。本 spec **延续同一套惯例**(加 flag、默认翻 HIQL、旧值显式可回退、旧值与现状逐位等价)，把上述三条也做成 flag 并默认对齐 HIQL。

## 2. 用户决策(已确认)

- **三个独立 flag**(方案 A)，每个默认翻 HIQL、旧值可回退；第三条用**一个总开关** `--value_rep_mode` 控制整套 goal 表征切换(不再拆细)。
- **第三条范围**：用户初选"goal+state 都对齐 HIQL"。**经代码核实后修正为：只改 goal 编码器(concat→goal-only)，state 侧保持原始**——见 §3 关键纠正。
- **scope**：只到 offline 两件套(`gc_value` + `high_actor`)的代码 + 测试 + 双段审查 + commit。重训/verify/起新 residual run 留用户本机后续决定，本轮**不碰** `train_chunk_residual`、**不动**正在跑的两个 residual run。

## 3. 关键设计事实(代码已核实)

### 3.1 ① done-mask(`hiql_gc_value.py:206-208`)
```python
success = torch.tensor(si == gi, dtype=torch.float32)
reward = success - 1.0
mask = (1.0 - success) * (1.0 - done_all[b])   # ← 公共 mask，在 value_loss_mode 分支之前
```
`mask` 是公共量(两个 `value_loss_mode` 分支 `shared_min`/`hiql` 都用)，**改一处两分支同时受益**。`mask` 计算不消耗 rng。

### 3.2 ② 优势 min(`hiql_high_actor.py:104-106`)
```python
vs1, vs2 = vf(s, g)
vw1, vw2 = vf(sw, g)
adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)   # ← min 聚合
```
聚合不消耗 rng。`z_tgt = vf.phi(s, sw)` 与 AWR 权重不变。

### 3.3 ③ goal 表征 + **关键纠正：HIQL 的 state 侧是 identity，没有 ψ 编码器**

核实 HIQL 真实构造：
- `hiql.py:243`：`value_state_encoder = None`；非 visual(state 输入)分支里 self **始终保持 None**，只有 `value_goal_encoder` 在 `use_rep=1` 时被赋值(`hiql.py:273-274`)。
- `special_networks.py:108-113`：`get_rep(None, targets)` → `return targets`(**identity**)。
- `special_networks.py:96-100`(`MonolithicVF`)：`phi=observations(原始state)`、`psi=goals(goal_reps)`，`value_net(concat([phi, psi]))`。

即 HIQL value = `value_net(concat[ 原始state, φ(goal-only) ])`——**state 侧就是原始 state，没有编码器**。

resfit 现状(`hiql_gc_value.py:72-77`)：`value(concat[ 原始state, φ(concat[g,s]) ])`——**state 侧两边本来就一致(都是原始 identity)**。

**结论**：真正差异只在 goal 编码器输入(HIQL=goal-only vs resfit=concat[g,s])。"对齐 HIQL" = **state 侧保持原始不动 + goal 编码器 concat→goal-only**。给 state 加 ψ 反而**偏离** HIQL，故不做。

(`use_layer_norm`/`value_loss_mode`/`goal_future_mode` 已在前几轮对齐，本 spec 不涉及。)

## 4. 设计

### 4.0 三个 flag 总览

| flag(脚本) | choices | 默认 | 旧值语义(逐位等价现状) | 新值语义(对齐 HIQL) |
|---|---|---|---|---|
| `--value_mask_mode`(`train_hiql_gc_value.py`) | `done_aware`/`hiql` | **hiql** | `mask=(1-success)*(1-done)` | `mask=(1-success)` |
| `--value_rep_mode`(`train_hiql_gc_value.py`) | `concat`/`goal_only` | **goal_only** | goal 编码器吃 `concat[g,s]`(输入 2·state_dim) | goal 编码器只吃 `g`(输入 state_dim) |
| `--adv_agg`(`train_hiql_high_actor.py`) | `min`/`mean` | **mean** | `adv=min(vw)-min(vs)` | `adv=0.5(vw1+vw2)-0.5(vs1+vs2)` |

三者各自独立、可逐条 A/B。state 侧在两个 `value_rep_mode` 下都保持原始(不变)。

### 4.1 ① done-mask

`train_gc_value` 增 keyword 参 `value_mask_mode='hiql'`(默认新行为)。`hiql_gc_value.py:208` 改为：
```python
if value_mask_mode == "done_aware":
    mask = (1.0 - success) * (1.0 - done_all[b])
else:  # "hiql"
    mask = (1.0 - success)
```
位置仍在 `value_loss_mode` 分支之前(公共)。`done_aware` 分支 = 现状原文。

### 4.2 ② 优势 min→mean

`train_high_actor` 增 `adv_agg='mean'`(默认新行为)。`hiql_high_actor.py:106` 改为：
```python
if adv_agg == "min":
    adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
else:  # "mean"
    adv = 0.5 * (vw1 + vw2) - 0.5 * (vs1 + vs2)
```
`min` 分支 = 现状原文。

### 4.3 ③ goal 表征 concat→goal_only(改网络结构)

`RelativeGoalEncoder` 增 `rep_mode`(默认 `'goal_only'`)：
```python
class RelativeGoalEncoder(nn.Module):
    def __init__(self, state_dim, rep_dim=10, hidden=256, use_layer_norm=False, rep_mode="goal_only"):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.rep_mode = rep_mode
        in_dim = state_dim if rep_mode == "goal_only" else 2 * state_dim
        self.net = _mlp(in_dim, hidden, rep_dim, use_layer_norm=use_layer_norm)

    def forward(self, targets, bases):
        inp = targets if self.rep_mode == "goal_only" else torch.cat([targets, bases], dim=-1)
        rep = self.net(inp)
        rep = rep / (rep.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return rep
```
- `GoalConditionedVF.__init__` 增 `rep_mode` 透传给 `RelativeGoalEncoder`；`v1/v2` 头维度 `state_dim+rep_dim` **不变**(φ 输出仍 rep_dim)。`forward(s,g)` 不变(仍 `concat[s, phi(s,g)]`)。
- `phi(s,g)=goal_encoder(g,s)`：`goal_only` 下 `forward` 取 `targets=g`、忽略 `bases=s`。
- `train_gc_value` 增 `value_rep_mode='goal_only'` 透传。
- `concat` 分支 = 现状逐位等价(输入维度 2·state_dim、`forward` 走 cat 路径)。

### 4.4 ckpt 向后兼容(provenance + 重建)

- `save_gc_value` payload 增 `"value_mask_mode"`、`"value_rep_mode"`(纯 provenance；`value_rep_mode` 还决定重建维度)。
- `load_gc_value`：`value_mask_mode = ckpt.get("value_mask_mode", "done_aware")`、`value_rep_mode = ckpt.get("value_rep_mode", "concat")`，按 `value_rep_mode` 重建 goal 编码器输入维度。**旧 ckpt(无键)→ 回退旧行为(concat，维度 2·state_dim)→ 仍可载**。
- `save_high_actor`/`load_high_actor` 增 `"adv_agg"`(纯 provenance；`get` 默认 `"min"`)。`adv_agg` 只影响训练，不影响推理与 .pt 形状。

## 5. 级联重训(写进 plan，留用户本机跑)

- ① 改 value 训练、③ 改 value 结构 → **重训 `gc_value`**(产 `three_piece_gc_value_hiql3.pt` 之类)。
- ② 改 high_actor 训练、且 high_actor 的 `z_tgt=vf.phi` 挂在新 value 上 → **级联重训 `high_actor`**(挂新 value，`--state30_cache` 免 MuJoCo 回放)。
- 默认翻 HIQL 后，需重训 **value+high_actor 两件套**才生效。基线两件套保留不删作对照。

## 6. 验证(只到离线 gate)

- `verify_gc_value.py`：状态侧 spearman ≈0.99、目标侧 spearman ≈−0.98、`V(s,g=s)` 自指 ≈0 长尾、双 critic 分化度——三件套对齐后跑，确认不退化。
- `verify_high_actor.py`：前向步 median≈25、前向占比高、塌末态≈0、off/expected≈1.0——确认 ② + 新 value 未破坏高层。
- 数值/图落 `outputs_chunk/`，结论写回 plan gate 段。全程无在线 rollout。

## 7. 接口与单测清单

- `hiql_gc_value.py`：`RelativeGoalEncoder(rep_mode=)`、`GoalConditionedVF(rep_mode=)`、`train_gc_value(value_mask_mode=, value_rep_mode=)`、`save/load_gc_value` 带两键。
- `hiql_high_actor.py`：`train_high_actor(adv_agg=)`、`save/load_high_actor` 带 `adv_agg`。
- `train_hiql_gc_value.py`：`--value_mask_mode {done_aware,hiql}`、`--value_rep_mode {concat,goal_only}`(透传 + 启动日志 echo)。
- `train_hiql_high_actor.py`：`--adv_agg {min,mean}`(透传 + echo)。
- 单测：
  1. **旧值逐位等价**：同 data/seed，`value_mask_mode=done_aware`/`value_rep_mode=concat` 训出的 `state_dict` 与不传(现状)逐张量 `allclose(atol=0)`；`adv_agg=min` 同理。
  2. **① 手算**：`hiql` 模式 `mask==1-success`(末帧 done=1 时仍 bootstrap)。
  3. **② 手算**：`mean` 模式 `adv==0.5(vw1+vw2)-0.5(vs1+vs2)`，与 `min` 在两 critic 交叉时不同。
  4. **③ 结构**：`goal_only` 下 goal 编码器第一层 `in_features==state_dim`；`phi(s,g)` 对 `s` 扰动不变(goal-only 忽略 s)；`concat` 下 `in_features==2*state_dim`、扰动 s 改变 φ。
  5. **save/load 往返**：存 mode→读回一致；旧 ckpt(无键)→ 回退 `done_aware`/`concat`/`min`，且 concat 维度可重建载入。
  6. **`hiql` 模式学到单调 V**(仿现有 progress 单测)：默认三 flag 下 `V(s,g=末态)` 末段 > 起段。

## 8. scope 边界 / 非目标

- 不动 `train_chunk_residual.py`、不重启正在跑的两个 residual run(它们用旧 aligned 两件套 ckpt)。
- 不给 state 侧加 ψ 编码器(§3.3：HIQL 本无此物)。
- 不改单任务 `hiql_value.py`(③a Φ 势能那条，保持正交)。
- 不在本轮做在线 rollout 评测(验证只到离线 gate)。
- 不自动重训/不自动起新 run(GPU/EGL/慢，留用户)。

## 9. 风险

- **默认翻 HIQL 放弃"逐位等价默认"**：但前几轮(LN/value_loss/target_mode)已是此惯例，本轮延续；三条均无独立 A/B 前置定论(直接信 HIQL 口径)。缓解：旧值一键回退；离线 verify 兜底确认不退化。
- **③ 改网络结构 → 旧 ckpt 维度不匹配**：缓解：`load_gc_value` 按 ckpt 记的 `value_rep_mode` 重建维度；旧档无键回退 concat(2·state_dim)；save/load 往返单测覆盖两 mode + 旧档。
- **默认等价被污染**：旧值分支若误插 rng 调用或改了现状语句，A/B 基线失真。缓解：旧值分支保现状原文 + 单测 1 逐张量 atol=0。
- **级联成本**：重训 value + high_actor + 两 verify。缓解：留用户本机跑；`--state30_cache` 免回放；基线两件套保留对照。
