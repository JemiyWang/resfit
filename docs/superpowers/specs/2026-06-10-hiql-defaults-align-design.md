# HIQL 默认值对齐 — 设计文档

- 日期: 2026-06-10
- 分支: `chunk-residual-validation`
- 参考项目: `/mnt/mnt/data/wjm/residual/HIQL`
- 承接: [[2026-06-10-hiql-alignment-deltas-design]]、[[2026-06-10-hiql-value-loss-align-design]]、[[2026-06-08-hiql-hierarchy-residual-design]]

## 1. 背景与动机

resfit 分层路(`--subgoal_conditioned`)对照参考 HIQL 后,收敛出一批"实现上偏离 HIQL"的点。
之前的设计哲学是:把每处偏离做成 **flag-gated**,**默认值保持与旧基线逐位等价**,只有显式开 flag
才启用对齐行为,A/B 验证后再逐个决定是否纳入默认。

到 2026-06-10,多处对齐改动已落码并部分跑完 A/B(见承接的两份 spec)。本任务的目标是:
**把训练脚本的默认值整体翻向 HIQL 口径**,即把"默认 = 旧基线"改成"默认 = 对齐 HIQL(折中口径)"。

注意本任务**只改 argparse 默认值 + 连带必须改的测试/写法**,不引入任何新算法、不碰无 flag 的硬差异。

## 2. 关键决策(已与用户确认)

1. **对齐口径 = 折中 + LN(用户 2026-06-10 追加决策:对齐优先)**:
   - A/B 已有结论中有益的按结论设默认:`geometric` value、`clamp_to_goal` high_actor。
   - `use_layer_norm` **改为默认开(=1)**:用户决策"对齐 HIQL 优先于 A/B 收益"。⚠️ A/B 实测开 LN 会让
     gc_value 的 V(s,g=s) 自指负尾退步(−0.96→−3.20),用户知情接受此风险;连带激活 ReLU→GELU,
     进一步对齐 HIQL 的 LayerNormMLP(GELU+LN)。
   - 未验证的也翻成 HIQL 口径但**标 TODO 待 A/B**:`value_loss_mode=hiql`、`high_p_randomgoal=0.3`。
2. **范围 = 只动有 flag 的默认值**:不碰无 flag 的硬差异(min→mean、concat-φ→state、mask 去 (1-done))、
   不动低层 AWR(架构级,与 [[project_resfit_gt_as_base_bug]] 决策冲突)。
3. **renorm 默认开**:`--renorm_subgoal` 默认改 True(对齐 HIQL eval 无条件投球面,实测无害,纯卫生)。
4. **测试处理**:改默认值会让"断言默认=旧行为"的单测失败。处理方式 = 把这些断言改成
   "显式传旧 flag 才测等价",并新增"默认值=HIQL 口径"的断言。**不删测试**。

## 3. 改动清单(6 处默认值)

| # | flag | 文件 | 旧默认 | 新默认 | 依据 | 改后需重训? |
|---|---|---|---|---|---|---|
| 1 | `--goal_future_mode` | `train_hiql_gc_value.py` | `stage_entry` | `geometric` | ✅ A/B 验证有益(填洞,V(s,g=s)负尾 −19→−0.96) | value |
| 2 | `--target_mode` | `train_hiql_high_actor.py` | `fixed_waypoint` | `clamp_to_goal` | ✅ A/B 验证有益(near 1.0 vs 12.0) | high_actor |
| 3 | `--value_loss_mode` | `train_hiql_gc_value.py` | `shared_min` | `hiql` | ⚠️ 折中:翻 HIQL 口径,**标 TODO 待 A/B** | value |
| 4 | `--high_p_randomgoal` | `train_hiql_high_actor.py` | `0.0` | `0.3` | ⚠️ 折中:翻 HIQL 口径,**标 TODO 待 A/B**(仅 clamp 下生效) | high_actor |
| 5 | `--renorm_subgoal` | `train_chunk_residual.py` | `False` | `True` | ◐ 对齐 HIQL、实测无害(z 偏差仅 1-2%) | 否(在线生效) |
| 6 | `--use_layer_norm` | `train_hiql_gc_value.py` | `0` | `1`(开) | ⚠️ 用户决策:对齐 HIQL 优先(A/B 实测 V(s,g=s) 退步 −0.96→−3.20,知情接受;连带激活 ReLU→GELU) | value |

所有改动**只改 `default=`,`choices`/旧选项全部保留**,旧行为仍可通过显式传 flag 复现。

## 4. 连带必须改

### 4.1 测试断言(否则改完即红)
现有一批单测断言"不带 flag = 旧行为逐位等价"(参数差=0.0 等)。改默认后这些会失败。
对每个受影响测试:
- 把"等价"断言改成**显式传旧 flag**(如 `--value_loss_mode shared_min`、`--goal_future_mode stage_entry`、
  `--target_mode fixed_waypoint`、`--high_p_randomgoal 0.0`、`--no_renorm_subgoal`)再断言与历史旧行为等价。
- **新增**断言:不带 flag 时默认值 = HIQL 口径(parser 默认值检查 + 行为检查)。

受影响测试(待实现时逐一核对):`tests/test_hiql_gc_value.py`、`tests/test_hiql_high_actor.py`、
`tests/test_hiql_subgoal_wiring.py`、`tests/test_hiql_value.py`(若涉及) 及 `verify_*` 脚本中硬编码默认的断言。

### 4.2 renorm argparse 写法
`--renorm_subgoal` 当前是 `action="store_true"`(默认 False),无法直接默认 True。改为
`action=argparse.BooleanOptionalAction, default=True`(Python 3.9+):
- 自动同时提供 `--renorm_subgoal`(显式开,幂等) 和 `--no-renorm_subgoal`(关)。**保留 `--renorm_subgoal`
  这个 flag 名 = 旧脚本不 break**(若哪里显式写了它,仍合法、仍是开)。
- 旧调用语义变更:之前**不写** = 关 → 现在不写 = **开**;要关需显式 `--no-renorm_subgoal`。
  **这是行为变更,需在 echo 里醒目打印当前 renorm 状态**。
- 实现时确认 argparse 版本支持 BooleanOptionalAction;若环境 Python <3.9,退回手动加互斥的
  `--renorm_subgoal`(store_true)/`--no_renorm_subgoal`(store_false,同 dest) 且 `default=True`。

### 4.3 provenance / 旧 ckpt 加载
`load_gc_value` 从 ckpt 读 `value_loss_mode`(不依赖 argparse 默认),旧 ckpt 无此键时 `get` 回退 `shared_min` —
**改默认不破坏旧 ckpt 加载**。仅新训练默认变 hiql,需保证 `save_gc_value` echo + provenance 写入新默认值。

## 5. 明确不做(scope 边界)

- 无 flag 硬差异:high_actor 优势聚合 `min-min`→HIQL `mean`(`hiql_high_actor.py:106`)、
  goal encoder `concat-φ`→HIQL `rep_type='state'`、value-loss mask 多乘 `(1-done)`。
- 低层换 AWR low actor(架构级,与 gt_as_base 决策冲突)。
- `way_steps=25`/`beta=1.0`:已与 HIQL 一致,不动。

## 6. 落地生效路径(改默认值 ≠ 立即生效)

现有 canonical 产物(`three_piece_gc_value_geom.pt` / `three_piece_high_actor_geom_clamp.pt`)是
geom+clamp 但 **shared_min + p_randomgoal=0**,不含本次折中项(#3 #4)。要让对齐默认真正生效需级联重训
(EGL 回放重活,**留用户本机跑**;本任务只交付代码默认+测试+命令草图,新产物用新文件名,旧的保留对照):

1. 重训 value(不带 flag = 新默认 geom+hiql+**LN**)→ `outputs_chunk/three_piece_gc_value_aligned.pt`
2. 级联重训 high_actor(挂 aligned value,默认 clamp+p_randomgoal=0.3)→ `outputs_chunk/three_piece_high_actor_aligned.pt`
3. 主训练 `train_chunk_residual --subgoal_conditioned` 挂两新 `.pt`(renorm 默认已开)

精确命令行(含 `CUDA_VISIBLE_DEVICES`/`OMP_NUM_THREADS`/数据/缓存路径)在 plan 阶段给定。

## 7. 验证 / Gate

**本任务可独立验证的(不需重训)**:
- 全测试绿:`pytest resfit/rl_finetuning/chunk_residual/tests`(默认=HIQL 口径新断言 + 显式旧 flag 等价断言均过)。
- parser 默认值断言:6 处 default 确为新值。
- 旧 ckpt 仍可载(provenance 回退路径测试)。

**重训后的 gate(用户本机跑)**:复用 `verify_gc_value.py`(看 ⑥ 双 critic 分化度 corr 是否比 shared_min 基线低)、
`verify_high_actor.py`(判据不变)。这是 #3 #4 折中项 A/B 的"待验"部分,不阻塞本任务代码合入。

## 8. 风险与未决

- **#6 `use_layer_norm=1` A/B 实测退步仍设默认**(用户决策:对齐 HIQL 优先于 A/B 收益)。风险:gc_value 的
  V(s,g=s) 自指负尾恶化(−0.96→−3.20),可能拖累高层"优势之差"信号质量。缓解:旧选项 `--use_layer_norm 0`
  保留可一键回退;重训后用 §7 gate(`verify_gc_value` 的 V(s,g=s) 分布)复核,若退步明显则提请用户重新决策。
- **#3 `value_loss_mode=hiql`、#4 `high_p_randomgoal=0.3` 未经 A/B 验证**就设为默认。风险:可能不如旧默认。
  缓解:旧选项保留可一键回退;spec/echo/CHANGELOG 标注"待验证默认";重训后用 §7 gate 复核。
- **renorm 行为变更**(默认开)对已有调用脚本有语义影响(不写 flag 从"关"变"开")。缓解:`--no_renorm_subgoal`
  关闭开关 + echo 醒目提示。
- 改默认后若有遗漏的"隐式依赖旧默认"的脚本(如 run_hiql_align_ab.sh)会行为漂移。缓解:实现时 grep 全仓
  显式传 flag 的调用点,确认它们要么显式传值(不受默认影响)、要么是有意跟随新默认。
