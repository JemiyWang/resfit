# 物体感知 HIQL value(③a' object-aware)Implementation Plan

> **For agentic workers:** TDD,逐 task 实现(`- [ ]` checkbox 跟踪)。每个 task:先写失败测试 → 实现 → 跑绿。

**Goal:** 给 ③a 的 value `V` 加双臂 eef 相对两 piece 的位置(18→30 维),从仿真特权状态现算、online/offline 同源,`--state_mode {eef,eef_piece}` 默认 eef 零回归。缓解 online 偏离流形时的 value 高估。

**Architecture:** 纯逻辑(`eef_rel_piece`/`rel_piece_stats`)集中在新 `object_state.py`(全可单测);sim 读取(`read_piece_positions`)是薄集成层。offline replay 与 online `_process_obs` 共用 `compute_eef_rel_piece` 保证同源。

**Spec:** `docs/superpowers/specs/2026-06-07-hiql-value-object-aware-design.md`

**决策(已 review):** ① rel = 世界系差 `eef-piece`;② 12 维双臂×两 piece 全要;③ 实现+冒烟到位,起不起等 B。

**测试命令:** `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest <路径> -q`(从仓库根)

---

## File Structure
- `chunk_residual/object_state.py`(新)— `PIECE_ROOT_BODIES`、`eef_rel_piece`(纯)、`read_piece_positions`(sim)、`compute_eef_rel_piece`(组合)、`rel_piece_stats`(纯)。
- `chunk_residual/offline_hdf5_buffer.py` — `assemble_state(obs, mode)` 支持 18/30;`STATE18_KEYS` 不动。
- `chunk_residual/offline_stage_replay.py` — replay 时算 rel_piece(set_state 后读 sim)。
- `chunk_residual/train_hiql_value.py` — `--state_mode`,replay 算 rel_piece stats + 存。
- `chunk_residual/hiql_value.py` — `save_value/load_value` 存/读 `state_mode` + rel_piece mean/std。
- `dexmg/environments/dexmg.py` — `_process_obs` 在 eef_piece 模式拼 rel_piece(从 sim)。
- `chunk_residual/hiql_potential.py` / `train_chunk_residual.py` — ③b 两端喂 30 维。
- `tests/test_object_state.py`、`tests/test_object_state_parity.py`(同源一致性)。

---

## Task 1: object_state.py 纯逻辑 + sim 读取
**Files:** Create `chunk_residual/object_state.py` + `tests/test_object_state.py`
- [ ] Step 1: 失败测试 — `eef_rel_piece` 顺序/值/dtype、`rel_piece_stats` mean/std、`PIECE_ROOT_BODIES` 常量。
- [ ] Step 2: 实现 4 个函数(纯逻辑 + `read_piece_positions`/`compute_eef_rel_piece` 集成层)。
- [ ] Step 3: `pytest tests/test_object_state.py -q` 绿。

## Task 2: assemble_state 支持 state_mode(offline)
**Files:** `offline_hdf5_buffer.py` + 测试
- [ ] 失败测试:`assemble_state(obs, mode="eef")` == 旧 `assemble_state18`(逐位);`mode="eef_piece"` 需额外传 rel_piece、输出 30 维。
- [ ] 实现 + 绿。**回归:旧 `assemble_state18` 保留、零变化。**

## Task 3: offline replay 算 rel_piece
**Files:** `offline_stage_replay.py` + smoke
- [ ] replay 逐帧 `set_state` 后调 `compute_eef_rel_piece(sim, eefs)`,与 stage 同一遍循环出 rel_piece 数组。
- [ ] smoke:真数据 1 条 demo replay 出 (T,12) rel_piece、无 NaN。

## Task 4: train_hiql_value --state_mode + rel_piece stats
**Files:** `train_hiql_value.py` + 测试
- [ ] `--state_mode`;eef_piece 时:replay 算 rel_piece → `rel_piece_stats` → 标准化拼进 30 维 state → 训 V → 存 stats。
- [ ] 测试:eef 模式与基础版逐位等价;eef_piece 产出 state_dim=30。

## Task 5: hiql_value save/load 存 state_mode + stats
**Files:** `hiql_value.py` + 测试
- [ ] `save_value`/`load_value` 多存 `state_mode`、`rel_piece_mean/std`;load 回来一致。

## Task 6: online _process_obs 拼 rel_piece
**Files:** `dexmg.py` + 测试(mock sim)
- [ ] eef_piece 模式:`_process_obs` 从 `self.env.sim` 调 `compute_eef_rel_piece`,标准化后拼进 `observation.state`(30 维)。
- [ ] 默认 eef:逐位不变(回归)。

## Task 7: online/offline 同源一致性(关键)
**Files:** `tests/test_object_state_parity.py`
- [ ] 构造/取一个 sim state,online 路径 vs offline replay 路径算出的 rel_piece **必须一致**。这是方案不失效的命门测试。

## Task 8: ③b 两端喂 30 维
**Files:** `hiql_potential.py` / `train_chunk_residual.py` + 测试
- [ ] HiqlPotential 按 value.pt 的 state_mode 决定喂 18/30;online 从 sim、offline 从 buffer,标准化口径一致。
- [ ] 断言 online/offline 喂给 Φ 的 state_dim 一致。

## Task 9: 端到端冒烟
- [ ] 小规模(几条 demo)训 30 维 value + ③b `--smoke`(照今天 hiql smoke 口径),链路通、scale 打印、offline buffer 用新 V 重算不报错。
