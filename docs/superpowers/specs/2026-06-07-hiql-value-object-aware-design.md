# 物体感知 HIQL value(③a' object-aware)Design Spec

**Status:** Draft — 待 review
**Date:** 2026-06-07
**接续:** `2026-06-07-hiql-value-design.md`(③a 基础版,18 维 eef)、`2026-06-07-hiql-potential-design.md`(③b)

## Goal

给 ③a 的 action-free value `V` 加入物体(piece)相对几何输入,把 state 从 **18 维(纯 eef 本体)→ 30 维**(+ 双臂 eef 相对两个 piece 的位置),让 `V` 不再"近视":能区分「手到位**且件对上**(真接近成功)」vs「手到位**但件没跟上**(假接近)」,缓解 online 偏离 demo 流形时的 value 高估。**default-off**,旧 18 维路径逐位零回归。

## Background

- ③a 现状:`V` 输入 = `STATE18_KEYS`(robot0/1 的 `eef_pos`+`eef_quat`+`gripper_qpos`,纯本体),不含任何物体信息。
- 局限(本会话 V 曲线分析):demo 上 eef↔物体强相关,`V` 借 eef **间接**判断、看着准;但 online 偏离流形(actor 学歪、手到位件没跟上)时相关断裂 → `V` 高估。这是 ③ 的首要局限。
- **命门(2026-06-07 探针):在线 robosuite obs 里没有任何 piece key**(`REL_PIECE_IN_ONLINE_OBS=[]`)。数据集虽现成有 `*_rel_piece_*`,但 online env 不输出 → **不能简单加 obs key**(一提取就 KeyError),必须从仿真特权状态现算。
- 特权合法性:`V` 是 PBS 的势函数 Φ,**只在仿真训练时算 shaping reward、不进策略输入、部署时丢弃** → 用特权(sim ground-truth piece pose)天经地义;`threepiece_stage` 检测器就是同类做法(读 `env.piece_*` + sim 谓词)。

## Design

### 1. 同源 rel_piece 计算(核心)
自定义一个函数,**online / offline 共用**,保证两端同分布:
```
compute_eef_rel_piece(sim, eef_pos_by_arm) -> np.ndarray  # (12,)
  # piece pose: sim.data.get_body_xpos("piece_1_root" / "piece_2_root")  ← 特权,已验证可读
  # 每臂 eef、每 piece: rel = eef_pos - piece_pos  (世界系差, 3 维)
  # 顺序: [eef0-p1, eef0-p2, eef1-p1, eef1-p2] = 4×3 = 12 维
```
**不复刻数据集 rel_piece 的原始公式**;offline 也 replay 重算(同函数)→ 天然同源(同 stage 的处理思路:offline 也 sim-replay 算、不用数据集现成 stage)。

### 2. 新 state 布局(30 维)
- `[0:18]` 原 STATE18(eef_pos/quat/gripper × 双臂)—— 不变。
- `[18:30]` eef_rel_piece(12)。
- 开关 `--state_mode {eef, eef_piece}` 控制,默认 `eef`(= 旧 18 维,逐位零回归)。

### 3. 标准化(关键难点)
新 12 维不在 LeRobot dataset stats 里 → 自算:
- 训 value 前,对全 1006 demo replay 算 rel_piece,统计 mean/std(12 维),**存进 value.pt**(连同 `state_mode`)。
- 原 18 维仍走现有 `StateStandardizer`(dataset stats);两段各自标准化后拼接喂 `V`。
- ③b online 推理时从 value.pt 读 rel_piece stats 做同口径标准化。

### 4. online / offline 接线
- **offline**:`offline_stage_replay` 已 `set_state` 进 sim → 顺手算 rel_piece;`assemble_state` 按 state_mode 输出 18 或 30 维。
- **online ③b**:`dexmg._process_obs` 在 `state_mode=eef_piece` 时从 `env.sim` 算 rel_piece 拼进 `observation.state`;`HiqlPotential` 喂 30 维。
- `hiql_value.ValueMLP` 已自适应 `state_dim`(`d=s.shape[1]`),无需改网络。

## Files(预计改动)
- **新** `chunk_residual/object_state.py`:`compute_eef_rel_piece` + piece root body 常量 + rel_piece stats 计算(纯逻辑,全可单测)。
- `offline_stage_replay.py` / `offline_hdf5_buffer.py`:replay 算 rel_piece;`assemble_state` 支持 state_mode。
- `dexmg.py::_process_obs` + `_get_expected_low_dim_keys`:state_mode=eef_piece 时拼 rel_piece(从 sim)。
- `train_hiql_value.py`:`--state_mode`,算+存 rel_piece stats。
- `hiql_value.py::save_value/load_value`:存/读 `state_mode` + rel_piece stats。
- `hiql_potential.py` / `train_chunk_residual.py`:③b 两端喂 30 维(online 从 sim、offline 从 buffer)。
- `tests/`:`compute_eef_rel_piece` 单测 + **online/offline 同源一致性测试** + 维度/回归测试。

## Risks
1. **online/offline 不同源 → V 失效**(最致命)。缓解:同一函数 + 一致性测试(同一 sim state,两端算出 bit-一致的 rel_piece)。
2. offline replay env 与 online env 的 piece body 命名/坐标系不一致。缓解:测试断言 `piece_*_root` 两边一致 + 同一函数。
3. 标准化漂移:online 的 rel_piece 会超出 demo stats 范围(这正是"偏离"要捕捉的)。决策:**不裁剪**,让 V 自然外推(正是要观察的高估缓解)。
4. **白训风险**:维度/标准化错会整轮白训。缓解:plan 里先小规模冒烟(几条 demo 算 stats + 训几百步 + ③b smoke)验证链路,再全量重训。

## Testing(TDD)
- `compute_eef_rel_piece` 纯函数单测(已知 sim state → 已知 rel)。
- **同源一致性**:构造一个 sim state,online 路径 vs offline replay 路径算出的 rel_piece 必须一致。
- 维度/回归:state_mode=eef_piece → state_dim==30;eef → 18 且与现有逐位等价。
- 端到端冒烟:小规模训 value(30 维)+ ③b smoke(照今天 hiql smoke 的口径)。

## Validation(成功判据)
复用今天的 V 曲线工具:跑一条 online **失败** episode,对比 `18维V` vs `30维V`——30 维 V 应在"手到位件没对上"段**不再虚高**(高估缓解)。

## Out of scope
- 姿态相对量(`eef_quat_rel_piece`):先只位置,不够再加。
- piece 间相对位姿(`piece1_rel_piece2`):本版用 eef-rel-piece 间接覆盖。
- 不碰正在跑的 B run(18 维);这是并行的下一版。
- ViT / 图像 value:更远期。

## Open questions(请 review 拍板)
1. **rel_piece 表示**:用世界系差(`eef-piece`)够,还是转 eef-frame(更平移/旋转不变但要 quat 运算)?**建议:先世界系差**(简单、够用)。
2. **维度**:12 维(双臂×两 piece)全要,还是每臂只对"当前目标 piece"?**建议:先全要**(信息全、维度也不大)。
3. **跑不跑实验**:这版实现+冒烟到位后,现在就起一臂,还是等 B run 结果再决定?**建议:实现+冒烟到位,起不起等 B**。
