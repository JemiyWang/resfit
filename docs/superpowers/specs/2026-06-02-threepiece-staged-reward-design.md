# ThreePiece stage 4→5 拆细 + collection-time staged 稠密奖励 — 设计

日期:2026-06-02
分支:chunk-residual-validation
接续:HANDOFF_2026-06-02.md(§6 负结果 / §7 下一步 item 1+2)

## 0. 一句话
给 chunk-RL 的 critic 一个**按 stage 推进的稠密奖励**(收集时 additive 加分,flag 门控),
并把瓶颈段 stage2→3 拆成两段以提高信用分辨率,验证能否解除"critic 没信号"导致的负结果。

## 1. 背景 & 假设
- 已证负结果根因:稀疏 {0,1} 奖励 + 在线极少成功 → critic 无信号(chunk 版 target_q≈0,
  原版单步 critic_loss≡0)→ actor 的 -Q 梯度是噪声 → 残差乱推毁基座。
- 奖励稀疏度探针(25 ep):最高 stage 分布 {0:2, 1:2, 2:16, 3:5},**64% 卡在 stage 2**。
  大量"对中间阶段成功、对最终失败"的轨迹在稀疏奖励下全是 0 —— 正是稠密 staged 奖励的用武之地。
- **可证伪假设**:加 staged 稠密奖励后,(a) critic_loss 不再恒 0;(b) target_q 出现 stage 结构;
  (c) success 能否超过基座开环 ~20-30%。任一不成立都是有价值的结果。

## 2. 范围
- 一体交付:stage 4→5 拆细 + collection-time staged 奖励,全程 TDD。
- staged 奖励用开关门控,**默认关闭**,保证既有负结果 baseline 逐位可复现。
- 长跑(300k)由用户启动;本切片交付到 代码+全绿单测+真环境探针+smoke+ready 命令 为止。

## 3. 明确不做(YAGNI)
- advantage-gated residual 止损(handoff item 3)
- stage-conditioned / goal-conditioned critic(handoff 4a)
- potential-based shaping(本切片先用最直接的 additive;若 shaping 偏置策略再议)
- 真·goal-conditioned HER(改 actor/critic 输入维度 + 采样器,handoff 已 deprioritized)
- flow vs raw 的 chunk 级 A/B —— 等 RL 先在某任务学得动再说

## 4. 组件改动(全部在 chunk_residual/ 自有文件;**零新增原仓库改动**)

### 4.1 stage_detectors.py — 4 段 → 5 段
新编号:`0 起步 / 1 piece1抓 / 2 piece1放好释放 / 3 piece2抓起 / 4 成功`

threepiece_stage(env) 逻辑(瞬时阶段;闩锁仍由 wrapper 负责):
```
if env._check_second_piece_is_assembled():            # == _check_success
    return 4
grasped2 = _grasped(env, env.piece_2)
if env._check_first_piece_is_assembled() and grasped2:
    return 3                                           # piece1 已装 且 正握 piece2
grasped1 = _grasped(env, env.piece_1)
if env._check_first_piece_is_assembled() and not grasped1:
    return 2
if grasped1:
    return 1
return 0
```
- NUM_STAGES["TwoArmThreePieceAssembly"] = 5
- 复用已修好的 _grasped(对 dict-gripper 取 .values())。
- **真·未知**:`env.piece_2` 属性名需真环境探针确认(detector 此前只用过 env.piece_1)。
  若属性名不同(如 env.pieces[1]),探针阶段修正并记录。

### 4.2 chunk_env_wrapper.py — collection-time staged 奖励
- 构造新增参数 `stage_reward_bonus: float = 0.0`(=0 即关闭 → 行为与现状逐位相同)。
- 纯逻辑小函数(可单测,放本文件或 stage_replay 旁):
  `staged_bonus(start_stage, end_stage, bonus) -> bonus * max(0, end_stage - start_stage)`
- step() 改动:
  - 开头记 `start_stage = self._stage`(入 chunk 前的闩锁值)。
  - 循环已维护 `max_in_chunk`(归零前峰值)。
  - `total_reward += staged_bonus(start_stage, max_in_chunk, self.stage_reward_bonus)`,
    在 return 前完成 → add_chunk_transition 存的是已加分 reward → MultiStepTransform 自动折进 n-step。
  - **不设 done**(纯 shaping,非终止)。final success 的 env +1 照常,与 staged bonus 叠加(additive)。

### 4.3 train_chunk_residual.py — CLI 接线
- 新增 `--staged_reward`(store_true)、`--stage_reward_bonus`(float,默认 1.0)。
- 训练 env 构造传 `stage_reward_bonus = args.stage_reward_bonus if args.staged_reward else 0.0`。
- **eval env 恒传 0.0**(保证 eval success_rate 不被 shaping 污染)。
- buffer schema 不变(reward / max_stage / observation.stage_id 字段都已在)。

## 5. 数据流 & 不变量
```
worker detector(0..4) → info["stage_id"]
  → wrapper latch self._stage / max_in_chunk
  → total_reward += bonus*Δ(Δ=max_in_chunk - start_stage)
  → add_chunk_transition → buffer.add(MultiStepTransform 烤 n-step)
  → (stage-balanced) 采样 → QAgent.update
```
- stage_id / max_stage 值域由 0..3 变宽到 0..4;stage-balanced 采样与 stage-diag 已按
  "present 阶段"动态分桶/聚合,**无需改**。

## 6. 边界条件
- done-mid-chunk:start_stage 是入 chunk 前闩锁值;max_in_chunk 是归零前峰值 → Δ 正确;
  下一 chunk 从 0 起。
- Δ 由闩锁单调性保证非负;staged_bonus 再 max(0,·) 兜底。
- eval bonus=0 → 即便 run_dexmg_evaluation 读 reward 也无污染。
- bonus=0 回归:wrapper 行为与改前逐位相同(回归测试守住)。

## 7. 测试(TDD,先写后实现)
detector(test_stage_detector.py):
- 新增:piece1已装+piece2握 → 3;success → 4。
- 改既有契约:success_is_3 → is_4;priority 测试 → 4;first_assembled_is_2 不变;NUM_STAGES==5。
- 扩 fake env:加 piece_2;_check_grasp 能按 object_geoms 区分 piece1/piece2(或 fake 加 grasp2 字段)。

staged_bonus 纯函数:
- Δ=0 → 0;Δ=2,bonus=1 → 2;Δ<0 → 0;bonus=0 → 0。

wrapper 集成:
- bonus>0:step 的 total_reward 含 bonus*Δ。
- bonus=0:逐位等于现状(回归)。
- done-mid-chunk:用归零前峰值算 Δ(配合 _StageVecEnv term_at)。

既有 35 个 stage 用例其余保持绿。

## 8. 交付边界 & 验证判据
交付物:
1. 代码(4.1–4.3)+ 全绿单测。
2. 真环境 stage 探针(仿 verify_stage_instant):确认 env.piece_2 属性名 & stage3 真触发,
   并打印 5 段瞬时分布。
3. smoke 跑通(--smoke,bonus>0 路径不崩)。
4. 一条 ready 300k 命令(GPU / MUJOCO_GL=egl / PYOPENGL_PLATFORM=egl / TMPDIR=/data2 /
   --staged_reward --stage_reward_bonus 1.0 --stage_balanced)。

验证判据(用户跑长跑后看):
- critic_loss 不再恒 0;
- [stage-diag] 的 target_q 出现 stage 结构(不再全 ≈0);
- eval success_rate 能否超过基座开环 ~20-30%。

## 9. 风险 & 缓解
- env.piece_2 属性名未知 → 探针先确认再下结论(已列入交付物)。
- shaping 偏置策略(additive 非 potential-based)→ 本切片接受;若 success 上不去且诊断显示
  策略被 shaping 带偏,下一步换 potential-based。
- bonus 量级(默认 1.0/段)可能淹没 final +1 → 量级是 CLI 可调旋钮;先用 1.0 求"唤醒 critic",
  再按诊断调。
