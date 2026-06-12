# 离线 HIQL value 接冻结 pi0 prefix 特征(图像+本体)作为 state(flag-gated)设计

- 日期:2026-06-12
- 状态:设计已定稿,待写实施计划
- 范围:**只做①离线 value 预训练**——给 `gc_value`(+ `high_actor`,以及共用汇聚点 `read_per_demo_states`)新增一种 `state_mode="pi0_feat"`,把"每帧 state"从 `eef`/`eef_piece` 换成"冻结 pi05 prefix 池化嵌入 ⊕ 原始本体"。**默认 `eef`/`eef_piece` 逐位不变**。明确**不做**:②在线子目标注入(rollout 实时 z)、③LIBERO 单臂 env/rollout/eval、pi0_feat vs eef_piece 的性能 A/B——各留后续独立 spec。

## 1. 背景与动机

resfit 现有 HIQL value(路线B:`gc_value` + `high_actor`)的 state 口径写死 `eef_piece`(30 维 = 双臂 eef18 + 物体相对位姿 rel_piece12)。其中 rel_piece12 不在数据里,要 **MuJoCo 逐帧 replay** 取物体世界 pose,且 `STATE18_KEYS` 硬绑 dexmg 双臂 obs 键。后果两条:
- **LIBERO 跑不了**:LIBERO 单臂、无 `robot1_*`,物体 pose 埋在 `states` 的自由关节 qpos 里、要解析 model 或 replay 才能切出。
- **不对齐真机部署**:rel_piece 是仿真特权信息,真机相机拿不到。

参考 HIQL 原版(`/mnt/mnt/data/wjm/residual/HIQL`)的 value 只吃**数据集原生观测**(state 域切前 N 维、像素域过 CNN 编码),从不 replay。本设计据此给 resfit 加一条**部署诚实**的 state 源:只用相机图像 + 本体 + prompt,这些真机都有;并**复用 base policy(pi05)已编码好的 image+proprio 表征**(冻结 pi05 的 prefix 池化嵌入),省去从零训视觉 backbone。一旦砍掉特权物体 pose,replay 依赖随之消失,LIBERO 也能训 value。

**为什么是 image+proprio 而非只 proprio**:砍掉特权物体 pose 后,只本体的 value "看不见物体",对 manipulation 判断进度太弱;图像是部署合法前提下把物体信息找回来的唯一途径。

## 2. 关键约束(命门)

1. **默认零影响**:不传 `--state_mode pi0_feat` 时,`eef`/`eef_piece` 全路径**逐位等价**于现状;现有测试与 parser 默认全绿。
2. **冻结复用,不重训 backbone**:pi05 `eval()` + `no_grad`,参数不进优化器;只训新加的小 MLP value 头 + φ(沿用现有结构)。
3. **离线缓存**:整条 demo 跑 pi05 很贵,嵌入必须缓存(仿 `state30_cache`),命中跳过 pi05 前向。
4. **同源**:offline 训出的 `gc_value.pt`/`high_actor.pt` 记 `state_mode + 提取签名`,后续(在线/high_actor)加载时断言一致,杜绝异源拼接。
5. **预处理一致**:图像预处理(resize/归一化/通道序)必须与 base policy 逐位同款,复用 openpi/pi05 adapter 现成预处理,不手搓。

## 3. 设计

### 3.1 `Pi0FeatureExtractor`(新文件 `pi0_feature.py`,纯特征器)
- `__init__(ckpt, device, image_keys, proprio_key, pooling="last", prompt=None)`:加载冻结 torch `PI0Pytorch`(pi05,`openpi.models_pytorch.pi0_pytorch`),`eval()`+`requires_grad_(False)`。
- `embed_batch(images: dict[str, Tensor], proprio: Tensor, prompts) -> Tensor[B, D]`:
  1. 按 base policy 同款预处理图像;
  2. `embed_prefix(images, img_masks, lang_tokens, lang_masks)`;
  3. paligemma **prefix-only 前向**(对应 `pi0_pytorch.sample_actions` 里填 KV cache 的那次 `paligemma_with_expert.forward(inputs_embeds=[prefix, None], ...)`),取 prefix hidden states;
  4. 池化:`last`=末有效 token / `mean`=有效 token 均值;
  5. **concat 原始本体** `proprio` → `[B, D_emb + D_proprio]`,cast float32。
- 属性 `feature_dim`;`signature() -> dict`(pi05 ckpt 标识、image_keys、proprio_key、pooling、prompt)供缓存/同源校验。
- **确定性**:断言无 dropout/采样;同输入同输出。

### 3.2 `pi0_feat_cache.py`(仿 `state30_cache`)
- `save_pi0_feat_cache(path, seqs, emb_stats, *, signature)`:存每条 demo 嵌入序列 + 嵌入归一化 `(mean,std)` + 完整签名。允许 **fp16** 存盘选项(磁盘体积 = D_emb 浮点 × 帧 × demo)。
- `pi0_feat_cache_reuse(path, *, signature, num_demos) -> (seqs, emb_stats) | None`:签名(含 dataset_id、num_demos、pi05 ckpt 标识、image_keys、proprio_key、pooling、prompt)**全一致**才命中,否则 None → 重算。`num_demos` 非 None 的部分量**不落盘当全量**。

### 3.3 `read_per_demo_states` 加 `pi0_feat` 分支(改 `train_hiql_value.py`)
保持现有契约 `-> (seqs, standardizer, aux_stats)`:
```
state_mode == "pi0_feat":
    命中 pi0_feat 缓存 → 直接返回 (seqs, standardizer, feat_stats)
    否则:逐 demo 读 hdf5 图像键 + proprio_key
          → Pi0FeatureExtractor.embed_batch(批量) 得每帧 [emb_raw ⊕ proprio_raw]
          → 对整条 [emb_raw ⊕ proprio_raw] 自算单组 (mean,std) 标准化
          → 拼成 seqs;num_demos 全量时落盘缓存
    返回 (seqs, standardizer, feat_stats)   # feat_stats=(mean,std) 占 aux_stats 槽(类比 rel_piece_stats)
```
- **归一化口径(命门)**:`pi0_feat` **不复用** `observation.state` 的 `StateStandardizer`(proprio_key 可配、未必等于 observation.state)——而是对整条 `[emb_raw ⊕ proprio_raw]` 自算**一组** `(mean,std)`。该 stats 同时:(a) 作为 `feat_stats` 返回;(b) 由 `save_gc_value`/`save_high_actor` 存进 ckpt 的 `mean/std` 字段,供后续在线对实时 `[emb⊕proprio]` 同款标准化。返回的 `standardizer` 在此模式仅占位(下游 pi0_feat 路不依赖它做 proprio 标准化)。
- `eef`/`eef_piece` 两支**原样不动**(仍走 `StateStandardizer` + `rel_piece_stats`)。

### 3.4 接线(4 处,全 flag-gated)
- `train_hiql_value.py` / `train_hiql_gc_value.py` / `train_hiql_high_actor.py`:`--state_mode` 加 `pi0_feat` 枚举;新增仅 `pi0_feat` 用的 `--pi0_ckpt/--pi0_image_keys/--pi0_proprio_key/--pi0_prompt/--pi0_pooling/--pi0_feat_cache`。
  - **注**:`train_hiql_gc_value.py` 现在**写死** `state_mode="eef_piece"`(`:108`)→ 改成读 `--state_mode`(默认仍 `eef_piece`,逐位等价)。
- 守卫 `validate_pi0_feat_cfg(args)`:`pi0_feat` 缺 `--pi0_ckpt`/图像键 → `ValueError`;非 `pi0_feat` 传了 `--pi0_*` → warn 忽略(仿现有 `validate_stage_cache`,在重活前 fail-fast)。
- `save_gc_value`/`save_high_actor`:存 `state_mode="pi0_feat"` + 签名;加载侧断言一致(现有 dim/state_mode assert 延伸)。

### 3.5 数据集映射
- image_keys:dexmg `agentview_image,robot0_eye_in_hand_image,robot1_eye_in_hand_image`;LIBERO `agentview_rgb,eye_in_hand_rgb` → 由 `--pi0_image_keys` 指定(沿用 pi05 adapter 的 image_key_map 思路)。
- proprio:dexmg 双臂 / LIBERO 单臂维度不同 → 直接读对应 `--pi0_proprio_key`,**不需双臂对齐**;维度进签名+ckpt。

### 3.6 不变量
- 默认路逐位等价;`pi0_feat` 下 `seqs[i].shape[1] == D_emb + D_proprio` 且与 ckpt 签名一致;φ([g,s]) / build_gc_data / train_gc_value / train_high_actor 结构不变(只是 state_dim 变)。

## 4. 测试(TDD,结构+回归级;不强依赖 GPU)

**回归锁**:现有 `eef`/`eef_piece` 测试全绿;`--state_mode` 默认仍 `eef`(value)/`eef_piece`(gc_value);整套 suite 通过。

**新单测(stub 假 pi05,不跑真权重)**:
1. `embed_batch` 形状/类型/确定性:输出 `[B, D_emb+proprio]`、float32、同输入同输出;`last` vs `mean` 池化有别;proprio 拼在正确切片。
2. flag 守卫:`pi0_feat` 缺 ckpt → `ValueError`;eef 模式传 `--pi0_*` 被忽略;parser 默认不变。
3. 缓存:save→reuse 往返一致;签名(pi05 id/image_keys/proprio_key/pooling/prompt/dataset_id/num_demos)不符 → reuse 返回 None;部分量不当全量落盘。
4. `read_per_demo_states` 的 `pi0_feat` 分支:小合成 hdf5(几帧假图+本体)+ stub extractor → 返回正确维度 seqs/standardizer/emb_stats;命中缓存跳过 extractor(spy 验证)。
5. 同源断言:签名/state_mode 不符的 ckpt 加载即报。

**集成冒烟(opt-in,真 pi05,默认 skip/标 slow)**:dexmg 与 LIBERO 各取几帧跑真 extractor → 产出极小 `gc_value.pt`,断言训几步不 NaN、V 有限。这是"两种 hdf5 都跑通"的完成判据,门控起来不拖累单测。

## 5. 要改/新增的文件
- 新增 `resfit/rl_finetuning/chunk_residual/pi0_feature.py`(Pi0FeatureExtractor)
- 新增 `resfit/rl_finetuning/chunk_residual/pi0_feat_cache.py`(缓存)
- 改 `train_hiql_value.py`(`read_per_demo_states` 加分支 + parser)
- 改 `train_hiql_gc_value.py`(state_mode 改可选 + parser + save 签名)
- 改 `train_hiql_high_actor.py`(parser + save 签名)
- 新增 tests:`tests/test_pi0_feature.py`、`tests/test_pi0_feat_cache.py`、`read_per_demo_states` pi0_feat 分支测试

## 6. 前置依赖与风险
- **torch pi05 可加载**(命门前置):本机有 `PI0Pytorch` 类 + `convert_jax_model_to_pytorch.py` + pi05 权重(如 `FPF_workspace/checkpoints/pi05_base`)。**实施计划第一步任务**先验证"加载冻结 torch pi05 + prefix 前向出特征"成立(只有 JAX 就先转换);不成立则回退方案2(JAX 特征器+cache 解耦,见 brainstorm 记录)。
- **预处理一致性**:图像预处理与 base policy 不一致 → 特征垃圾。必须复用现成预处理。
- **缓存体积**:D_emb 较大(pi05 gemma width)→ 帧×demo 累积可观,提供 fp16 缓存选项。
- **pi05 state 内部通道**:pi05 跳过 suffix state token(`pi0_pytorch.py:244 if not self.pi05`),故 proprio 是否已隐含在 prefix 不确定 → 设计统一**显式 concat 原始 proprio**,无论 pi05 内部如何都保证本体在场(冗余但安全)。
