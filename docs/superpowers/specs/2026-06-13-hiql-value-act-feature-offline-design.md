# 离线 HIQL value 接冻结 ACT encoder 特征(图像+本体)作为 state(flag-gated)设计

- 日期:2026-06-13
- 状态:设计已定稿,待用户审阅 → 写实施计划
- 范围:**只做①离线 value 预训练**——给 `gc_value`(+ `high_actor`,及共用汇聚点 `read_per_demo_states`)新增一种 `state_mode="act_feat"`,把"每帧 state"从 `eef`/`eef_piece` 换成"冻结 ACT transformer encoder 池化嵌入 ⊕ 原始本体"。**默认 `eef`/`eef_piece` 逐位不变**。明确**不做**:②在线子目标注入(rollout 实时 z)、act_feat vs eef_piece 性能 A/B、LIBERO——各留后续独立 spec。
- 姊妹设计:`2026-06-12-hiql-value-pi0-feature-offline-design.md`(pi0/pi05 版)。本设计是其"base=ACT"的对等变体,并因 ACT 在 residual 环境内、又小又快而大幅简化(无环境分离、无 JAX→torch 转换、就地 cache-or-build)。

## 1. 背景与动机

resfit 现有 HIQL value(路线B:`gc_value` + `high_actor`)的 state 口径有两档(`train_hiql_value.py:90-91`):`eef`(18 维双臂 eef_pos+eef_quat+gripper_qpos)与 `eef_piece`(30 维 = eef18 + 物体相对位姿 rel_piece12)。其中 rel_piece12 **不在数据里**,要 MuJoCo 逐帧 replay 从 sim 读物体 root body 世界 pose(`object_state.py:39-48` `get_body_xpos`),且**在线 rollout 也实时从 sim 读**(`hiql_subgoal.py` → `compute_eef_rel_piece_from_env`)。后果:

- **特权信息、不对齐部署**:rel_piece 是仿真特权,真机相机拿不到 → 当前 value/subgoal 路本质 sim-only。
- **pi0_feat 只覆盖 base=pi0**:姊妹设计用冻结 pi05 prefix 特征替换特权 state,但只在 base=pi0 时可用;而 resfit 的 dexmg 主线(three-piece / threading / pouring / lift-tray)base 是 **ACT**(`chunk_act_base.py` 全程操作 `ACTPolicy`)。

本设计给 base=ACT 的主线一条**部署诚实**的 state 源:只用相机图像 + 本体(真机都有),**复用 base policy(ACT)已编码好的 image+proprio 融合表征**(冻结 ACT transformer encoder 的池化输出),省去从零训视觉 backbone。砍掉特权物体 pose 后,replay 依赖随之消失。

**为什么是 image+proprio 而非只 proprio**:砍掉物体 pose 后只本体的 value "看不见物体",对 manipulation 判断进度太弱;图像是部署合法前提下把物体信息找回来的途径。

**为什么取 encoder 输出而非 ResNet 原始特征**:对标 pi0_feat 取 paligemma prefix 池化隐状态(出动作前的多模态融合表征);ACT 的直接类比是 transformer **encoder 输出**(decoder 交叉注意力的"记忆"),它已融合 image+state,比纯 ResNet feature map 表征更强。

## 2. 关键约束(命门)

1. **默认零影响**:不传 `--state_mode act_feat` 时,`eef`/`eef_piece` 全路径**逐位等价**于现状;现有测试与 parser 默认全绿。
2. **冻结复用,不重训 backbone**:ACT `eval()` + `no_grad`,参数不进优化器;只训新加的小 MLP value 头 + φ(沿用现有结构)。
3. **离线缓存**:ACT 过全 demo(几十万帧 × 3 相机)虽便宜但非免费;嵌入缓存(仿 `state30_cache`),命中跳过 ACT 前向。提供 fp16 存盘选项。
4. **同源**:offline 训出的 `gc_value.pt`/`high_actor.pt` 记 `state_mode + 提取签名`,加载时断言一致,杜绝异源拼接。
5. **预处理一致**:图像/本体预处理必须与 ACT 逐位同款 —— 直接复用 ACT 的 `normalize_inputs` + 图像重组,不手搓。
6. **不改 ACT**:沿用 `chunk_act_base.py` 纪律,只复用 ACT 组件、不修改其源码。

## 3. 环境(相对 pi0_feat 的核心简化)

ACT(`ACTPolicy`)在 resfit 的 `residual` 环境(`/mnt/mnt/data/envs/residual`)内可直接 import、又小又快。因此**不需要** pi0_feat 那套环境分离(openpi venv)、JAX→torch 转换、cache-required 跨环境约束。`act_feat` 的嵌入计算 + 缓存**就地在 residual 进程内**完成,与现有 `state30_cache.load_or_build_state30`(进程内 MuJoCo replay 建缓存)同一路子。

## 4. 设计

### 4.1 `ActFeatureExtractor`(新文件 `act_feature.py`,纯特征器,不改 ACT)
- `__init__(act_policy, device, image_keys, proprio_key, pooling="mean")`:持有冻结 `ACTPolicy`,`eval()` + `requires_grad_(False)`。
- `embed_batch(raw_obs: dict) -> Tensor[B, D_emb + D_proprio]`:
  1. **复用 ACT 现成预处理**:`act_policy.normalize_inputs(raw_obs)` + 图像重组 `batch["observation.images"]=[batch[k] for k in image_features]`(同 `modeling_act.py:164-167` / `chunk_act_base.py:14-16`);
  2. **跑到 encoder 输出为止**:复用 ACT `model` 内 backbone + transformer encoder,取 `encoder_out`(`modeling_act.py:582` 的输出),**不进 decoder、不出动作**;`use_vae=False` 下 latent=零 → 确定性;
  3. **池化**:`encoder_out` 是一组 token(latent 位 + state 位 + 各相机 image patch token),做 **mean pooling**(无因果序集合,均值最自然)→ `[B, D_emb]`;
  4. **concat 原始本体** `raw_obs[proprio_key]`(18 维 eef)→ `[B, D_emb + D_proprio]`,cast float32。
- 属性 `feature_dim`;`signature() -> dict`(ACT ckpt 标识、image_keys、proprio_key、pooling)供缓存/同源校验。
- **确定性**:`eval()` 关 dropout、无 VAE 采样;断言同输入同输出。
- **实现注**:取 `encoder_out` 需要一个干净 hook。ACT `model.forward`(`modeling_act.py:469-601`)内部算完 `encoder_out` 后即进 decoder;helper 复用其 encoder 段(backbone + `encoder_in_tokens` 组装 + `self.encoder(...)`)到 `encoder_out`,不改 ACT 源码(必要时用 ACT 暴露的子模块逐步调用)。Task 1 spike 先把这条取数路径在真 ACT 上验通(产出有限、确定性特征)。

### 4.2 `act_feat_cache.py`(仿 `state30_cache`,纯 numpy)
- `save_act_feat_cache(path, seqs, emb_stats, *, signature, fp16=False)`:存每条 demo 嵌入序列 + 整条 `[emb⊕proprio]` 归一化 `(mean,std)` + 完整签名;`fp16` 选项压磁盘。
- `act_feat_cache_reuse(path, *, signature, num_demos) -> (seqs, emb_stats) | None`:签名(dataset_id、num_demos、ACT ckpt 标识、image_keys、proprio_key、pooling)**全一致**才命中,否则 None → 重算。`num_demos` 非 None 的部分量**不落盘当全量**。

### 4.3 `read_per_demo_states` 加 `act_feat` 分支(改 `train_hiql_value.py`),cache-or-build(进程内)
保持现有契约 `-> (seqs, standardizer, aux_stats)`:
```
state_mode == "act_feat":
    命中 act_feat_cache → 返回 (seqs, None, emb_stats)   # seqs 已标准化;standardizer 占位 None
    未命中 → 用冻结 ACT(ActFeatureExtractor)遍历 hdf5 算嵌入 → 标准化 → 落盘 → 返回
```
- **归一化口径(命门)**:`act_feat` **不复用** `observation.state` 的 `StateStandardizer`。对整条 `[emb_raw ⊕ proprio_raw]` 算**一组** `(mean,std)`;`save_gc_value`/`save_high_actor` 把它存进 ckpt `mean/std`,供后续在线同款标准化。
- `eef`/`eef_piece` 两支**原样不动**(仍走 `StateStandardizer` + `rel_piece_stats`)。

### 4.4 接线(全 flag-gated)
- `train_hiql_value.py` / `train_hiql_gc_value.py` / `train_hiql_high_actor.py`:`--state_mode` 加 `act_feat` 枚举 + `--act_feat_cache`、`--act_base_ckpt`、`--act_image_keys`、`--act_proprio_key`、`--pooling`(默认 `mean`)。
  - **注**:`train_hiql_gc_value.py` 现在**写死** `state_mode="eef_piece"`(`:108`)→ 改成读 `--state_mode`(默认仍 `eef_piece`,逐位等价)。
- 守卫 `validate_act_feat_cfg(args)`:`act_feat` 缺 `--act_feat_cache` 且无法 build(缺 `--act_base_ckpt`)→ `ValueError`;非 `act_feat` 传了 `--act_*` → warn 忽略。
- **同 ACT 基座**:`act_feat` 默认用该 run 的同一个 ACT base ckpt;ckpt 标识进签名。
- `save_gc_value`/`save_high_actor`:存 `state_mode="act_feat"` + 签名;加载侧断言一致(延伸现有 dim/state_mode assert)。

### 4.5 数据集映射
- image_keys:dexmg `agentview_image, robot0_eye_in_hand_image, robot1_eye_in_hand_image`(pouring 为 `agentview + 左右手腕眼`)→ 由 `--act_image_keys` 指定,沿用 ACT `config.image_features`。
- proprio:直接读 `--act_proprio_key`(默认 `observation.state`,18 维 eef);维度进签名 + ckpt。

### 4.6 不变量
- 默认路逐位等价;`act_feat` 下 `seqs[i].shape[1] == D_emb + D_proprio` 且与 ckpt 签名一致;φ([g,s]) / build_gc_data / train_gc_value / train_high_actor 结构不变(只是 state_dim 变)。

## 5. 测试(TDD,结构+回归级,不依赖 GPU)

**回归锁**:现有 `eef`/`eef_piece` 测试全绿;`--state_mode` 默认仍 `eef`(value)/`eef_piece`(gc_value);整套 suite 通过。

**新单测(stub 假 ACT,不跑真权重)**:
1. `embed_batch` 形状/类型/确定性:输出 `[B, D_emb+proprio]`、float32、同输入同输出;mean 池化正确;proprio 拼在正确切片。
2. flag 守卫:`act_feat` 缺 cache 且缺 `--act_base_ckpt` → `ValueError`;eef 模式传 `--act_*` 被忽略;parser 默认不变。
3. 缓存:save→reuse 往返一致;签名(ACT id/image_keys/proprio_key/pooling/dataset_id/num_demos)不符 → reuse 返回 None;部分量不当全量落盘。
4. `read_per_demo_states` 的 `act_feat` 分支:小合成 hdf5(几帧假图+本体)+ stub extractor → 返回正确维度 seqs/standardizer/emb_stats;命中缓存跳过 extractor(spy 验证)。
5. 同源断言:签名/state_mode 不符的 ckpt 加载即报。

**集成冒烟(opt-in,真 ACT,默认 skip/标 slow)**:dexmg 取几帧跑真 extractor → 产出极小 `gc_value.pt`,断言训几步不 NaN、V 有限。这是"取数路径在真 ACT 上跑通"的完成判据。

## 6. 要改/新增的文件
- 新增 `resfit/rl_finetuning/chunk_residual/act_feature.py`(`ActFeatureExtractor` + 纯函数 pool/concat/signature;不改 ACT)
- 新增 `resfit/rl_finetuning/chunk_residual/act_feat_cache.py`(缓存,纯 numpy)
- 改 `train_hiql_value.py`(`read_per_demo_states` 加 cache-or-build 分支 + parser `--act_*` + `validate_act_feat_cfg`)
- 改 `train_hiql_gc_value.py`(state_mode 改可选 + parser + save 签名)
- 改 `train_hiql_high_actor.py`(state 源 dispatch + parser + save 签名)
- 新增 tests:`tests/test_act_feature.py`、`tests/test_act_feat_cache.py`、`tests/test_read_per_demo_states_act_feat.py`、`tests/test_act_feat_cli_wiring.py`(均 residual 环境可跑)

## 7. 前置依赖与风险
- **encoder_out 取数路径**:需在不改 ACT 的前提下干净取到 `encoder_out`。Task 1 spike 先在真 ACT 上验通(`modeling_act.py:546-582` 段),确认池化特征有限、确定性;不成立则回退"只池化 ResNet feature map"。
- **预处理一致性**:图像/本体预处理与 ACT 不一致 → 特征垃圾。必须复用 ACT 的 `normalize_inputs`/图像重组。
- **缓存体积**:D_emb = ACT `dim_model`(lerobot ACT 默认 512,以本 ckpt config 为准),帧×demo 累积可观,提供 fp16 缓存选项。
- **同 base 一致性**:act_feat 用的 ACT ckpt 应与残差 run 的 base 同一个;签名记录之,避免特征-策略权重错配。
- **build 时算力**:首次 build 需 ACT base 已加载 + GPU 过全 demo(一次性,之后命中缓存)。
