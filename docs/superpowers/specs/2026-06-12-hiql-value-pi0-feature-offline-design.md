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

### 3.0 环境分工(2026-06-12 修订:实测 `residual` 环境 import 不了 `openpi`)
实测:resfit 的 `residual` 环境(`/mnt/mnt/data/envs/residual`)有 torch/safetensors 但**无 `openpi`**;能 import `PI0Pytorch` 的是 **openpi 环境 `/mnt/mnt/data/chj/openpi/.venv`**。且本机**无 torch pi05 ckpt**,只有 JAX/orbax `pi05_base`(`/mnt/mnt/data/FPF_workspace/checkpoints/pi05_base`)。故采纳**环境分离(方案 X)**:
- **建缓存在 openpi 环境**:新增脚本 `build_pi0_feat_cache.py`,用 openpi venv 跑——先(一次性)`convert_jax_model_to_pytorch.py` 把 `pi05_base` JAX→torch,再加载冻结 torch pi05、遍历 hdf5、出**已标准化**的 pi0_feat 缓存 npz(seqs + feat_stats + 签名)。
- **value 训练在 residual 环境**:`read_per_demo_states(pi0_feat)` / gc_value / high_actor **只读缓存**(cache-required),**全程不 import openpi**;缓存缺失即报错引导先在 openpi 环境生成。
- `pi0_feature.py` 的纯函数(pool/concat/signature)只依赖 torch/numpy,两环境都可 import;其中加载真 pi05 的 `from_checkpoint`/`_model_prefix_pool` 走**惰性 import openpi**,仅 build 脚本(openpi venv)触达。
- **标准化口径搬到 build 时**:整条 `[emb⊕proprio]` 的 mean/std 在 build 脚本对全量算并写进缓存;residual 侧只加载,不再自算(§3.3 据此修订)。

### 3.1 `Pi0FeatureExtractor`(新文件 `pi0_feature.py`,纯特征器;由 build 脚本在 openpi 环境用)
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
保持现有契约 `-> (seqs, standardizer, aux_stats)`,但 **residual 侧 cache-required**(不在 residual 进程内跑 pi05):
```
state_mode == "pi0_feat":   # residual 环境
    cache_path 缺失/未命中 → raise(引导:先在 openpi 环境跑 build_pi0_feat_cache.py)
    命中 → 返回 (seqs, None, feat_stats)   # 缓存里 seqs 已标准化;standardizer 占位 None
```
- **不在 residual 进程内构造 Extractor、不读图像、不 import openpi**。嵌入计算 + 标准化全在 build 脚本(openpi 环境,§3.0)完成并写入缓存。
- **归一化口径(命门)**:`pi0_feat` **不复用** `observation.state` 的 `StateStandardizer`(proprio_key 可配)。build 脚本对整条 `[emb_raw ⊕ proprio_raw]` 算**一组** `(mean,std)` 写进缓存;`save_gc_value`/`save_high_actor` 把它存进 ckpt `mean/std`,供后续在线同款标准化。
- `eef`/`eef_piece` 两支**原样不动**(仍走 `StateStandardizer` + `rel_piece_stats`)。

### 3.4 接线(全 flag-gated)
- **build 脚本(openpi 环境)**`build_pi0_feat_cache.py` 的 flags:`--pi0_ckpt`(torch pi05 目录)`--hdf5 --dataset --image_keys --proprio_key --prompt --pooling --out_cache [--num_demos]`。
- **residual 侧训练** `train_hiql_value.py` / `train_hiql_gc_value.py` / `train_hiql_high_actor.py`:`--state_mode` 加 `pi0_feat` 枚举 + `--pi0_feat_cache`(指向 build 产物,**须已存在**)。**不加 `--pi0_ckpt` 等**(那是 build 脚本的)。
  - **注**:`train_hiql_gc_value.py` 现在**写死** `state_mode="eef_piece"`(`:108`)→ 改成读 `--state_mode`(默认仍 `eef_piece`,逐位等价)。
- 守卫 `validate_pi0_feat_cfg(args)`(residual 侧):`pi0_feat` 缺 `--pi0_feat_cache` 或文件不存在 → `ValueError`(引导先 build);非 `pi0_feat` 传了 `--pi0_feat_cache` → warn 忽略。
- `save_gc_value`/`save_high_actor`:存 `state_mode="pi0_feat"` + 缓存里读出的签名;加载侧断言一致(现有 dim/state_mode assert 延伸)。

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
- 新增 `resfit/rl_finetuning/chunk_residual/verify_pi0_prefix_feature.py`(spike;openpi 环境跑;含 load_frozen_pi05/build_observation/prefix_feature 真实底稿)
- 新增 `resfit/rl_finetuning/chunk_residual/pi0_feature.py`(纯函数 + Pi0FeatureExtractor;惰性 import openpi)
- 新增 `resfit/rl_finetuning/chunk_residual/pi0_feat_cache.py`(缓存,纯 numpy)
- 新增 `resfit/rl_finetuning/chunk_residual/build_pi0_feat_cache.py`(**openpi 环境**入口:转换/加载 pi05 → 出缓存)
- 改 `train_hiql_value.py`(`read_per_demo_states` 加 cache-required 分支 + parser `--pi0_feat_cache` + `validate_pi0_feat_cfg`)
- 改 `train_hiql_gc_value.py`(state_mode 改可选 + parser + save 签名)
- 改 `train_hiql_high_actor.py`(state 源 dispatch + parser + save 签名)
- 新增 tests:`tests/test_pi0_feature.py`、`tests/test_pi0_feat_cache.py`、`tests/test_read_per_demo_states_pi0_feat.py`、`tests/test_pi0_feat_cli_wiring.py`(均 residual 环境可跑,不 import openpi)

## 6. 前置依赖与风险(2026-06-12 实测更新)
- **环境(已实测)**:`residual`(`/mnt/mnt/data/envs/residual`)有 torch/safetensors、**无 openpi**;openpi torch 栈在 **`/mnt/mnt/data/chj/openpi/.venv`**(可 import `PI0Pytorch`)。→ 据此定 §3.0 环境分离。
- **无 torch pi05 ckpt**(命门前置):本机只有 JAX/orbax `pi05_base`(`/mnt/mnt/data/FPF_workspace/checkpoints/pi05_base`,确认 orbax)。**Task 1 spike 先在 openpi 环境**:`convert_jax_model_to_pytorch.py --checkpoint_dir <含'pi05'的目录> --output_path <out> --config_name <pi05 config>` 转 torch,再 load + prefix 前向验有限特征。不成立则回退方案2。注:转换/spike 需 GPU 授权;部署同款 finetuned pi05 在远端 `/home/dex`,首版用通用 `pi05_base` 证机制即可(特征-策略权重一致性留作后续 refinement)。
- **build 脚本环境依赖**:openpi venv 需有 h5py/numpy(待 Task 1 一并确认)。
- **预处理一致性**:图像预处理与 base policy 不一致 → 特征垃圾。必须复用现成预处理。
- **缓存体积**:D_emb 较大(pi05 gemma width)→ 帧×demo 累积可观,提供 fp16 缓存选项。
- **pi05 state 内部通道**:pi05 跳过 suffix state token(`pi0_pytorch.py:244 if not self.pi05`),故 proprio 是否已隐含在 prefix 不确定 → 设计统一**显式 concat 原始 proprio**,无论 pi05 内部如何都保证本体在场(冗余但安全)。
