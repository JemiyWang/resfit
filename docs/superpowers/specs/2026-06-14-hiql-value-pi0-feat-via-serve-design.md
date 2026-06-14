# 离线 HIQL value 接 pi05 prefix 特征——经 serve 透出(C 方案)设计

- 日期:2026-06-14
- 状态:设计已定稿,待写实施计划
> **⚠️ 2026-06-14 修订(应用户要求,零碰 openpi 仓)**:原设计把 serve 侧 wrapper(`feature_policy.py`/`serve_with_feat.py`/其测试)放 openpi 仓。改为**全部放 resfit 仓的 `pi0_serve/` 目录**(`/mnt/mnt/data/resfit/pi0_serve/`,与 `resfit/` 包平级、自包含),**运行时借 openpi 的 `.venv` 解释器**(`/mnt/mnt/data/chj/openpi/.venv/bin/python`,因需 import openpi 跑 pi05 prefix 前向;已实测从 resfit cwd 可 import openpi、`make_attn_mask` 在 `openpi.models.pi0`)。openpi 仓**零文件、零 commit**。下文 §3.0/§3.1/§3.2/§5 凡写 `src/openpi/...`/`scripts/...` 一律改读 `pi0_serve/...`,commit 一律到 resfit 仓。build/cache/train(resfit 包内,residual 环境)不变。

- 范围:**只做①serve 透特征 + ②离线 pi0_feat 训 `gc_value`**。给 base policy 的 serve(openpi)加一层零侵入 wrapper,在 `infer` 返回里**额外透出 pi05 prefix 池化特征**;resfit 侧(residual 环境)用 `openpi_client` 调 serve 批量出**特征缓存**,再以 `state_mode="pi0_feat"` 训出 `gc_value`,验证"pi05 图像特征当 HIQL value 的 state 比 `eef_piece` 好不好"。**明确不做**:`high_actor`、在线 subgoal 注入(rollout 实时出 z)、pi0_feat vs eef_piece 的正式性能 A/B——各留后续独立 spec。

## 1. 背景与动机

resfit 路线B(`gc_value` + `high_actor`)的 value state 现支持 `eef_piece`(30 维 = 双臂 eef18 + 物体相对位姿 rel_piece12,sim 特权 replay 取)与 `act_feat`(冻结 ACT encoder 特征)。pi05 当 base 时,既想给 value 一个**含图像**的 state、又**部署诚实**(不依赖 sim 特权),自然想到"冻结 pi05 prefix 特征当 state"(pi0_feat,离线 spec 见 `2026-06-12-hiql-value-pi0-feature-offline-design.md`)。

那份离线 spec 的特征器走"在 openpi 环境用 torch/JAX 进程内加载 pi05 抽特征"。但本机实测两条约束:
- **residual 环境(`/mnt/mnt/data/envs/residual`)无 `jax`/`flax`/`openpi` 本体**,只有 `torch` + `openpi_client`(websocket 客户端);能 import `openpi`/JAX 的是 openpi 的 `.venv`。即 **pi05 前向不可能在 resfit 项目内跑**。
- 本机 pi05 权重只有 JAX/orbax(`pi05_base`、`pi0_libero` 等),无 torch safetensors;serve(`serve_policy.py`)跑的就是 JAX policy server。

**C 方案的洞察**:resfit 跑残差**本来就靠 serve 提供 base 动作**(`openpi_client.WebsocketClientPolicy` 连 serve,见 `lerobot/policies/pi05/load_pi05.py`)。既然 serve 进程已经在用同一份 pi05 跑前向,就让它在 `infer` 返回里**顺带透出 prefix 池化特征**——resfit 侧只多取一个字段。这样:
- **不转 torch、不在 resfit 内跑 pi05**(residual 环境用 `openpi_client` 调 serve 即可,连 openpi 环境都不必)。
- **特征与 base 动作天然同源**:特征由 serve(base policy 本体)用**同一份权重 + 同一套预处理**算出,逐位一致,直接消掉离线特征器 vs base 预处理对齐的命门。
- **离线/在线统一**:同一套"serve 透特征"机制,本 spec 用于离线 build 缓存,将来在线注入(下一个 spec)实时调同一个 serve。本 spec 只落地离线半边。

## 2. 关键约束(命门)

1. **协议不破坏、向后兼容**:openpi 的 websocket 协议是"server 把 `policy.infer(obs)` 返回的 dict 用 msgpack pack 发回"(`websocket_policy_server.py:61,71`)。透特征 = 让返回 dict **多一个 key** `prefix_feat`。现有 client 只取 `result["actions"]`(`libero_pi05_adapter.py:46`),多出的字段被忽略,**残差那条路零影响**。
2. **零侵入 openpi 上游**:用 `FeaturePolicy` wrapper 包住原 `Policy`,放在 **openpi 仓新增的 untracked 脚本/模块**里,**不改任何已 tracked 的 openpi 原始文件**(`policy.py`/`websocket_policy_server.py`/`pi0.py` 等全不动)。
3. **预处理同源**:wrapper 算 prefix 特征时,喂 `embed_prefix` 的 `Observation` 必须经原 policy 的 `_input_transform`(算 action 同一套),否则特征与 base 不同源。
4. **冻结复用**:wrapper 只前向、不训练;pi05 参数本就 eval/no-grad。特征前向是**独立第二次**(不复用 `sample_actions` 内部 prefix),换取 wrapper 零侵入;离线 build 一次性,多一次前向可忽略。
5. **离线缓存 + cache-required**:整条 demo 调 serve 很贵,特征必须缓存;resfit 侧训练只读缓存、缺失即 `raise`,全程不连 serve、不 import openpi。
6. **同源签名**:缓存与 `gc_value.pt` 记 `serve_ckpt_id`(人工锚定权重身份)+ pooling/prompt/image_keys/proprio_key/dataset/num_demos;后续(在线)加载断言一致,杜绝异源。
7. **默认零影响**:不传 `--state_mode pi0_feat` 时,`eef`/`eef_piece`/`act_feat` 全路径逐位等价;现有测试与 parser 默认全绿。

## 3. 设计

### 3.0 组件与环境分工
```
[1] openpi 仓(新增 untracked,不改原始文件) —— 在 openpi .venv(有 jax/openpi)
    scripts/serve_policy_with_feat.py:起 serve,用 FeaturePolicy 包原 policy
    src/openpi/serving/feature_policy.py:FeaturePolicy wrapper + pool 纯函数
        │ websocket(msgpack dict 加 key prefix_feat,向后兼容)
        ▼
[2] resfit 仓 / residual 环境(openpi_client 调 serve,不需 openpi 环境)
    build_pi0_feat_cache_via_serve.py:遍历 demo 每帧 → client.infer 取 prefix_feat
        → ⊕ proprio → 全量标准化 → 缓存 npz(签名含 serve_ckpt_id)
        │ 缓存 npz
        ▼
[3] resfit 仓 / residual 环境(只读缓存,不碰 serve)
    read_per_demo_states 加 pi0_feat 分支(cache-required)
    → train_hiql_gc_value --state_mode pi0_feat → gc_value.pt → verify_gc_value 看 gate
```

### 3.1 `FeaturePolicy`(openpi 仓新增 `src/openpi/serving/feature_policy.py`)
- 构造:持有原 `Policy` 实例 `inner`、`pooling`(`last`/`mean`)。复用 `inner._model`、`inner._input_transform`(同源命门)。
- `infer(obs) -> dict`:
  1. `outputs = inner.infer(obs)`(原样:actions/state/policy_timing,不动)。
  2. `obs_model = inner._input_transform(obs)`(与算 action 同款预处理 → `Observation`)。
  3. `tok, mask, ar = model.embed_prefix(obs_model)`(现成方法,image+prompt,**不含 state**,`pi0.py:118`)。
  4. `(prefix_out, _), _ = model.PaliGemma.llm([tok, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask,1)-1)`(= `sample_actions:317-320` 那次 prefix 前向,但接住被丢弃的 `prefix_out`;模板见 `pi0_value.py:266 _forward_backbone`)。
  5. `pooled = pool(prefix_out, mask, pooling)`(`[B,D]`,B=1)。
  6. `outputs["prefix_feat"] = np.asarray(pooled[0])`;返回。
- `pool(prefix_out, mask, pooling)` 纯函数:`last`=`prefix_out[arange(B), mask.sum(1)-1]`(末**有效** token,非 `[:,-1]`);`mean`=按 mask 加权均值(排除 padding)。
- **确定性**:no-grad、无采样;同输入同输出。

### 3.2 `scripts/serve_policy_with_feat.py`(openpi 仓新增)
- 复用 `serve_policy.py` 的 policy 构造路径(`create_trained_policy` / checkpoint),拿到原 `Policy` 后用 `FeaturePolicy(policy, pooling=...)` 包一层,再交给 `websocket_policy_server` 起服务。flags:沿用 serve_policy 的 config/dir/port + 新增 `--pooling`(default `last`)。
- 不改 `serve_policy.py` 本身;这是并列的新入口。

### 3.3 `build_pi0_feat_cache_via_serve.py`(resfit 仓新增,residual 环境)
- 用 `openpi_client.WebsocketClientPolicy(host, port)` 连 serve(同 `load_pi05.py` 的连法,含 `ping_interval=None` 防冷启动 JIT 掉线)。
- 遍历源 hdf5 各 demo 各帧:发**原始 obs**(图像 + prompt,与残差 rollout 时发给 serve 的同款;`image_key_map` 沿用 pi05 adapter)→ `feat = client.infer(obs)["prefix_feat"]`。**注:build 侧不做任何图像预处理**——resize/归一化/tokenize 全由 serve 端 `FeaturePolicy` 的 `inner._input_transform` 统一做(命门 3),这保证 build 期特征与将来在线期特征经同一套预处理、严格同源。→ `⊕ proprio`(从 hdf5 读 `--proprio_key`,如双臂 `STATE18_KEYS` 拼的 18 维)→ 收集 `[T, D_emb+D_proprio]`。
- 全量算一组 `(mean, std)`,标准化各 demo;`save_pi0_feat_cache(out, seqs_std, (mean,std), signature=sig)`。
- 签名 `sig`(见 3.4)。flags:`--host --port --hdf5 --dataset --image_keys --proprio_key --prompt --pooling --serve_ckpt_id --out_cache [--num_demos]`。

### 3.4 `pi0_feat_cache.py`(resfit 仓新增,镜像 `act_feat_cache.py`)
- `save_pi0_feat_cache(path, seqs, emb_stats, *, signature, fp16=False)`:存(已标准化)嵌入序列 + `(mean,std)` + 签名 json;`fp16` 仅压 seqs 存盘。
- `pi0_feat_cache_reuse(path, *, signature, num_demos) -> (seqs, stats) | None`:签名全等(含 `num_demos`)才命中。
- `load_pi0_feat_cache(path)`:读 `(seqs, (mean,std), sig)`,无校验。
- 签名字段:`{serve_ckpt_id, serve_metadata, image_keys, proprio_key, pooling, prompt, dataset_id, num_demos}`。`serve_ckpt_id` 由 build 显式传(serve 的 `policy_metadata` 不保证含唯一权重标识,不依赖自动识别);`serve_metadata` = 握手下发的 metadata(有啥记啥,辅助/诊断)。

### 3.5 `read_per_demo_states` 加 `pi0_feat` 分支(改 `train_hiql_value.py`)
- 保持契约 `-> (seqs, standardizer, aux_stats)`;**cache-required**:
  ```
  state_mode == "pi0_feat":
      cache 缺失/未命中 → raise(引导先在 residual 环境跑 build_pi0_feat_cache_via_serve.py)
      命中 → 返回 (seqs, None, feat_stats)   # 缓存里 seqs 已标准化;standardizer 占位 None
  ```
- 不连 serve、不 import openpi、不读图像。`eef`/`eef_piece`/`act_feat` 三支原样不动。

### 3.6 CLI 接线(全 flag-gated)
- `train_hiql_value.py`:`--state_mode` 加 `pi0_feat` 枚举 + `--pi0_feat_cache`(须已存在)+ `validate_pi0_feat_cfg`(pi0_feat 缺 cache → `ValueError` 引导先 build;非 pi0_feat 传 `--pi0_feat_cache` → warn 忽略)。
- `train_hiql_gc_value.py`:`--state_mode` 枚举加 `pi0_feat`(默认仍 `eef_piece`,逐位等价)+ `--pi0_feat_cache` + **复用 `train_hiql_value.validate_pi0_feat_cfg`** 守卫;`save_gc_value` 存 `state_mode="pi0_feat"` + `mean/std`(=feat_stats)+ 缓存里读出的签名;`load_gc_value` 读回放 info(缺键回退 None)。
- **本 spec 不动 `train_hiql_high_actor.py`**(high_actor 不在范围)。

### 3.7 数据集映射
- image_keys:dexmg `agentview_image,robot0_eye_in_hand_image,robot1_eye_in_hand_image`;由 `--image_keys` 指定(沿用 pi05 adapter image_key_map)。
- proprio:`--proprio_key` 读对应维度(dexmg 双臂 18;不需双臂对齐),维度进签名 + ckpt。

### 3.8 不变量
- 默认路(eef/eef_piece/act_feat)逐位等价;`pi0_feat` 下 `seqs[i].shape[1] == D_emb + D_proprio` 且与签名一致;`build_gc_data`/`train_gc_value`/φ 结构不变(只是 `state_dim` 变)。

## 4. 测试(TDD;不强依赖 GPU/真 serve)

**回归锁**:现有 eef/eef_piece/act_feat 测试全绿;`--state_mode` 默认不变;整套 suite 通过。

**openpi 侧单测(openpi 环境,stub model,不跑真权重)**:
1. `pool` 纯函数:last 取末有效 token、mean 按 mask 排除 padding、未知 pooling raise。
2. `FeaturePolicy.infer`:stub inner + stub model → 返回 dict 含 `actions` 与 `prefix_feat[D]`;`prefix_feat` 不含 proprio;走了 `inner._input_transform`(spy);last/mean 有别。

**resfit 侧单测(residual 环境,stub client/extractor,不连 serve)**:
3. build 脚本:stub client(假 `{actions, prefix_feat}`)→ 遍历 demo/拼 proprio/全量标准化/写缓存;签名正确;命中缓存跳过 client(spy)。
4. `pi0_feat_cache` save→reuse 往返一致;签名(serve_ckpt_id/image_keys/proprio_key/pooling/prompt/dataset/num_demos)不符 → None;部分量(num_demos≠None)不当全量。
5. `read_per_demo_states` 的 pi0_feat 分支:命中缓存返回标准化 seqs + stats、standardizer=None;缺失 raise。
6. CLI:`--state_mode pi0_feat` 接受;`validate_pi0_feat_cfg` 缺 cache → ValueError、eef 模式忽略 `--pi0_feat_cache`;gc_value 默认仍 `eef_piece`。
7. gc_value save/load 往返:pi0_feat 时存的 `mean/std`=feat_stats、`state_mode`/签名读回一致。

**集成冒烟(opt-in,默认 skip,env gate `PI0_FEAT_VIA_SERVE_SMOKE=1`)**:
- 起真 pi05-with-feat serve → build 脚本对几帧出缓存 → `train_hiql_gc_value --state_mode pi0_feat` 训几步 → 断言 V 有限、`gc_value.pt` 落地。这是"端到端从真 serve 拿到特征并训出 value"的完成判据。

## 5. 要改/新增的文件
**openpi 仓(新增 untracked,不改原始文件)**:
- `src/openpi/serving/feature_policy.py`(FeaturePolicy + pool 纯函数)
- `scripts/serve_policy_with_feat.py`(包 FeaturePolicy 起 serve 的并列入口)
- `tests/test_feature_policy.py`(openpi 环境)

**resfit 仓**:
- 新增 `rl_finetuning/chunk_residual/pi0_feat_cache.py`
- 新增 `rl_finetuning/chunk_residual/build_pi0_feat_cache_via_serve.py`(residual 环境调 serve)
- 改 `rl_finetuning/chunk_residual/train_hiql_value.py`(read 分支 cache-required + parser + validate_pi0_feat_cfg)
- 改 `rl_finetuning/chunk_residual/train_hiql_gc_value.py`(state_mode 加 pi0_feat + save/load 签名)
- 新增 tests:`tests/test_pi0_feat_cache.py`、`tests/test_build_pi0_feat_via_serve.py`、`tests/test_read_per_demo_states_pi0_feat.py`、`tests/test_pi0_feat_cli_wiring.py`、opt-in `tests/test_pi0_feat_via_serve_smoke.py`

## 6. 前置依赖与风险
- **环境(已实测)**:residual 环境有 `torch`+`openpi_client`、**无 jax/openpi 本体**;openpi `.venv` 有 jax/openpi。→ openpi 侧 wrapper/测试在 openpi 环境;resfit 侧全程只 `openpi_client` 调 serve。
- **真实 API 钉死(Task 1 spike)**:`FeaturePolicy` 里 `embed_prefix`/`PaliGemma.llm([tok,None])` 的真实返回结构需在 openpi 环境对真 pi05 跑通一次(底稿 `pi0_value._forward_backbone` 已证 `prefix_output[0]` 可取);不成立就地修到跑通。注:`PaliGemma.llm` 是 2-config(`[paligemma, action_expert]`),prefix-only 传 `[tok, None]`,取第一返回的 prefix 分量。
- **serve 吞吐**:离线对整数据集逐帧 RPC(dexmg three_piece 量级数万帧);serve 在 GPU 上比 CPU 直跑快,但仍是数万次 websocket 往返,属一次性开销。可选:build 支持小批/进度日志;`ping_interval=None` 防冷启动掉线。
- **proprio 维度/键**:dexmg 双臂与 LIBERO 单臂 proprio 不同 → `--proprio_key` 指定,维度进签名。
- **同源锚点**:`serve_ckpt_id` 是人工锚;务必在 build 与(未来)在线用同一个值,否则缓存特征与在线特征异源而签名无法发现。
