# 在线 HIQL 子目标注入支持 act_feat(冻结 ACT 特征在线提)设计

- 日期:2026-06-13
- 状态:设计已定稿,待用户审阅 → 写实施计划
- 范围:打通 **②在线子目标注入** 对 `state_mode="act_feat"` 的支持,使 `train_chunk_residual --subgoal_conditioned` 能消费 act_feat 训出的 gc_value/high_actor,从而可跑"等配置 `three_piece_aligned_bp_bc01`、仅把 value 输入换成 act_feat"的主训与 A/B。
- 前置:离线 act_feat(`state_mode=act_feat` 的 gc_value/high_actor + 缓存)已落地并端到端验证(见 `docs/superpowers/specs/2026-06-13-hiql-value-act-feature-offline-design.md` 与记忆 `project_resfit_act_feat_value_offline`)。本设计在其上加在线注入。

## 1. 背景与堵点

resfit HIQL 路线B 的在线子目标注入(`hiql_subgoal.HiqlSubgoal` + `train_chunk_residual` rollout)目前**硬绑 eef_piece**:
- `hiql_subgoal.py:47` 断言 `info["state_mode"] == "eef_piece"`;
- `subgoal_online(state18_std, rel_raw)` 用 env 每步透出的 **sim 特权 rel_piece** 拼 30 维 state;
- `train_chunk_residual` 为此把 `env_state_mode/eval_state_mode` 设成 `eef_piece`(让 env 经 `info["rel_piece"]` 透出 rel),goal30 取自 `--subgoal_state30_cache` 的 medoid。

因此加载 act_feat 的 gc_value(state_mode=act_feat,state_dim=530)在主训启动即 `AssertionError`。act_feat 当前只能离线。

**关键可复用资产**:主训本就加载冻结 ACT `base_policy`(给残差出 base 动作)。act_feat 的在线特征可直接复用它,无需另加载模型,且与离线缓存同一个 ACT(同源天然成立)。

## 2. 关键约束(命门)

1. **eef_piece 路径逐位不变**:不传 act_feat gc_value 时,在线/离线 eef_piece 全路径与现状逐位等价。
2. **同源—图像预处理(命门 A)**:offline 缓存的特征来自 hdf5 图(已 `HWC-uint8 → CHW-float[0,1]`);online 必须喂 ACT **同款**预处理的图。env obs 的图(base_policy 本就在吃)须是 CHW float[0,1];否则在线 extractor 入口补同款变换。一致性靠 §5 smoke 验。
3. **同源—proprio 约定(命门 B)**:proprio 全栈统一为 **dataset-标准化**(env 的 `StateStandardizer`)。offline act_feat build 改为对 18 维 proprio 先 dataset-标准化再进 extractor;online 直接用已标准化的 `obs["observation.state"]`。两边 act_feat `(mean,std)` 据此一致。
4. **复用 base_policy 安全**:extractor 走 `base_policy.model(...)` 内部 forward + encoder hook,不碰 `select_action` 的 per-env 动作 queue。
5. **同源签名**:主训断言 `--act_feat_cache` 的 act_feat_signature 与 gc_value/high_actor ckpt 的签名核心字段一致(沿用 `assert_act_feat_pair_consistent` 思路)。

## 3. 设计

### 3.1 `HiqlSubgoal` 按 state_mode 分发(改 `hiql_subgoal.py`)
- `from_ckpts(gc_value_ckpt, high_actor_ckpt, *, goal, device, renorm_subgoal, base_policy=None)`:
  - 读 `info["state_mode"]`;断言 `state_mode in {"eef_piece","act_feat"}`(替换原 eef_piece-only 断言)。
  - eef_piece:与现状一致(rel_mean/std 来自 ckpt)。
  - act_feat:断言 `base_policy is not None`;建 `self.extractor = ActFeatureExtractor(base_policy, image_keys=<来自签名>, proprio_key=<签名>, pooling=<签名>)`;`self.feat_mean/std` = ckpt 的 `mean/std`(act_feat 整条 530 的标准化量)。
  - `goal`:已是对应维度的 medoid 向量(eef_piece 30 / act_feat 530),由调用方按模式构建后传入(§3.2)。
- `subgoal_online(obs, rel_raw=None)` 内部按 `self.state_mode` 分发:
  - eef_piece:`s30 = build_state30(obs["observation.state"], rel_raw)`(现状)。
  - act_feat:`feat = self.extractor.embed_batch(obs)`(530 raw);`s = (feat - feat_mean)/feat_std`;`z = high_actor(s, goal).mean`;renorm 同现状。
- `representative_goal`(原 `representative_goal30`):泛化命名,对任意维 seqs 取 medoid(逻辑不变;保留旧名做别名以防引用)。

### 3.2 `train_chunk_residual.py` 接线
- subgoal init(634-658)先 `peek` gc_value ckpt 的 state_mode(`load_gc_value` 取 info,或加轻量读签名),分发:
  - **eef_piece**:现状不变(env_state_mode/eval_state_mode=eef_piece;goal30 取自 state30 cache)。
  - **act_feat**:
    - `env_state_mode = eval_state_mode = "eef"`(不透出 rel_piece);
    - goal530 = `representative_goal(act_feat_seqs)`,`act_feat_seqs` 从 `--act_feat_cache`(新 flag)经 `load_act_feat_cache` 读(已标准化的 530 序列);
    - `HiqlSubgoal.from_ckpts(..., goal=goal530, base_policy=base_policy)`;
    - 断言 cache 签名核心字段 == gc/high ckpt 的 act_feat_signature(同源)。
- 新 flag `--act_feat_cache`(act_feat 子目标必需;eef_piece 忽略+warn)。`--subgoal_state30_cache` 对 act_feat 不需要。
- rollout 调用点:`subgoal.subgoal_online(obs, cur_rel)`(传完整 obs;cur_rel 仅 eef_piece 用,act_feat 忽略)。`next_rel` 同理。obs 已含 `observation.images.*` 与 `observation.state`(base_policy 本就吃)。
- env_state_mode 守卫:现有 `assert env_state_mode == "eef_piece"`(637)放宽为"act_feat→eef / eef_piece→eef_piece"。

### 3.3 离线 proprio 同源修正(改 `train_hiql_value.py`,命门 B)
- act_feat build 路:对 `assemble_state18` 的 18 维 proprio 先用 dataset `StateStandardizer`(`meta.stats["observation.state"]`)标准化,再进 extractor 的 concat。
- 即 act_feat build 需加载 dataset stats(dexmg 有;act_feat 只服务 dexmg ACT base,不破坏部署诚实的核心——图像仍是合法观测)。cache-hit 路不变。
- **重建缓存**(真缓存尚未建,无损)。签名不变(proprio_key 仍 observation.state);但缓存内容(标准化口径)变,故旧缓存需重建。

### 3.4 一致性 smoke(命门 A/B 硬判据,§5)
新增 opt-in 脚本/测试:同一条 demo,offline 从 hdf5 提的 530 特征 vs online 从 env obs(真 env reset 到该 demo 初态或对齐状态)提的 530 特征,断言 allclose(合理容差)。并打印 env obs 图的 dtype/shape/range 确认 == CHW float[0,1]。过了才跑 500k 主训。

## 4. 不变量
- eef_piece 在线/离线逐位等价;`--subgoal_conditioned` 默认仍 eef_piece(取决于 gc_value ckpt mode)。
- act_feat 在线:`obs["observation.subgoal"]` 仍是 10 维 z;actor/critic 的 observation.state 仍 18 维;env 不透出 rel_piece。
- 同源:offline 特征 ≈ online 特征(smoke 容差内);签名一致断言通过。

## 5. 测试
**回归锁**:eef_piece 在线接线相关单测(`test_hiql_subgoal_wiring` 等)全绿;eef_piece 主训冒烟不变。

**新单测(stub,不跑真 ACT/env)**:
1. `HiqlSubgoal.from_ckpts` act_feat 分支:用 stub gc_value/high_actor(state_mode=act_feat,state_dim=530)+ stub base_policy → 建成,`subgoal_online(stub_obs)` 出 [B, rep_dim];eef_piece 分支不受影响。
2. 分发正确:state_mode=act_feat 不读 rel;eef_piece 仍读 rel。
3. `train_chunk_residual` 的 subgoal init act_feat 分支(可抽 helper 测):env_state_mode→eef、goal 从 act_feat cache、签名一致断言。
4. proprio dataset-标准化:offline build 的 18 维已标准化(对拍 StateStandardizer)。

**集成 smoke(opt-in,真 ACT+env,默认 skip)**:§3.4 的 offline↔online 特征一致性 + 短主训(几十步)不 NaN、subgoal z 有限。

## 6. 要改/新增的文件
- 改 `hiql_subgoal.py`(from_ckpts 分发 + subgoal_online 分发 + extractor 持有 + representative_goal 泛化)
- 改 `train_chunk_residual.py`(subgoal init 分发 + `--act_feat_cache` flag + env_state_mode 放宽 + rollout 调用点传 obs)
- 改 `train_hiql_value.py`(act_feat build proprio dataset-标准化)
- 新增 `tests/test_hiql_subgoal_act_feat.py`(stub 单测)+ 扩 `test_act_feat_cli_wiring`/相关
- 新增 `verify_act_feat_online_consistency.py`(opt-in 一致性 smoke)

## 7. 前置依赖与风险
- **命门 A/B 未验即跑 500k = 烧钱**:必须先过一致性 smoke(offline↔online 特征 allclose)再长跑。
- **env obs 图格式**:若 env 给的不是 CHW float[0,1],需在线 extractor 入口补变换(smoke 会暴露)。
- **重建缓存**:proprio 约定改了,旧 act_feat 缓存(若已建)须删重建;gc_value/high_actor 也须用新缓存重训。
- **每步 ACT 前向**:in-loop +1 encoder forward/step(ACT 小,可接受);num_envs=1 训练 rollout。
- **peek state_mode**:主训需在建 subgoal 前知道 gc_value 的 mode;`load_gc_value` 已返回 info["state_mode"],直接用。
