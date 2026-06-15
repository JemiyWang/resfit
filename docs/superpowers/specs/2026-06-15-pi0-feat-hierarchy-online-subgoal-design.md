# pi0_feat 完整分层:high_actor + 在线子目标注入(②③)设计

- 日期:2026-06-15
- 状态:设计已定稿,待写实施计划
- 范围:**给 `state_mode=pi0_feat` 接完整 HIQL 分层的后两段**——② high_actor(高层策略)离线训 pi0_feat;③ 在线 rollout 时高层出子目标 z 注入低层残差策略。**目标 = (a) 接通+smoke**:完整链路"pi0表征→V+高层→子目标→低层"真正打通、能起残差 RL(几步不塌)。**明确不做(b)**:残差 RL 长训 + eval 看 LIBERO 成功率(独立实验阶段,另开 spec)。

## 1. 背景与动机

pi0_feat 的第一段(pi0 prefix 图像特征 ⊕ 本体 → V/`gc_value`)已落地且 **gate PASS**(2026-06-15:38 demos,V 沿专家轨迹 spearman median=0.994、目标侧 -1.000、到达态≈0,见 [[project_resfit_pi0_feat_via_serve]])——证明 pi05 图像特征当 value state 质量健康。现接完整分层:让高层策略 `high_actor` 也用 pi0_feat state 训出,在线 rollout 时高层出子目标 z 喂低层残差。

**完全照 `act_feat` 已趟通的分层模式**(act_feat 的 ②③ 已实现:`train_hiql_high_actor` 的 act_feat dispatch + `hiql_subgoal` 的 act_feat 在线分支 + `train_chunk_residual --subgoal_conditioned` 接线)。唯一本质差异是**在线特征来源**:act_feat 在 residual 进程内用 `ActFeatureExtractor.embed_batch` 提(ACT 在 residual 环境);pi0_feat residual 跑不了 pi05,特征只能从 **serve** 取——而残差 rollout 的 base policy 本来就连同一个 pi0_libero serve、serve 的 FeaturePolicy 每次 infer 都返回 prefix_feat,**复用它即可零额外推理**。

## 2. 关键约束(命门)

1. **在线特征零额外推理(方案A)**:base policy(`LiberoPi05Adapter`)每步 `select_action` 调 serve、serve 返回 `{actions, prefix_feat}`。adapter 存"最近一次 prefix_feat"并暴露;残差主循环取它喂 high_actor 出 z。**不**另起 serve client 重复 infer。queue 模式下 prefix_feat 在 chunk 边界更新→z 在 chunk 内不变(可接受,同 base action chunk 内不变)。
2. **同源(在线 z = 离线训练口径)**,四条:
   - ① **prefix_feat**:离线 build 与在线 rollout 取自**同一 pi0_libero serve**(同权重/预处理/pooling)→天然同源;`serve_ckpt_id`+`pooling` 签名校验保证在线 serve 与 build 一致。
   - ② **proprio(真命门)**:离线 proprio=`read_libero_demo` 的 `demo["state"]`(8维);在线 proprio=残差 env 的 **raw** `observation.state`(8维)。须 (a) env-state 与 demo-state **同维度同语义**;(b) 用 **raw** proprio,**不用** `chunk_env_wrapper` 经 `state_standardizer` 标准化后的(那是另一套 stats)。smoke 时实测对齐。
   - ③ **feat_mean/std**:build 对整条 `[prefix⊕proprio]`(2056) 全量算的 `(mean,std)` 存进 ckpt;在线用 ckpt 这组 stats 标准化 raw `[prefix⊕proprio]`。
   - ④ **goal medoid**:固定 goal=pi0_feat 缓存 seqs 末态 medoid(2056 标准化空间),`HiqlSubgoal.from_ckpts` 时从缓存 seqs 算(同 act_feat 口径)。
3. **默认零影响**:不走 pi0_feat 时,act_feat/eef_piece 的离线训练与在线注入逐位等价;现有测试全绿。
4. **范围只到接通+smoke**:不调参、不长训、不 eval 成功率。

## 3. 设计

### 3.0 整体架构与数据流
```
② 离线(residual,只读缓存):
   pi0_feat 全量缓存 + pi0_feat gc_value.pt
        → train_hiql_high_actor --state_mode pi0_feat → high_actor.pt
   (clamp_to_goal 口径无 stage 依赖,LIBERO 兼容)

③ 在线(残差 rollout, train_chunk_residual --subgoal_conditioned):
   每步:
   base_policy.select_action(obs) → serve {actions, prefix_feat} → adapter._last_prefix_feat
        │                                                              │
   残差走 actions+残差                              主循环取 last_prefix_feat() ⊕ raw proprio(obs.state)
                                                      → (−feat_mean)/feat_std (缓存 stats)
                                                      → hiql_subgoal.subgoal_online(pi0_feat 分支)
                                                      → high_actor(s, goal_medoid).mean → z(10)
                                                      → obs["observation.subgoal"]=z → 低层残差吃
```
③ 的接线管道(`train_chunk_residual.py:872` 每步 `obs["observation.subgoal"]=subgoal_online(...)`、低层吃 `observation.subgoal`)完全复用 act_feat,只换特征来源。

### 3.1 `LiberoPi05Adapter` 暴露 prefix_feat(改 `libero_pi05_adapter.py`)
- `_infer_chunk(obs)`(:42-48 现仅取 `result["actions"]`):同时 `self._last_prefix_feat = result.get("prefix_feat")`。
- 加 `last_prefix_feat()` → 返回 `self._last_prefix_feat`(初始 None);`__init__` 设 `self._last_prefix_feat = None`。
- queue 模式:`_infer_chunk` 在 chunk 边界调,故 `_last_prefix_feat` 边界更新。dexmg 的 `Pi05PolicyAdapter`(kai0)本次**不动**(只 LIBERO 路)。

### 3.2 `hiql_subgoal` 加 pi0_feat 分支(改 `hiql_subgoal.py`)
- `from_ckpts`(:61-79):`assert sm in (...)` 加 `"pi0_feat"`;pi0_feat 时**不建 ActFeatureExtractor**,只取 ckpt 的 `feat_mean/std`(info["mean"]/["std"])+ 用传入 seqs 算 `goal` medoid。**故 pi0_feat 时 `base_policy` 参数不需要(传 None)**;区别于 act_feat 必须真 ACT base_policy(要 hook encoder)。`HiqlSubgoal.__init__` 加 pi0_feat 分支(存 feat_mean/std,不需 extractor)。
- `subgoal_online(obs, rel_raw=None, prefix_feat=None)`(:93-106)加分支:
  ```python
  if self.state_mode == "pi0_feat":
      assert prefix_feat is not None, "pi0_feat 在线需 prefix_feat(从 base policy 取)"
      proprio = torch.as_tensor(np.asarray(obs["observation.state"]), dtype=torch.float32, device=self.device)
      pf = torch.as_tensor(np.asarray(prefix_feat), dtype=torch.float32, device=self.device)
      if pf.ndim == 1: pf = pf.unsqueeze(0)
      if proprio.ndim == 1: proprio = proprio.unsqueeze(0)
      feat = torch.cat([pf, proprio], dim=-1)              # [B,2056] raw
      s = (feat - self.feat_mean) / self.feat_std          # 缓存 stats 标准化
  ```
  其余(g expand、`z=self.ha(s,g).mean`、renorm)与现有共用。

### 3.3 `train_chunk_residual` 接 pi0_feat(改 `train_chunk_residual.py:872` 那处 + subgoal 构造)
- subgoal 构造(约 :700-742):`--subgoal_conditioned` + gc_value 的 state_mode=pi0_feat 时,`HiqlSubgoal.from_ckpts(..., base_policy=None)`(pi0_feat 不建 extractor);act_feat/eef_piece 路原样传 base_policy。goal medoid 从 pi0_feat 缓存 seqs 算(seqs 由 `read_per_demo_states(...,"pi0_feat")` 从缓存读)。
- 每步注入(:872):
  ```python
  if args.subgoal_conditioned:
      if subgoal.state_mode == "pi0_feat":
          pf = base_policy.last_prefix_feat()
          obs["observation.subgoal"] = subgoal.subgoal_online(obs, prefix_feat=pf).to(device)
      else:  # eef_piece/act_feat 原样
          obs["observation.subgoal"] = subgoal.subgoal_online(obs, cur_rel).to(device)
  ```
- eval 路同样处理(若 eval 也 subgoal_conditioned)。

### 3.4 high_actor 离线训 pi0_feat(改 `train_hiql_high_actor.py`)
- 加 `--state_mode pi0_feat` 枚举 + `add_pi0_feat_args(p)` + `validate_pi0_feat_cfg(args)`(复用 train_hiql_value 的)。
- main dispatch(:81-91)加 `elif args.state_mode == "pi0_feat"`:照刚做的 gc_value pi0_feat dispatch——`load_pi0_feat_cache` 读缓存签名 → 断言核心字段 → `read_per_demo_states(..., "pi0_feat", pi0_feat_cache, pi0_feat_signature=cache_sig)` 取 seqs/feat_stats。
- stage:`clamp_to_goal`(默认)不读 stage;`stage_entries` 走 `stage_entries_aligned(..., stage_cache=None, ...)` 回退末态(geometric 不读)。**LIBERO 无 hdf5 → 须确认 `stage_entries_aligned` 在 stage_cache=None 时不读 hdf5**(若读,改成 pi0_feat 时传空 stage_entries,见风险)。
- `save_high_actor(..., state_mode="pi0_feat", act_feat_signature=None, pi0_feat_signature=pi0_sig, mean=feat_mean, std=feat_std)`:`save/load_high_actor` 加可选 `pi0_feat_signature`(类比 act_feat_signature);`feat_mean/std`(= `read_per_demo_states` pi0_feat 分支返回的 feat_stats)存入 ckpt(供在线 subgoal_online 同源标准化)。
- **前置**:high_actor 需 pi0_feat 的 `gc_value.pt`。gate 当时用直接脚本未 save;须先用 `train_hiql_gc_value --state_mode pi0_feat`(geometric,stage_cache=None)存出 `gc_value.pt`(同样须确认 LIBERO 无 hdf5 兼容)。

### 3.5 不变量
- 默认 act_feat/eef_piece 路逐位等价;pi0_feat 在线 z 维度=rep_dim(10);在线 `[prefix⊕proprio]` 维=2056 且经 ckpt feat_mean/std 标准化与离线同源。

## 4. 测试(TDD)

**单测(stub,不需真 serve/GPU)**:
1. `LiberoPi05Adapter`:stub `policy.infer` 返回 `{actions, prefix_feat}` → `_infer_chunk` 后 `last_prefix_feat()` 返回该 prefix_feat;infer 不含 prefix_feat 时返回 None 不崩。
2. `hiql_subgoal` pi0_feat:stub ckpt(feat_mean/std + goal)构造 `HiqlSubgoal(state_mode="pi0_feat")` → `subgoal_online(obs{observation.state}, prefix_feat=...)` 出 z[B,10];验证标准化用的是 feat_mean/std、raw `[prefix⊕proprio]` 拼接顺序(prefix 在前)。
3. `train_hiql_high_actor`:`--state_mode pi0_feat` parser 接受、`validate_pi0_feat_cfg` 守卫;main monkeypatch(`read_per_demo_states`/`load_pi0_feat_cache`/`save_high_actor`)验证走 pi0_feat 路 + save 含 pi0_feat_signature + feat_stats。
4. `train_chunk_residual` 接线:monkeypatch `base_policy.last_prefix_feat` + `subgoal.subgoal_online`,验证 subgoal_conditioned + pi0_feat 时每步取 prefix_feat 传 subgoal_online(spy)。
5. `save/load_high_actor` 的 pi0_feat_signature 往返。
6. **回归锁**:act_feat/eef_piece 的 subgoal/high_actor 测试全绿。

**opt-in 集成 smoke(真 serve + 真 LIBERO env,默认 skip,env gate)**:
- 前置:`train_hiql_gc_value --state_mode pi0_feat`(全量缓存)存 `gc_value.pt`;`train_hiql_high_actor --state_mode pi0_feat` 存 `high_actor.pt`。
- 在线:起残差 rollout(`--subgoal_conditioned`,pi0_feat)几步 → 断言 `observation.subgoal` 维=10、z 有限、残差几步不塌。
- **命门② 实测**:打印在线 raw `observation.state` 与离线 `demo["state"]` 的维度/范围,确认同源。

## 5. 要改/新增的文件
- 改 `libero_pi05_adapter.py`(_infer_chunk 存 + last_prefix_feat getter)
- 改 `hiql_subgoal.py`(from_ckpts + __init__ + subgoal_online 的 pi0_feat 分支)
- 改 `train_chunk_residual.py`(subgoal 注入处 pi0_feat 取 base prefix_feat;subgoal 构造)
- 改 `train_hiql_high_actor.py`(--state_mode pi0_feat dispatch + CLI + validate)
- 改 `hiql_high_actor.py`(save/load_high_actor 加 pi0_feat_signature + feat_stats)
- 新增 tests:`test_adapter_prefix_feat.py`(或并入 libero adapter 测)、`test_hiql_subgoal_pi0_feat.py`、`test_pi0_feat_high_actor_cli.py`、`test_train_chunk_residual_pi0_feat_subgoal.py`、opt-in `test_pi0_feat_online_subgoal_smoke.py`

## 6. 前置依赖与风险
- **stage/hdf5 兼容(高优先核实)**:`train_hiql_gc_value`/`train_hiql_high_actor` 的 pi0_feat dispatch 调 `stage_entries_aligned(...)`——须确认 stage_cache=None + geometric 下**不读 dexmg hdf5**(LIBERO 无)。若仍读 hdf5,改成 pi0_feat 时直接传 `[empty for seqs]` 当 stage_entries(绕过 stage_entries_aligned)。
- **命门② env-state vs demo-state 同源**:残差 `libero_env` 的 `observation.state` 与 LeRobot `demo["state"]` 须同维同义;不一致则在线 z 错。smoke 必测;若不一致需对齐层。
- **base_policy 类型**:pi0_feat 在线只支持 LIBERO 路(LiberoPi05Adapter);dexmg(Pi05PolicyAdapter)需另适配(本 spec 不含)。
- **serve 须起着**:在线 rollout 依赖 pi0_libero serve(with FeaturePolicy)在线;`serve_ckpt_id`/`pooling` 与 build/high_actor 训练一致。
- **执行序**:先存 gc_value.pt → high_actor.pt(离线)→ 再在线 rollout smoke。
