# LiberoPi05Adapter per-env prefix_feat(向量化 eval)设计

> 状态:待实现。范围聚焦、单一计划可覆盖。承接 `2026-06-16-libero-eval-online-subgoal-injection`(集成 smoke 暴露)。

## 背景与问题

LIBERO eval 在线子目标注入已落地并经 smoke 验证(`--eval_num_envs 1` 端到端跑通,`eval success_rate=0.333`、best.pt 落盘)。但 **默认 `--eval_num_envs 8`(向量化 eval)会崩**:

```
hiql_subgoal.py:119  feat = torch.cat([pf, proprio], dim=-1)
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 1 but got size 8 ...
```

根因:`LiberoPi05Adapter.select_action`(libero_pi05_adapter.py:60)逐 env 循环,每个 env 队列空时各自 serve infer,但 `_infer_chunk` 每次**覆盖**同一个 `self._last_prefix_feat`(last-writer-wins)。跑完 b=8 个 env,`last_prefix_feat()` 只剩"最后一个重规划 env"的单帧特征(batch 1)。注入时 pf=[1,2048] 撞 proprio=[8,8]。训练 num_envs=1 永不暴露;向量化 eval + pi0_feat 是首次真跑。

## 目标

`last_prefix_feat()` 在向量化场景返回与当前 obs batch 对齐的 per-env 特征 `[b, 2048]`,使默认 `--eval_num_envs 8` 的 pi0_feat eval 正常;单 env(训练)与既有直调 `_infer_chunk` 行为保持向后兼容(零回归)。

## 范围

**做**:① 改 `LiberoPi05Adapter`:`select_action` 维护 per-env prefix_feat、`last_prefix_feat()` 就位时返回堆叠 `[b,2048]`;② 单元测试。
**不做**:改 `run_libero_evaluation`/注入点(已正确,无需动);改 `hiql_subgoal.subgoal_online`(已支持 ndim==2 的 pf);训练路逻辑;dexmg/act_feat。

## 关键事实(已核实)

- **预热时机**:`ChunkResidualEnvWrapper.reset()`(chunk_env_wrapper.py:134)经 `_base_chunk_flat → select_action(raw_obs)` 对**所有 b 个 env** 预热(reset 后队列全空 → 全部重 infer)。故 `run_libero_evaluation` 首帧注入(env.reset 后、首个 env.step 前)时 per-env 特征已就位 `[b,2048]`。
- **serve 单帧特征**:每次 `_infer_chunk` 对单 env obs 调 serve,`prefix_feat` 形状 `(2048,)`(1D)。
- **subgoal_online 形状处理**(hiql_subgoal.py:115-118):pf.ndim==1 → unsqueeze 成 [1,2048];ndim==2 → 原样。故返回 [b,2048] 直接可用;返回 [1,2048] 与旧的 [2048]→unsqueeze 等价。
- **既有测试约束**:`test_adapter_prefix_feat.py` 直调 `_infer_chunk` 后断言 `last_prefix_feat()` 形状 `(2048,)`(及 serve 不透特征时 None)。这条**直调路径**必须保留 → `last_prefix_feat()` 在 per-env 未就位时回退单帧 `_last_prefix_feat`。

## 设计

`LiberoPi05Adapter` 改三处(libero_pi05_adapter.py):

1. `__init__` 加 `self._prefix_feat_per_env = []`(与 `_queues` 同步增长)。
2. `_ensure_queues(b)` 在 append deque 的同时 append 一个 `None` 到 `_prefix_feat_per_env`,保持等长。
3. `select_action` 循环里,env i 重 infer 后记下该 env 特征:`self._prefix_feat_per_env[i] = self._last_prefix_feat`(`_infer_chunk` 仍照旧 set 单帧 `_last_prefix_feat`,直调测试不变)。未重 infer 的 env(队列非空)保留上次特征 —— 与 action 队列语义一致(对应上次 base 重规划)。
4. `last_prefix_feat()`:
   - per-env 未就位(空,或含 None)→ 回退单帧 `self._last_prefix_feat`(覆盖直调 `_infer_chunk` 测试 + serve 不透特征 None);
   - 全部就位 → `np.stack([np.asarray(f) for f in self._prefix_feat_per_env])` = `[b,2048]`。

```python
def last_prefix_feat(self):
    """per-env 就位时返回堆叠 [b,2048](向量化 eval);未就位(直调 _infer_chunk/无 select_action/
    serve 不透特征)回退单帧 _last_prefix_feat(向后兼容 + None)。"""
    if not self._prefix_feat_per_env or any(f is None for f in self._prefix_feat_per_env):
        return self._last_prefix_feat
    return np.stack([np.asarray(f, dtype=np.float32) for f in self._prefix_feat_per_env])
```

## 数据流

eval(b=8):env.reset → wrapper `_base_chunk_flat` → `select_action(8 envs)` → 8 队列全空 → 8 次 `_infer_chunk` → `_prefix_feat_per_env=[f0..f7]` → 首帧 `last_prefix_feat()=[8,2048]` → 注入 cat proprio[8,8] → [8,2056]。后续步:仅队列空的 env 重 infer 更新各自槽位,其余沿用 → 始终 [8,2048]。
训练(b=1):`_prefix_feat_per_env=[f0]` → `[1,2048]`,与旧 [2048]→unsqueeze 等价。

## 错误处理

serve 不透特征:`_infer_chunk` 写 None → `_prefix_feat_per_env[i]=None` → `any None` → 回退单帧 None → 注入器 `_inject_subgoal` 的 None 守卫抛清晰 ValueError(已有)。`reset()` 跨 episode 不清 `_prefix_feat_per_env`(下个 select_action 会全量覆盖),但 `reset()` 清队列保证下次全部重 infer。

## 测试

`tests/test_libero_pi05_adapter.py`(或 `test_adapter_prefix_feat.py`)新增:
- `test_last_prefix_feat_multi_env_stacks_per_env`:stub policy 每次 infer 返回**可辨识**特征(如 `np.full(2048, call_idx)`),`select_action(_raw_obs(3))` 后 `last_prefix_feat().shape==(3,2048)` 且第 i 行可辨识对应第 i 次 infer。
- `test_last_prefix_feat_single_env_shape`:`select_action(_raw_obs(1))` 后形状 `(1,2048)`。
- `test_last_prefix_feat_carries_over_non_reinferred_env`:execute_horizon≥2,`select_action(_raw_obs(2))` 调两次;第二次队列非空不重 infer → `last_prefix_feat()` 仍是首次两 env 的特征(验证沿用,不被清)。
- **回归**:`test_adapter_prefix_feat.py` 两个直调 `_infer_chunk` 测试仍绿(per-env 未就位回退单帧 (2048,)/None);`test_libero_pi05_adapter.py` 既有多 env/队列测试全绿。

## 成功标准

1. 新单测过 + 既有适配器测试零回归 + 聚焦回归全绿。
2. 重跑集成 smoke 用**默认 `--eval_num_envs 8`**:跨过 eval(无 batch 不匹配)、打印 success_rate、best.pt 落盘。

## 相关

- 注入点:`libero_eval.py:_inject_subgoal`(本次不改)。
- 预热:`chunk_env_wrapper.py:129 reset → 134 _base_chunk_flat`。
- 形状消费:`hiql_subgoal.py:103 subgoal_online`。
