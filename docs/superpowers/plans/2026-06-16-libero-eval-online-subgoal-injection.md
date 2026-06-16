# LIBERO eval 路在线子目标注入 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `run_libero_evaluation` 加在线子目标注入,使 `--subgoal_conditioned` 的 LIBERO pi0_feat 残差训练在 eval 阶段不再 `KeyError: 'observation.subgoal'`。

**Architecture:** 镜像 dexmg eval 的 `_eval_inject_subgoal`,但用 libero 本地小函数(不 import `evaluate_dexmg`,避免被 robosuite-1.5 拖崩)。`run_libero_evaluation` 加 `subgoal=/base_policy=` 两个默认 None 的关键字参数,在 reset 后给首帧 obs、每次 step 后给 next_obs 注入 `observation.subgoal`;`subgoal=None` 时旧路径逐字节不变。调用处必须传 `eval_base_policy`(非训练 `base_policy`)。

**Tech Stack:** Python/PyTorch;测试 pytest。所有命令从 `/mnt/mnt/data/resfit` 用 `/mnt/mnt/data/envs/residual/bin/python` 跑单测(纯 torch,不需 libero 环境)。

**Spec:** `docs/superpowers/specs/2026-06-16-libero-eval-online-subgoal-injection-design.md`

## 文件结构

| 文件 | 职责 | 改动 |
|---|---|---|
| `resfit/rl_finetuning/chunk_residual/libero_eval.py` | env-无关 eval rollout | 加 `_inject_subgoal` helper + `run_libero_evaluation` 加 2 参 + 2 处注入 + 1 处 guard |
| `resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py` | 单测 | Task 1 加 4 个测试 + 3 个 stub |
| `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` | 训练主流程 eval block | Task 2:libero eval 调用传 `subgoal`/`base_policy=eval_base_policy` |

---

### Task 1: libero_eval.py 注入 subgoal(TDD)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_eval.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py`

**背景:** `run_libero_evaluation` 现签名 `(*, env, agent, num_episodes, device="cpu")`,rollout 循环里 `agent.act(obs)` 在 actor `subgoal_conditioned=True` 时要 `obs["observation.subgoal"]`,缺失即崩。本任务加注入。复用测试文件已有的 `_ScriptedVecEnv`(每次 reset/step 返回新 obs dict,含 `observation.state`)和 `_StubAgent`。

- [ ] **Step 1: 写失败测试(加到 `test_libero_eval.py` 末尾)**

```python
# --- subgoal 在线注入(pi0_feat) ---

class _RecordingAgent(_StubAgent):
    """记录每次 act 收到的 obs 里 observation.subgoal 的形状(无则记 None)。"""
    def __init__(self):
        super().__init__()
        self.seen_subgoal = []

    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
        sg = obs.get("observation.subgoal")
        self.seen_subgoal.append(None if sg is None else tuple(sg.shape))
        return super().act(obs, eval_mode=eval_mode, stddev=stddev, cpu=cpu)


class _StubSubgoalPi0Feat:
    state_mode = "pi0_feat"
    rep_dim = 10

    def subgoal_online(self, obs, rel_raw=None, prefix_feat=None):
        assert prefix_feat is not None          # 注入必须经 base_policy.last_prefix_feat()
        b = obs["observation.state"].shape[0]
        return torch.zeros(b, self.rep_dim)


class _StubSubgoalEefPiece(_StubSubgoalPi0Feat):
    state_mode = "eef_piece"                     # 非 pi0_feat → 应 fail-fast


class _StubBasePolicy:
    def last_prefix_feat(self):
        return torch.zeros(2048)


def test_injects_subgoal_each_act_when_conditioned():
    agent = _RecordingAgent()
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0, 1.0])
    run_libero_evaluation(env=env, agent=agent, num_episodes=2, device="cpu",
                          subgoal=_StubSubgoalPi0Feat(), base_policy=_StubBasePolicy())
    assert len(agent.seen_subgoal) == 2
    assert all(s == (1, 10) for s in agent.seen_subgoal)   # 每次 act 都注入了 (B, rep_dim)


def test_no_subgoal_omits_key():
    agent = _RecordingAgent()
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0, 1.0])
    run_libero_evaluation(env=env, agent=agent, num_episodes=2, device="cpu")   # subgoal 默认 None
    assert len(agent.seen_subgoal) == 2
    assert all(s is None for s in agent.seen_subgoal)      # 旧路径不注入,零回归


def test_subgoal_requires_base_policy():
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0])
    with pytest.raises(ValueError):
        run_libero_evaluation(env=env, agent=_StubAgent(), num_episodes=1, device="cpu",
                              subgoal=_StubSubgoalPi0Feat(), base_policy=None)


def test_subgoal_rejects_non_pi0_feat():
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0])
    with pytest.raises(NotImplementedError):
        run_libero_evaluation(env=env, agent=_StubAgent(), num_episodes=1, device="cpu",
                              subgoal=_StubSubgoalEefPiece(), base_policy=_StubBasePolicy())
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py::test_injects_subgoal_each_act_when_conditioned -v`
Expected: FAIL with `TypeError: run_libero_evaluation() got an unexpected keyword argument 'subgoal'`

- [ ] **Step 3: 实现 — 加 helper + 2 参 + guard + 2 处注入**

在 `libero_eval.py`,`def run_libero_evaluation` **之前**插入 helper:

```python
def _inject_subgoal(subgoal, obs, base_policy):
    """LIBERO eval 子目标注入:返回该 obs 的 z(调用方写进 obs['observation.subgoal'])。
    对称 evaluate_dexmg._eval_inject_subgoal,但 LIBERO 无 rel_piece,当前只支持 pi0_feat。"""
    if subgoal.state_mode == "pi0_feat":
        return subgoal.subgoal_online(obs, prefix_feat=base_policy.last_prefix_feat())
    raise NotImplementedError(
        f"LIBERO eval 子目标注入目前只支持 state_mode=pi0_feat,得到 {subgoal.state_mode!r}")
```

把 `run_libero_evaluation` 签名改为(加 2 个默认 None 的关键字参数):

```python
def run_libero_evaluation(*, env, agent, num_episodes: int, device: str = "cpu",
                          subgoal=None, base_policy=None) -> dict:
```

在函数体 `agent.eval()` **之前**加 guard:

```python
    if subgoal is not None and base_policy is None:
        raise ValueError("run_libero_evaluation: subgoal!=None 时必须传 base_policy(读 last_prefix_feat)")
    agent.eval()
```

把 `obs, _ = env.reset()` 改为(reset 后注入首帧):

```python
    obs, _ = env.reset()
    if subgoal is not None:
        obs["observation.subgoal"] = _inject_subgoal(subgoal, obs, base_policy)
```

把循环里 `next_obs, reward, terminated, truncated, _info = env.step(actions)` 改为(step 后注入续帧):

```python
            next_obs, reward, terminated, truncated, _info = env.step(actions)
            if subgoal is not None:
                next_obs["observation.subgoal"] = _inject_subgoal(subgoal, next_obs, base_policy)
```

(其余 rollout/统计逻辑**完全不动**。)

- [ ] **Step 4: 跑测试验证通过(含既有 5 个测试不回归)**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py -v`
Expected: PASS(新 4 个 + 既有 `test_success_rate_and_returns_single_env_multistep`、`test_uses_eval_mode_and_restores_train_mode`、`test_stops_exactly_at_num_episodes_multi_env`、`test_mean_successful_length_excludes_failed_episodes`、`test_restores_train_mode_on_exception` 全绿)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/libero_eval.py resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py
git commit -m "feat(libero_eval): run_libero_evaluation 在线注入 observation.subgoal(pi0_feat)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: train_chunk_residual eval block 传 subgoal + eval_base_policy

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(libero eval 调用处,约 991-994 行)

**背景:** main() 内的 eval block wiring 无法独立单元测(要构造整个 main 上下文),由"全量回归不破坏 + 集成 smoke"覆盖。两个要传的变量在该行均已在作用域:`subgoal`(约 712 行附近定义)、`eval_base_policy`(651 行定义,663 行已用于建 `eval_env`)。**命门:必须传 `eval_base_policy` 而非训练 `base_policy`** —— LIBERO 用 queue 模式,eval 单独建了 base 实例,`last_prefix_feat()` 只有读 eval_env 真正在跑的那个实例才对应当前 eval obs。

- [ ] **Step 1: 确认两个变量在调用处作用域内**

Run: `cd /mnt/mnt/data/resfit && grep -n -E 'eval_base_policy|^    subgoal = |run_libero_evaluation' resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
Expected: 看到 `eval_base_policy =`(约 651)、`subgoal =` 的定义行(在 700+ 行)、以及 `run_libero_evaluation(` 调用行(约 993),确认定义都在调用之前。

- [ ] **Step 2: 改调用处传 2 参**

把(约 991-994 行):

```python
            if args.env_family == "libero":
                from resfit.rl_finetuning.chunk_residual.libero_eval import run_libero_evaluation
                m = run_libero_evaluation(env=eval_env, agent=agent,
                                          num_episodes=args.eval_num_episodes, device=args.device)
```

改为:

```python
            if args.env_family == "libero":
                from resfit.rl_finetuning.chunk_residual.libero_eval import run_libero_evaluation
                m = run_libero_evaluation(env=eval_env, agent=agent,
                                          num_episodes=args.eval_num_episodes, device=args.device,
                                          subgoal=(subgoal if args.subgoal_conditioned else None),
                                          base_policy=eval_base_policy)
```

- [ ] **Step 3: 全量聚焦回归(确认无破坏)**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_libero_eval.py resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_pi0_feat_subgoal.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py -q`
Expected: 全绿(libero_eval 新 4 个 + 既有相关测试)。

- [ ] **Step 4: import 健全性自检(语法/作用域)**

Run: `cd /mnt/mnt/data/resfit && /mnt/mnt/data/envs/residual/bin/python -c "import ast; ast.parse(open('resfit/rl_finetuning/chunk_residual/train_chunk_residual.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat(train_chunk_residual): libero eval 传 subgoal + eval_base_policy(在线子目标注入)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## 实现后:集成 smoke(人工,不入 pytest)

两任务通过后,**复用昨天那条崩在 eval 的 task3 跑法**直接重跑验证(serve 已在 GPU:8000、task8 三件套已就绪、`outputs_chunk/libero_task3_pi0feat_offcache` 可秒复用)。从 `/mnt/mnt/data/resfit`,用 **`resfit-libero`** env(有 libero 模块,不是 residual env):

- 沿用上次 `smoke_task3_v2` 的完整命令行(同 suite/task_id=3、`--subgoal_conditioned`、`--gc_value_ckpt`/`--high_actor_ckpt`/`--pi0_feat_cache`、`--pi0_prompt` 为 task3 的任务语言、`--offline_buffer_cache outputs_chunk/libero_task3_pi0feat_offcache`、`--smoke`),其余参照 spec 示例。
- 起 serve(若已停):`cd pi0_serve && CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false /mnt/mnt/data/chj/openpi/.venv/bin/python serve_with_feat.py --config pi0_libero --dir <ckpt29999> --port 8000 --pooling last`

**Expected:** 离线 buffer 秒命中复用 → 训练混采若干步 → **进 eval 不再 `KeyError: 'observation.subgoal'`**,打印 `[env_steps N] eval success_rate=...`、`best.pt` 落盘。成功即解锁正式训练(去掉 `--smoke`、加 `--total_env_steps 500000` 等,复用同一 offcache)。

> 若 eval 这次报别的错(如 z 形状/设备),属 Task 1 注入与 actor 期望维度不符,回到 Task 1 核对 `subgoal_online` 返回维度与 `actor.subgoal_dim`。

---

## 自审清单

**1. Spec 覆盖:** 注入 helper(Task 1 Step 3)、2 参 + guard(Task 1)、reset/step 两处注入(Task 1)、调用处传 `eval_base_policy` 命门(Task 2)、fail-fast 错误处理(Task 1 的 ValueError/NotImplementedError 测试)、零回归(Task 1 `test_no_subgoal_omits_key` + 既有 5 测试)、集成 smoke。全覆盖。

**2. 占位扫描:** 无 TBD/TODO;每步含完整测试与实现代码。

**3. 类型一致:** `_inject_subgoal(subgoal, obs, base_policy)`(Task 1 helper)↔ 两处调用同签名;`subgoal_online(obs, prefix_feat=...)` 与 `hiql_subgoal.py:103` 真实签名一致;`base_policy.last_prefix_feat()` 与 `LiberoPi05Adapter`(commit 074195f)一致;`run_libero_evaluation` 新增 `subgoal=/base_policy=` ↔ Task 2 调用处同名传参。
