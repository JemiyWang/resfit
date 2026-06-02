# cl=1 queue 基座模式 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 chunk_residual 在 chunk_length=1 时走 ACT 原生 action queue 复刻原版 step 级残差基座(含不 clamp),同时保留 staged 奖励等 chunk 管线机制。

**Architecture:** 给 `ChunkResidualEnvWrapper` 加 `base_action_mode`("replan" 默认不变 / "queue" 走 `base_policy.select_action`),queue 模式 assert chunk_length==1 且 combined 动作不 clamp;`train_chunk_residual.py` 加 `--base_action_mode` 并在 queue 模式给 eval 单独的 base_policy 实例(避免 queue 状态互踩)。

**Tech Stack:** Python, PyTorch, torchrl, lerobot ACTPolicy, pytest, conda env `residual`。

**约定:**
- 测试命令:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -v`
- 仓库:`/data2/RL/residual-offpolicy-rl`,分支 `chunk-residual-validation`。
- **commit 由用户自己管**(见 HANDOFF);计划里的 commit 步骤标注为可选,可批量提交。
- 相关文件:
  - 改:`resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`
  - 改:`resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
  - 测:`resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`
- 参考(原版双实例做法):`resfit/rl_finetuning/scripts/train_residual_td3.py:265-270, 373`

---

## Task 0: 加测试夹具(queue 假基座)

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`(在文件末尾追加)

- [ ] **Step 1: 在测试文件末尾追加 queue 假基座夹具**

```python
from collections import deque


class _FakeQueueBase(_FakeBase):
    """模拟 ACT select_action 的内部 queue:每 PLAN_LEN 步重规划一次,每次 pop 一个。
    每次重规划用递增的 plan_id 填充(整段值=plan_id),便于断言'同一规划顺序取'。
    继承 _FakeBase 拿到 config / get_action_chunk(queue 模式不会用到后者)。"""
    PLAN_LEN = 2

    def __init__(self):
        self.queue = deque()
        self.model_calls = 0     # 重规划次数(= 跑底层 model 的次数)
        self.reset_calls = 0
        self.plan_id = 0

    def reset(self, env_ids=None):
        self.queue.clear()
        self.reset_calls += 1

    def select_action(self, raw_obs):
        b = raw_obs["observation.state"].shape[0]
        if len(self.queue) == 0:
            self.model_calls += 1
            self.plan_id += 1
            for _ in range(self.PLAN_LEN):
                self.queue.append(torch.full((b, D), float(self.plan_id)))
        return self.queue.popleft()


class _FakeVecEnvTermAt1:
    """第 1 步即 terminate 的单环境(配 chunk_length=1 测 done 清队)。"""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(D,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-np.inf, np.inf, (1, 3), dtype=np.float32),
            "observation.images.cam": gym.spaces.Box(0, 1, (1, 3, 4, 4), dtype=np.float32),
        })

    def _obs(self):
        return {"observation.state": torch.zeros(1, 3),
                "observation.images.cam": torch.zeros(1, 3, 4, 4)}

    def reset(self, **kw):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), torch.tensor([1.0]), torch.tensor([True]), torch.tensor([False]), {}
```

- [ ] **Step 2: 跑现有测试确认夹具不破坏导入**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -v`
Expected: 现有 4 个测试仍 PASS(只是加了夹具,无新测试)。

- [ ] **Step 3: 提交(可选,你自己管 commit)**

```bash
git add resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
git commit -m "test(chunk_residual): add queue fake base fixtures"
```

---

## Task 1: 构造参数 base_action_mode + queue 仅限 cl=1 的 assert

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py:52-72`(构造函数)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`

- [ ] **Step 1: 写失败测试**

```python
def test_queue_mode_requires_chunk_length_one():
    import pytest
    env = _FakeVecEnv()
    with pytest.raises(AssertionError):
        ChunkResidualEnvWrapper(env, _FakeQueueBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=2, base_action_mode="queue")


def test_queue_mode_constructs_at_cl1():
    env = _FakeVecEnv()
    w = ChunkResidualEnvWrapper(env, _FakeQueueBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, base_action_mode="queue")
    assert w.base_action_mode == "queue"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_queue_mode_constructs_at_cl1 -v`
Expected: FAIL(`__init__` 不接受 `base_action_mode` → TypeError)。

- [ ] **Step 3: 改构造函数**

把 `chunk_env_wrapper.py` 的构造签名与函数体改为(在 `gamma` 后加参,在 assert chunk_length 后加 mode assert,并存字段):

```python
    def __init__(self, vec_env, base_policy, action_scaler, state_standardizer,
                 chunk_length: int, stage_reward_bonus: float = 0.0,
                 reward_shaping_mode: str = "none", gamma: float = 0.99,
                 base_action_mode: str = "replan"):
        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.chunk_length = chunk_length
        self.stage_reward_bonus = stage_reward_bonus
        self.reward_shaping_mode = reward_shaping_mode
        self.gamma = gamma
        self.base_action_mode = base_action_mode
        assert chunk_length >= 1, "chunk_length must be >= 1"
        assert base_action_mode in ("replan", "queue"), \
            f"unknown base_action_mode: {base_action_mode!r}"
        if base_action_mode == "queue":
            assert chunk_length == 1, "base_action_mode='queue' 仅支持 chunk_length==1"
        self.action_dim = vec_env.action_space.shape[-1]
        self.num_envs = getattr(vec_env, "num_envs", 1)
        self.flat_dim = chunk_length * self.action_dim
        self._last_base_flat = None
        self._get_chunk = getattr(base_policy, "get_action_chunk", None)
        self._stage = 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_queue_mode_requires_chunk_length_one resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_queue_mode_constructs_at_cl1 -v`
Expected: 两个 PASS。

- [ ] **Step 5: 提交(可选)**

```bash
git add resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
git commit -m "feat(chunk_residual): add base_action_mode param with queue cl=1 guard"
```

---

## Task 2: queue 路径(_base_chunk_flat 走 select_action,顺序取、done 清队)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py:75-81`(`_base_chunk_flat`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`

- [ ] **Step 1: 写失败测试**

```python
def test_queue_mode_uses_select_action_and_sequential_pop():
    # PLAN_LEN=2:reset 取 plan1 第1步;step1 取 plan1 第2步(不重规划);step2 触发 plan2。
    env = _FakeVecEnvNoTerm()
    base = _FakeQueueBase()
    w = ChunkResidualEnvWrapper(env, base, _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, base_action_mode="queue")
    obs, _ = w.reset()
    assert obs["observation.base_action"].shape == (1, D)        # cl=1 → 单步 D 维
    assert torch.allclose(obs["observation.base_action"], torch.full((1, D), 1.0))  # plan1
    assert base.model_calls == 1                                  # 只规划过一次

    next_obs, *_ = w.step(torch.zeros(1, D))
    assert torch.allclose(next_obs["observation.base_action"], torch.full((1, D), 1.0))  # plan1 第2步
    assert base.model_calls == 1                                  # 顺序取,未重规划

    next_obs2, *_ = w.step(torch.zeros(1, D))
    assert torch.allclose(next_obs2["observation.base_action"], torch.full((1, D), 2.0))  # plan2
    assert base.model_calls == 2                                  # queue 空 → 重规划


def test_queue_mode_clears_queue_on_done():
    env = _FakeVecEnvTermAt1()
    base = _FakeQueueBase()
    w = ChunkResidualEnvWrapper(env, base, _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, base_action_mode="queue")
    w.reset()
    resets_before = base.reset_calls
    _, _, terminated, _, _ = w.step(torch.zeros(1, D))
    assert terminated.item() is True
    assert base.reset_calls == resets_before + 1                  # done 触发 base.reset() 清队
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_queue_mode_uses_select_action_and_sequential_pop -v`
Expected: FAIL(queue 模式未实现,`_base_chunk_flat` 仍走 get_action_chunk → base_action 是 0.5 而非 1.0)。

- [ ] **Step 3: 改 `_base_chunk_flat` 加 queue 分支**

把 `_base_chunk_flat` 改为:

```python
    # ---- 取基座动作并归一化到 [-1,1] 展平 ----
    def _base_chunk_flat(self, raw_obs):
        if self.base_action_mode == "queue":                     # cl=1:走 ACT 原生 action queue
            base_action = self.base_policy.select_action(raw_obs)  # [B, D]
            base_n = self.action_scaler.scale(base_action)
            return base_n.reshape(base_n.shape[0], -1)            # [B, D] (=L*D, L=1)
        # replan(默认):每边界重跑模型取前 chunk_length 步
        if self._get_chunk is not None:                          # fake/可替换路径
            chunk_raw = self._get_chunk(raw_obs, self.chunk_length)
        else:
            chunk_raw = get_action_chunk(self.base_policy, raw_obs, self.chunk_length)
        chunk_n = self.action_scaler.scale(chunk_raw)            # [B,L,D] -> [-1,1]
        return chunk_n.reshape(chunk_n.shape[0], -1)             # [B, L*D]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_queue_mode_uses_select_action_and_sequential_pop resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_queue_mode_clears_queue_on_done -v`
Expected: 两个 PASS。

- [ ] **Step 5: 提交(可选)**

```bash
git add resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
git commit -m "feat(chunk_residual): queue base path via select_action"
```

---

## Task 3: queue 模式 combined 动作不 clamp(replan 仍 clamp)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py:108`(step 内 combined 计算)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`

- [ ] **Step 1: 写失败测试**

```python
def test_queue_mode_does_not_clamp_combined():
    # base=1.0(scale 后仍 1.0),residual=0.5 → combined=1.5;queue 模式不 clamp → env 收到 1.5
    env = _FakeVecEnvNoTerm()
    w = ChunkResidualEnvWrapper(env, _FakeQueueBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, base_action_mode="queue")
    w.reset()
    w.step(torch.full((1, D), 0.5))
    assert torch.allclose(env.last_actions[0], torch.full((1, D), 1.5), atol=1e-5)


def test_replan_mode_still_clamps_combined():
    # base=0.5,residual=0.8 → 1.3;replan 模式 clamp → env 收到 1.0
    env = _FakeVecEnvNoTerm()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=L)  # replan 默认
    w.reset()
    w.step(torch.full((1, L * D), 0.8))
    assert torch.allclose(env.last_actions[0], torch.full((1, D), 1.0), atol=1e-5)
```

注:`_FakeVecEnvNoTerm` 需记录 `last_actions`。若其当前实现未记录,在 Step 3 一并补上
`self.last_actions = []`(`reset` 清空)与 `step` 里 `self.last_actions.append(action.clone())`。

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_queue_mode_does_not_clamp_combined -v`
Expected: FAIL(当前恒 clamp → env 收到 1.0 而非 1.5;或因 `_FakeVecEnvNoTerm` 无 last_actions 报 AttributeError)。

- [ ] **Step 3: 给 `_FakeVecEnvNoTerm` 补 last_actions,并改 step 的 combined 计算**

3a. 在测试文件 `_FakeVecEnvNoTerm.__init__` 末尾加 `self.last_actions = []`;`reset` 里加 `self.last_actions = []`;`step` 开头加 `self.last_actions.append(action.clone())`。

3b. 把 `chunk_env_wrapper.py` step 内这行:

```python
        combined_flat = torch.clamp(self._last_base_flat + residual_flat, -1.0, 1.0)  # [B,L*D]
```

改为:

```python
        if self.base_action_mode == "queue":
            combined_flat = self._last_base_flat + residual_flat                       # 等同原版,不 clamp
        else:
            combined_flat = torch.clamp(self._last_base_flat + residual_flat, -1.0, 1.0)  # [B,L*D]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_queue_mode_does_not_clamp_combined resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_replan_mode_still_clamps_combined -v`
Expected: 两个 PASS。

- [ ] **Step 5: 跑整个 wrapper 测试文件确认无回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -v`
Expected: 全部 PASS(原 4 + 新增 7 = 11)。

- [ ] **Step 6: 提交(可选)**

```bash
git add resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
git commit -m "feat(chunk_residual): no clamp in queue mode (match original)"
```

---

## Task 4: train 脚本 --base_action_mode + queue 模式独立 eval base_policy

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:152-182`(argparse + env 构造)

- [ ] **Step 1: 加 argparse 参数**

在 `--output_dir` 那条之后(约 L153)加:

```python
    p.add_argument("--base_action_mode", choices=["replan", "queue"], default="replan",
                   help="基座动作来源:replan(每边界重跑模型取前chunk步)|queue(ACT原生action queue,仅cl=1,复刻原版step级)")
```

- [ ] **Step 2: queue 模式 assert + 透传 + 独立 eval 基座**

把当前 env / eval_env 构造段(约 L171-182):

```python
    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device)
    shaping_mode = resolve_shaping_mode(args.reward_shaping, args.staged_reward)
    print(f"[reward-shaping] mode={shaping_mode} bonus={args.stage_reward_bonus} gamma={args.gamma}")
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length,
                                  stage_reward_bonus=args.stage_reward_bonus,
                                  reward_shaping_mode=shaping_mode, gamma=args.gamma)
    eval_vec = create_vectorized_env(env_name=args.task, num_envs=args.eval_num_envs,
                                     device=args.device)
    eval_env = ChunkResidualEnvWrapper(eval_vec, base_policy, action_scaler, state_standardizer,
                                       chunk_length=args.chunk_length,
                                       reward_shaping_mode="none")   # eval 不加 shaping,指标纯净
```

改为:

```python
    if args.base_action_mode == "queue":
        assert args.chunk_length == 1, "--base_action_mode queue 仅支持 --chunk_length 1"
    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device)
    shaping_mode = resolve_shaping_mode(args.reward_shaping, args.staged_reward)
    print(f"[reward-shaping] mode={shaping_mode} bonus={args.stage_reward_bonus} gamma={args.gamma}")
    print(f"[base-action] mode={args.base_action_mode}")
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length,
                                  stage_reward_bonus=args.stage_reward_bonus,
                                  reward_shaping_mode=shaping_mode, gamma=args.gamma,
                                  base_action_mode=args.base_action_mode)
    eval_vec = create_vectorized_env(env_name=args.task, num_envs=args.eval_num_envs,
                                     device=args.device)
    # queue 有状态(per-env action queue):eval(num_envs>1)与训练(num_envs=1)共享同一 base_policy
    # 会互踩 queue。queue 模式给 eval 单独的 base_policy 实例(对齐原版 train_residual_td3 双实例)。
    eval_base_policy = (build_base_policy(args.base_wandb_id, args.device)
                        if args.base_action_mode == "queue" else base_policy)
    eval_env = ChunkResidualEnvWrapper(eval_vec, eval_base_policy, action_scaler, state_standardizer,
                                       chunk_length=args.chunk_length,
                                       reward_shaping_mode="none",   # eval 不加 shaping,指标纯净
                                       base_action_mode=args.base_action_mode)
```

- [ ] **Step 3: 冒烟验证(无 GPU 训练,只验证启动与 env 步通)**

Run:
```bash
cd /data2/RL/residual-offpolicy-rl && CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run --no-capture-output -n residual python -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task ThreePieceAssembly --base_wandb_id <ThreePiece 本地 base 目录> \
  --dataset <ThreePiece dataset> --chunk_length 1 --base_action_mode queue \
  --reward_shaping staged --stage_balanced --smoke
```
Expected: 打印 `[base-action] mode=queue`,完成 2 步 chunk 冒烟无异常退出(`--smoke` 跑极少步)。
注:`<...>` 占位用 HANDOFF 里 ThreePiece 的本地 base 目录与 dataset 名替换(沿用之前 run 的取值)。

- [ ] **Step 4: queue + cl>1 的 assert 验证**

Run:
```bash
cd /data2/RL/residual-offpolicy-rl && conda run --no-capture-output -n residual python -u -m \
  resfit.rl_finetuning.chunk_residual.train_chunk_residual --chunk_length 5 --base_action_mode queue --smoke
```
Expected: 立即 AssertionError "仅支持 --chunk_length 1"(在加载数据前就该触发;若数据加载在前则在 env 构造处触发)。

- [ ] **Step 5: 提交(可选)**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat(chunk_residual): --base_action_mode + separate eval base in queue mode"
```

---

## Task 5: 全量回归 + 正式 run

**Files:** 无(验证 + 起跑)

- [ ] **Step 1: 跑整个 chunk_residual 测试套件**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -v`
Expected: 原 77 passed + 本轮新增 7 = 84 passed(数字以实际为准,关键是 0 failed)。

- [ ] **Step 2: 起正式 run(cl=1 queue step 级 + staged)**

按 HANDOFF 踩坑清单:独立 `--output_dir`、`--no-capture-output`、`-u`/`PYTHONUNBUFFERED=1`、`setsid nohup`、哨兵盯真 python PID。示例:
```bash
cd /data2/RL/residual-offpolicy-rl && CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONUNBUFFERED=1 setsid nohup conda run --no-capture-output -n residual python -u -m \
  resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task ThreePieceAssembly --base_wandb_id <本地 base> --dataset <dataset> \
  --chunk_length 1 --base_action_mode queue --reward_shaping staged --stage_balanced \
  --output_dir outputs_chunk/cl1_queue_staged \
  > outputs_chunk/cl1_queue_staged.log 2>&1 &
```
Expected: 日志出现 `[base-action] mode=queue`,`ActionScaler initialized` 仅 1 次(grep -c =1),周期性 `[env_steps ...] eval success_rate=...`。

- [ ] **Step 3: 记录运行态 + 更新项目记忆**

把 run 的 GPU/pid/output_dir/log 记下;在 `project_dexmg_stage_detector` 记忆追加 queue 模式与本 run 状态。

---

## Self-Review(已执行)

**Spec 覆盖:** spec §3.1(base_action_mode + queue 分支 + assert + 不 clamp)→ Task 1/2/3;§3.2(train 参数 + 独立 eval 基座)→ Task 4;§5 测试 5 条 → Task 1(assert cl>1)/Task 2(select_action 使用、顺序取、done 清队)/Task 3(不 clamp);§6 验收 → Task 4 冒烟 + Task 5 回归/行为核对。覆盖完整。

**占位符:** Task 3/4 的 `<ThreePiece 本地 base 目录>` / `<dataset>` 是运行期真实取值(沿用 HANDOFF 之前 run),非代码占位;已标注来源。其余无 TBD/TODO。

**类型/命名一致:** `base_action_mode` 字段名、`"replan"/"queue"` 取值、`select_action` 调用、`_FakeQueueBase`/`_FakeVecEnvTermAt1` 夹具名在 Task 0-4 间一致。
