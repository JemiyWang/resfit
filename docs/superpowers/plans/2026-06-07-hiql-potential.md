# HIQL value 当 PBS 势函数 Φ(③b)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ③a 学好的冻结 value.pt 当 PBS 势函数 Φ,接进 online(chunk_env_wrapper)+ offline(build_offline_buffer)两端的 reward shaping,以 `--potential_source {stage,hiql}` 默认 stage 逐位等价 baseline。

**Architecture:** 新模块 `hiql_potential.py` 封装「PBS 公式 + 加载冻结 V + 缩放 + 从标准化 state 算 Φ」;online wrapper 与 offline builder 持有同一个 `HiqlPotential`(train 构一次传两端)保证两端 Φ 一致;现状 `shaping_reward` 的 potential 分支重构成复用 `potential_shaping`(逐位等价)。Φ 尺度 = `V(state) * auto_scale * phi_scale`,`auto_scale=(num_stages-1)/(v_max-v_min)` 来自 ③a 的 v_stats,只缩放不平移。

**Tech Stack:** PyTorch、numpy、③a 的 `hiql_value.load_value`/`ValueMLP`、现有 `chunk_env_wrapper`/`offline_hdf5_buffer`/`offline_stage_replay`/`train_chunk_residual`、conda env `residual`、pytest。

**Spec:** `docs/superpowers/specs/2026-06-07-hiql-potential-design.md`

**通用测试命令:** `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest <路径> -v`(所有命令从仓库根跑)

---

## File Structure

- `resfit/rl_finetuning/chunk_residual/hiql_potential.py`(新)—— `potential_shaping`(纯 PBS 函数)+ `HiqlPotential`(加载冻结 V + 缩放 + phi)。
- `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`(改)—— `shaping_reward` potential 分支复用 `potential_shaping`;`ChunkResidualEnvWrapper` 加 `potential` 参数 + `_start_state_std` 维护 + `step` hiql 分支。
- `resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py`(改)—— `transition_rewards`/`transition_fields` 加 `potential`+`state_seq`。
- `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`(改)—— `build_offline_buffer` 加 `potential`,读 state_n 提前,传入。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(改)—— 3 个 flag + 构 `HiqlPotential` + 传两端 + 断言。
- 测试:`tests/test_hiql_potential.py`(新);`tests/test_chunk_env_wrapper.py`(加 1 个 online 测试)。

**default-off 等价主要靠回归**:`potential=None`/`potential_source=stage` 时代码路径与现状一致 -> 现有 `test_chunk_env_wrapper.py` / `test_shaping_reward.py` / `test_staged_reward.py` 全绿即等价证明。每个改动 task 都要跑这些回归。

---

### Task 1: potential_shaping(通用 PBS 纯函数)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/hiql_potential.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_hiql_potential.py`:

```python
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping


def test_potential_shaping_not_done():
    # F = bonus*(gamma*phi_next - phi_start)
    f = potential_shaping(1.0, 2.0, bonus=1.0, gamma=0.99, done=False)
    assert abs(f - (0.99 * 2.0 - 1.0)) < 1e-6


def test_potential_shaping_done_zeroes_next():
    # done -> phi_next=0 -> F = -bonus*phi_start
    f = potential_shaping(3.0, 9.9, bonus=2.0, gamma=0.99, done=True)
    assert abs(f - (2.0 * (0.0 - 3.0))) < 1e-6


def test_potential_shaping_accepts_tensor_scalar():
    # online 传 [1] tensor 标量,应与 float 同结果
    f = potential_shaping(torch.tensor([1.0]), torch.tensor([2.0]),
                          bonus=1.0, gamma=0.99, done=False)
    assert abs(float(f) - (0.99 * 2.0 - 1.0)) < 1e-6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -v`
Expected: FAIL — `ImportError: cannot import name 'potential_shaping'`

- [ ] **Step 3: 写最小实现**

创建 `hiql_potential.py`(文件开头 + 函数):

```python
"""把 ③a 的冻结 value 当 PBS 势函数 Φ(模块 ③b)。

设计见 docs/superpowers/specs/2026-06-07-hiql-potential-design.md。
potential_shaping 是通用 PBS 公式(Φ 可为 int stage 或 V(state)*scale);
HiqlPotential 加载冻结 value 并把标准化 state 映射到 Φ。
"""
import torch

from resfit.rl_finetuning.chunk_residual.hiql_value import load_value


def potential_shaping(phi_start, phi_next, *, bonus, gamma, done):
    """通用 PBS 整形:F = bonus*(gamma*phi_next - phi_start),done 时 phi_next=0。

    phi_start/phi_next 可为 float 或单元素张量(online b=1);返回 float。
    """
    pn = 0.0 if done else float(phi_next)
    return bonus * (gamma * pn - float(phi_start))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_potential.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat(hiql-phi): potential_shaping 通用 PBS 纯函数 + 单测(③b)"
```

---

### Task 2: shaping_reward 重构(potential 分支复用 potential_shaping)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py:36-39`(potential 分支)
- Test: 回归现有 `tests/test_shaping_reward.py`

**Context:** 现状 potential 分支为:
```python
    if mode == "potential":
        phi_start = float(int(start_stage))
        phi_next = 0.0 if done else float(int(end_stage))    # 终止 Φ=0
        return bonus * (gamma * phi_next - phi_start)
```
改成复用 `potential_shaping`(传 int stage 当 phi)——数值逐位不变。这是纯重构,正确性靠现有 `test_shaping_reward.py` 回归。

- [ ] **Step 1: 先跑现有 shaping 测试,确认基线绿**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_shaping_reward.py resfit/rl_finetuning/chunk_residual/tests/test_staged_reward.py -v`
Expected: 全绿(记下数量,作为重构后对照)。

- [ ] **Step 2: 改 potential 分支复用 potential_shaping**

在 `chunk_env_wrapper.py` 顶部 import 区(在 `from resfit...chunk_act_base import get_action_chunk` 之后)加:
```python
from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping
```
把 `shaping_reward` 的 potential 分支(36-39 行)替换为:
```python
    if mode == "potential":
        return potential_shaping(int(start_stage), int(end_stage),
                                 bonus=bonus, gamma=gamma, done=done)
```
（注意:`hiql_potential.py` 顶部 `from ...hiql_value import load_value`,而 `chunk_env_wrapper` 现在 import `hiql_potential` —— 确认无循环 import：`hiql_value` 不 import `chunk_env_wrapper`，`hiql_potential` 不 import `chunk_env_wrapper`，所以 `chunk_env_wrapper -> hiql_potential -> hiql_value` 是单向链，安全。但 `offline_hdf5_buffer` import `chunk_env_wrapper.shaping_reward`，链仍单向。）

- [ ] **Step 3: 跑回归确认逐位等价**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_shaping_reward.py resfit/rl_finetuning/chunk_residual/tests/test_staged_reward.py resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -v`
Expected: 全绿,数量与 Step 1 一致(potential 数值未变)。若有 ImportError(循环),检查 Step 2 的 import 链。

- [ ] **Step 4: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py
git commit -m "refactor(hiql-phi): shaping_reward potential 分支复用 potential_shaping(逐位等价)(③b)"
```

---

### Task 3: HiqlPotential(加载冻结 V + 缩放 + phi)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_potential.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`

- [ ] **Step 1: 写失败测试**

往 `tests/test_hiql_potential.py` 追加(import 加 `HiqlPotential`,并 import ③a 的 save 工具造假 ckpt):

```python
from resfit.rl_finetuning.chunk_residual.hiql_potential import HiqlPotential
from resfit.rl_finetuning.chunk_residual.hiql_value import ValueMLP, save_value


def _make_fake_ckpt(tmp_path, vmin, vmax):
    m = ValueMLP(state_dim=3, hidden=8)
    p = str(tmp_path / "value.pt")
    save_value(p, m, v_stats={"min": vmin, "max": vmax, "mean": 0.5 * (vmin + vmax)},
               mean=torch.zeros(3), std=torch.ones(3), dataset_id="dummy")
    return p, m


def test_hiqlpotential_auto_scale(tmp_path):
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    # num_stages=5 -> 动态范围目标 [0,4];auto_scale=(5-1)/(2-0)=2.0;phi_scale=1
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    assert abs(pot.scale - 2.0) < 1e-6


def test_hiqlpotential_phi_equals_v_times_scale(tmp_path):
    p, model = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    x = torch.randn(4, 3)
    with torch.no_grad():
        expected = model(x).squeeze(-1) * 2.0
    out = pot.phi(x)
    assert out.shape == (4,)
    assert torch.allclose(out, expected, atol=1e-5)


def test_hiqlpotential_phi_scale_multiplies(tmp_path):
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=0.5, device="cpu")
    assert abs(pot.scale - 1.0) < 1e-6   # 2.0 * 0.5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -v`
Expected: FAIL — `ImportError: cannot import name 'HiqlPotential'`

- [ ] **Step 3: 写最小实现**

在 `hiql_potential.py` 追加:

```python
class HiqlPotential:
    """加载 ③a 的冻结 value,把标准化 state 映射到 PBS 势函数值 phi = V(state)*scale。

    scale = auto_scale * phi_scale,auto_scale=(num_stages-1)/(v_max-v_min) 让 V 动态范围
    匹配现状整数 stage Φ 的 [0,num_stages-1]。只缩放不平移(PBS 下平移会引入存活项)。
    """

    def __init__(self, model, scale, device="cpu"):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scale = float(scale)
        self.device = device

    @classmethod
    def from_ckpt(cls, path, *, num_stages, phi_scale=1.0, device="cpu"):
        model, info = load_value(path, map_location=device)
        vmin, vmax = info["v_stats"]["min"], info["v_stats"]["max"]
        auto_scale = (num_stages - 1) / max(vmax - vmin, 1e-6)
        return cls(model, scale=auto_scale * phi_scale, device=device)

    @torch.no_grad()
    def phi(self, state_std):
        """state_std: [B,state_dim] 已标准化 -> [B] 势函数值 = V(state)*scale。"""
        return self.model(state_std.to(self.device)).squeeze(-1) * self.scale
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -v`
Expected: PASS(6 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_potential.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat(hiql-phi): HiqlPotential 加载冻结 V + auto_scale + phi + 单测(③b)"
```

---

### Task 4: offline transition_rewards/transition_fields 加 potential

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py`(`transition_rewards` 111-127、`transition_fields` 75-99)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`

- [ ] **Step 1: 写失败测试**

往 `tests/test_hiql_potential.py` 追加(import numpy + transition_rewards):

```python
import numpy as np
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import transition_rewards


class _StubPot:
    """phi(state_seq[T,D]) = 每行第一元素,模拟 V(state)。"""
    def phi(self, state_seq):
        return state_seq[:, 0]


def test_transition_rewards_hiql_uses_v():
    instant = np.array([0, 1, 2])                 # 3 帧 -> 2 transition,T=3
    state_seq = torch.tensor([[10.], [20.], [30.]])  # phi = [10,20,30]
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.99,
                           success=True, potential=_StubPot(), state_seq=state_seq)
    exp0 = potential_shaping(10.0, 20.0, bonus=1.0, gamma=0.99, done=False)        # t=0 not done
    exp1 = 1.0 + potential_shaping(20.0, 30.0, bonus=1.0, gamma=0.99, done=True)   # t=1 done(base+1)
    assert abs(r[0] - exp0) < 1e-5
    assert abs(r[1] - exp1) < 1e-5


def test_transition_rewards_none_matches_stage_baseline():
    # potential=None(默认)走现状 latch 路径,逐位等价
    instant = np.array([0, 1, 2])
    from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import shaping_reward
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.99, success=True)
    exp0 = shaping_reward(0, 1, mode="potential", bonus=1.0, gamma=0.99, done=False)
    exp1 = 1.0 + shaping_reward(1, 2, mode="potential", bonus=1.0, gamma=0.99, done=True)
    assert abs(r[0] - exp0) < 1e-5 and abs(r[1] - exp1) < 1e-5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k "transition_rewards" -v`
Expected: FAIL — `TypeError: transition_rewards() got an unexpected keyword argument 'potential'`

- [ ] **Step 3: 写实现**

把 `offline_hdf5_buffer.py` 的 `transition_rewards`(111-127)替换为:

```python
def transition_rewards(instant_stages, *, bonus: float, mode: str,
                       gamma: float, success: bool = True,
                       potential=None, state_seq=None) -> np.ndarray:
    """一条 demo 的 T 帧瞬时 stage → T-1 个 transition 的总 reward。

    与线上 cl=1 一致:每步 reward = base 稀疏 + shaping。
    potential=None:Φ=闩锁 stage(现状)。potential 非空(③b):Φ=potential.phi(state_seq)(V*scale),
    用通用 potential_shaping;两端用同一个 potential 保证 Φ 一致。
    """
    latch = latch_from_instant(instant_stages)
    T = len(latch)
    rewards = np.empty(T - 1, dtype=np.float32)
    phi = None
    if potential is not None:
        assert state_seq is not None and len(state_seq) == T, \
            "potential 模式需 state_seq 且长度=T"
        phi = potential.phi(state_seq)            # [T]
    for t in range(T - 1):
        done = success and (t == T - 2)
        base = float(done)
        if potential is None:
            shaped = shaping_reward(int(latch[t]), int(latch[t + 1]),
                                    mode=mode, bonus=bonus, gamma=gamma, done=done)
        else:
            shaped = potential_shaping(phi[t], phi[t + 1],
                                       bonus=bonus, gamma=gamma, done=done)
        rewards[t] = base + shaped
    return rewards
```
在 `offline_hdf5_buffer.py` 顶部 import 区(在 `from ...chunk_env_wrapper import shaping_reward` 那行之后)加:
```python
from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping
```
再把 `transition_fields`(75-99)签名与对 `transition_rewards` 的调用加上透传:签名末尾加 `potential=None, state_seq=None`,把 `reward` 那行改为:
```python
        "reward": transition_rewards(instant, bonus=bonus, mode=mode, gamma=gamma,
                                     success=success, potential=potential, state_seq=state_seq),
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -v`
Expected: PASS(8 passed)。再跑现有 offline 回归:
`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -k "offline or hdf5 or shaping" -q` 应全绿。

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/offline_hdf5_buffer.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat(hiql-phi): offline transition_rewards/fields 支持 V(state) Φ(potential 参数)(③b)"
```

---

### Task 5: offline build_offline_buffer 接线(读 state_n 提前 + 传 potential)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`(`build_offline_buffer` 139-216)

**Context:** 现状循环里 `transition_fields(instant, ...)`(约 181-182)在读标准化 `state_n`(约 184-186)之前调。要 (a) 给 `build_offline_buffer` 加 `potential=None` 参数,(b) 把读 `state_n` 提到 `transition_fields` 之前,(c) 调用 `transition_fields(..., potential=potential, state_seq=state_n)`。这是接线改动,正确性靠 Task 4 单测 + manual smoke;无新单测(集成需真 hdf5)。

- [ ] **Step 1: 改 build_offline_buffer**

打开 `offline_stage_replay.py`,在 `build_offline_buffer` 的签名(约 139-141)末尾加参数 `potential=None`。

在循环体内,把读 `state_n` 的那两行(约 184-186):
```python
                obs_arrays = {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS}
                state_n = state_standardizer.standardize(
                    torch.as_tensor(assemble_state18(obs_arrays), dtype=torch.float32)).cpu()
```
**移动到** `T = len(instant); if T < 2: continue` 之后、`transition_fields(...)` 调用**之前**。然后把 `transition_fields` 调用(约 181-182)改为:
```python
                fld = transition_fields(instant, bonus=bonus, mode=mode, gamma=gamma,
                                        success=True, potential=potential, state_seq=state_n)
```
（注意 `state_n` 现在在 `transition_fields` 之前已就绪;其余使用 `state_n` 的灌装代码不变。）

- [ ] **Step 2: import 冒烟(确认无语法/循环问题)**

Run: `conda run -n residual python -c "from resfit.rl_finetuning.chunk_residual.offline_stage_replay import build_offline_buffer; print('ok')"`
Expected: 打印 `ok`(无 ImportError)。

- [ ] **Step 3: 跑相关回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -k "offline or hdf5 or shaping or hiql" -q`
Expected: 全绿(build_offline_buffer 的 potential 默认 None,现状路径不变)。

- [ ] **Step 4: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/offline_stage_replay.py
git commit -m "feat(hiql-phi): build_offline_buffer 读 state_n 提前 + 传 potential 给 transition_fields(③b)"
```

---

### Task 6: online chunk_env_wrapper 接线(potential 参数 + _start_state_std + step hiql 分支)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`(`__init__` 54-66、`reset` 110-117、`step` 119-179)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`

- [ ] **Step 1: 写失败测试**

往 `tests/test_chunk_env_wrapper.py` 追加(文件已 import torch/np/gym + fake;复用 `_FakeBase`/`_IdentityScaler`/`_IdentityStd`):

```python
class _StateRampEnv:
    """obs.state 每步递增(不 terminate),便于区分 phi(start)/phi(next) 并验证 start 推进。"""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(D,), dtype=np.float32)
        self._k = 0

    def _obs(self):
        return {"observation.state": torch.full((1, 3), float(self._k)),
                "observation.images.cam": torch.zeros(1, 3, 4, 4)}

    def reset(self, **kw):
        self._k = 0
        return self._obs(), {}

    def step(self, action):
        self._k += 1
        return self._obs(), torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {}


class _StubPotential:
    """phi(state[B,3]) = 每行第一元素(= ramp 的 _k),模拟 V(state)。"""
    def phi(self, state_std):
        return state_std[:, 0]


def test_step_hiql_potential_uses_v_and_advances_start():
    from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping
    env = _StateRampEnv()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=1, reward_shaping_mode="potential",
                                stage_reward_bonus=1.0, gamma=0.99, potential=_StubPotential())
    w.reset()                                  # start state _k=0 -> phi 0
    _, r1, *_ = w.step(torch.zeros(1, D))      # end _k=1 -> phi 1;env reward=0
    assert abs(float(r1) - potential_shaping(0.0, 1.0, bonus=1.0, gamma=0.99, done=False)) < 1e-5
    _, r2, *_ = w.step(torch.zeros(1, D))      # start 应推进到上次 end(_k=1),end=_k=2
    assert abs(float(r2) - potential_shaping(1.0, 2.0, bonus=1.0, gamma=0.99, done=False)) < 1e-5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py::test_step_hiql_potential_uses_v_and_advances_start -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'potential'`

- [ ] **Step 3: 写实现**

在 `chunk_env_wrapper.py` 顶部已（Task 2）import 了 `potential_shaping`。

(a) `__init__`(54-66):签名末尾加 `potential=None`;函数体里(`self.base_action_mode = base_action_mode` 之后)加:
```python
        self.potential = potential          # None=Φ用stage(现状);HiqlPotential=Φ用V(state)
        self._start_state_std = None         # 本 chunk 起点的标准化 state(potential 模式用)
```

(b) `reset`(110-117):把 `return self._augment(raw_obs, base_flat), info` 改为:
```python
        aug = self._augment(raw_obs, base_flat)
        self._start_state_std = aug["observation.state"]
        return aug, info
```

(c) `step`(119-179):把 shaping 那段(163-166)替换为:
```python
        chunk_done = bool((terminated | truncated).any())
        if self.potential is None:
            total_reward = total_reward + shaping_reward(
                start_stage, max_in_chunk, mode=self.reward_shaping_mode,
                bonus=self.stage_reward_bonus, gamma=self.gamma, done=chunk_done)
        else:
            end_state_std = self.state_standardizer.standardize(raw_obs["observation.state"])
            phi_start = self.potential.phi(self._start_state_std)
            phi_next = self.potential.phi(end_state_std)
            total_reward = total_reward + potential_shaping(
                phi_start, phi_next, bonus=self.stage_reward_bonus,
                gamma=self.gamma, done=chunk_done)
```
并在 `step` 末尾(`aug_obs = self._augment(...)` 之后、`return` 之前)加更新起点 state:
```python
        self._start_state_std = aug_obs["observation.state"]   # 本 chunk 末 = 下个 chunk 起点
```

- [ ] **Step 4: 跑测试确认通过 + 回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -v`
Expected: 新测试 PASS,且现有所有 wrapper 测试仍绿(potential 默认 None -> 现状路径不变 = default-off 等价证明)。

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
git commit -m "feat(hiql-phi): online wrapper potential 参数 + _start_state_std + step hiql 分支(③b)"
```

---

### Task 7: train_chunk_residual CLI + 构 HiqlPotential + 传两端 + 断言

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(build_parser 约 262-267 区;shaping_mode 解析约 343;wrapper 构造约 346-350;build_offline_buffer 调用约 454-459)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py`

- [ ] **Step 1: 写失败测试(CLI)**

往 `tests/test_hiql_potential.py` 追加:

```python
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def test_parser_potential_source_defaults():
    args = build_parser().parse_args(["--task", "TwoArmThreePieceAssembly"])
    assert args.potential_source == "stage"
    assert args.hiql_value_ckpt is None
    assert args.phi_scale == 1.0


def test_parser_potential_source_hiql():
    args = build_parser().parse_args(
        ["--task", "TwoArmThreePieceAssembly", "--potential_source", "hiql",
         "--hiql_value_ckpt", "v.pt", "--phi_scale", "0.5"])
    assert args.potential_source == "hiql"
    assert args.hiql_value_ckpt == "v.pt"
    assert args.phi_scale == 0.5
```
（注:`--task` 是必填项;若 `build_parser` 还有其它 required flag 导致 parse 失败,按报错补上最小必填参数即可。）

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -k "parser_potential" -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'potential_source'`

- [ ] **Step 3: 写实现**

(a) 在 `build_parser` 的 reward shaping flag 区(约 262-267,`--reward_shaping`/`--stage_reward_bonus` 附近)加:
```python
    p.add_argument("--potential_source", choices=["stage", "hiql"], default="stage",
                   help="PBS 势函数 Φ 来源(③b):stage=整数 stage(默认,逐位等价);hiql=V(state)*scale")
    p.add_argument("--hiql_value_ckpt", default=None,
                   help="--potential_source hiql 时 ③a 产出的 value.pt 路径")
    p.add_argument("--phi_scale", type=float, default=1.0,
                   help="hiql Φ 的额外缩放乘子(在 auto_scale 之上;默认 1.0)")
```

(b) 在 `shaping_mode = resolve_shaping_mode(...)`(约 343)之后、构 env 之前,加构造 potential + 断言:
```python
    potential = None
    if args.potential_source == "hiql":
        import os
        assert shaping_mode == "potential", \
            "--potential_source hiql 需 --reward_shaping potential"
        assert args.hiql_value_ckpt and os.path.exists(args.hiql_value_ckpt), \
            f"--hiql_value_ckpt 不存在: {args.hiql_value_ckpt!r}"
        from resfit.rl_finetuning.chunk_residual.hiql_potential import HiqlPotential
        potential = HiqlPotential.from_ckpt(
            args.hiql_value_ckpt, num_stages=num_stages,
            phi_scale=args.phi_scale, device=args.device)
        print(f"[hiql-phi] potential on; ckpt={args.hiql_value_ckpt} scale={potential.scale:.4f}")
```
（`num_stages` 在该脚本里已定义,与 stage_budget 用的同一个变量;若它定义在此行之后,把上面这段移到 `num_stages` 定义之后、env 构造之前。）

(c) `ChunkResidualEnvWrapper(...)` 构造(约 346-350)加 `potential=potential`。

(d) `build_offline_buffer(...)` 调用(约 454-459)加 `potential=potential`。

- [ ] **Step 4: 跑测试确认通过 + 全回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -v`
Expected: PASS(10 passed)。

全回归:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全绿(含 stage_budget/demo_bc/relabel/wandb/hiql_value/shaping/wrapper/staged)。

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py
git commit -m "feat(hiql-phi): train CLI(--potential_source/--hiql_value_ckpt/--phi_scale)+ 构 HiqlPotential 传两端 + 断言(③b)"
```

- [ ] **Step 6: Manual 两端一致冒烟(可选但推荐,需真 value.pt)**

若已用 ③a 训出 `value.pt`,跑一个极短训练冒烟(确认 online+offline 两端都能用 hiql Φ 起来、不报错):
```bash
cd /data2/RL/residual-offpolicy-rl && CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id dexmg-threepiece-bc/7zklm69g \
  --dataset ankile/dexmg-two-arm-three-piece-assembly --chunk_length 1 --base_action_mode queue \
  --reward_shaping potential --potential_source hiql --hiql_value_ckpt outputs_chunk/three_piece_value.pt \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/three_piece_stages.npz \
  --offline_dataset_path deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --smoke --wandb_mode offline 2>&1 | tail -20
```
Expected: 打印 `[hiql-phi] potential on; ... scale=...`,offline buffer 构建与若干训练 step 不报错。若环境/数据问题(metadata/GPU)导致,记录但不阻塞 task 完成。

---

## 验收(全部 Task 完成后)

- 新测试:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_potential.py -v` -> 10 passed。
- online 新测试:`test_chunk_env_wrapper.py::test_step_hiql_potential_uses_v_and_advances_start` 绿。
- **全回归**:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q` 全绿(default-off 等价:potential=None / potential_source=stage 路径未变)。
- default-off:不传 `--potential_source`(=stage)-> 两端 potential=None -> 逐位等价 baseline。
- 两端一致:online 与 offline 用同一个 `HiqlPotential`(train 构一次传两端)。
- A/B(run 阶段,需先 ③a 训 value.pt):A=`--reward_shaping potential`;B=`+ --potential_source hiql --hiql_value_ckpt value.pt`,扫 `--phi_scale`。
