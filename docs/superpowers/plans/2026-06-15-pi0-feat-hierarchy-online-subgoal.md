# pi0_feat 完整分层(high_actor + 在线子目标注入)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `state_mode=pi0_feat` 接完整 HIQL 分层后两段——high_actor 离线训 + 在线 rollout 高层出子目标 z 注入低层残差,让"pi0表征→V+高层→子目标→低层"完整链路打通、能起残差 RL(接通+smoke)。

**Architecture:** 完全照 act_feat 已趟通的分层模式;唯一差异是在线特征来源——act_feat 进程内 `ActFeatureExtractor.embed_batch`,pi0_feat 改成取 base policy(`LiberoPi05Adapter`)最近一次 serve infer 返回的 `prefix_feat`(零额外推理)。同源靠 ckpt 存的 feat_mean/std + 同一 serve 的 prefix_feat。

**Tech Stack:** Python / PyTorch / conda env `residual` / pytest。LIBERO 路(pi0_libero serve);stage_cache=None(geometric)下不读 hdf5,LIBERO 兼容。

参照 spec: `docs/superpowers/specs/2026-06-15-pi0-feat-hierarchy-online-subgoal-design.md`。跑测: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest <path> -v`。

---

## File Structure

- Modify `resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py` — `_infer_chunk` 存 `_last_prefix_feat` + `last_prefix_feat()` getter。
- Modify `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py` — `save/load_high_actor` 加可选 `pi0_feat_signature`。
- Modify `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py` — `--state_mode pi0_feat` dispatch(照 gc_value)。
- Modify `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py` — `from_ckpts`/`__init__`/`subgoal_online` 的 pi0_feat 分支。
- Modify `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` — subgoal 注入处 pi0_feat 取 base prefix_feat。
- Tests: `tests/test_adapter_prefix_feat.py`、`tests/test_hiql_subgoal_pi0_feat.py`、`tests/test_pi0_feat_high_actor_cli.py`、`tests/test_train_chunk_residual_pi0_feat_subgoal.py`、opt-in `tests/test_pi0_feat_online_subgoal_smoke.py`。

---

## Task 1: `LiberoPi05Adapter` 暴露 prefix_feat

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_adapter_prefix_feat.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from resfit.rl_finetuning.chunk_residual.libero_pi05_adapter import LiberoPi05Adapter


class _StubPolicy:
    def __init__(self, with_feat=True):
        self.with_feat = with_feat
    def infer(self, obs):
        out = {"actions": np.zeros((5, 7), np.float32)}
        if self.with_feat:
            out["prefix_feat"] = np.arange(2048, dtype=np.float32)
        return out


def test_adapter_stores_and_exposes_prefix_feat():
    a = LiberoPi05Adapter.from_policy(_StubPolicy(with_feat=True), prompt="x", action_dim=7,
                                      device="cpu", execute_horizon=5)
    assert a.last_prefix_feat() is None                       # 未 infer 前
    a._infer_chunk({"observation/image": np.zeros((224, 224, 3), np.uint8)})
    pf = a.last_prefix_feat()
    assert pf is not None and np.asarray(pf).shape == (2048,)


def test_adapter_prefix_feat_none_when_serve_omits_it():
    a = LiberoPi05Adapter.from_policy(_StubPolicy(with_feat=False), prompt="x", action_dim=7,
                                      device="cpu", execute_horizon=5)
    a._infer_chunk({"observation/image": np.zeros((224, 224, 3), np.uint8)})
    assert a.last_prefix_feat() is None                       # serve 没透特征→None,不崩
```

- [ ] **Step 2: 跑测验证失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_adapter_prefix_feat.py -v`
Expected: FAIL(`AttributeError: ... 'last_prefix_feat'`)。

- [ ] **Step 3: 改 `libero_pi05_adapter.py`**

`__init__`(约 :22-30)末尾加:
```python
        self._last_prefix_feat = None
```
`_infer_chunk`(约 :42-48)里 `result = self.policy.infer(obs)` 之后加一行存特征:
```python
        result = self.policy.infer(obs)
        self._last_prefix_feat = result.get("prefix_feat")   # 方案A:复用 base infer 的 prefix_feat
```
(保留原有 `result["actions"]` 取法不变。)加方法:
```python
    def last_prefix_feat(self):
        """最近一次 serve infer 返回的 prefix_feat(queue 边界更新);serve 未透特征则 None。"""
        return self._last_prefix_feat
```

- [ ] **Step 4: 跑测验证通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_adapter_prefix_feat.py -v`
Expected: PASS(2 项)。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/libero_pi05_adapter.py resfit/rl_finetuning/chunk_residual/tests/test_adapter_prefix_feat.py && \
git commit -m "feat(libero_adapter): expose last_prefix_feat (reuse base serve infer for pi0_feat online)"
```

---

## Task 2: `save/load_high_actor` 加 pi0_feat_signature

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`(`save_high_actor`/`load_high_actor`,约 :127-167)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py`(本 task 先建,放 save/load 往返测)

- [ ] **Step 1: 先读现有 `save_high_actor`/`load_high_actor`** 看 act_feat_signature 怎么存/读(对称加 pi0_feat_signature)。

- [ ] **Step 2: 写失败测试**

```python
import numpy as np
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
    HighActor, save_high_actor, load_high_actor,
)


def test_high_actor_pi0_feat_signature_roundtrip(tmp_path):
    ha = HighActor(state_dim=2056, rep_dim=10, hidden=64)
    pt = str(tmp_path / "ha.pt")
    sig = {"serve_ckpt_id": "pi0_libero", "pooling": "last", "proprio_key": "observation.state"}
    save_high_actor(pt, ha, gc_value_ckpt="gc.pt", way_steps=5, beta=1.0,
                    state_mode="pi0_feat", mean=np.zeros(2056, np.float32),
                    std=np.ones(2056, np.float32), pi0_feat_signature=sig)
    model, info = load_high_actor(pt)
    assert info["state_mode"] == "pi0_feat"
    assert info["pi0_feat_signature"]["serve_ckpt_id"] == "pi0_libero"
    assert np.asarray(info["mean"]).shape == (2056,)
```

(注:`HighActor.__init__` 与 `save_high_actor` 的真实参数名以现有代码为准;若 `save_high_actor` 现有签名不同[如 mean/std 叫别的],按现有对称加 `pi0_feat_signature`,并相应改测试参数名。)

- [ ] **Step 3: 跑测失败 → 改 `hiql_high_actor.py`**

`save_high_actor` 签名末尾加 `pi0_feat_signature=None`;payload 里 `if pi0_feat_signature is not None: payload["pi0_feat_signature"] = pi0_feat_signature`(类比现有 act_feat_signature 写法)。`load_high_actor` 的 info 加 `info["pi0_feat_signature"] = ckpt.get("pi0_feat_signature")`。

- [ ] **Step 4: 跑测通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py::test_high_actor_pi0_feat_signature_roundtrip -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py && \
git commit -m "feat(high_actor): save/load pi0_feat_signature (round-trip)"
```

---

## Task 3: `train_hiql_high_actor` 的 pi0_feat dispatch

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py`(parser + main dispatch,约 :60-96)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_high_actor_cli.py`

- [ ] **Step 1: 先读** `train_hiql_high_actor.py` 的 build_parser + main 的 act_feat dispatch(对称加 pi0_feat) + `train_hiql_gc_value.py` 的 pi0_feat dispatch(:123-177,照搬逻辑)。

- [ ] **Step 2: 写失败测试**

```python
import pytest
from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser
from resfit.rl_finetuning.chunk_residual.train_hiql_value import validate_pi0_feat_cfg


def test_high_actor_parser_accepts_pi0_feat():
    a = build_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--gc_value_ckpt", "gc.pt",
                                   "--state_mode", "pi0_feat", "--pi0_feat_cache", "c.npz",
                                   "--pi0_serve_ckpt_id", "pi0_libero", "--pi0_image_keys", "a",
                                   "--pi0_proprio_key", "observation.state"])
    assert a.state_mode == "pi0_feat"


def test_high_actor_parser_default_state_mode_unchanged():
    a = build_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--gc_value_ckpt", "gc.pt"])
    assert a.state_mode == "eef_piece"
```

- [ ] **Step 3: 跑测失败 → 改 `train_hiql_high_actor.py`**

- parser:`--state_mode` choices 加 `"pi0_feat"`;`from ...train_hiql_value import add_pi0_feat_args, validate_pi0_feat_cfg` 后 `add_pi0_feat_args(p)`。
- main 开头 `validate_pi0_feat_cfg(args)`(若 state_mode!=pi0_feat 则 noop)。
- main 取 seqs 的 dispatch 加 pi0_feat 分支(照 train_hiql_gc_value:123-177):
  ```python
      if args.state_mode == "pi0_feat":
          from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
          _, _, _cache_sig = load_pi0_feat_cache(args.pi0_feat_cache)
          for k, v in (("serve_ckpt_id", args.pi0_serve_ckpt_id), ("image_keys", args.pi0_image_keys),
                       ("proprio_key", args.pi0_proprio_key), ("pooling", args.pi0_pooling),
                       ("prompt", args.pi0_prompt)):
              if _cache_sig.get(k) != v:
                  raise ValueError(f"缓存签名 {k}={_cache_sig.get(k)!r} 与 CLI {v!r} 不符")
          seqs, _, aux_stats = read_per_demo_states(
              args.hdf5, args.dataset, "pi0_feat", num_demos=args.num_demos,
              pi0_feat_cache=args.pi0_feat_cache, pi0_feat_signature=_cache_sig)
          feat_mean, feat_std = aux_stats
          pi0_sig = _cache_sig
      else:
          # eef_piece/act_feat 原样
          ...
  ```
- `stage_entries`:pi0_feat 时用 `stage_entries_aligned(args.hdf5, stage_cache=None, num_demos=..., seq_lens=...)`(stage_cache=None 不读 hdf5,见 :49-50);effective_num_demos 取 `_cache_sig.get("num_demos")`(自洽,同 gc_value)。
- save:`save_high_actor(..., state_mode="pi0_feat", mean=feat_mean, std=feat_std, pi0_feat_signature=pi0_sig)`(eef/act_feat 分支不动)。

- [ ] **Step 4: 跑测通过 + 回归**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_high_actor_cli.py resfit/rl_finetuning/chunk_residual/tests/ -k "high_actor or hiql" -v`
Expected: 新 2 项 PASS;既有 high_actor 用例全绿。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_high_actor_cli.py && \
git commit -m "feat(high_actor train): --state_mode pi0_feat dispatch (mirror gc_value, stage_cache=None no hdf5)"
```

---

## Task 4: `hiql_subgoal` 的 pi0_feat 分支

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py`(`__init__`:28-58、`from_ckpts`:60-79、`subgoal_online`:92-106)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py`(追加)

- [ ] **Step 1: 写失败测试(stub high_actor + feat_stats,验证 subgoal_online 出 z)**

```python
import numpy as np
import torch
from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal


class _StubHA(torch.nn.Module):
    rep_dim = 10
    state_dim = 2056
    def forward(self, s, g):
        b = s.shape[0]
        class _D:  # 仿 .mean 接口
            mean = torch.zeros(b, 10)
        return _D()


def test_subgoal_online_pi0_feat_outputs_z_standardized():
    ha = _StubHA()
    goal = np.zeros(2056, np.float32)
    feat_mean = np.zeros(2056, np.float32); feat_std = np.ones(2056, np.float32)
    sg = HiqlSubgoal(gc_value=None, high_actor=ha, goal=goal, device="cpu",
                     state_mode="pi0_feat", feat_stats=(feat_mean, feat_std))
    obs = {"observation.state": np.ones(8, np.float32)}
    z = sg.subgoal_online(obs, prefix_feat=np.ones(2048, np.float32))
    assert z.shape[-1] == 10                                  # rep_dim
    assert torch.isfinite(z).all()


def test_subgoal_online_pi0_feat_needs_prefix_feat():
    ha = _StubHA()
    sg = HiqlSubgoal(gc_value=None, high_actor=ha, goal=np.zeros(2056, np.float32), device="cpu",
                     state_mode="pi0_feat", feat_stats=(np.zeros(2056, np.float32), np.ones(2056, np.float32)))
    import pytest
    with pytest.raises(AssertionError):
        sg.subgoal_online({"observation.state": np.ones(8, np.float32)}, prefix_feat=None)
```

(注:`HiqlSubgoal.__init__` 现有签名 `(gc_value, high_actor, goal, device, renorm_subgoal, *, state_mode, rel_stats, extractor, feat_stats)`;gc_value=None 时 `self.rep_dim = high_actor.rep_dim`[见 :40-42]。)

- [ ] **Step 2: 跑测失败 → 改 `hiql_subgoal.py`**

`__init__` 加 pi0_feat 分支(在 act_feat elif 后):
```python
        elif state_mode == "pi0_feat":
            assert feat_stats is not None, "pi0_feat 须给 feat_stats(缓存的 2056 mean/std)"
            self.feat_mean = torch.as_tensor(feat_stats[0], dtype=torch.float32, device=device)
            self.feat_std = torch.as_tensor(feat_stats[1], dtype=torch.float32, device=device)
```
(注意把原 `else: raise ValueError` 留在最后。)

`from_ckpts` 的 `assert sm in ("eef_piece", "act_feat")` 改成加 `"pi0_feat"`;在 act_feat 分支后加:
```python
        if sm == "pi0_feat":
            return cls(gc, ha, goal, device=device, renorm_subgoal=renorm_subgoal,
                       state_mode="pi0_feat", feat_stats=(info["mean"], info["std"]))
```
(pi0_feat 不需 base_policy/extractor。)

`subgoal_online(obs, rel_raw=None, prefix_feat=None)` 加 pi0_feat 分支(在 eef_piece/act_feat 之外):
```python
        if self.state_mode == "pi0_feat":
            assert prefix_feat is not None, "pi0_feat 在线需 prefix_feat(从 base policy last_prefix_feat 取)"
            proprio = torch.as_tensor(np.asarray(obs["observation.state"]), dtype=torch.float32, device=self.device)
            pf = torch.as_tensor(np.asarray(prefix_feat), dtype=torch.float32, device=self.device)
            if pf.ndim == 1: pf = pf.unsqueeze(0)
            if proprio.ndim == 1: proprio = proprio.unsqueeze(0)
            feat = torch.cat([pf, proprio], dim=-1)            # [B,2056] raw(prefix 在前)
            s = (feat - self.feat_mean) / self.feat_std
        elif self.state_mode == "eef_piece":
            ...   # 原样
        else:    # act_feat 原样
            ...
        g = self.goal.unsqueeze(0).expand(s.shape[0], -1)
        z = self.ha(s, g).mean
        if self.renorm_subgoal:
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return z
```

- [ ] **Step 3: 跑测通过 + 回归**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py resfit/rl_finetuning/chunk_residual/tests/ -k "subgoal" -v`
Expected: 新测 PASS;既有 subgoal(act_feat/eef_piece)全绿。

- [ ] **Step 4: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/hiql_subgoal.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py && \
git commit -m "feat(hiql_subgoal): pi0_feat branch (from_ckpts + subgoal_online with prefix_feat, no extractor)"
```

---

## Task 5: `train_chunk_residual` 接 pi0_feat 子目标注入

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(subgoal 构造 ~:700-742 + 注入 ~:872)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_pi0_feat_subgoal.py`

- [ ] **Step 1: 先读** train_chunk_residual.py 的 `--subgoal_conditioned` 构造(from_ckpts 调用)+ :872 每步注入处的 act_feat 写法。

- [ ] **Step 2: 写失败测试(纯逻辑:抽一个 helper 函数测注入选择)**

把"每步算 subgoal"抽成可测 helper(若现有是内联,新建小函数):
```python
# 在 train_chunk_residual.py 加(或测现有注入逻辑):
def compute_online_subgoal(subgoal, obs, base_policy, cur_rel=None):
    """按 state_mode 选特征来源出 z:pi0_feat 取 base prefix_feat,否则走 obs/rel。"""
    if subgoal.state_mode == "pi0_feat":
        return subgoal.subgoal_online(obs, prefix_feat=base_policy.last_prefix_feat())
    return subgoal.subgoal_online(obs, cur_rel)
```
测试:
```python
import numpy as np, torch
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import compute_online_subgoal


class _SG:
    state_mode = "pi0_feat"
    def subgoal_online(self, obs, rel_raw=None, prefix_feat=None):
        assert prefix_feat is not None
        return torch.zeros(1, 10)


class _Base:
    def last_prefix_feat(self): return np.ones(2048, np.float32)


def test_compute_online_subgoal_pi0_feat_pulls_base_prefix_feat():
    z = compute_online_subgoal(_SG(), {"observation.state": np.ones(8, np.float32)}, _Base())
    assert z.shape[-1] == 10


def test_compute_online_subgoal_act_feat_uses_obs_rel():
    class _SG2:
        state_mode = "act_feat"
        def subgoal_online(self, obs, rel_raw=None, prefix_feat=None):
            assert prefix_feat is None        # act_feat 不走 prefix_feat
            return torch.zeros(1, 10)
    z = compute_online_subgoal(_SG2(), {"x": 1}, _Base(), cur_rel=None)
    assert z.shape[-1] == 10
```

- [ ] **Step 3: 跑测失败 → 在 train_chunk_residual.py 加 `compute_online_subgoal` helper**(代码见上),并把 :872 那处注入改成调它:
```python
      if args.subgoal_conditioned:
          obs["observation.subgoal"] = compute_online_subgoal(subgoal, obs, base_policy, cur_rel).to(device)
```
(eval 路若也注入,同样改。)subgoal 构造处:pi0_feat 时 `from_ckpts(..., base_policy=None)` + goal 从 pi0_feat 缓存 seqs medoid(读现有 act_feat 构造对称加;pi0_feat 时 base_policy 传 None)。

- [ ] **Step 4: 跑测通过 + 回归(import 自检 train_chunk_residual)**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_pi0_feat_subgoal.py -v && conda run -n residual python -c "import resfit.rl_finetuning.chunk_residual.train_chunk_residual"`
Expected: 2 PASS;import 无异常。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_train_chunk_residual_pi0_feat_subgoal.py && \
git commit -m "feat(train_chunk_residual): online pi0_feat subgoal (pull base last_prefix_feat -> high_actor -> z)"
```

---

## Task 6: 全量回归 + opt-in 在线 smoke

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_online_subgoal_smoke.py`(默认 skip)

- [ ] **Step 1: 写 opt-in smoke(env gate,真 serve + 真 LIBERO env)**

```python
import os
import numpy as np
import pytest

RUN = os.environ.get("PI0_FEAT_HIER_SMOKE") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="opt-in:需真 serve+LIBERO env+预训 gc/high_actor,设 PI0_FEAT_HIER_SMOKE=1")


def test_online_subgoal_z_injected_and_finite():
    """前置(手动): train_hiql_gc_value/high_actor --state_mode pi0_feat 出 gc.pt/ha.pt;serve 起着。
    环境变量:PI0_GC_PT, PI0_HA_PT, PI0_SERVE_HOST/PORT, PI0_FEAT_CACHE(算 goal medoid 的全量缓存)。
    验证:HiqlSubgoal.from_ckpts 出 z 维=10、有限;并打印在线 obs.state 维度(命门②对齐)。"""
    from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal, representative_goal
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    seqs, _, _ = load_pi0_feat_cache(os.environ["PI0_FEAT_CACHE"])
    goal = representative_goal(seqs)
    sg = HiqlSubgoal.from_ckpts(os.environ["PI0_GC_PT"], os.environ["PI0_HA_PT"],
                                goal=goal, device="cpu")
    obs = {"observation.state": np.zeros(seqs[0].shape[1] - 2048, np.float32)}  # proprio 维=2056-2048
    z = sg.subgoal_online(obs, prefix_feat=np.ones(2048, np.float32))
    assert z.shape[-1] == sg.rep_dim and np.isfinite(z.detach().numpy()).all()
    print(f"[smoke] z dim={z.shape[-1]} proprio dim={obs['observation.state'].shape}")
```

- [ ] **Step 2: 跑(默认 skip)**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_online_subgoal_smoke.py -v`
Expected: 1 SKIPPED。手动 opt-in 跑见 docstring;真在线 rollout(残差几步不塌)由用户起 serve+env 手动验。

- [ ] **Step 3: 全量回归(确认 act_feat/eef_piece 零影响)**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全绿(既有 + 本计划新增)。**任何既有变红 → 停下修到逐位等价**。

- [ ] **Step 4: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_online_subgoal_smoke.py && \
git commit -m "test: opt-in pi0_feat online subgoal smoke + full regression green"
```

---

## Self-Review(写完计划自查)

**Spec 覆盖**:§3.1 adapter→Task1;§3.4 high_actor 训练→Task2(save/load)+Task3(dispatch);§3.2 hiql_subgoal→Task4;§3.3 train_chunk_residual→Task5;§4 测试(单测+opt-in smoke+回归)→各 Task Step + Task6;§2 命门(prefix_feat 同源/proprio raw/feat_stats/goal medoid)→Task4 subgoal_online + Task5 注入 + smoke 实测②。全覆盖。

**占位符扫描**:无 TBD/TODO。几处"以现有 act_feat/save_high_actor 签名为准对称加"是**真实模式复用**(给了 file:line + pi0_feat 分支完整代码),非占位;实施时先读对称点再加。

**类型/命名一致**:`last_prefix_feat()`、`pi0_feat_signature`、`subgoal_online(obs, rel_raw=None, prefix_feat=None)`、`compute_online_subgoal(subgoal, obs, base_policy, cur_rel)`、`HiqlSubgoal(state_mode="pi0_feat", feat_stats=)`、`feat_mean/feat_std`(=aux_stats) 跨 Task 一致。

**stage/hdf5 已核实**:stage_cache=None(:49-50)不读 hdf5,LIBERO pi0_feat 训练(Task3)兼容,--hdf5 传 dummy 即可。
