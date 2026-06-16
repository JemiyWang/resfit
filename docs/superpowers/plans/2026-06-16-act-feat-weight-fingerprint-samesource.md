# act_feat 权重指纹同源校验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 act_feat 路加「ACT 权重内容指纹」同源校验,确保离线 build cache/训 gc_value 与在线推理用的是同一份 ACT 权重,不一致默认硬失败(可逃生),且不破坏任何现有产物。

**Architecture:** 新增纯函数 `act_weight_fingerprint`(全量 state_dict 的确定性 sha256)+ 纯校验函数 `assert_act_base_samesource`(三分支:旧产物 warn/skip、同源静默、异源 raise/逃生)。指纹**旁挂**存进 act_feat cache npz 和 gc_value ckpt(不动 `act_feat_signature`,旧缓存零失效)。离线在 build cache / 训 gc_value 时算并写入;在线对 base_policy 重算后调校验函数,替换掉原先不可靠的 `base_wandb_id` vs `act_ckpt_id` 字符串比较。

**Tech Stack:** Python, PyTorch, numpy, pytest。仓库根 `/mnt/mnt/data/resfit`,包根 `resfit/`,conda 环境 `residual`。从仓库根跑测试:`conda run -n residual python -m pytest <path> -v`。

**Spec:** `docs/superpowers/specs/2026-06-16-act-feat-weight-fingerprint-samesource-design.md`

---

## File Structure

| 文件 | 职责 | 改动 |
|---|---|---|
| `resfit/rl_finetuning/chunk_residual/act_feature.py` | ACT 特征器 + 纯函数 | **加** `act_weight_fingerprint`、`assert_act_base_samesource`、`ActFeatureExtractor.weight_fingerprint` |
| `resfit/rl_finetuning/chunk_residual/act_feat_cache.py` | cache 存取 | sha 旁挂字段;`_load`/`load_act_feat_cache` 返回 4 元组 |
| `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py` | gc_value 存取 | `save_gc_value`/`load_gc_value` 加 `act_weight_sha` |
| `resfit/rl_finetuning/chunk_residual/train_hiql_value.py` | 离线读特征 + setup | `setup_act_feat` 返回 5 元组;build 时把 sha 写进 cache |
| `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py` | 训 gc_value | 解包 5 元组 + 把 sha 传进 `save_gc_value` |
| `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py` | 训 high_actor | 解包 5 元组(吸收第 5 项) |
| `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` | 在线残差训练 | 加 `--allow_act_base_mismatch`;731 行 4 解包;替换 739-747 校验块 |
| `resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py` | 现有测试 | `setup_act_feat` 解包改 5 元组 |
| `tests/test_act_weight_fingerprint.py`(新) | 单测 | 指纹 + 校验函数 |

---

## Task 1: 权重指纹纯函数 + extractor 方法

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/act_feature.py`(在文件尾部追加 `act_weight_fingerprint`;在 `ActFeatureExtractor` 内加方法)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py`(新建)

- [ ] **Step 1: 写失败测试**

新建 `resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py`:

```python
import torch
from torch import nn

from resfit.rl_finetuning.chunk_residual.act_feature import act_weight_fingerprint


def _tiny():
    torch.manual_seed(0)
    m = nn.Linear(4, 3)
    m.register_buffer("ctr", torch.tensor([7], dtype=torch.int64))  # 整型缓冲
    return m


def test_fingerprint_deterministic():
    m = _tiny()
    assert act_weight_fingerprint(m) == act_weight_fingerprint(m)


def test_fingerprint_sensitive_to_weight_change():
    m = _tiny()
    h0 = act_weight_fingerprint(m)
    with torch.no_grad():
        m.weight.add_(1e-3)
    assert act_weight_fingerprint(m) != h0


def test_fingerprint_handles_int_buffer():
    # 含 int64 buffer 不崩,且改 buffer 值 → hash 变(身份敏感)
    m = _tiny()
    h0 = act_weight_fingerprint(m)
    m.ctr.fill_(9)
    assert act_weight_fingerprint(m) != h0


def test_fingerprint_is_hexdigest_str():
    h = act_weight_fingerprint(_tiny())
    assert isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py -v`
Expected: FAIL —— `ImportError: cannot import name 'act_weight_fingerprint'`

- [ ] **Step 3: 写实现**

在 `act_feature.py` 顶部 import 区加 `import hashlib`(`import torch` 已有)。文件尾部追加:

```python
def act_weight_fingerprint(policy) -> str:
    """对 policy.state_dict() 的确定性 sha256 指纹(hexdigest)。

    确定性:按 key 排序;浮点张量先 detach().cpu().float().contiguous() 去掉
    设备/内存布局/当前精度差异;整型/bool 缓冲按原 dtype 取字节。key/dtype/shape/bytes
    全部进哈希。保证:同一份内存权重 → 同一 hash;改任一权重 → hash 变。
    不保证 fp16 存档 vs fp32 存档相等(那本就是两份不同的值)。
    """
    h = hashlib.sha256()
    sd = policy.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k]
        if not torch.is_tensor(t):
            continue
        t = t.detach().cpu()
        if t.is_floating_point():
            t = t.float()
        t = t.contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(repr(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()
```

在 `ActFeatureExtractor` 类内(`signature` 方法旁)加:

```python
    def weight_fingerprint(self) -> str:
        """对冻结的 self.act 算权重指纹(同源校验用)。"""
        return act_weight_fingerprint(self.act)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/act_feature.py resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py
git commit -m "feat(act_feat): act_weight_fingerprint + extractor.weight_fingerprint

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: 同源校验纯函数 `assert_act_base_samesource`

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/act_feature.py`(追加纯函数)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py`(同 Task 1 文件,追加)

- [ ] **Step 1: 写失败测试**(追加到 `test_act_weight_fingerprint.py` 末尾)

```python
import warnings
import pytest

from resfit.rl_finetuning.chunk_residual.act_feature import assert_act_base_samesource


def test_samesource_old_artifact_warns_and_skips():
    # gv_sha=None(旧产物)→ warn,不 raise
    with pytest.warns(UserWarning):
        assert_act_base_samesource(gv_sha=None, cache_sha=None, base_sha="x")


def test_samesource_match_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # 任何 warning 都会失败
        assert_act_base_samesource(gv_sha="a", cache_sha="a", base_sha="a")  # 不抛即通过


def test_samesource_offline_inconsistent_raises():
    # cache 与 gc_value 两边都有指纹且不等 → 离线异源
    with pytest.raises(ValueError):
        assert_act_base_samesource(gv_sha="a", cache_sha="b", base_sha="a")


def test_samesource_online_mismatch_raises_by_default():
    with pytest.raises(ValueError):
        assert_act_base_samesource(gv_sha="a", cache_sha="a", base_sha="b")


def test_samesource_online_mismatch_warns_with_escape():
    with pytest.warns(UserWarning):
        assert_act_base_samesource(gv_sha="a", cache_sha="a", base_sha="b", allow_mismatch=True)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py -v -k samesource`
Expected: FAIL —— `ImportError: cannot import name 'assert_act_base_samesource'`

- [ ] **Step 3: 写实现**(追加到 `act_feature.py`,`act_weight_fingerprint` 之后)

```python
def assert_act_base_samesource(*, gv_sha, cache_sha, base_sha, allow_mismatch=False):
    """act_feat 在线 base ACT 同源校验(纯逻辑,便于单测)。

    - cache_sha 与 gv_sha 两边都有且不等 → 离线就异源(cache≠gc_value),raise。
    - gv_sha is None(旧 cache/gc_value 无指纹)→ 无法校验,warn 并跳过。
    - base_sha == gv_sha → 同源,静默返回。
    - base_sha != gv_sha → 默认 raise;allow_mismatch=True 则降级 warn。
    """
    import warnings
    if cache_sha and gv_sha and cache_sha != gv_sha:
        raise ValueError(
            f"[act_feat] cache 与 gc_value 权重指纹不符: {cache_sha} vs {gv_sha}(离线就异源)")
    if gv_sha is None:
        warnings.warn(
            "[act_feat] 产物无权重指纹(旧 cache/gc_value),无法校验在线 base 同源;"
            "务必先过一致性 smoke", stacklevel=2)
        return
    if base_sha == gv_sha:
        return
    if allow_mismatch:
        warnings.warn(
            f"[act_feat] 在线 base 权重指纹 != 离线({base_sha} vs {gv_sha}),"
            "--allow_act_base_mismatch 已放行", stacklevel=2)
        return
    raise ValueError(
        f"[act_feat] 在线 base_policy 权重 != 离线 build cache 的 ACT"
        f"(指纹 {base_sha} vs {gv_sha});确认同源,或加 --allow_act_base_mismatch 放行")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py -v`
Expected: PASS(9 passed —— Task1 的 4 + Task2 的 5)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/act_feature.py resfit/rl_finetuning/chunk_residual/tests/test_act_weight_fingerprint.py
git commit -m "feat(act_feat): assert_act_base_samesource pure check (三分支)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: act_feat cache 旁挂 sha 字段

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/act_feat_cache.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cache.py`(追加)

- [ ] **Step 1: 写失败测试**(追加到 `test_act_feat_cache.py` 末尾)

```python
from resfit.rl_finetuning.chunk_residual.act_feat_cache import load_act_feat_cache


def test_load_returns_sha_when_present(tmp_path):
    p = str(tmp_path / "c.npz")
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)),
                        signature=SIG, act_weight_sha="deadbeef")
    seqs, stats, sig, sha = load_act_feat_cache(p)
    assert sha == "deadbeef"
    assert sig == SIG and len(seqs) == 1


def test_load_returns_none_sha_when_absent(tmp_path):
    # 旧缓存(不带 sha)→ 第 4 项 None,且 reuse 仍照常命中(向后兼容)
    p = str(tmp_path / "c.npz")
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=SIG)
    seqs, stats, sig, sha = load_act_feat_cache(p)
    assert sha is None
    assert act_feat_cache_reuse(p, signature=SIG, num_demos=None) is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cache.py -v -k sha`
Expected: FAIL —— `save_act_feat_cache() got an unexpected keyword argument 'act_weight_sha'`

- [ ] **Step 3: 写实现**

改 `act_feat_cache.py`:

`save_act_feat_cache` 加参数并存字段:

```python
def save_act_feat_cache(path, seqs, emb_stats, *, signature, fp16=False, act_weight_sha=None):
    """存每条 demo 的(已标准化)嵌入序列 + (mean,std) + 签名 json + 可选权重指纹。
    fp16 仅压 seqs 存盘。act_weight_sha 旁挂,不进 signature(不影响 reuse 命中)。"""
    mean, std = emb_stats
    payload = {
        "n": np.int64(len(seqs)),
        "emb_mean": np.asarray(mean, dtype=np.float32),
        "emb_std": np.asarray(std, dtype=np.float32),
        "signature": np.asarray(json.dumps(_norm_sig(signature), sort_keys=True)),
    }
    if act_weight_sha is not None:
        payload["act_weight_sha"] = np.asarray(str(act_weight_sha))
    dt = np.float16 if fp16 else np.float32
    for i, s in enumerate(seqs):
        payload[f"s{i}"] = np.asarray(s, dtype=dt)
    np.savez_compressed(path, **payload)
```

`_load` 多返回 sha(容缺):

```python
def _load(path):
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        seqs = [np.asarray(z[f"s{i}"], dtype=np.float32) for i in range(n)]
        stats = (np.asarray(z["emb_mean"], np.float32), np.asarray(z["emb_std"], np.float32))
        sig = json.loads(str(z["signature"]))
        sha = str(z["act_weight_sha"]) if "act_weight_sha" in z.files else None
    return seqs, stats, sig, sha
```

`load_act_feat_cache`(返回 4 元组):

```python
def load_act_feat_cache(path):
    """读缓存 → (seqs[float32], (mean,std), signature_dict, act_weight_sha_or_None)。无签名校验,仅读取。"""
    return _load(path)
```

`act_feat_cache_reuse`(内部 4 解包,返回仍是 2 元组,比较逻辑不动):

```python
def act_feat_cache_reuse(path, *, signature, num_demos):
    """签名全等(含 num_demos)才命中,否则 None。sha 不参与命中比较。"""
    if not (path and os.path.exists(path)):
        return None
    seqs, stats, sig, _sha = _load(path)
    want = _norm_sig(signature)
    want["num_demos"] = num_demos
    if sig != want:
        return None
    return seqs, stats
```

- [ ] **Step 4: 跑测试确认通过(含原有回归)**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cache.py -v`
Expected: PASS(原有 5 + 新 2 = 7 passed)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/act_feat_cache.py resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cache.py
git commit -m "feat(act_feat): cache 旁挂 act_weight_sha + load 返回 4 元组(向后兼容)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: gc_value ckpt 旁挂 sha

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py:269-321`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_gc_value_act_weight_sha.py`(新建)

- [ ] **Step 1: 写失败测试**

新建 `resfit/rl_finetuning/chunk_residual/tests/test_gc_value_act_weight_sha.py`:

```python
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
    GoalConditionedVF, save_gc_value, load_gc_value,
)


def _model():
    return GoalConditionedVF(state_dim=6, rep_dim=4, hidden=8)


def test_save_load_act_weight_sha(tmp_path):
    p = str(tmp_path / "gc.pt")
    save_gc_value(p, _model(), v_stats={}, mean=torch.zeros(6), std=torch.ones(6),
                  dataset_id="ds", state_mode="act_feat",
                  act_feat_signature={"act_ckpt_id": "A"}, act_weight_sha="cafef00d")
    _, info = load_gc_value(p)
    assert info["act_weight_sha"] == "cafef00d"


def test_load_old_ckpt_sha_none(tmp_path):
    # 不传 act_weight_sha(旧档)→ load 后为 None
    p = str(tmp_path / "gc.pt")
    save_gc_value(p, _model(), v_stats={}, mean=torch.zeros(6), std=torch.ones(6),
                  dataset_id="ds", state_mode="eef_piece")
    _, info = load_gc_value(p)
    assert info["act_weight_sha"] is None
```

> 已核对构造签名:`GoalConditionedVF(state_dim, rep_dim=10, hidden=256, use_layer_norm=False, rep_mode="concat")`,上面 `GoalConditionedVF(state_dim=6, rep_dim=4, hidden=8)` 可直接用。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_gc_value_act_weight_sha.py -v`
Expected: FAIL —— `save_gc_value() got an unexpected keyword argument 'act_weight_sha'`

- [ ] **Step 3: 写实现**

`save_gc_value`(`hiql_gc_value.py:269-272`)签名加 `act_weight_sha=None`:

```python
def save_gc_value(path, model, *, v_stats, mean, std, dataset_id,
                  state_mode="eef_piece", rel_piece_stats=None,
                  value_loss_mode="shared_min", value_mask_mode="done_aware",
                  act_feat_signature=None, pi0_feat_signature=None, act_weight_sha=None):
```

在 `if pi0_feat_signature is not None:` 块(298-299)之后、`torch.save` 之前加:

```python
    if act_weight_sha is not None:
        payload["act_weight_sha"] = act_weight_sha
```

`load_gc_value` 在 `info["act_feat_signature"] = ...`(319)之后加:

```python
    info["act_weight_sha"] = ckpt.get("act_weight_sha")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_gc_value_act_weight_sha.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_gc_value_act_weight_sha.py
git commit -m "feat(act_feat): gc_value ckpt 旁挂 act_weight_sha(save/load,向后兼容)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: setup_act_feat 返回 sha + build 时写进 cache

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`(`setup_act_feat` 238-258、`read_per_demo_states` 107-108、`main` 333)
- Modify: `resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py:60`(解包改 5 元组)

- [ ] **Step 1: 改 `read_per_demo_states` 的 cache 写入(build 路写 sha)**

`train_hiql_value.py:107-108`,把 `save_act_feat_cache` 调用改为带 sha:

```python
        if act_feat_cache and num_demos is None:
            save_act_feat_cache(act_feat_cache, seqs_std, (mean, std), signature=sig,
                                act_weight_sha=act_extractor.weight_fingerprint())
            print(f"[read_per_demo_states] 已写 act_feat 缓存 {act_feat_cache}")
```

- [ ] **Step 2: 改 `setup_act_feat` 返回 5 元组**

`train_hiql_value.py:238-258`,三处 return 各加第 5 项 sha:

- 非 act_feat 早返(244):`return None, None, None, None, None`
- cache_ready 分支(246-251):

```python
    if cache_ready:
        from resfit.rl_finetuning.chunk_residual.act_feat_cache import load_act_feat_cache
        _, _, sig, sha = load_act_feat_cache(args.act_feat_cache)
        ckpt = str(args.act_base_ckpt) if args.act_base_ckpt else sig.get("act_ckpt_id")
        image_keys = args.act_image_keys if args.act_image_keys is not None else sig.get("image_keys")
        return None, ckpt, image_keys, sig, sha
```

- build 分支(258):

```python
    return ext, str(args.act_base_ckpt), image_keys, ext.signature(str(args.act_base_ckpt)), ext.weight_fingerprint()
```

同时更新 docstring 第一行为「返回 (extractor, act_ckpt_id, image_keys, signature_or_None, act_weight_sha_or_None)」。

- [ ] **Step 3: 改本文件 `main` 的解包(333)**

```python
    extractor, act_ckpt_id, image_keys, _, _ = setup_act_feat(args)
```

- [ ] **Step 4: 改现有测试解包**

`tests/test_act_feat_cli_wiring.py:60`:

```python
    ext, ckpt, image_keys, got_sig, _got_sha = setup_act_feat(args)
```

- [ ] **Step 5: 跑现有 act_feat 测试确认不回归**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py -v`
Expected: PASS(全绿;若某用例本就因缺真 ACT 跳过,保持 skip 不变)

- [ ] **Step 6: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py
git commit -m "feat(act_feat): setup_act_feat 返回 act_weight_sha + build 时写进 cache

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: gc_value/high_actor 训练器解包 5 元组 + 透传 sha

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py:144, 175-179`
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py:104`

- [ ] **Step 1: 改 train_hiql_gc_value 解包 + 透传**

`train_hiql_gc_value.py:144`:

```python
        extractor, act_ckpt_id, image_keys, act_sig, act_sha = setup_act_feat(args)
```

> 注:pi0_feat 分支(`else:` 之前)不调 setup_act_feat,`act_sha` 不在其作用域。在该分支里 `act_sig = None` 那一行**紧跟一行** `act_sha = None`(两支都定义 `act_sha`,否则 save_gc_value 传参 NameError)。

`train_hiql_gc_value.py:175-179`,`save_gc_value(...)` 末尾加 `act_weight_sha=act_sha`:

```python
    save_gc_value(args.output, model, v_stats=v_stats,
                  mean=mean, std=std,
                  dataset_id=args.dataset, state_mode=args.state_mode, rel_piece_stats=rel_stats,
                  value_loss_mode=args.value_loss_mode, value_mask_mode=args.value_mask_mode,
                  act_feat_signature=act_sig, pi0_feat_signature=pi0_sig, act_weight_sha=act_sha)
```

- [ ] **Step 2: 改 train_hiql_high_actor 解包(吸收第 5 项)**

`train_hiql_high_actor.py:104`:

```python
        extractor, act_ckpt_id, image_keys, act_sig, _act_sha = setup_act_feat(args)
```

> high_actor 不存 sha(在线只读 gc_value);此处仅吸收返回值避免解包报错。

- [ ] **Step 3: 跑相关测试 + 编译确认无解包错**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_high_actor_cli.py -v && conda run -n residual python -c "import ast; [ast.parse(open(f).read()) for f in ['resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py','resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py']]; print('parse-ok')"`
Expected: PASS + `parse-ok`

- [ ] **Step 4: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py
git commit -m "feat(act_feat): gc_value 训练透传 act_weight_sha;high_actor 吸收 5 元组

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: 在线校验接入 train_chunk_residual + 逃生 flag

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(argparse ~453;731 行解包;替换 739-747 校验块)

- [ ] **Step 1: 加 argparse flag**

在 `train_chunk_residual.py:453`(`--act_feat_cache` 那一行)之后加:

```python
    p.add_argument("--allow_act_base_mismatch", action="store_true",
                   help="act_feat:在线 base 权重指纹 != 离线时,把硬失败降级为 warning 放行")
```

- [ ] **Step 2: 改 731 行 cache 解包为 4 元组**

`train_chunk_residual.py:731`:

```python
            _seqs, _stats, _cache_sig, _cache_sha = load_act_feat_cache(args.act_feat_cache)
```

- [ ] **Step 3: 替换 739-747(不可靠的 id 比较)为指纹校验**

把 `train_chunk_residual.py:739-747` 整段(从 `import warnings` 那行到 `stacklevel=2)` 结束)替换为:

```python
            from resfit.rl_finetuning.chunk_residual.act_feature import (
                act_weight_fingerprint, assert_act_base_samesource)
            assert_act_base_samesource(
                gv_sha=_gc_info.get("act_weight_sha"),
                cache_sha=_cache_sha,
                base_sha=act_weight_fingerprint(base_policy),
                allow_mismatch=args.allow_act_base_mismatch)
```

> 保留上下文:735-738 行原有的 4 字段签名硬校验(`act_ckpt_id/image_keys/proprio_key/pooling`)**不动**,本步只换掉它后面那段 id 比较 warning。新块不再 import os(原注释的 UnboundLocalError 隐患随之消失)。

- [ ] **Step 4: 编译 + 冒烟 argparse**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -c "import ast; ast.parse(open('resfit/rl_finetuning/chunk_residual/train_chunk_residual.py').read()); print('parse-ok')" && conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual --help 2>&1 | grep -- "--allow_act_base_mismatch"`
Expected: `parse-ok` + 能 grep 到 `--allow_act_base_mismatch` 帮助行

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat(act_feat): 在线用权重指纹校验 base 同源(替换不可靠 id 比较)+ --allow_act_base_mismatch

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: 全量回归 + 端到端 smoke 说明

**Files:** 无代码改动(验证 + 文档)

- [ ] **Step 1: 跑 chunk_residual 全量单测**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests -v`
Expected: 全绿(在原基线 passed 数基础上 +新增用例;无新 fail/error;原有 skip 保持)

- [ ] **Step 2: 记录 smoke 步骤(交用户人工跑,不入 pytest)**

在本 plan 末尾「Smoke」一节已写明三场景;实现完成后提醒用户:用一个 act_feat `--subgoal_conditioned` 配置(同源 base)应静默通过;故意指向另一份 base ACT 应 raise;加 `--allow_act_base_mismatch` 应 warn 并继续。

- [ ] **Step 3: 无新增提交**(本任务仅验证;若 Step1 暴露需修复项,回到对应 Task 修)

---

## Smoke(人工,实现后由用户跑)

前提:已有一份 act_feat 的 cache + 同源 gc_value(用本改动后的代码新建,带 sha),以及对应的 base ACT。

1. **同源**:`train_chunk_residual ... --subgoal_conditioned --state_mode 走 act_feat gc_value --base_wandb_id <与离线同一份 ACT>` → 不应有指纹相关 warning/raise,正常进训练。
2. **异源(默认拦)**:把 `--base_wandb_id` 指向**另一份** ACT → 应 `ValueError: 在线 base_policy 权重 != 离线 build cache 的 ACT`。
3. **逃生**:在场景 2 基础上加 `--allow_act_base_mismatch` → 应只 warn 并继续训练。
4. **旧产物兼容**:用一份改动前建的旧 cache/gc_value(无 sha)→ 应 warn「产物无权重指纹…务必先过一致性 smoke」,不 raise。

---

## 命门 / 注意

- **cache 读取元数 3→4**:`_load`/`load_act_feat_cache` 全部解包点必须改齐(Task 3 改 `act_feat_cache_reuse` 内部;Task 5 改 `setup_act_feat`;Task 7 改 731)。`pi0_feat_cache.py` 有**独立的** `_load`,不受影响。grep 校验:`grep -rn "load_act_feat_cache(\|= _load(" resfit/rl_finetuning/chunk_residual`(排除 pi0)。
- **setup_act_feat 4→5**:4 个调用点全改(train_hiql_value/gc_value/high_actor + test_act_feat_cli_wiring),漏一个就 `ValueError: not enough values to unpack`。
- **指纹对象**:在线必须对 `base_policy`(真正喂进 extractor 的那个 ACT)算;离线 build 对 `act_extractor.act` 算(`weight_fingerprint()` 已封装)。
- **act_sha 作用域**(Task 6):pi0_feat 分支不经 setup_act_feat,务必在 if/else 之前 `act_sha = None` 初始化,否则 save_gc_value 传参时 NameError。
- **不动 `act_feat_signature`**:全程不往 4 字段签名里加 sha,否则旧缓存 reuse 全 miss → 违反向后兼容决策。
