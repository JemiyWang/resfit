# HIQL value pi0_feat via serve (C 方案) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 base policy 的 serve(openpi)加零侵入 FeaturePolicy wrapper,在 `infer` 返回里透出 pi05 prefix 池化特征;resfit 侧(residual 环境)用 `openpi_client` 调 serve 出特征缓存,以 `state_mode="pi0_feat"` 离线训出 `gc_value`,验证"pi05 图像特征当 HIQL value 的 state 有没有用"。

**Architecture:** 纯逻辑(池化/拼接/标准化/缓存/签名)与模型/serve 耦合层分离。openpi 侧 `FeaturePolicy` 用依赖注入持有原 `Policy`,真前向走可注入函数(测试塞 stub)。resfit 侧 cache-required:训练只读缓存、不连 serve、不 import openpi。默认 `eef`/`eef_piece`/`act_feat` 逐位不变。

**Tech Stack:** Python / JAX (openpi 侧,`.venv`) / PyTorch (resfit 侧,conda `residual`) / `openpi_client` websocket / h5py / numpy / pytest。

跑测命令:
- openpi 侧(openpi 环境): `cd /mnt/mnt/data/chj/openpi && .venv/bin/python -m pytest <path> -v`
- resfit 侧(residual 环境): `conda run -n residual python -m pytest <path> -v`

参照 spec: `docs/superpowers/specs/2026-06-14-hiql-value-pi0-feat-via-serve-design.md`。

---

## File Structure

**openpi 仓(新增 untracked,不改原始文件)**
- `src/openpi/serving/feature_policy.py` — `pool_prefix`(纯函数) + `make_prefix_feat_fn`(真前向,惰性 import) + `FeaturePolicy`(注入式 wrapper) + `wrap_with_feature`。
- `scripts/serve_policy_with_feat.py` — 复用 serve_policy 构造 policy → `wrap_with_feature` → 起 websocket server 的并列入口。
- `tests/test_feature_policy.py` — pool / FeaturePolicy / wrap 的单测(stub,不跑真权重)。

**resfit 仓**
- `resfit/rl_finetuning/chunk_residual/pi0_feat_cache.py` — `save/reuse/load` + 签名(镜像 `act_feat_cache.py`)。
- `resfit/rl_finetuning/chunk_residual/build_pi0_feat_cache_via_serve.py` — 纯函数 `assemble_pi0_feat_seqs` + `pi0_feat_signature` + `main`(stub client 可测)。
- `resfit/rl_finetuning/chunk_residual/train_hiql_value.py` — `read_per_demo_states` 加 `pi0_feat` 分支 + `validate_pi0_feat_cfg` + parser。
- `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py` — `--state_mode pi0_feat` + save/load 签名。
- tests: `test_pi0_feat_cache.py`、`test_build_pi0_feat_via_serve.py`、`test_read_per_demo_states_pi0_feat.py`、`test_pi0_feat_cli_wiring.py`、`test_pi0_feat_via_serve_smoke.py`(opt-in)。

---

## Task 1: `pool_prefix` 纯函数(openpi 侧)

**Files:**
- Create: `/mnt/mnt/data/chj/openpi/src/openpi/serving/feature_policy.py`
- Test: `/mnt/mnt/data/chj/openpi/tests/test_feature_policy.py`

- [ ] **Step 1: 写失败测试**

```python
import jax.numpy as jnp
import pytest
from openpi.serving.feature_policy import pool_prefix


def test_pool_last_picks_last_valid_token():
    out = jnp.array([[[1., 1], [2, 2], [9, 9]], [[3, 3], [4, 4], [5, 5]]])
    mask = jnp.array([[1, 1, 0], [1, 1, 1]])
    got = pool_prefix(out, mask, "last")
    assert jnp.allclose(got, jnp.array([[2., 2], [5, 5]]))   # 末有效 token,非 [:, -1]


def test_pool_mean_ignores_padding():
    out = jnp.array([[[1., 1], [3, 3], [9, 9]]])
    mask = jnp.array([[1, 1, 0]])
    got = pool_prefix(out, mask, "mean")
    assert jnp.allclose(got, jnp.array([[2., 2]]))           # (1+3)/2,padding 不计


def test_pool_unknown_raises():
    with pytest.raises(ValueError):
        pool_prefix(jnp.zeros((1, 1, 2)), jnp.ones((1, 1)), "bogus")
```

- [ ] **Step 2: 跑测验证失败**

Run: `cd /mnt/mnt/data/chj/openpi && .venv/bin/python -m pytest tests/test_feature_policy.py -v`
Expected: FAIL (`ModuleNotFoundError: openpi.serving.feature_policy`)。

- [ ] **Step 3: 写 `feature_policy.py` 的纯函数部分**

```python
"""零侵入 serve 特征器:在 Policy.infer 返回里透出 pi05 prefix 池化特征。

pool_prefix 纯函数(只依赖 jax)可单测;真前向 make_prefix_feat_fn 惰性 import pi0,
只在真 serve 进程触达。设计见 specs/2026-06-14-hiql-value-pi0-feat-via-serve-design.md。
"""
from __future__ import annotations

import jax.numpy as jnp


def pool_prefix(prefix_out, mask, pooling="last"):
    """prefix_out [B,L,D] + mask [B,L](bool/0-1) → [B,D]。

    last = 末**有效** token(按 mask.sum-1,非 [:, -1]);mean = 有效 token 均值(排除 padding)。
    """
    if pooling == "last":
        last_idx = mask.astype(jnp.int32).sum(axis=1) - 1
        return prefix_out[jnp.arange(prefix_out.shape[0]), last_idx]
    if pooling == "mean":
        m = mask.astype(prefix_out.dtype)[..., None]
        return (prefix_out * m).sum(axis=1) / jnp.clip(m.sum(axis=1), 1.0, None)
    raise ValueError(f"unknown pooling {pooling!r}")
```

- [ ] **Step 4: 跑测验证通过**

Run: `cd /mnt/mnt/data/chj/openpi && .venv/bin/python -m pytest tests/test_feature_policy.py -v`
Expected: PASS (3 项)。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/chj/openpi && git add src/openpi/serving/feature_policy.py tests/test_feature_policy.py && \
git commit -m "feat(serve): pool_prefix pure fn (pi05 prefix feature pooling, last/mean by mask)"
```

---

## Task 2: `FeaturePolicy` wrapper + 真前向(openpi 侧)

**Files:**
- Modify: `/mnt/mnt/data/chj/openpi/src/openpi/serving/feature_policy.py`
- Test: `/mnt/mnt/data/chj/openpi/tests/test_feature_policy.py`

- [ ] **Step 1: 追加失败测试(stub inner,注入假 prefix_feat_fn)**

```python
import numpy as np
from openpi.serving.feature_policy import FeaturePolicy, wrap_with_feature


class _StubInner:
    metadata = {"ckpt": "stub"}

    def __init__(self):
        self.infer_calls = 0
        self.transform_calls = 0
        self._model = object()

    def _input_transform(self, x):
        self.transform_calls += 1
        return x

    def infer(self, obs, **kw):
        self.infer_calls += 1
        return {"actions": np.zeros((2, 7), np.float32), "state": np.asarray(obs["state"])}


def test_feature_policy_adds_prefix_feat_and_keeps_actions():
    inner = _StubInner()
    fp = FeaturePolicy(inner, pooling="last",
                       prefix_feat_fn=lambda _inner, _obs, _pool: np.arange(5, dtype=np.float32))
    out = fp.infer({"state": np.ones(3, np.float32)})
    assert inner.infer_calls == 1
    assert np.allclose(out["actions"], 0.0)              # 原 actions 透传
    assert np.allclose(out["prefix_feat"], np.arange(5))  # 新字段
    assert out["prefix_feat"].dtype == np.float32


def test_feature_policy_metadata_passthrough():
    inner = _StubInner()
    fp = FeaturePolicy(inner, prefix_feat_fn=lambda *a: np.zeros(2, np.float32))
    assert fp.metadata == {"ckpt": "stub"}


def test_wrap_with_feature_returns_feature_policy():
    inner = _StubInner()
    fp = wrap_with_feature(inner, pooling="mean", prefix_feat_fn=lambda *a: np.zeros(2, np.float32))
    assert isinstance(fp, FeaturePolicy) and fp.pooling == "mean"
```

- [ ] **Step 2: 跑测验证失败**

Run: `cd /mnt/mnt/data/chj/openpi && .venv/bin/python -m pytest tests/test_feature_policy.py -v`
Expected: FAIL (`ImportError: cannot import name 'FeaturePolicy'`)。

- [ ] **Step 3: 在 `feature_policy.py` 追加 FeaturePolicy + 真前向 + wrap**

```python
import numpy as np


def make_prefix_feat_fn():
    """真前向(惰性 import pi0):obs → 复现 Policy.infer 预处理 → embed_prefix → prefix llm 前向 → pool。

    真 API(make_attn_mask import 路径、embed_prefix/PaliGemma.llm 返回结构)底稿:
    pi0.sample_actions:317-320 + pi0_value._forward_backbone:266-273。返回 [D] np.float32。
    单测不触达此(走注入 stub);真权重在 opt-in smoke / 起真 serve 时验证。
    """
    import jax
    import jax.numpy as _jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    def _fn(inner, obs, pooling):
        inputs = jax.tree.map(lambda x: x, obs)                       # copy(同 Policy.infer:70)
        inputs = inner._input_transform(inputs)                       # 同款预处理(同源命门)
        inputs = jax.tree.map(lambda x: _jnp.asarray(x)[None, ...], inputs)  # 加 batch(同 :74)
        observation = _model.Observation.from_dict(inputs)            # 同 :90
        model = inner._model
        tok, mask, ar = model.embed_prefix(observation)               # image+prompt,不含 state
        (prefix_out, _), _ = model.PaliGemma.llm(
            [tok, None], mask=make_attn_mask(mask, ar), positions=_jnp.cumsum(mask, 1) - 1)
        pooled = pool_prefix(prefix_out, mask, pooling)               # [1, D]
        return np.asarray(pooled[0], dtype=np.float32)

    return _fn


class FeaturePolicy:
    """包住原 Policy,在 infer 返回里加 prefix_feat。零侵入:不改原 Policy/协议。

    prefix_feat_fn(inner, obs, pooling) -> [D];默认 = make_prefix_feat_fn()(真前向)。
    """

    def __init__(self, inner, pooling="last", prefix_feat_fn=None):
        self.inner = inner
        self.pooling = pooling
        self._prefix_feat_fn = prefix_feat_fn or make_prefix_feat_fn()

    @property
    def metadata(self):
        return self.inner.metadata

    def infer(self, obs, **kw):
        outputs = self.inner.infer(obs, **kw)                         # 原 actions/state 不动
        feat = self._prefix_feat_fn(self.inner, obs, self.pooling)
        outputs["prefix_feat"] = np.asarray(feat, dtype=np.float32)
        return outputs


def wrap_with_feature(inner, pooling="last", prefix_feat_fn=None):
    """把一个 Policy 包成透特征的 FeaturePolicy。"""
    return FeaturePolicy(inner, pooling=pooling, prefix_feat_fn=prefix_feat_fn)
```

- [ ] **Step 4: 跑测验证通过**

Run: `cd /mnt/mnt/data/chj/openpi && .venv/bin/python -m pytest tests/test_feature_policy.py -v`
Expected: PASS (6 项)。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/chj/openpi && git add src/openpi/serving/feature_policy.py tests/test_feature_policy.py && \
git commit -m "feat(serve): FeaturePolicy wrapper (inject-able prefix feat fn, metadata passthrough)"
```

---

## Task 3: `serve_policy_with_feat.py` 启动脚本(openpi 侧)

**Files:**
- Create: `/mnt/mnt/data/chj/openpi/scripts/serve_policy_with_feat.py`
- Test: `/mnt/mnt/data/chj/openpi/tests/test_feature_policy.py`

- [ ] **Step 1: 追加失败测试(验证脚本 main 把 policy 包成 FeaturePolicy 再起 server)**

```python
def test_serve_with_feat_wraps_and_serves(monkeypatch):
    import scripts.serve_policy_with_feat as m
    captured = {}

    def fake_create(args):
        return _StubInner()

    def fake_serve(policy, host, port):
        captured["policy"] = policy
        captured["port"] = port

    monkeypatch.setattr(m, "_create_policy", fake_create)
    monkeypatch.setattr(m, "_serve", fake_serve)
    m.run(_FakeArgs(port=8000, pooling="last"))
    from openpi.serving.feature_policy import FeaturePolicy
    assert isinstance(captured["policy"], FeaturePolicy)
    assert captured["policy"].pooling == "last" and captured["port"] == 8000


class _FakeArgs:
    def __init__(self, port, pooling):
        self.port = port
        self.pooling = pooling
```

- [ ] **Step 2: 跑测验证失败**

Run: `cd /mnt/mnt/data/chj/openpi && .venv/bin/python -m pytest tests/test_feature_policy.py::test_serve_with_feat_wraps_and_serves -v`
Expected: FAIL (`ModuleNotFoundError: scripts.serve_policy_with_feat`)。

- [ ] **Step 3: 写 `serve_policy_with_feat.py`**

```python
"""起 pi05 serve,但用 FeaturePolicy 包原 policy → infer 返回额外带 prefix_feat。

并列入口,不改 serve_policy.py。用法(openpi 环境):
  uv run scripts/serve_policy_with_feat.py --config <cfg> --dir <ckpt> --port 8000 --pooling last
"""
from __future__ import annotations

import dataclasses

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.serving.feature_policy import wrap_with_feature
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str
    dir: str
    port: int = 8000
    host: str = "0.0.0.0"
    pooling: str = "last"
    default_prompt: str | None = None


def _create_policy(args: "Args"):
    return _policy_config.create_trained_policy(
        _config.get_config(args.config), args.dir, default_prompt=args.default_prompt)


def _serve(policy, host, port):
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=host, port=port, metadata=policy.metadata).serve_forever()


def run(args: "Args"):
    inner = _create_policy(args)
    policy = wrap_with_feature(inner, pooling=args.pooling)
    _serve(policy, args.host, args.port)


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
```

注:`WebsocketPolicyServer` 的真实构造签名(host/port/metadata 参数名)以 `serve_policy.py` 现有调用为准,Step 4 跑通脚本 import 时若不符就地对齐。单测走 `_serve` monkeypatch,不依赖真实签名。

- [ ] **Step 4: 跑测验证通过 + import 自检**

Run: `cd /mnt/mnt/data/chj/openpi && .venv/bin/python -m pytest tests/test_feature_policy.py -v && .venv/bin/python -c "import scripts.serve_policy_with_feat"`
Expected: 全 PASS;import 无异常(若 `WebsocketPolicyServer` 签名不符,按 serve_policy.py 修 `_serve`)。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/chj/openpi && git add scripts/serve_policy_with_feat.py tests/test_feature_policy.py && \
git commit -m "feat(serve): serve_policy_with_feat entry (wrap base policy to emit prefix_feat)"
```

---

## Task 4: `pi0_feat_cache.py`(resfit 侧)

**Files:**
- Create: `/mnt/mnt/data/resfit/resfit/rl_finetuning/chunk_residual/pi0_feat_cache.py`
- Test: `/mnt/mnt/data/resfit/resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import (
    save_pi0_feat_cache, pi0_feat_cache_reuse, load_pi0_feat_cache,
)

SIG = {"serve_ckpt_id": "pi05_base", "serve_metadata": {"a": 1}, "image_keys": ["agentview_image"],
       "proprio_key": "state18", "pooling": "last", "prompt": "do it", "dataset_id": "ds", "num_demos": None}


def _seqs():
    return [np.arange(6, dtype=np.float32).reshape(3, 2), np.ones((2, 2), np.float32)]


def test_save_reuse_roundtrip(tmp_path):
    p = str(tmp_path / "f.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    got = pi0_feat_cache_reuse(p, signature=SIG, num_demos=None)
    assert got is not None
    seqs, (mean, std) = got
    assert len(seqs) == 2 and np.allclose(seqs[0], _seqs()[0])
    assert np.allclose(mean, 0) and np.allclose(std, 1)


def test_reuse_signature_mismatch_returns_none(tmp_path):
    p = str(tmp_path / "f.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    assert pi0_feat_cache_reuse(p, signature=dict(SIG, pooling="mean"), num_demos=None) is None


def test_reuse_partial_num_demos_returns_none(tmp_path):
    p = str(tmp_path / "f.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    assert pi0_feat_cache_reuse(p, signature=SIG, num_demos=1) is None


def test_reuse_missing_file_returns_none(tmp_path):
    assert pi0_feat_cache_reuse(str(tmp_path / "nope.npz"), signature=SIG, num_demos=None) is None


def test_load_returns_seqs_stats_sig(tmp_path):
    p = str(tmp_path / "f.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    seqs, stats, sig = load_pi0_feat_cache(p)
    assert len(seqs) == 2 and sig["serve_ckpt_id"] == "pi05_base"
```

- [ ] **Step 2: 跑测验证失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache.py -v`
Expected: FAIL (`ModuleNotFoundError`)。

- [ ] **Step 3: 写 `pi0_feat_cache.py`(镜像 `act_feat_cache.py`)**

```python
"""pi0_feat 嵌入缓存(纯 numpy,仿 act_feat_cache)。签名全等(含 num_demos)才命中。"""
import json
import os

import numpy as np


def _norm_sig(sig):
    return json.loads(json.dumps(sig, sort_keys=True))


def save_pi0_feat_cache(path, seqs, emb_stats, *, signature, fp16=False):
    """存每条 demo 的(已标准化)嵌入序列 + (mean,std) + 签名 json。fp16 仅压 seqs 存盘。"""
    mean, std = emb_stats
    payload = {
        "n": np.int64(len(seqs)),
        "emb_mean": np.asarray(mean, dtype=np.float32),
        "emb_std": np.asarray(std, dtype=np.float32),
        "signature": np.asarray(json.dumps(_norm_sig(signature), sort_keys=True)),
    }
    dt = np.float16 if fp16 else np.float32
    for i, s in enumerate(seqs):
        payload[f"s{i}"] = np.asarray(s, dtype=dt)
    np.savez_compressed(path, **payload)


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        seqs = [np.asarray(z[f"s{i}"], dtype=np.float32) for i in range(n)]
        stats = (np.asarray(z["emb_mean"], np.float32), np.asarray(z["emb_std"], np.float32))
        sig = json.loads(str(z["signature"]))
    return seqs, stats, sig


def load_pi0_feat_cache(path):
    """读缓存 → (seqs[float32], (mean,std), signature_dict)。无签名校验,仅读取。"""
    return _load(path)


def pi0_feat_cache_reuse(path, *, signature, num_demos):
    """签名全等(含 num_demos)才命中,否则 None。"""
    if not (path and os.path.exists(path)):
        return None
    seqs, stats, sig = _load(path)
    want = _norm_sig(signature)
    want["num_demos"] = num_demos
    if sig != want:
        return None
    return seqs, stats
```

- [ ] **Step 4: 跑测验证通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache.py -v`
Expected: PASS (5 项)。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/pi0_feat_cache.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache.py && \
git commit -m "feat: pi0_feat embedding cache (signature-gated, full-only reuse)"
```

---

## Task 5: build 脚本纯逻辑 + main(resfit 侧)

**Files:**
- Create: `/mnt/mnt/data/resfit/resfit/rl_finetuning/chunk_residual/build_pi0_feat_cache_via_serve.py`
- Test: `/mnt/mnt/data/resfit/resfit/rl_finetuning/chunk_residual/tests/test_build_pi0_feat_via_serve.py`

- [ ] **Step 1: 写失败测试(纯函数 + main 用 stub client/hdf5)**

```python
import h5py
import numpy as np
from resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve import (
    assemble_pi0_feat_seqs, pi0_feat_signature, build_main,
)


def test_assemble_concats_proprio_and_standardizes():
    raw = [np.ones((2, 3), np.float32), np.full((2, 3), 3.0, np.float32)]   # 每 demo prefix_feat(T,3)
    prop = [np.zeros((2, 2), np.float32), np.full((2, 2), 4.0, np.float32)] # proprio(T,2)
    seqs, mean, std = assemble_pi0_feat_seqs(raw, prop)
    assert seqs[0].shape == (2, 5)                          # 3 + 2
    allcat = np.concatenate(seqs, axis=0)
    assert np.allclose(allcat.mean(0), 0, atol=1e-5)        # 全量标准化
    assert mean.shape == (5,) and std.shape == (5,)


def test_signature_changes_with_pooling():
    s1 = pi0_feat_signature("ds", None, image_keys=["a"], proprio_key="p", pooling="last",
                            prompt="x", serve_ckpt_id="ck", serve_metadata={})
    s2 = pi0_feat_signature("ds", None, image_keys=["a"], proprio_key="p", pooling="mean",
                            prompt="x", serve_ckpt_id="ck", serve_metadata={})
    assert s1 != s2


class _StubClient:
    def __init__(self):
        self.calls = 0
        self.metadata = {"server": "stub"}

    def get_server_metadata(self):
        return self.metadata

    def infer(self, obs):
        self.calls += 1
        return {"actions": np.zeros((1, 7), np.float32), "prefix_feat": np.ones(3, np.float32)}


def _make_hdf5(path, demos=2, T=3):
    with h5py.File(path, "w") as f:
        for i in range(demos):
            g = f.create_group(f"data/demo_{i}")
            g.create_dataset("obs/agentview_image", data=np.zeros((T, 4, 4, 3), np.uint8))
            g.create_dataset("obs/state18", data=np.full((T, 2), float(i + 1), np.float32))


def test_build_main_writes_cache_and_uses_client(tmp_path):
    hdf5 = str(tmp_path / "t.hdf5"); _make_hdf5(hdf5)
    cache = str(tmp_path / "c.npz")
    client = _StubClient()
    build_main(client, hdf5=hdf5, dataset_id="ds", image_keys=["agentview_image"],
               proprio_key="state18", prompt="x", pooling="last", serve_ckpt_id="pi05_base",
               out_cache=cache, num_demos=None)
    assert client.calls == 6                                # 2 demos * 3 frames
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    seqs, (mean, std), sig = load_pi0_feat_cache(cache)
    assert len(seqs) == 2 and seqs[0].shape[1] == 3 + 2     # prefix3 + proprio2
    assert sig["serve_ckpt_id"] == "pi05_base"
```

- [ ] **Step 2: 跑测验证失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_build_pi0_feat_via_serve.py -v`
Expected: FAIL (`ModuleNotFoundError`)。

- [ ] **Step 3: 写 `build_pi0_feat_cache_via_serve.py`**

```python
"""遍历源 hdf5,经 serve(openpi_client)取 pi05 prefix 特征 ⊕ proprio → 标准化 → pi0_feat 缓存。

residual 环境跑(只 openpi_client,不 import openpi);发原始 obs,预处理全在 serve 端(同源命门)。
用法:先在 openpi 环境起 serve_policy_with_feat.py,再:
  conda run -n residual python -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host H --port P --hdf5 X --dataset DS --image_keys agentview_image,... --proprio_key state18 \
    --prompt "..." --pooling last --serve_ckpt_id pi05_base --out_cache OUT [--num_demos N]
"""
from __future__ import annotations

import argparse

import numpy as np

from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import sorted_demo_keys
from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import save_pi0_feat_cache


def pi0_feat_signature(dataset_id, num_demos, *, image_keys, proprio_key, pooling, prompt,
                       serve_ckpt_id, serve_metadata):
    """缓存/同源校验签名(可 JSON 序列化、稳定排序)。"""
    return {
        "dataset_id": str(dataset_id), "num_demos": num_demos,
        "image_keys": list(image_keys), "proprio_key": str(proprio_key),
        "pooling": str(pooling), "prompt": str(prompt),
        "serve_ckpt_id": str(serve_ckpt_id), "serve_metadata": serve_metadata or {},
    }


def assemble_pi0_feat_seqs(raw_feats, propres):
    """list[(T,D_emb)] prefix 特征 + list[(T,D_p)] proprio → (seqs_std, mean, std)。

    逐 demo 尾拼 proprio → 全量算一组 (mean,std) → 标准化各 demo。
    """
    cat = [np.concatenate([np.asarray(f, np.float32), np.asarray(p, np.float32)], axis=1)
           for f, p in zip(raw_feats, propres)]
    allcat = np.concatenate(cat, axis=0)
    mean = allcat.mean(0).astype(np.float32)
    std = np.maximum(allcat.std(0), 1e-6).astype(np.float32)
    seqs = [((c - mean) / std).astype(np.float32) for c in cat]
    return seqs, mean, std


def _server_metadata(client):
    """尽力取 serve 握手 metadata(辅助签名);取不到则空 dict。"""
    for attr in ("get_server_metadata", "metadata"):
        m = getattr(client, attr, None)
        if callable(m):
            return dict(m() or {})
        if isinstance(m, dict):
            return dict(m)
    return {}


def build_main(client, *, hdf5, dataset_id, image_keys, proprio_key, prompt, pooling,
               serve_ckpt_id, out_cache, num_demos=None):
    """核心流程(client 注入,便于 stub 测试)。"""
    import h5py
    raw_feats, propres = [], []
    with h5py.File(hdf5, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
        if num_demos is not None:
            eps = eps[:num_demos]
        for ep in eps:
            grp = f[f"data/{ep}"]
            imgs = {k: grp[f"obs/{k}"][()] for k in image_keys}
            proprio = np.asarray(grp[f"obs/{proprio_key}"][()], np.float32)
            T = proprio.shape[0]
            feats = []
            for t in range(T):
                obs = {k: imgs[k][t] for k in image_keys}
                obs["prompt"] = prompt
                feats.append(np.asarray(client.infer(obs)["prefix_feat"], np.float32))
            raw_feats.append(np.stack(feats, axis=0))
            propres.append(proprio)
    seqs, mean, std = assemble_pi0_feat_seqs(raw_feats, propres)
    sig = pi0_feat_signature(dataset_id, num_demos, image_keys=image_keys, proprio_key=proprio_key,
                             pooling=pooling, prompt=prompt, serve_ckpt_id=serve_ckpt_id,
                             serve_metadata=_server_metadata(client))
    save_pi0_feat_cache(out_cache, seqs, (mean, std), signature=sig)
    print(f"[build_pi0_feat] wrote {out_cache}: {len(seqs)} demos, dim={seqs[0].shape[1]}")


def _connect(host, port):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import websockets.sync.client as _wsc
    if not getattr(_wsc.connect, "_no_ping_patched", False):
        _orig = _wsc.connect

        def _no_ping(*a, **k):
            k.setdefault("ping_interval", None)
            return _orig(*a, **k)

        _no_ping._no_ping_patched = True
        _wsc.connect = _no_ping
    return WebsocketClientPolicy(host=host, port=port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--image_keys", type=lambda s: s.split(","), required=True)
    ap.add_argument("--proprio_key", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--serve_ckpt_id", required=True)
    ap.add_argument("--out_cache", required=True)
    ap.add_argument("--num_demos", type=int, default=None)
    args = ap.parse_args()
    client = _connect(args.host, args.port)
    build_main(client, hdf5=args.hdf5, dataset_id=args.dataset, image_keys=args.image_keys,
               proprio_key=args.proprio_key, prompt=args.prompt, pooling=args.pooling,
               serve_ckpt_id=args.serve_ckpt_id, out_cache=args.out_cache, num_demos=args.num_demos)


if __name__ == "__main__":
    main()
```

注:`client.infer(obs)` 的真实 obs schema(图像键名、prompt 字段、是否需 image_mask)以 `load_pi05.py`/adapter 现有发法为准;Step 4 跑真 serve smoke(Task 8)时对齐。单测走 stub client 不依赖真 schema。

- [ ] **Step 4: 跑测验证通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_build_pi0_feat_via_serve.py -v`
Expected: PASS (3 项)。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/build_pi0_feat_cache_via_serve.py resfit/rl_finetuning/chunk_residual/tests/test_build_pi0_feat_via_serve.py && \
git commit -m "feat: build_pi0_feat_cache_via_serve (openpi_client → prefix_feat ⊕ proprio → standardized cache)"
```

---

## Task 6: `read_per_demo_states` 加 `pi0_feat` 分支(resfit 侧)

**Files:**
- Modify: `/mnt/mnt/data/resfit/resfit/rl_finetuning/chunk_residual/train_hiql_value.py`(`read_per_demo_states`,约 :26-104)
- Test: `/mnt/mnt/data/resfit/resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_pi0_feat.py`

- [ ] **Step 1: 写失败测试(cache-required:命中读、缺失 raise)**

```python
import numpy as np
import pytest
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import save_pi0_feat_cache

SIG = {"serve_ckpt_id": "pi05_base", "serve_metadata": {}, "image_keys": ["agentview_image"],
       "proprio_key": "state18", "pooling": "last", "prompt": "x", "dataset_id": "ds", "num_demos": None}


def test_pi0_feat_cache_hit_returns_seqs_and_stats(tmp_path):
    cache = str(tmp_path / "c.npz")
    seqs = [np.zeros((3, 5), np.float32), np.zeros((2, 5), np.float32)]
    save_pi0_feat_cache(cache, seqs, (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=SIG)
    out_seqs, standardizer, feat_stats = read_per_demo_states(
        "ignored.hdf5", "ds", "pi0_feat", pi0_feat_cache=cache,
        pi0_feat_signature=dict(SIG, num_demos=None))
    assert len(out_seqs) == 2 and out_seqs[0].shape[1] == 5
    assert standardizer is None
    mean, std = feat_stats
    assert mean.shape == (5,)


def test_pi0_feat_cache_missing_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        read_per_demo_states("ignored.hdf5", "ds", "pi0_feat",
                             pi0_feat_cache=str(tmp_path / "nope.npz"),
                             pi0_feat_signature=dict(SIG, num_demos=None))
```

- [ ] **Step 2: 跑测验证失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_pi0_feat.py -v`
Expected: FAIL (`got an unexpected keyword argument 'pi0_feat_cache'`)。

- [ ] **Step 3: 改 `read_per_demo_states`(签名加两参 + pi0_feat 早返回分支)**

在 `read_per_demo_states(...)` 签名追加 `pi0_feat_cache=None, pi0_feat_signature=None`。在函数体最前(现有 act_feat 短路分支之前)插入:

```python
    if state_mode == "pi0_feat":
        from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import pi0_feat_cache_reuse
        assert pi0_feat_signature is not None, "pi0_feat 需 pi0_feat_signature(build 时写的同款签名)"
        nd = pi0_feat_signature.get("num_demos", num_demos)
        hit = pi0_feat_cache_reuse(pi0_feat_cache, signature=pi0_feat_signature, num_demos=nd)
        if hit is None:
            raise RuntimeError(
                f"pi0_feat 缓存未命中/缺失: {pi0_feat_cache}。请先在 residual 环境跑 "
                "build_pi0_feat_cache_via_serve.py(serve 须先起)生成缓存。")
        seqs, feat_stats = hit
        return seqs, None, feat_stats        # 缓存里 seqs 已标准化;standardizer 占位 None
```

(eef/eef_piece/act_feat 三支原样不动;pi0_feat 在其前 return。)

- [ ] **Step 4: 跑测验证通过 + 既有 value 测试不回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_pi0_feat.py resfit/rl_finetuning/chunk_residual/tests/ -k "value or state30 or hiql or act_feat" -v`
Expected: 新 2 项 PASS;既有 value/state30/hiql/act_feat 用例全绿。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_pi0_feat.py && \
git commit -m "feat: read_per_demo_states pi0_feat branch (cache-required, no serve/openpi import)"
```

---

## Task 7: CLI 接线 + `validate_pi0_feat_cfg` + gc_value save/load(resfit 侧)

**Files:**
- Modify: `train_hiql_value.py`(parser `pi0_feat` 枚举 + `--pi0_feat_*` flags + `validate_pi0_feat_cfg`)
- Modify: `train_hiql_gc_value.py`(`--state_mode pi0_feat` + main dispatch + `save_gc_value` 透 signature)
- Test: `/mnt/mnt/data/resfit/resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py`

- [ ] **Step 1: 写失败测试**

```python
import pytest
from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    build_parser as v_parser, validate_pi0_feat_cfg,
)
from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser as gc_parser


def test_gc_parser_state_mode_default_unchanged():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d"])
    assert a.state_mode == "eef_piece"


def test_pi0_feat_choice_accepted():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--state_mode", "pi0_feat",
                                "--pi0_feat_cache", "c.npz", "--pi0_serve_ckpt_id", "pi05_base",
                                "--pi0_image_keys", "agentview_image", "--pi0_proprio_key", "state18"])
    assert a.state_mode == "pi0_feat" and a.pi0_image_keys == ["agentview_image"]


def test_validate_requires_cache(tmp_path):
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--state_mode", "pi0_feat",
                                "--pi0_serve_ckpt_id", "ck", "--pi0_image_keys", "a", "--pi0_proprio_key", "p"])
    with pytest.raises(ValueError):
        validate_pi0_feat_cfg(a)                          # 缺 --pi0_feat_cache(或不存在) → 报错


def test_validate_noop_for_eef_modes():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d"])
    validate_pi0_feat_cfg(a)                              # eef_piece 不报
```

- [ ] **Step 2: 跑测验证失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py -v`
Expected: FAIL (`cannot import name 'validate_pi0_feat_cfg'` / `pi0_feat` 非法 choice)。

- [ ] **Step 3: 实现 parser + validate + gc_value 接线**

`train_hiql_value.py` 加:
```python
def _add_pi0_feat_args(p):
    p.add_argument("--pi0_feat_cache", default=None, help="pi0_feat 嵌入缓存 npz(仅 pi0_feat,须已存在)")
    p.add_argument("--pi0_serve_ckpt_id", default=None, help="serve 权重身份锚(仅 pi0_feat,同源签名)")
    p.add_argument("--pi0_image_keys", type=lambda s: s.split(","), default=None, help="逗号分隔图像键")
    p.add_argument("--pi0_proprio_key", default=None, help="本体 obs 键(仅 pi0_feat)")
    p.add_argument("--pi0_prompt", default="", help="pi05 prompt(仅 pi0_feat)")
    p.add_argument("--pi0_pooling", choices=["last", "mean"], default="last")


def validate_pi0_feat_cfg(args):
    """pi0_feat 须给已存在的 --pi0_feat_cache + serve_ckpt_id/image_keys/proprio_key;否则 ValueError。
    非 pi0_feat 传 --pi0_* 忽略。"""
    import os
    import warnings
    if getattr(args, "state_mode", None) != "pi0_feat":
        if getattr(args, "pi0_feat_cache", None):
            warnings.warn("非 pi0_feat 模式,--pi0_* 被忽略", stacklevel=2)
        return
    missing = [k for k in ("pi0_serve_ckpt_id", "pi0_image_keys", "pi0_proprio_key")
               if not getattr(args, k, None)]
    if missing:
        raise ValueError(f"--state_mode pi0_feat 需提供 {missing}")
    if not (args.pi0_feat_cache and os.path.exists(args.pi0_feat_cache)):
        raise ValueError("--state_mode pi0_feat 需 --pi0_feat_cache(已存在);请先跑 build_pi0_feat_cache_via_serve.py")
```
把 `train_hiql_value.py` 的 `--state_mode choices` 加 `"pi0_feat"`,并在 build_parser 里调 `_add_pi0_feat_args(p)`。

`train_hiql_gc_value.py`:`--state_mode` choices 加 `"pi0_feat"`(默认仍 `eef_piece`);`from ...train_hiql_value import _add_pi0_feat_args, validate_pi0_feat_cfg` 后 `_add_pi0_feat_args(p)`;`main()` 开头 `validate_pi0_feat_cfg(args)`;pi0_feat 时构造签名并经 `read_per_demo_states(..., pi0_feat_cache=args.pi0_feat_cache, pi0_feat_signature=sig)` 取 seqs,`save_gc_value(..., state_mode="pi0_feat", mean=mean, std=std, rel_piece_stats=None, pi0_feat_signature=sig)`。`save_gc_value`/`load_gc_value` 加可选 `pi0_feat_signature=None`(写入/读回 payload,缺键 None,类比 act_feat_signature)。签名构造复用 `build_pi0_feat_cache_via_serve.pi0_feat_signature`(serve_metadata 传 `{}` 占位,离线训练侧不连 serve;命门:与 build 写入缓存的签名一致,故 build 与 train 都用 dataset/num_demos/keys/pooling/prompt/serve_ckpt_id 同值,serve_metadata 不参与训练侧重算——**改进:reuse 时签名以缓存内为准,训练侧只需传 cache 路径**;见 Step 实现注)。

> **实现注(消歧义)**:为避免训练侧重算 serve_metadata 与缓存不一致,`read_per_demo_states` 的 pi0_feat 分支**改为:不传整签名,只传 cache 路径 + 轻校验字段**。即 Task 6 的分支调整为 `pi0_feat_cache_reuse(path, signature=None宽松, ...)`——但为保持 Task 6 已写接口,这里采用:gc_value 先 `load_pi0_feat_cache` 读出缓存内签名,用它当 `pi0_feat_signature` 传回 `read_per_demo_states`(自洽命中),并断言缓存签名的 `serve_ckpt_id/image_keys/proprio_key/pooling/prompt` 与 CLI 传入一致(不一致 raise,防张冠李戴)。`serve_metadata` 仅诊断、不参与一致性断言。

- [ ] **Step 4: 跑测验证通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py -v`
Expected: PASS (4 项)。

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py && \
git commit -m "feat: --state_mode pi0_feat CLI + validate guard + gc_value save/load signature"
```

---

## Task 8: 全量回归 + opt-in 集成 smoke

**Files:**
- Test: `/mnt/mnt/data/resfit/resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_via_serve_smoke.py`(默认 skip)

- [ ] **Step 1: 写集成冒烟(env gate,真 serve)**

```python
import os
import numpy as np
import pytest

RUN = os.environ.get("PI0_FEAT_VIA_SERVE_SMOKE") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="opt-in:需真 pi05-with-feat serve + hdf5,设 PI0_FEAT_VIA_SERVE_SMOKE=1")


def test_end_to_end_serve_to_gc_value(tmp_path):
    """起真 serve_policy_with_feat 后,build 几帧缓存 → 训几步 gc_value,断言 V 有限。
    环境变量:PI0_SERVE_HOST/PORT、PI0_FEAT_HDF5、PI0_FEAT_IMGKEYS、PI0_FEAT_PROPRIO、PI0_FEAT_PROMPT。"""
    from resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve import _connect, build_main
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    cache = str(tmp_path / "c.npz")
    client = _connect(os.environ["PI0_SERVE_HOST"], int(os.environ["PI0_SERVE_PORT"]))
    build_main(client, hdf5=os.environ["PI0_FEAT_HDF5"], dataset_id="smoke",
               image_keys=os.environ["PI0_FEAT_IMGKEYS"].split(","),
               proprio_key=os.environ["PI0_FEAT_PROPRIO"], prompt=os.environ.get("PI0_FEAT_PROMPT", ""),
               pooling="last", serve_ckpt_id="smoke", out_cache=cache, num_demos=2)
    seqs, _, _ = load_pi0_feat_cache(cache)
    data = build_gc_data(seqs, [np.empty(0, np.int64) for _ in seqs])
    _model, v_stats = train_gc_value(data, steps=5)
    assert np.isfinite(v_stats["mean"]) and np.isfinite(v_stats["min"])
```

- [ ] **Step 2: 跑(默认 skip)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_via_serve_smoke.py -v`
Expected: 1 SKIPPED(未设 env)。手动 opt-in 见 docstring。

- [ ] **Step 3: 全量回归(resfit + openpi 各自)**

Run:
```
conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q
cd /mnt/mnt/data/chj/openpi && .venv/bin/python -m pytest tests/test_feature_policy.py -q
```
Expected: 全绿(既有用例 + 本计划新增)。**任何既有用例变红 → 停下修到逐位等价**。

- [ ] **Step 4: Commit**

```bash
cd /mnt/mnt/data/resfit && git add resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_via_serve_smoke.py && \
git commit -m "test: opt-in pi0_feat-via-serve integration smoke + full regression green"
```

---

## Self-Review(写完计划自查)

**Spec 覆盖**:§3.1 FeaturePolicy→Task2;§3.2 serve_with_feat→Task3;§3.3 build→Task5;§3.4 缓存/签名→Task4;§3.5 read 分支→Task6;§3.6 CLI/validate/save→Task7;§4 测试(stub 单测 + opt-in smoke + 回归)→各 Task Step1 + Task8。pool 纯函数(§3.1)→Task1。全覆盖。high_actor/在线注入按 spec 不在范围,无对应 task(正确)。

**占位符扫描**:无 TBD/TODO。两处"以真 serve/真权重为准就地对齐"(FeaturePolicy 真前向 import 路径、build 的 obs schema、WebsocketPolicyServer 签名)是**真权重/真服务现场钉死**(无法离线臆测),已给完整底稿代码 + 明确"单测走 stub 不依赖"+ Task8 真服务验证,非占位逃避。

**类型/命名一致**:`pool_prefix`、`FeaturePolicy(inner,pooling,prefix_feat_fn)`、`wrap_with_feature`、`save/load/pi0_feat_cache_reuse`、`pi0_feat_signature(dataset,num_demos,*,image_keys,proprio_key,pooling,prompt,serve_ckpt_id,serve_metadata)`、`assemble_pi0_feat_seqs`、`build_main`、`read_per_demo_states(...,pi0_feat_cache,pi0_feat_signature)`、`validate_pi0_feat_cfg`、`save_gc_value(...,pi0_feat_signature)` 跨 Task 一致。

**已修歧义**:Task7 实现注明确了"训练侧不重算 serve_metadata、以缓存内签名自洽命中 + 核心字段一致性断言",消除 build 与 train 两侧签名可能不一致的风险。
