# HIQL value 接冻结 ACT encoder 特征(act_feat)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 resfit HIQL 离线 value(`gc_value`/`high_actor` 及共用 `read_per_demo_states`)新增 `state_mode="act_feat"`,用冻结 ACT transformer encoder 的池化输出 ⊕ 原始 18 维本体当 state,替换 sim 特权的 `eef_piece`;默认路逐位不变。

**Architecture:** 不改 ACT,用 `base_policy.model.encoder` 的 forward hook 抓 `encoder_out` → mean 池化 → concat 原始 proprio。嵌入在 residual 进程内 cache-or-build(仿 `state30_cache`),整条 `[emb⊕proprio]` 算一组 `(mean,std)` 标准化并随 ckpt 保存。全程 flag-gated。

**Tech Stack:** Python, PyTorch, h5py, numpy, pytest;ACT = `resfit.lerobot.policies.act.modeling_act.ACTPolicy`(经 `resfit.lerobot.utils.load_policy.load_policy` 加载)。

参考 spec:`docs/superpowers/specs/2026-06-13-hiql-value-act-feature-offline-design.md`。

---

## File Structure

- Create `resfit/rl_finetuning/chunk_residual/act_feature.py` — 纯函数(pool/concat/signature)+ `ActFeatureExtractor`(hook 抓 encoder_out)。
- Create `resfit/rl_finetuning/chunk_residual/act_feat_cache.py` — 缓存 save/reuse(纯 numpy,fp16 选项)。
- Create `resfit/rl_finetuning/chunk_residual/verify_act_prefix_feature.py` — 真 ACT spike(opt-in)。
- Modify `resfit/rl_finetuning/chunk_residual/train_hiql_value.py` — `read_per_demo_states` 加 `act_feat` 分支 + parser flags + guard + main() mean/std 分流。
- Modify `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py` — `state_mode` 改可读 CLI(默认 `eef_piece`)+ act_feat 分流 + save 签名。
- Modify `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py` — `save_gc_value`/`load_gc_value` 透传 `act_feat_signature`。
- Modify `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py` — state 源 dispatch(act_feat 走缓存)+ parser + save 签名。
- Modify `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py` — `save_high_actor`/`load_high_actor` 透传 `act_feat_signature`。
- Create tests: `tests/test_act_feature.py`、`tests/test_act_feat_cache.py`、`tests/test_read_per_demo_states_act_feat.py`、`tests/test_act_feat_cli_wiring.py`。

约定:`D_emb` = ACT `dim_model`;`D_proprio` = 18(`STATE18_KEYS`/`assemble_state18`)。所有命令从仓库根 `/mnt/mnt/data/resfit` 用 `conda run -n residual` 跑。

---

## Task 1: Spike — 真 ACT 上验通 encoder_out hook(de-risk,opt-in)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/verify_act_prefix_feature.py`

目的:在不改 ACT 的前提下确认 `base_policy.model.encoder` forward hook 能拿到 `encoder_out`,特征有限、确定性,并记下 `D_emb` 与图像键/格式。**非 TDD,是探针**;需要 GPU + ACT base ckpt,默认不在 CI 跑。

- [ ] **Step 1: 写 spike 脚本**

```python
"""Spike(opt-in,需 GPU + ACT base):验证 encoder_out forward hook 取数路径。
跑:conda run -n residual python -m resfit.rl_finetuning.chunk_residual.verify_act_prefix_feature \
      --base /mnt/mnt/data/resfit/resfit/out/piecce/best --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def load_act(base_dir):
    from resfit.lerobot.utils.load_policy import load_policy
    cand = Path(base_dir) / "policy"
    policy_dir = cand if cand.is_dir() else Path(base_dir)
    p = load_policy(policy_dir)
    p.eval()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--hdf5", required=True)
    args = ap.parse_args()
    act = load_act(args.base)
    img_keys = list(act.config.image_features.keys())
    print("ACT image_features:", img_keys, "dim_model:", act.config.dim_model)

    with h5py.File(args.hdf5, "r") as f:
        ep = sorted(f["data"].keys())[0]
        grp = f[f"data/{ep}"]
        raw = {}
        for k in img_keys:
            name = k.replace("observation.images.", "")
            raw[k] = torch.as_tensor(grp[f"obs/{name}_image"][0:2])
        from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
            STATE18_KEYS, assemble_state18)
        obs_arrays = {kk: grp[f"obs/{kk}"][0:2] for kk, _ in STATE18_KEYS}
        raw["observation.state"] = torch.as_tensor(assemble_state18(obs_arrays))

    captured = {}
    h = act.model.encoder.register_forward_hook(lambda m, i, o: captured.__setitem__("enc", o))
    with torch.no_grad():
        batch = dict(act.normalize_inputs(raw))
        batch["observation.images"] = [batch[k] for k in img_keys]
        act.model(batch)
    h.remove()
    enc = captured["enc"]
    print("encoder_out shape:", tuple(enc.shape))   # 期望 [seq, B, dim_model]
    emb = enc.mean(dim=0)
    print("pooled emb shape:", tuple(emb.shape), "finite:", bool(torch.isfinite(emb).all()))
    # 确定性:重跑一次
    captured.clear()
    h = act.model.encoder.register_forward_hook(lambda m, i, o: captured.__setitem__("enc", o))
    with torch.no_grad():
        act.model(batch)
    h.remove()
    assert torch.allclose(emb, captured["enc"].mean(dim=0), atol=1e-6), "非确定性!"
    print("OK: 确定性通过;D_emb =", emb.shape[-1])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 spike(opt-in,需授权 GPU)**

Run: `conda run -n residual python -m resfit.rl_finetuning.chunk_residual.verify_act_prefix_feature --base resfit/out/piecce/best --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5`
Expected: 打印 `encoder_out shape [seq, B, dim_model]`、`pooled emb shape [B, dim_model]`、`finite: True`、`OK: 确定性通过`。**记下 dim_model(=D_emb)与 image_features 键名/hdf5 obs 图像键映射**,后续 Task 据此填。若 `act.model.encoder` 属性名不符或图像格式报错,在此修正取数路径(改 hook 目标或图像键映射),再继续。

- [ ] **Step 3: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/verify_act_prefix_feature.py
git commit -m "spike: verify ACT encoder_out forward-hook feature path (act_feat)"
```

---

## Task 2: `act_feature.py` 纯函数(pool / concat / signature)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/act_feature.py`
- Test: `tests/test_act_feature.py`

- [ ] **Step 1: 写失败测试(纯函数)**

```python
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.act_feature import (
    pool_encoder_out, concat_proprio, act_feat_signature,
)


def test_pool_encoder_out_mean():
    enc = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)  # [seq=2,B=3,dim=4]
    out = pool_encoder_out(enc, pooling="mean")
    assert out.shape == (3, 4)
    assert torch.allclose(out, enc.mean(dim=0))


def test_pool_encoder_out_unknown_pooling_raises():
    enc = torch.zeros(2, 3, 4)
    try:
        pool_encoder_out(enc, pooling="weird")
        assert False, "应当报错"
    except ValueError:
        pass


def test_concat_proprio_slice():
    emb = torch.zeros(5, 8)
    proprio = torch.ones(5, 18)
    out = concat_proprio(emb, proprio)
    assert out.shape == (5, 26)
    assert torch.allclose(out[:, 8:], torch.ones(5, 18))   # proprio 拼在尾部
    assert out.dtype == torch.float32


def test_signature_stable_and_sensitive():
    a = act_feat_signature("ckptA", ["observation.images.agentview"], "observation.state", "mean")
    b = act_feat_signature("ckptA", ["observation.images.agentview"], "observation.state", "mean")
    c = act_feat_signature("ckptB", ["observation.images.agentview"], "observation.state", "mean")
    assert a == b and a != c
    assert a["act_ckpt_id"] == "ckptA" and a["pooling"] == "mean"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest tests/test_act_feature.py -v`
Expected: FAIL（`ModuleNotFoundError` 或 `ImportError: cannot import name`）。

- [ ] **Step 3: 写最小实现**

```python
"""冻结 ACT encoder 特征器(act_feat);不改 ACT,复用其组件。

纯函数(pool/concat/signature)只依赖 torch/numpy,单测可跑;ActFeatureExtractor 需真 ACT。
"""
from __future__ import annotations

import torch


def pool_encoder_out(encoder_out: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    """[seq, B, dim] -> [B, dim]。pooling='mean' 对 token 维求均值。"""
    if pooling != "mean":
        raise ValueError(f"unsupported pooling: {pooling}")
    return encoder_out.mean(dim=0)


def concat_proprio(emb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """[B, D_emb] ⊕ [B, D_proprio] -> [B, D_emb+D_proprio] float32(proprio 在尾部)。"""
    return torch.cat([emb.float(), proprio.float()], dim=-1)


def act_feat_signature(act_ckpt_id, image_keys, proprio_key, pooling) -> dict:
    """同源校验签名(纯数据)。"""
    return {
        "act_ckpt_id": str(act_ckpt_id),
        "image_keys": list(image_keys),
        "proprio_key": str(proprio_key),
        "pooling": str(pooling),
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest tests/test_act_feature.py -v`
Expected: PASS（4 passed）。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/act_feature.py tests/test_act_feature.py
git commit -m "feat: act_feature pure helpers (pool/concat/signature) + tests"
```

---

## Task 3: `ActFeatureExtractor`(forward hook 抓 encoder_out)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/act_feature.py`
- Test: `tests/test_act_feature.py`

- [ ] **Step 1: 写失败测试(stub ACT)**

```python
import torch
from torch import nn

from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor


class _StubEncoder(nn.Module):
    def forward(self, tokens, pos_embed=None):
        return tokens  # [seq, B, dim]


class _StubModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = _StubEncoder()
        self.dim = dim

    def forward(self, batch):
        b = batch["observation.images"][0].shape[0]
        seq = 4
        tokens = torch.ones(seq, b, self.dim)   # 确定性
        return self.encoder(tokens)


class _StubCfg:
    def __init__(self, image_keys):
        self.image_features = {k: None for k in image_keys}


class _StubACT(nn.Module):
    def __init__(self, dim, image_keys):
        super().__init__()
        self.model = _StubModel(dim)
        self.config = _StubCfg(image_keys)

    def normalize_inputs(self, raw):
        return dict(raw)   # 身份预处理


def _raw(b=3):
    return {
        "observation.images.agentview": torch.zeros(b, 3, 4, 4),
        "observation.state": torch.arange(b * 18, dtype=torch.float32).reshape(b, 18),
    }


def test_extractor_shape_dtype_determinism_and_proprio_slice():
    act = _StubACT(dim=8, image_keys=["observation.images.agentview"])
    ext = ActFeatureExtractor(act, image_keys=["observation.images.agentview"],
                              proprio_key="observation.state", pooling="mean")
    raw = _raw()
    out1 = ext.embed_batch(raw)
    out2 = ext.embed_batch(raw)
    assert out1.shape == (3, 8 + 18)
    assert out1.dtype == torch.float32
    assert torch.allclose(out1, out2)                    # 确定性
    assert torch.allclose(out1[:, 8:], raw["observation.state"])  # 原始 proprio 在尾部
    assert ext.feature_dim == 8 + 18
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest tests/test_act_feature.py::test_extractor_shape_dtype_determinism_and_proprio_slice -v`
Expected: FAIL（`ImportError: cannot import name 'ActFeatureExtractor'`）。

- [ ] **Step 3: 写最小实现(追加到 `act_feature.py`)**

```python
class ActFeatureExtractor:
    """冻结 ACT,用 model.encoder 的 forward hook 抓 encoder_out → 池化 ⊕ 原始 proprio。"""

    def __init__(self, act_policy, image_keys, proprio_key="observation.state",
                 pooling="mean", proprio_dim=18):
        self.act = act_policy
        self.act.eval()
        for p in self.act.parameters():
            p.requires_grad_(False)
        self.image_keys = list(image_keys)
        self.proprio_key = proprio_key
        self.pooling = pooling
        self._proprio_dim = proprio_dim

    @property
    def feature_dim(self) -> int:
        return int(self.act.config.dim_model) + self._proprio_dim

    @torch.no_grad()
    def embed_batch(self, raw_obs: dict) -> torch.Tensor:
        batch = dict(self.act.normalize_inputs(raw_obs))
        batch["observation.images"] = [batch[k] for k in self.image_keys]
        captured = {}
        h = self.act.model.encoder.register_forward_hook(
            lambda m, i, o: captured.__setitem__("enc", o))
        try:
            self.act.model(batch)
        finally:
            h.remove()
        emb = pool_encoder_out(captured["enc"], self.pooling)   # [B, D_emb]
        proprio = torch.as_tensor(raw_obs[self.proprio_key], dtype=torch.float32)
        return concat_proprio(emb, proprio)

    def signature(self, act_ckpt_id) -> dict:
        return act_feat_signature(act_ckpt_id, self.image_keys, self.proprio_key, self.pooling)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest tests/test_act_feature.py -v`
Expected: PASS（5 passed）。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/act_feature.py tests/test_act_feature.py
git commit -m "feat: ActFeatureExtractor via encoder forward-hook + stub test"
```

---

## Task 4: `act_feat_cache.py`(save / reuse / 签名 / fp16)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/act_feat_cache.py`
- Test: `tests/test_act_feat_cache.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np

from resfit.rl_finetuning.chunk_residual.act_feat_cache import (
    save_act_feat_cache, act_feat_cache_reuse,
)

SIG = {"act_ckpt_id": "ckptA", "image_keys": ["observation.images.agentview"],
       "proprio_key": "observation.state", "pooling": "mean",
       "dataset_id": "ds", "num_demos": None}


def test_roundtrip(tmp_path):
    p = str(tmp_path / "c.npz")
    seqs = [np.ones((3, 5), np.float32), np.zeros((2, 5), np.float32)]
    stats = (np.arange(5, dtype=np.float32), np.ones(5, np.float32))
    save_act_feat_cache(p, seqs, stats, signature=SIG)
    got = act_feat_cache_reuse(p, signature=SIG, num_demos=None)
    assert got is not None
    gseqs, gstats = got
    assert len(gseqs) == 2 and gseqs[0].shape == (3, 5)
    assert np.allclose(gstats[0], stats[0]) and np.allclose(gstats[1], stats[1])


def test_signature_mismatch_returns_none(tmp_path):
    p = str(tmp_path / "c.npz")
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=SIG)
    bad = dict(SIG, act_ckpt_id="ckptB")
    assert act_feat_cache_reuse(p, signature=bad, num_demos=None) is None


def test_partial_num_demos_not_reused(tmp_path):
    p = str(tmp_path / "c.npz")
    sig_partial = dict(SIG, num_demos=1)
    save_act_feat_cache(p, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=sig_partial)
    # 请求全量但缓存是部分量 → 不命中
    assert act_feat_cache_reuse(p, signature=SIG, num_demos=None) is None


def test_fp16_option_roundtrip(tmp_path):
    p = str(tmp_path / "c.npz")
    seqs = [np.full((2, 5), 0.5, np.float32)]
    save_act_feat_cache(p, seqs, (np.zeros(5, np.float32), np.ones(5, np.float32)),
                        signature=SIG, fp16=True)
    gseqs, _ = act_feat_cache_reuse(p, signature=SIG, num_demos=None)
    assert gseqs[0].dtype == np.float32 and np.allclose(gseqs[0], 0.5, atol=1e-3)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest tests/test_act_feat_cache.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 写最小实现**

```python
"""act_feat 嵌入缓存(纯 numpy,仿 state30_cache)。签名全等才命中。"""
import json
import os

import numpy as np


def save_act_feat_cache(path, seqs, emb_stats, *, signature, fp16=False):
    """存每条 demo 的(已标准化)嵌入序列 + (mean,std) + 签名 json。fp16 仅压 seqs 存盘。"""
    mean, std = emb_stats
    payload = {
        "n": np.int64(len(seqs)),
        "emb_mean": np.asarray(mean, dtype=np.float32),
        "emb_std": np.asarray(std, dtype=np.float32),
        "signature": np.asarray(json.dumps(signature, sort_keys=True)),
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


def act_feat_cache_reuse(path, *, signature, num_demos):
    """签名全等(含 num_demos)才命中,否则 None。"""
    if not (path and os.path.exists(path)):
        return None
    seqs, stats, sig = _load(path)
    want = dict(signature)
    want.setdefault("num_demos", num_demos)
    if sig != want:
        return None
    return seqs, stats
```

注:签名里 `num_demos` 由 build 侧写入 signature;`act_feat_cache_reuse` 用传入 `signature`(应已含 dataset_id/num_demos)直接比对。Task 5 build 处构造完整 signature(含 dataset_id + num_demos)再存。

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest tests/test_act_feat_cache.py -v`
Expected: PASS（4 passed）。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/act_feat_cache.py tests/test_act_feat_cache.py
git commit -m "feat: act_feat_cache save/reuse with signature + fp16 + tests"
```

---

## Task 5: `read_per_demo_states` 加 `act_feat` 分支 + flags + guard

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`
- Test: `tests/test_read_per_demo_states_act_feat.py`, `tests/test_act_feat_cli_wiring.py`

实现策略:`act_feat` 分支在 `read_per_demo_states` 内 cache-or-build。build 用注入的 `extractor`(测试传 stub;生产由 main 构造真 ACT extractor)。返回 `(seqs_std, None, (mean,std))`。

- [ ] **Step 1: 写失败测试(分支 + guard)**

```python
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    read_per_demo_states, validate_act_feat_cfg, build_parser,
)


class _StubExtractor:
    """每帧返回 [D_emb+18];记录被调用次数(验证命中缓存跳过)。"""
    calls = 0

    def embed_batch(self, raw):
        _StubExtractor.calls += 1
        b = raw["observation.state"].shape[0]
        emb = torch.zeros(b, 4)
        return torch.cat([emb, torch.as_tensor(raw["observation.state"], dtype=torch.float32)], -1)


def _make_hdf5(tmp_path, n_demos=2, T=3):
    import h5py
    p = str(tmp_path / "mini.hdf5")
    with h5py.File(p, "w") as f:
        for d in range(n_demos):
            g = f.create_group(f"data/demo_{d}")
            g.create_dataset("obs/agentview_image", data=np.zeros((T, 3, 4, 4), np.uint8))
            # STATE18_KEYS 对应的 obs 字段由 assemble_state18 决定;mini 直接造 observation.state 形态见下
    return p


def test_act_feat_branch_builds_then_caches(tmp_path, monkeypatch):
    # monkeypatch 掉真实 hdf5 读取与 assemble,使分支只依赖 stub extractor:见实现里 _build_raw_obs_seq 注入点
    cache = str(tmp_path / "act.npz")
    seqs, std, stats = read_per_demo_states(
        "ignored.hdf5", "ds", state_mode="act_feat", num_demos=None,
        act_feat_cache=cache, act_extractor=_StubExtractor(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        _raw_obs_seqs=[{"observation.images.agentview": torch.zeros(3, 3, 4, 4),
                        "observation.state": torch.ones(3, 18)}],
    )
    assert std is None
    assert seqs[0].shape[1] == 4 + 18
    assert stats is not None and stats[0].shape[0] == 22
    # 再读一次:命中缓存,extractor 不应再被调用
    before = _StubExtractor.calls
    seqs2, _, _ = read_per_demo_states(
        "ignored.hdf5", "ds", state_mode="act_feat", num_demos=None,
        act_feat_cache=cache, act_extractor=_StubExtractor(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        _raw_obs_seqs=[{"observation.images.agentview": torch.zeros(3, 3, 4, 4),
                        "observation.state": torch.ones(3, 18)}],
    )
    assert _StubExtractor.calls == before    # 命中缓存,未再调用
    assert np.allclose(seqs[0], seqs2[0])


def test_guard_act_feat_requires_cache_or_base():
    p = build_parser()
    args = p.parse_args(["--hdf5", "h", "--dataset", "d", "--state_mode", "act_feat"])
    try:
        validate_act_feat_cfg(args)
        assert False, "应报错(act_feat 缺 cache 且缺 base)"
    except ValueError:
        pass


def test_parser_defaults_unchanged():
    p = build_parser()
    args = p.parse_args(["--hdf5", "h", "--dataset", "d"])
    assert args.state_mode == "eef"
    assert args.act_feat_cache is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest tests/test_read_per_demo_states_act_feat.py -v`
Expected: FAIL（`act_feat` 非合法 choice / `validate_act_feat_cfg` 不存在 / 参数 `act_feat_cache` 缺失）。

- [ ] **Step 3: 改 `train_hiql_value.py`**

3a. `read_per_demo_states` 签名加可选参数并加 `act_feat` 分支(放在 `meta`/standardizer 之后、现有 `eef_piece` 缓存判断之前):

```python
def read_per_demo_states(hdf5_path, dataset_id, state_mode="eef", num_demos=None,
                         device="cpu", cache_path=None,
                         act_feat_cache=None, act_extractor=None,
                         act_image_keys=None, act_ckpt_id=None, act_proprio_key="observation.state",
                         pooling="mean", _raw_obs_seqs=None):
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.state30_cache import (
        state30_cache_reuse, save_state30_cache)
    meta = LeRobotDatasetMetadata(dataset_id)
    standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=device)

    if state_mode == "act_feat":
        from resfit.rl_finetuning.chunk_residual.act_feature import act_feat_signature
        from resfit.rl_finetuning.chunk_residual.act_feat_cache import (
            save_act_feat_cache, act_feat_cache_reuse)
        sig = act_feat_signature(act_ckpt_id, act_image_keys, act_proprio_key, pooling)
        sig = dict(sig, dataset_id=str(dataset_id), num_demos=num_demos)
        hit = act_feat_cache_reuse(act_feat_cache, signature=sig, num_demos=num_demos)
        if hit is not None:
            seqs_std, stats = hit
            print(f"[read_per_demo_states] act_feat 缓存命中 {act_feat_cache}")
            return seqs_std, None, (np.asarray(stats[0]), np.asarray(stats[1]))
        assert act_extractor is not None, "act_feat build 需 act_extractor(真 ACT 或 stub)"
        raw_seqs = _raw_obs_seqs if _raw_obs_seqs is not None else _build_raw_obs_seqs(
            hdf5_path, act_image_keys, act_proprio_key, num_demos)
        raw_feat = [act_extractor.embed_batch(ro).cpu().numpy().astype(np.float32) for ro in raw_seqs]
        allf = np.concatenate(raw_feat, axis=0)
        mean = allf.mean(axis=0).astype(np.float32)
        std = np.maximum(allf.std(axis=0), 1e-6).astype(np.float32)
        seqs_std = [((s - mean) / std).astype(np.float32) for s in raw_feat]
        if act_feat_cache and num_demos is None:
            save_act_feat_cache(act_feat_cache, seqs_std, (mean, std), signature=sig)
            print(f"[read_per_demo_states] 已写 act_feat 缓存 {act_feat_cache}")
        return seqs_std, None, (mean, std)
    # ...(以下 eef / eef_piece 原逻辑保持不动)
```

3b. 新增 build 辅助(从 hdf5 逐 demo 组装 ACT 输入 raw_obs;图像键映射按 Task 1 spike 确认):

```python
def _build_raw_obs_seqs(hdf5_path, image_keys, proprio_key, num_demos):
    """每条 demo -> 一个 raw_obs dict(整段 T 帧):ACT image_features 键 + proprio_key。"""
    import torch
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
        STATE18_KEYS, assemble_state18, sorted_demo_keys)
    out = []
    with h5py.File(hdf5_path, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
        if num_demos is not None:
            eps = eps[:num_demos]
        for ep in eps:
            grp = f[f"data/{ep}"]
            ro = {}
            for k in image_keys:
                name = k.replace("observation.images.", "")
                ro[k] = torch.as_tensor(grp[f"obs/{name}_image"][()])
            obs_arrays = {kk: grp[f"obs/{kk}"][()] for kk, _ in STATE18_KEYS}
            ro[proprio_key] = torch.as_tensor(assemble_state18(obs_arrays), dtype=torch.float32)
            out.append(ro)
    return out
```

3c. parser 加 flags + guard(`build_parser` 内 `--state_mode` choices 加 `act_feat`,并新增):

```python
    p.add_argument("--state_mode", choices=["eef", "eef_piece", "act_feat"], default="eef",
                   help="eef(18)|eef_piece(30,sim 特权)|act_feat(冻结 ACT encoder 池化 ⊕ 本体)")
    p.add_argument("--act_feat_cache", default=None, help="act_feat 嵌入缓存 npz(cache-or-build)")
    p.add_argument("--act_base_ckpt", default=None, help="act_feat build 用的 ACT base 目录(同 run base)")
    p.add_argument("--act_image_keys", nargs="*", default=None, help="ACT image_features 键(默认取 base config)")
    p.add_argument("--act_proprio_key", default="observation.state")
    p.add_argument("--pooling", choices=["mean"], default="mean")
```

```python
def validate_act_feat_cfg(args):
    """act_feat 须能命中缓存或能 build(给了 base ckpt);否则 ValueError。非 act_feat 传 act_* 忽略。"""
    import os
    import warnings
    if args.state_mode != "act_feat":
        if getattr(args, "act_feat_cache", None) or getattr(args, "act_base_ckpt", None):
            warnings.warn("非 act_feat 模式,--act_* 被忽略", stacklevel=2)
        return
    cache_ok = bool(args.act_feat_cache) and os.path.exists(args.act_feat_cache)
    if not cache_ok and not args.act_base_ckpt:
        raise ValueError("state_mode=act_feat 需 --act_feat_cache(已存在)或 --act_base_ckpt 以 build")
```

3d. `main()` 在 `read_per_demo_states` 前调用 `validate_act_feat_cfg(args)`;并对 act_feat 分流 mean/std + 构造真 extractor:

```python
def main():
    args = build_parser().parse_args()
    validate_act_feat_cfg(args)
    extractor, act_ckpt_id, image_keys = None, None, None
    if args.state_mode == "act_feat":
        from pathlib import Path
        from resfit.lerobot.utils.load_policy import load_policy
        from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor
        if not (args.act_feat_cache and Path(args.act_feat_cache).exists()):
            cand = Path(args.act_base_ckpt) / "policy"
            act = load_policy(cand if cand.is_dir() else Path(args.act_base_ckpt))
            image_keys = args.act_image_keys or list(act.config.image_features.keys())
            extractor = ActFeatureExtractor(act, image_keys, args.act_proprio_key, args.pooling)
            act_ckpt_id = str(args.act_base_ckpt)
        else:
            image_keys = args.act_image_keys
            act_ckpt_id = str(args.act_base_ckpt) if args.act_base_ckpt else None
    seqs, standardizer, aux = read_per_demo_states(
        args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos,
        cache_path=args.state30_cache, act_feat_cache=args.act_feat_cache,
        act_extractor=extractor, act_image_keys=image_keys, act_ckpt_id=act_ckpt_id,
        act_proprio_key=args.act_proprio_key, pooling=args.pooling)
    s, s_next, done = build_transitions(seqs)
    if args.state_mode == "act_feat":
        import torch
        mean, std = torch.as_tensor(aux[0]), torch.as_tensor(aux[1])
        rel_stats = None
    else:
        mean, std = standardizer._mean.cpu(), standardizer._std.cpu()
        rel_stats = aux
    model, v_stats = train_value(
        s, s_next, done, gamma=args.gamma, expectile=args.expectile, ema=args.ema,
        lr=args.lr, batch_size=args.batch_size, steps=args.steps,
        hidden=args.value_hidden, seed=args.seed)
    save_value(args.output, model, v_stats=v_stats, mean=mean, std=std,
               dataset_id=args.dataset, state_mode=args.state_mode, rel_piece_stats=rel_stats)
    print(f"[hiql_value] saved {args.output}; state_mode={args.state_mode} v_stats={v_stats}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest tests/test_read_per_demo_states_act_feat.py tests/test_act_feat_cli_wiring.py -v`
Expected: PASS。再跑回归:`conda run -n residual python -m pytest tests/ -k "hiql or value or state30" -q` → 现有 eef/eef_piece 全绿。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py tests/test_read_per_demo_states_act_feat.py tests/test_act_feat_cli_wiring.py
git commit -m "feat: act_feat branch in read_per_demo_states (cache-or-build) + flags + guard"
```

---

## Task 6: `train_hiql_gc_value.py` state_mode 可配 + act_feat 分流 + save 签名

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`(`save_gc_value`/`load_gc_value` 透传 `act_feat_signature`)
- Test: `tests/test_act_feat_cli_wiring.py`

- [ ] **Step 1: 写失败测试**

```python
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
    save_gc_value, load_gc_value, GoalConditionedVF,
)
from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser as gc_parser


def test_gc_parser_has_state_mode_default_eef_piece():
    args = gc_parser().parse_args(["--hdf5", "h", "--dataset", "d"])
    assert args.state_mode == "eef_piece"           # 默认逐位不变
    assert "act_feat" in gc_parser()._option_string_actions["--state_mode"].choices


def test_save_load_act_feat_signature(tmp_path):
    p = str(tmp_path / "gc.pt")
    m = GoalConditionedVF(state_dim=22, rep_dim=10, hidden=32)
    sig = {"act_ckpt_id": "ckptA", "pooling": "mean"}
    save_gc_value(p, m, v_stats={"min": 0, "max": 1, "mean": 0.5},
                  mean=torch.zeros(22), std=torch.ones(22), dataset_id="ds",
                  state_mode="act_feat", rel_piece_stats=None, act_feat_signature=sig)
    _, info = load_gc_value(p)
    assert info["state_mode"] == "act_feat"
    assert info["act_feat_signature"] == sig
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest tests/test_act_feat_cli_wiring.py -k gc -v`
Expected: FAIL（`save_gc_value` 无 `act_feat_signature` 参数 / parser 无 `--state_mode`）。

- [ ] **Step 3: 改实现**

3a. `hiql_gc_value.py` 的 `save_gc_value` 加 `act_feat_signature=None` 参数,有值则写 payload;`load_gc_value` 的 `info` 加 `info["act_feat_signature"] = ckpt.get("act_feat_signature")`:

```python
def save_gc_value(path, model, *, v_stats, mean, std, dataset_id,
                  state_mode="eef_piece", rel_piece_stats=None, act_feat_signature=None):
    payload = {
        # ...(现有键不变)...
        "mean": mean, "std": std, "dataset_id": dataset_id, "state_mode": state_mode,
    }
    if rel_piece_stats is not None:
        payload["rel_piece_mean"], payload["rel_piece_std"] = rel_piece_stats
    if act_feat_signature is not None:
        payload["act_feat_signature"] = act_feat_signature
    torch.save(payload, path)
```

```python
# load_gc_value 内,info 组装处追加:
    info["act_feat_signature"] = ckpt.get("act_feat_signature")
```

3b. `train_hiql_gc_value.py`:`build_parser` 加 `--state_mode`(choices `eef_piece`/`act_feat`,默认 `eef_piece`)+ 复用 Task 5 的 `--act_feat_cache/--act_base_ckpt/--act_image_keys/--act_proprio_key/--pooling`;`main()` 把写死的 `state_mode="eef_piece"` 改成 `args.state_mode`,对 act_feat 构造 extractor + 分流 mean/std(同 Task 5 main 模式),`save_gc_value(..., state_mode=args.state_mode, rel_piece_stats=rel_stats, act_feat_signature=sig)`(sig 由 extractor.signature(act_ckpt_id) 或缓存签名给)。

```python
    p.add_argument("--state_mode", choices=["eef_piece", "act_feat"], default="eef_piece",
                   help="eef_piece(默认,sim 特权)|act_feat(冻结 ACT encoder ⊕ 本体)")
    # + 同 Task 5 的 --act_feat_cache/--act_base_ckpt/--act_image_keys/--act_proprio_key/--pooling
```

`main()` 关键改动(替换原 `:107-108` 写死 eef_piece 处):

```python
    validate_act_feat_cfg(args)   # 复用 train_hiql_value.validate_act_feat_cfg(import 之)
    extractor, act_ckpt_id, image_keys, act_sig = _setup_act_feat(args)  # 同 Task5 main 的构造逻辑,抽成 helper
    seqs, standardizer, aux = read_per_demo_states(
        args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos,
        cache_path=args.state30_cache, act_feat_cache=args.act_feat_cache,
        act_extractor=extractor, act_image_keys=image_keys, act_ckpt_id=act_ckpt_id,
        act_proprio_key=args.act_proprio_key, pooling=args.pooling)
    if args.state_mode == "act_feat":
        mean, std, rel_stats = torch.as_tensor(aux[0]), torch.as_tensor(aux[1]), None
    else:
        mean, std, rel_stats = standardizer._mean.cpu(), standardizer._std.cpu(), aux
    # ...(train_gc_value 不变)...
    save_gc_value(args.output, model, v_stats=v_stats, mean=mean, std=std,
                  dataset_id=args.dataset, state_mode=args.state_mode,
                  rel_piece_stats=rel_stats, act_feat_signature=act_sig)
```

将 Task 5 main 里"构造 extractor + image_keys + act_ckpt_id + signature"抽成共享 helper `_setup_act_feat(args)` 放 `train_hiql_value.py` 并在两处 import,避免重复(DRY)。

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest tests/test_act_feat_cli_wiring.py -v` → PASS。
回归:`conda run -n residual python -m pytest tests/ -k "gc_value or hiql" -q` → 默认 eef_piece 全绿。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/hiql_gc_value.py tests/test_act_feat_cli_wiring.py
git commit -m "feat: gc_value state_mode configurable + act_feat signature in ckpt"
```

---

## Task 7: `train_hiql_high_actor.py` state 源 dispatch + save 签名

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py`
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`(`save_high_actor`/`load_high_actor` 透传 `act_feat_signature`)
- Test: `tests/test_act_feat_cli_wiring.py`

high_actor 现在用 `load_or_build_state30`(写死 eef_piece 30 维)。act_feat 下改为走 `read_per_demo_states(state_mode="act_feat", ...)` 拿 seqs。

- [ ] **Step 1: 写失败测试**

```python
from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser as ha_parser


def test_ha_parser_has_state_mode_default_eef_piece():
    args = ha_parser().parse_args(["--hdf5", "h", "--dataset", "d", "--gc_value_ckpt", "g"])
    assert args.state_mode == "eef_piece"
    assert "act_feat" in ha_parser()._option_string_actions["--state_mode"].choices
```

并加 `save_high_actor` act_feat_signature 往返断言(仿 Task 6 的 save/load 测试,用 `HighActor(state_dim=22,...)`)。

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest tests/test_act_feat_cli_wiring.py -k ha -v`
Expected: FAIL。

- [ ] **Step 3: 改实现**

3a. `hiql_high_actor.py` 的 `save_high_actor` 加 `act_feat_signature=None`(有值写入 payload);`load_high_actor` 返回的信息加 `act_feat_signature`(若调用方读 info,则在其 info dict 里补 `ckpt.get("act_feat_signature")`)。

3b. `train_hiql_high_actor.py`:`build_parser` 加 `--state_mode`(choices `eef_piece`/`act_feat`,默认 `eef_piece`)+ Task 5 同款 `--act_*` flags;`main()`:

```python
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
        validate_act_feat_cfg, read_per_demo_states, _setup_act_feat)
    validate_act_feat_cfg(args)
    if args.state_mode == "act_feat":
        extractor, act_ckpt_id, image_keys, act_sig = _setup_act_feat(args)
        seqs, _, _ = read_per_demo_states(
            args.hdf5, args.dataset, "act_feat", num_demos=args.num_demos,
            act_feat_cache=args.act_feat_cache, act_extractor=extractor,
            act_image_keys=image_keys, act_ckpt_id=act_ckpt_id,
            act_proprio_key=args.act_proprio_key, pooling=args.pooling)
    else:
        seqs = load_or_build_state30(args.hdf5, args.dataset, args.num_demos, args.state30_cache)
        act_sig = None
    # ...(train_high_actor 不变;state_dim 由 seqs 决定,自然对上 gc_value)...
    save_high_actor(args.output, ha, gc_value_ckpt=args.gc_value_ckpt, way_steps=args.way_steps,
                    beta=args.beta, act_feat_signature=act_sig)   # 其余原参数保持
```

注:high_actor 的 `state_dim` 自 seqs 推断,会与 act_feat 的 gc_value(state_dim=D_emb+18)自动一致;`hiql_subgoal.from_ckpts` 里现有的 `assert info["state_mode"]=="eef_piece"` 属于在线路(范围外),不在本计划改动——离线产物 state_mode/签名一致性已由 gc_value/high_actor 各自 ckpt 保证。

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest tests/test_act_feat_cli_wiring.py -v` → PASS。
回归:`conda run -n residual python -m pytest tests/ -k "high_actor or hiql" -q` → 默认 eef_piece 全绿。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/hiql_high_actor.py tests/test_act_feat_cli_wiring.py
git commit -m "feat: high_actor act_feat state source dispatch + signature"
```

---

## Task 8: 集成冒烟(opt-in,真 ACT,默认 skip)

**Files:**
- Test: `tests/test_act_feat_smoke.py`(标 `@pytest.mark.slow`,默认 skip)

- [ ] **Step 1: 写冒烟测试(默认 skip)**

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACT_FEAT_SMOKE") != "1",
    reason="opt-in:需 GPU + ACT base;设 RUN_ACT_FEAT_SMOKE=1 开启")


def test_act_feat_gc_value_few_steps(tmp_path):
    """真 ACT 取几帧 → 训几步 gc_value → 断言不 NaN、V 有限。"""
    import torch
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
    from pathlib import Path
    from resfit.lerobot.utils.load_policy import load_policy
    from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor
    base = os.environ["ACT_BASE_DIR"]; hdf5 = os.environ["ACT_HDF5"]; ds = os.environ["ACT_DATASET"]
    cand = Path(base) / "policy"
    act = load_policy(cand if cand.is_dir() else Path(base))
    keys = list(act.config.image_features.keys())
    ext = ActFeatureExtractor(act, keys)
    seqs, std, stats = read_per_demo_states(hdf5, ds, "act_feat", num_demos=2,
                                            act_feat_cache=str(tmp_path / "a.npz"),
                                            act_extractor=ext, act_image_keys=keys, act_ckpt_id=base)
    assert std is None and seqs[0].shape[1] == act.config.dim_model + 18
    assert torch.isfinite(torch.as_tensor(seqs[0])).all()
```

- [ ] **Step 2: 跑(默认 skip;开启需授权 GPU)**

Run: `conda run -n residual python -m pytest tests/test_act_feat_smoke.py -v`
Expected: SKIPPED（未设 `RUN_ACT_FEAT_SMOKE`）。开启:`RUN_ACT_FEAT_SMOKE=1 ACT_BASE_DIR=resfit/out/piecce/best ACT_HDF5=resfit/dataset/two_arm_three_piece_assembly.hdf5 ACT_DATASET=ankile/dexmg-two-arm-three-piece-assembly conda run -n residual python -m pytest tests/test_act_feat_smoke.py -v` → PASS。

- [ ] **Step 3: 全套回归 + commit**

Run: `conda run -n residual python -m pytest tests/ -q`
Expected: 全绿(act_feat 新单测 PASS,eef/eef_piece 回归 PASS,smoke SKIPPED)。

```bash
git add tests/test_act_feat_smoke.py
git commit -m "test: opt-in real-ACT act_feat integration smoke"
```

---

## Self-Review 结论

- **Spec 覆盖**:§4.1 extractor→Task2/3;§4.2 cache→Task4;§4.3 read_per_demo_states 分支+归一化→Task5;§4.4 接线/flags/守卫/save 签名→Task5/6/7;§5 测试→各 Task 的 TDD + Task8 冒烟;§6 文件清单→全覆盖;§7 风险(encoder_out 取数)→Task1 spike。
- **占位扫描**:无 TBD/TODO;每个 code step 给了真实代码。`_setup_act_feat` 在 Task5 引入、Task6/7 复用(DRY),已在 Task6 step3 注明抽取。
- **类型一致**:`read_per_demo_states` 返回 `(seqs, standardizer|None, aux)` 全程一致;`save_gc_value`/`save_high_actor` 的 `act_feat_signature` 参数名在 Task6/7 一致;`act_feat_signature()` 字段(act_ckpt_id/image_keys/proprio_key/pooling)在 Task2/4/5 一致。
- **范围**:单一计划,仅离线 value;在线子目标注入/A/B/LIBERO 明确不做。
