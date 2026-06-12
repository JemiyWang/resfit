# 离线 HIQL value 接冻结 pi0 prefix 特征(pi0_feat)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 resfit 离线 HIQL value(gc_value + high_actor)新增 `state_mode="pi0_feat"`:用冻结 torch pi05 的 prefix 池化嵌入 ⊕ 原始本体当 state,默认 `eef`/`eef_piece` 逐位不变。

**Architecture:** 纯逻辑(池化/拼接/标准化/缓存)与模型耦合层分离。`Pi0FeatureExtractor` 用依赖注入持有已加载的 `PI0Pytorch`(测试塞 stub),真权重加载走 `from_checkpoint` classmethod。`read_per_demo_states` 加 `pi0_feat` 分支(注入 extractor),其余训练栈结构不变。

**Tech Stack:** Python / PyTorch / h5py / numpy / pytest;openpi `models_pytorch.pi0_pytorch.PI0Pytorch` + safetensors;conda env `residual`。

参照 spec:`docs/superpowers/specs/2026-06-12-hiql-value-pi0-feature-offline-design.md`。

> **⚠️ 2026-06-12 修订(环境分离,见 spec §3.0)**:实测 `residual` 环境无 `openpi`、openpi torch 栈在 `/mnt/mnt/data/chj/openpi/.venv`,且本机无 torch pi05(仅 JAX `pi05_base`)。故:**特征器/缓存构建移到 openpi 环境的 `build_pi0_feat_cache.py`**(先 JAX→torch 转换);**residual 侧 value 训练只读缓存(cache-required,不 import openpi、不构造 extractor)**。Task 1 是 openpi 环境的转换+spike,跑通后**据真实 API 回填 build 脚本与下列 Task 的细节**(Task 4/6/7 的"进程内 extractor"相应改成"读缓存");Task 2/3/5 的纯逻辑与 CLI 守卫基本不变(守卫改为校验 `--pi0_feat_cache` 存在)。下列任务文本为修订前版本,执行时以本横幅 + spec §3.0/§3.3/§3.4 为准。

跑测命令(全计划统一):
```
conda run -n residual python -m pytest <path>::<test> -v
```

---

## File Structure

- Create `resfit/rl_finetuning/chunk_residual/pi0_feature.py` — `pool_prefix`/`concat_proprio`(纯函数)+ `Pi0FeatureExtractor`(注入 model;`from_checkpoint`;`embed_demo`)+ `pi0_feat_signature`。
- Create `resfit/rl_finetuning/chunk_residual/pi0_feat_cache.py` — `save_pi0_feat_cache`/`pi0_feat_cache_reuse`(镜像 state30_cache)。
- Create `resfit/rl_finetuning/chunk_residual/verify_pi0_prefix_feature.py` — 真 pi05 spike 脚本(Task 1)。
- Modify `train_hiql_value.py` — `read_per_demo_states` 加 `pi0_feat` 分支 + parser flags + `validate_pi0_feat_cfg`。
- Modify `train_hiql_gc_value.py` — `--state_mode`(默认 eef_piece)+ 接 extractor + save 签名。
- Modify `train_hiql_high_actor.py` — state 源 dispatch(pi0_feat)+ parser + save 签名。
- Create tests:`tests/test_pi0_feature.py`、`tests/test_pi0_feat_cache.py`、`tests/test_read_per_demo_states_pi0_feat.py`、`tests/test_pi0_feat_cli_wiring.py`。

---

## Task 1: Spike — 验证冻结 torch pi05 加载 + prefix 出特征(真权重,opt-in)

**目的(spec §6 命门前置):** 钉死"`PI0Pytorch(cfg)` + `safetensors.load_model` → `embed_prefix` → `paligemma_with_expert.forward(inputs_embeds=[prefix,None])` → 池化 → 有限嵌入"这条真实路径,产出后续 extractor 直接复用的代码。不成立则停下回退方案2。

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/verify_pi0_prefix_feature.py`

- [ ] **Step 1: 写 spike 脚本**

```python
"""真 pi05 spike:加载冻结 torch pi05,跑 prefix-only 前向,验证出有限特征。
非测试套件的一部分(需真权重 + GPU)。手动跑:
  conda run -n residual python -m resfit.rl_finetuning.chunk_residual.verify_pi0_prefix_feature \
    --pi0_ckpt <dir with model.safetensors> --device cuda
"""
import argparse, torch, safetensors.torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models import pi0_config  # Pi0Config(pi05=True, ...)


def load_frozen_pi05(ckpt_dir, device):
    cfg = pi0_config.Pi0Config(pi05=True)          # 与转换时同款 variant/action_dim
    model = PI0Pytorch(cfg).to(device).eval()
    safetensors.torch.load_model(model, f"{ckpt_dir}/model.safetensors", device=str(device))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def prefix_feature(model, observation, device):
    images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(
        observation, train=False)
    prefix_embs, pad, att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    att2d = make_att_2d_masks(pad, att)
    pos = torch.cumsum(pad, dim=1) - 1
    att4d = model._prepare_attention_masks_4d(att2d)
    (prefix_out, _), _ = model.paligemma_with_expert.forward(
        attention_mask=att4d, position_ids=pos, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=False)
    last_idx = pad.long().sum(1) - 1
    return prefix_out[torch.arange(prefix_out.shape[0]), last_idx].float(), pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi0_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model = load_frozen_pi05(args.pi0_ckpt, args.device)
    obs = model.config.fake_obs() if hasattr(model.config, "fake_obs") else None
    assert obs is not None, "用 base policy 同款 obs 构造;若 config 无 fake_obs,手工构造一个 Observation"
    feat, pad = prefix_feature(model, obs, args.device)
    print(f"[spike] prefix feature shape={tuple(feat.shape)} finite={bool(torch.isfinite(feat).all())} "
          f"prefix_len={int(pad.sum(1).max())}")
    assert torch.isfinite(feat).all() and feat.ndim == 2


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 手动跑 spike(需真权重),记录结果**

Run: `conda run -n residual python -m resfit.rl_finetuning.chunk_residual.verify_pi0_prefix_feature --pi0_ckpt <ckpt_dir> --device cuda`
Expected: 打印 `prefix feature shape=(1, <width>) finite=True ...`,无异常。
- 若 `Pi0Config`/`fake_obs`/`_preprocess_observation` 的真实签名与此处不符,**就地把脚本改到能跑通**(这正是 spike 的目的:钉死真实 API)。把最终可跑的 `load_frozen_pi05` / `prefix_feature` 两函数作为 Task 2 `Pi0FeatureExtractor.from_checkpoint` / `embed_demo` 的实现底稿。
- 若加载/前向根本不成立(权重缺失、转换失败)→ **停止本计划**,回到 brainstorm 方案2(JAX 特征器)。

- [ ] **Step 3: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/verify_pi0_prefix_feature.py
git commit -m "spike: verify frozen torch pi05 prefix feature extraction"
```

---

## Task 2: `Pi0FeatureExtractor` 纯逻辑 + 注入式骨架

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/pi0_feature.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feature.py`

- [ ] **Step 1: 写失败测试(纯函数 pool/concat + stub extractor)**

```python
import numpy as np
import pytest
import torch
from resfit.rl_finetuning.chunk_residual.pi0_feature import (
    pool_prefix, concat_proprio, Pi0FeatureExtractor, pi0_feat_signature,
)


def test_pool_prefix_last_picks_last_valid_token():
    # B=2,L=3,D=2;第0条有效长度2,第1条有效长度3
    out = torch.tensor([[[1., 1], [2, 2], [9, 9]], [[3, 3], [4, 4], [5, 5]]])
    pad = torch.tensor([[1, 1, 0], [1, 1, 1]])
    got = pool_prefix(out, pad, pooling="last")
    assert torch.allclose(got, torch.tensor([[2., 2], [5, 5]]))


def test_pool_prefix_mean_ignores_padding():
    out = torch.tensor([[[1., 1], [3, 3], [9, 9]]])
    pad = torch.tensor([[1, 1, 0]])
    got = pool_prefix(out, pad, pooling="mean")
    assert torch.allclose(got, torch.tensor([[2., 2]]))      # (1+3)/2,padding 不计


def test_pool_prefix_unknown_raises():
    with pytest.raises(ValueError):
        pool_prefix(torch.zeros(1, 1, 2), torch.ones(1, 1), pooling="bogus")


def test_concat_proprio_appends_at_tail_float32():
    pooled = torch.zeros(2, 4)
    proprio = torch.ones(2, 3)
    got = concat_proprio(pooled, proprio)
    assert got.shape == (2, 7) and got.dtype == torch.float32
    assert torch.all(got[:, 4:] == 1.0)


class _StubModel:
    """假 pi05:embed_demo 用的最小接口。给定 T 帧 → 返回 (T, width) 末token特征。"""
    width = 5
    def prefix_pool(self, images, proprio):           # extractor 调它拿 (T, width)
        T = proprio.shape[0]
        return torch.arange(T * self.width, dtype=torch.float32).reshape(T, self.width)


def test_embed_demo_concats_pool_and_proprio_raw():
    ex = Pi0FeatureExtractor(model=_StubModel(), image_keys=["agentview_rgb"],
                             proprio_key="ee_states", pooling="last", prompt="do it")
    imgs = {"agentview_rgb": np.zeros((3, 4, 4, 3), np.uint8)}
    proprio = np.ones((3, 2), np.float32)
    out = ex.embed_demo(imgs, proprio)                # (T, width+proprio)
    assert out.shape == (3, 5 + 2)
    assert np.allclose(out[:, 5:], 1.0)               # proprio 原样拼尾
    assert ex.feature_dim == 7


def test_signature_changes_with_config():
    s1 = pi0_feat_signature("ds", None, image_keys=["a"], proprio_key="p",
                            pooling="last", prompt="x", ckpt_id="ck1")
    s2 = pi0_feat_signature("ds", None, image_keys=["a"], proprio_key="p",
                            pooling="mean", prompt="x", ckpt_id="ck1")
    assert s1 != s2
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feature.py -v`
Expected: FAIL(`ModuleNotFoundError: pi0_feature`)。

- [ ] **Step 3: 实现 `pi0_feature.py`**

```python
"""冻结 pi05 prefix 特征器(state_mode=pi0_feat 的 state 源)。
纯逻辑(pool_prefix/concat_proprio/signature)可单测;模型耦合走注入 + from_checkpoint。
真实加载/前向底稿见 verify_pi0_prefix_feature.py(Task 1 spike)。"""
import json
import numpy as np
import torch


def pool_prefix(prefix_out, pad_masks, pooling="last"):
    """prefix_out [B,L,D] + pad_masks [B,L] → [B,D]。last=末有效 token;mean=有效均值。"""
    if pooling == "last":
        last_idx = pad_masks.long().sum(dim=1) - 1
        return prefix_out[torch.arange(prefix_out.shape[0], device=prefix_out.device), last_idx]
    if pooling == "mean":
        m = pad_masks.to(prefix_out.dtype).unsqueeze(-1)
        return (prefix_out * m).sum(1) / m.sum(1).clamp_min(1.0)
    raise ValueError(f"unknown pooling {pooling!r}")


def concat_proprio(pooled, proprio):
    """[B,D_emb] ⊕ [B,D_proprio] → [B,D_emb+D_proprio] float32。"""
    return torch.cat([pooled.float(), proprio.float()], dim=-1)


def pi0_feat_signature(dataset_id, num_demos, *, image_keys, proprio_key, pooling, prompt, ckpt_id):
    """缓存/同源校验签名(可 JSON 序列化、稳定排序)。"""
    return {
        "dataset_id": str(dataset_id), "num_demos": num_demos,
        "image_keys": list(image_keys), "proprio_key": str(proprio_key),
        "pooling": str(pooling), "prompt": str(prompt), "ckpt_id": str(ckpt_id),
    }


class Pi0FeatureExtractor:
    """持有已加载的冻结 pi05(注入);embed_demo 把一条 demo 的图像+本体→ (T, D) 原始嵌入。"""

    def __init__(self, model, *, image_keys, proprio_key, pooling="last", prompt="",
                 ckpt_id="unknown", device="cpu", batch_size=32):
        self.model = model
        self.image_keys = list(image_keys)
        self.proprio_key = proprio_key
        self.pooling = pooling
        self.prompt = prompt
        self.ckpt_id = ckpt_id
        self.device = device
        self.batch_size = batch_size
        self._feat_dim = None

    @classmethod
    def from_checkpoint(cls, ckpt_dir, *, device, image_keys, proprio_key,
                        pooling="last", prompt="", **kw):
        """真权重加载(底稿=verify_pi0_prefix_feature.load_frozen_pi05)。"""
        from resfit.rl_finetuning.chunk_residual.verify_pi0_prefix_feature import load_frozen_pi05
        model = load_frozen_pi05(ckpt_dir, device)
        return cls(model=model, image_keys=image_keys, proprio_key=proprio_key,
                   pooling=pooling, prompt=prompt, ckpt_id=str(ckpt_dir), device=device, **kw)

    @property
    def feature_dim(self):
        if self._feat_dim is None:
            raise RuntimeError("feature_dim 在首次 embed_demo 后可用")
        return self._feat_dim

    def embed_demo(self, images, proprio):
        """images: {key:(T,H,W,3)};proprio:(T,Dp) → (T, D_emb+Dp) 原始(未标准化)np.float32。
        stub model 暴露 prefix_pool(images,proprio)->(T,D_emb);真 model 走 _model_prefix_pool。"""
        proprio_t = torch.as_tensor(np.asarray(proprio), dtype=torch.float32)
        if hasattr(self.model, "prefix_pool"):                  # 测试 stub 路径
            pooled = self.model.prefix_pool(images, proprio_t)
        else:                                                   # 真 pi05 路径
            pooled = self._model_prefix_pool(images, proprio_t)
        out = concat_proprio(pooled, proprio_t).cpu().numpy().astype(np.float32)
        self._feat_dim = out.shape[1]
        return out

    @torch.no_grad()
    def _model_prefix_pool(self, images, proprio_t):
        """真 pi05:逐 batch 构造 Observation → prefix 前向 → pool。
        实现底稿=verify_pi0_prefix_feature.prefix_feature(按 spike 跑通后的真实 obs 构造补全)。"""
        from resfit.rl_finetuning.chunk_residual.verify_pi0_prefix_feature import build_observation, prefix_feature
        feats = []
        T = proprio_t.shape[0]
        for i in range(0, T, self.batch_size):
            obs = build_observation(self.model, images, proprio_t, i, i + self.batch_size,
                                    self.image_keys, self.prompt, self.device)
            f, _ = prefix_feature(self.model, obs, self.device)
            feats.append(pool_prefix(*f) if isinstance(f, tuple) else f)
        return torch.cat(feats, dim=0)
```

注:`_model_prefix_pool` 依赖 spike 里补全的 `build_observation`(把图像+本体切片造成 base policy 同款 `Observation`)与 `prefix_feature`。Task 1 跑通后把这两个函数补进 `verify_pi0_prefix_feature.py`。单测不触达此路径(走 stub 的 `prefix_pool`)。

- [ ] **Step 4: Run → 通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feature.py -v`
Expected: PASS(6 项)。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/pi0_feature.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feature.py
git commit -m "feat: Pi0FeatureExtractor pool/concat + injection skeleton (pi0_feat state source)"
```

---

## Task 3: `pi0_feat_cache.py`(镜像 state30_cache)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/pi0_feat_cache.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest
from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import (
    save_pi0_feat_cache, pi0_feat_cache_reuse,
)

SIG = {"dataset_id": "ds", "num_demos": None, "image_keys": ["a"], "proprio_key": "p",
       "pooling": "last", "prompt": "x", "ckpt_id": "ck1"}


def _seqs():
    return [np.arange(6, dtype=np.float32).reshape(3, 2), np.ones((2, 2), np.float32)]


def test_save_reuse_roundtrip(tmp_path):
    p = str(tmp_path / "feat.npz")
    stats = (np.zeros(2, np.float32), np.ones(2, np.float32))
    save_pi0_feat_cache(p, _seqs(), stats, signature=SIG)
    got = pi0_feat_cache_reuse(p, signature=SIG, num_demos=None)
    assert got is not None
    seqs, (mean, std) = got
    assert len(seqs) == 2 and np.allclose(seqs[0], _seqs()[0])
    assert np.allclose(mean, 0) and np.allclose(std, 1)


def test_reuse_signature_mismatch_returns_none(tmp_path):
    p = str(tmp_path / "feat.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    bad = dict(SIG, pooling="mean")
    assert pi0_feat_cache_reuse(p, signature=bad, num_demos=None) is None


def test_reuse_partial_num_demos_returns_none(tmp_path):
    p = str(tmp_path / "feat.npz")
    save_pi0_feat_cache(p, _seqs(), (np.zeros(2, np.float32), np.ones(2, np.float32)), signature=SIG)
    assert pi0_feat_cache_reuse(p, signature=SIG, num_demos=1) is None


def test_reuse_missing_file_returns_none(tmp_path):
    assert pi0_feat_cache_reuse(str(tmp_path / "nope.npz"), signature=SIG, num_demos=None) is None
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache.py -v`
Expected: FAIL(`ModuleNotFoundError`)。

- [ ] **Step 3: 实现 `pi0_feat_cache.py`**

```python
"""pi0_feat 嵌入缓存(仿 state30_cache):签名全一致 + 全量才命中。"""
import json
import os
import numpy as np


def save_pi0_feat_cache(path, seqs, feat_stats, *, signature):
    mean, std = feat_stats
    payload = {f"seq_{i}": np.asarray(s, dtype=np.float32) for i, s in enumerate(seqs)}
    payload["n_seqs"] = np.asarray(len(seqs))
    payload["feat_mean"] = np.asarray(mean, dtype=np.float32)
    payload["feat_std"] = np.asarray(std, dtype=np.float32)
    payload["signature"] = np.asarray(json.dumps(signature, sort_keys=True))
    np.savez_compressed(path, **payload)


def pi0_feat_cache_reuse(path, *, signature, num_demos):
    """命中规则:文件在 + 全量(num_demos is None,因 feat mean/std 对全量算)+ 签名逐键一致。"""
    if num_demos is not None or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        if "signature" not in z.files:
            return None
        if str(z["signature"]) != json.dumps(signature, sort_keys=True):
            return None
        n = int(z["n_seqs"])
        seqs = [z[f"seq_{i}"] for i in range(n)]
        feat_stats = (z["feat_mean"], z["feat_std"])
    return seqs, feat_stats
```

- [ ] **Step 4: Run → 通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache.py -v`
Expected: PASS(4 项)。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/pi0_feat_cache.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cache.py
git commit -m "feat: pi0_feat embedding cache (signature-gated, full-only reuse)"
```

---

## Task 4: `read_per_demo_states` 加 `pi0_feat` 分支

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`(`read_per_demo_states`,约 :24-82)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_pi0_feat.py`

- [ ] **Step 1: 写失败测试(合成 hdf5 + stub extractor + cache 命中跳过)**

```python
import h5py
import numpy as np
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states


class _StubExtractor:
    image_keys = ["agentview_rgb"]
    proprio_key = "ee_states"
    ckpt_id = "ck1"; pooling = "last"; prompt = "x"
    def __init__(self): self.calls = 0
    def embed_demo(self, images, proprio):
        self.calls += 1
        T = proprio.shape[0]
        return np.concatenate([np.ones((T, 3), np.float32), np.asarray(proprio, np.float32)], axis=1)


def _make_hdf5(path, demos=2, T=4):
    with h5py.File(path, "w") as f:
        for i in range(demos):
            g = f.create_group(f"data/demo_{i}")
            g.create_dataset("obs/agentview_rgb", data=np.zeros((T, 4, 4, 3), np.uint8))
            g.create_dataset("obs/ee_states", data=np.full((T, 2), float(i + 1), np.float32))


def test_pi0_feat_branch_returns_standardized_seqs_and_stats(tmp_path):
    hdf5 = str(tmp_path / "t.hdf5"); _make_hdf5(hdf5)
    ex = _StubExtractor()
    seqs, standardizer, feat_stats = read_per_demo_states(
        hdf5, dataset_id="ds", state_mode="pi0_feat", extractor=ex,
        cache_path=str(tmp_path / "c.npz"))
    assert len(seqs) == 2 and seqs[0].shape[1] == 3 + 2          # emb3 + proprio2
    mean, std = feat_stats
    assert mean.shape == (5,) and std.shape == (5,)
    allcat = np.concatenate(seqs, axis=0)
    assert np.allclose(allcat.mean(0), 0, atol=1e-5)            # 已标准化
    assert standardizer is None                                 # pi0_feat 不建 StateStandardizer


def test_pi0_feat_cache_hit_skips_extractor(tmp_path):
    hdf5 = str(tmp_path / "t.hdf5"); _make_hdf5(hdf5)
    cache = str(tmp_path / "c.npz")
    ex1 = _StubExtractor()
    read_per_demo_states(hdf5, "ds", "pi0_feat", extractor=ex1, cache_path=cache)
    assert ex1.calls == 2
    ex2 = _StubExtractor()
    read_per_demo_states(hdf5, "ds", "pi0_feat", extractor=ex2, cache_path=cache)
    assert ex2.calls == 0                                       # 命中缓存,未再调 extractor
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_pi0_feat.py -v`
Expected: FAIL(`read_per_demo_states() got an unexpected keyword argument 'extractor'`)。

- [ ] **Step 3: 改 `read_per_demo_states`(加 `extractor=None` 参 + pi0_feat 分支;eef/eef_piece 不动)**

在签名加 `extractor=None`。把现有 `StateStandardizer` 构造**移进** eef/eef_piece 路径之前的保留位置不变;新增 pi0_feat 早返回分支(放在 `meta=LeRobotDatasetMetadata(...)` 之前,避免 LIBERO 无 LeRobot 注册时报错):

```python
def read_per_demo_states(hdf5_path, dataset_id, state_mode="eef", num_demos=None,
                         device="cpu", cache_path=None, extractor=None):
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import sorted_demo_keys
    if state_mode == "pi0_feat":
        from resfit.rl_finetuning.chunk_residual.pi0_feature import pi0_feat_signature
        from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import (
            save_pi0_feat_cache, pi0_feat_cache_reuse)
        assert extractor is not None, "state_mode=pi0_feat 需传 extractor(Pi0FeatureExtractor)"
        sig = pi0_feat_signature(dataset_id, num_demos, image_keys=extractor.image_keys,
                                 proprio_key=extractor.proprio_key, pooling=extractor.pooling,
                                 prompt=extractor.prompt, ckpt_id=extractor.ckpt_id)
        if cache_path is not None:
            hit = pi0_feat_cache_reuse(cache_path, signature=sig, num_demos=num_demos)
            if hit is not None:
                seqs, feat_stats = hit
                return seqs, None, feat_stats
        raw = []
        with h5py.File(hdf5_path, "r") as f:
            eps = sorted_demo_keys(list(f["data"].keys()))
            if num_demos is not None:
                eps = eps[:num_demos]
            for ep in eps:
                grp = f[f"data/{ep}"]
                imgs = {k: grp[f"obs/{k}"][()] for k in extractor.image_keys}
                proprio = grp[f"obs/{extractor.proprio_key}"][()]
                raw.append(extractor.embed_demo(imgs, proprio))
        allcat = np.concatenate(raw, axis=0)
        mean = allcat.mean(0).astype(np.float32)
        std = allcat.std(0).clip(1e-6).astype(np.float32)
        seqs = [((s - mean) / std).astype(np.float32) for s in raw]
        if cache_path is not None and num_demos is None:
            save_pi0_feat_cache(cache_path, seqs, (mean, std), signature=sig)
        return seqs, None, (mean, std)
    # --- 以下 eef/eef_piece 原样不变 ---
    ...
```

(eef/eef_piece 路径里原有 `import numpy as np` 与 `meta=LeRobotDatasetMetadata(...)` 等保持不动;pi0_feat 分支已在其前 return。)

- [ ] **Step 4: Run → 通过(并跑既有 value 测试确认 eef 路未回归)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_pi0_feat.py resfit/rl_finetuning/chunk_residual/tests/ -k "value or state30 or hiql" -v`
Expected: 新 2 项 PASS;既有 value/state30/hiql 用例全绿。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_pi0_feat.py
git commit -m "feat: read_per_demo_states pi0_feat branch (extractor-injected, cached, self-standardized)"
```

---

## Task 5: CLI 接线 + `validate_pi0_feat_cfg` 守卫(train_hiql_value / gc_value / high_actor parser)

**Files:**
- Modify: `train_hiql_value.py`(parser 加 `pi0_feat` 枚举 + `--pi0_*` flags + `validate_pi0_feat_cfg`)
- Modify: `train_hiql_gc_value.py`(parser:`--state_mode` 默认 `eef_piece` + `--pi0_*`)
- Modify: `train_hiql_high_actor.py`(parser:`--state_mode` 默认 `eef_piece` + `--pi0_*`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py`

- [ ] **Step 1: 写失败测试**

```python
import pytest
from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    build_parser as v_parser, validate_pi0_feat_cfg,
)
from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser as gc_parser
from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser as ha_parser


def test_value_parser_state_mode_default_unchanged():
    a = v_parser().parse_args(["--hdf5", "x", "--dataset", "d"])
    assert a.state_mode == "eef"


def test_gc_parser_state_mode_default_eef_piece():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d"])
    assert a.state_mode == "eef_piece"               # 默认逐位等价于写死时


def test_pi0_feat_choice_accepted():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--state_mode", "pi0_feat",
                                "--pi0_ckpt", "ck", "--pi0_image_keys", "agentview_rgb",
                                "--pi0_proprio_key", "ee_states"])
    assert a.state_mode == "pi0_feat" and a.pi0_image_keys == ["agentview_rgb"]


def test_validate_pi0_feat_cfg_requires_ckpt_and_keys():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--state_mode", "pi0_feat"])
    with pytest.raises(ValueError):
        validate_pi0_feat_cfg(a)


def test_validate_pi0_feat_cfg_noop_for_eef_modes():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d"])
    validate_pi0_feat_cfg(a)                          # eef_piece 不报
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py -v`
Expected: FAIL(`cannot import name 'validate_pi0_feat_cfg'` / `pi0_feat` 非法 choice)。

- [ ] **Step 3: 实现**

在 `train_hiql_value.py` 加 `validate_pi0_feat_cfg` 与 parser 项;gc_value/high_actor 复用同一函数 + 加相同 flags。

`train_hiql_value.py`:
```python
def validate_pi0_feat_cfg(args):
    """state_mode=pi0_feat 必须给 --pi0_ckpt 与 --pi0_image_keys/--pi0_proprio_key;
    否则报错。非 pi0_feat 模式不校验(传了 --pi0_* 也忽略)。"""
    if args.state_mode != "pi0_feat":
        return
    missing = [k for k in ("pi0_ckpt", "pi0_image_keys", "pi0_proprio_key")
               if not getattr(args, k, None)]
    if missing:
        raise ValueError(f"--state_mode pi0_feat 需提供 {missing}(冻结 pi05 特征器)")


def _add_pi0_feat_args(p):
    p.add_argument("--pi0_ckpt", default=None, help="冻结 torch pi05 ckpt 目录(仅 pi0_feat)")
    p.add_argument("--pi0_image_keys", type=lambda s: s.split(","), default=None,
                   help="逗号分隔图像 obs 键(仅 pi0_feat),如 agentview_rgb,eye_in_hand_rgb")
    p.add_argument("--pi0_proprio_key", default=None, help="本体 obs 键(仅 pi0_feat)")
    p.add_argument("--pi0_prompt", default="", help="pi05 prompt(仅 pi0_feat)")
    p.add_argument("--pi0_pooling", choices=["last", "mean"], default="last")
    p.add_argument("--pi0_feat_cache", default=None, help="pi0_feat 嵌入缓存 npz(仅 pi0_feat)")
```
并把 value parser 的 `--state_mode choices` 改为 `["eef", "eef_piece", "pi0_feat"]`,调 `_add_pi0_feat_args(p)`。

`train_hiql_gc_value.py`:把写死的 `state_mode="eef_piece"` 改为 parser 项 `--state_mode choices=["eef_piece","pi0_feat"] default="eef_piece"`,并 `from ...train_hiql_value import _add_pi0_feat_args, validate_pi0_feat_cfg` 后 `_add_pi0_feat_args(p)`。

`train_hiql_high_actor.py`:同 gc_value,加 `--state_mode default="eef_piece"` + `_add_pi0_feat_args(p)`。

- [ ] **Step 4: Run → 通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py -v`
Expected: PASS(5 项)。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py
git commit -m "feat: --state_mode pi0_feat CLI flags + validate_pi0_feat_cfg guard (default eef/eef_piece unchanged)"
```

---

## Task 6: gc_value main 接 pi0_feat(构造 extractor + 存 mean/std=feat_stats + 签名)

**Files:**
- Modify: `train_hiql_gc_value.py`(`main()`,约 :103-131)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py`(追加一项,monkeypatch 验证接线)

- [ ] **Step 1: 追加失败测试(用 monkeypatch 替掉 extractor 与 read,验证 main 走 pi0_feat 接线、save 用 feat_stats)**

```python
def test_gc_value_main_pi0_feat_wires_extractor_and_saves_feat_stats(tmp_path, monkeypatch):
    import numpy as np
    import resfit.rl_finetuning.chunk_residual.train_hiql_gc_value as m

    captured = {}
    def fake_read(hdf5, dataset, state_mode, num_demos=None, cache_path=None, extractor=None):
        captured["state_mode"] = state_mode
        captured["extractor"] = extractor
        seqs = [np.zeros((3, 6), np.float32), np.zeros((2, 6), np.float32)]
        return seqs, None, (np.zeros(6, np.float32), np.ones(6, np.float32))
    def fake_from_ckpt(ckpt_dir, **kw):
        return type("E", (), {"image_keys": kw["image_keys"], "proprio_key": kw["proprio_key"],
                              "pooling": kw["pooling"], "prompt": kw["prompt"], "ckpt_id": ckpt_dir})()
    saved = {}
    def fake_save(path, model, **kw):
        saved.update(kw)
    monkeypatch.setattr(m, "read_per_demo_states", fake_read)
    monkeypatch.setattr(m, "Pi0FeatureExtractor", type("P", (), {"from_checkpoint": staticmethod(fake_from_ckpt)}))
    monkeypatch.setattr(m, "save_gc_value", fake_save)
    monkeypatch.setattr(m, "train_gc_value", lambda *a, **k: (object(), {"min": 0, "max": 0, "mean": 0}))

    argv = ["--hdf5", "x", "--dataset", "ds", "--state_mode", "pi0_feat", "--pi0_ckpt", "ck",
            "--pi0_image_keys", "agentview_rgb", "--pi0_proprio_key", "ee_states",
            "--steps", "1", "--output", str(tmp_path / "gc.pt")]
    monkeypatch.setattr("sys.argv", ["prog", *argv])
    m.main()
    assert captured["state_mode"] == "pi0_feat" and captured["extractor"] is not None
    assert saved["state_mode"] == "pi0_feat"
    assert np.allclose(saved["mean"], 0) and np.allclose(saved["std"], 1)   # 存的是 feat_stats
    assert saved.get("extractor_signature") is not None
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest "resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py::test_gc_value_main_pi0_feat_wires_extractor_and_saves_feat_stats" -v`
Expected: FAIL(main 未走 pi0_feat 接线 / save 无 extractor_signature)。

- [ ] **Step 3: 改 gc_value `main()`**

顶部 import 加 `from resfit.rl_finetuning.chunk_residual.pi0_feature import Pi0FeatureExtractor, pi0_feat_signature`。`main()` 改成:

```python
def main():
    args = build_parser().parse_args()
    validate_pi0_feat_cfg(args)
    validate_stage_cache(args.stage_cache, needs_stage=gc_value_needs_stage(args))
    extractor = None
    if args.state_mode == "pi0_feat":
        extractor = Pi0FeatureExtractor.from_checkpoint(
            args.pi0_ckpt, device=args.device if hasattr(args, "device") else "cpu",
            image_keys=args.pi0_image_keys, proprio_key=args.pi0_proprio_key,
            pooling=args.pi0_pooling, prompt=args.pi0_prompt)
    seqs, standardizer, aux_stats = read_per_demo_states(
        args.hdf5, args.dataset, args.state_mode, num_demos=args.num_demos,
        cache_path=(args.pi0_feat_cache if args.state_mode == "pi0_feat" else args.state30_cache),
        extractor=extractor)
    seq_lens = [len(s) for s in seqs]
    stage_entries = stage_entries_aligned(args.hdf5, args.stage_cache, args.num_demos, seq_lens)
    data = build_gc_data(seqs, stage_entries)
    model, v_stats = train_gc_value(data, ...)        # 原参数不变
    if args.state_mode == "pi0_feat":
        mean, std = aux_stats
        sig = pi0_feat_signature(args.dataset, args.num_demos, image_keys=extractor.image_keys,
                                 proprio_key=extractor.proprio_key, pooling=extractor.pooling,
                                 prompt=extractor.prompt, ckpt_id=extractor.ckpt_id)
        save_gc_value(args.output, model, v_stats=v_stats, mean=mean, std=std,
                      dataset_id=args.dataset, state_mode="pi0_feat", rel_piece_stats=None,
                      value_loss_mode=args.value_loss_mode, value_mask_mode=args.value_mask_mode,
                      extractor_signature=sig)
    else:
        # 原 eef/eef_piece 存档分支不变(standardizer._mean/_std + rel_piece)
        ...
```
`save_gc_value` 签名加可选 `extractor_signature=None`,写入 payload(类比 rel_piece_stats 的可选写法)。`load_gc_value` 读出放进 info(缺键回退 None)。

- [ ] **Step 4: Run → 通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py -v`
Expected: PASS(全部)。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py
git commit -m "feat: gc_value main wires pi0_feat (extractor + feat_stats save + signature)"
```

---

## Task 7: high_actor main 接 pi0_feat(state 源 dispatch)

**Files:**
- Modify: `train_hiql_high_actor.py`(`main()`,约 :60-90)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py`(追加一项)

- [ ] **Step 1: 追加失败测试(monkeypatch 验证 pi0_feat 时走 read_per_demo_states+extractor,而非 load_or_build_state30)**

```python
def test_high_actor_main_pi0_feat_uses_extractor_state_source(tmp_path, monkeypatch):
    import numpy as np
    import resfit.rl_finetuning.chunk_residual.train_hiql_high_actor as m
    used = {"state30": False, "read": False}
    def fake_state30(*a, **k): used["state30"] = True; return [np.zeros((3, 6), np.float32)]
    def fake_read(hdf5, dataset, state_mode, num_demos=None, cache_path=None, extractor=None):
        used["read"] = True; return [np.zeros((3, 6), np.float32)], None, (np.zeros(6, np.float32), np.ones(6, np.float32))
    monkeypatch.setattr(m, "load_or_build_state30", fake_state30)
    monkeypatch.setattr(m, "read_per_demo_states", fake_read, raising=False)
    monkeypatch.setattr(m, "Pi0FeatureExtractor",
                        type("P", (), {"from_checkpoint": staticmethod(lambda c, **k: type("E", (), {**k, "ckpt_id": c, "image_keys": k["image_keys"], "proprio_key": k["proprio_key"], "pooling": k["pooling"], "prompt": k["prompt"]})())}), raising=False)
    monkeypatch.setattr(m, "load_gc_value", lambda *a, **k: (type("V", (), {"state_dim": 6, "rep_dim": 10})(), {"dataset_id": "ds", "state_mode": "pi0_feat"}))
    monkeypatch.setattr(m, "build_gc_data", lambda *a, **k: {"states": np.zeros((3, 6), np.float32), "s_idx": np.arange(3)})
    monkeypatch.setattr(m, "train_high_actor", lambda *a, **k: object())
    monkeypatch.setattr(m, "save_high_actor", lambda *a, **k: None)
    argv = ["--hdf5", "x", "--dataset", "ds", "--gc_value_ckpt", "v.pt", "--state_mode", "pi0_feat",
            "--pi0_ckpt", "ck", "--pi0_image_keys", "agentview_rgb", "--pi0_proprio_key", "ee_states",
            "--steps", "1", "--output", str(tmp_path / "ha.pt")]
    monkeypatch.setattr("sys.argv", ["prog", *argv])
    m.main()
    assert used["read"] and not used["state30"]      # pi0_feat 走 extractor 路,不 replay state30
```

- [ ] **Step 2: Run → 失败**

Run: `conda run -n residual python -m pytest "resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py::test_high_actor_main_pi0_feat_uses_extractor_state_source" -v`
Expected: FAIL。

- [ ] **Step 3: 改 high_actor `main()`(state 源 dispatch)**

import 加 `from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states, validate_pi0_feat_cfg` 与 `from resfit.rl_finetuning.chunk_residual.pi0_feature import Pi0FeatureExtractor`。把取 `seqs` 的那段改成:

```python
    validate_pi0_feat_cfg(args)
    if args.state_mode == "pi0_feat":
        extractor = Pi0FeatureExtractor.from_checkpoint(
            args.pi0_ckpt, device="cpu", image_keys=args.pi0_image_keys,
            proprio_key=args.pi0_proprio_key, pooling=args.pi0_pooling, prompt=args.pi0_prompt)
        seqs, _, _ = read_per_demo_states(args.hdf5, args.dataset, "pi0_feat",
                                          num_demos=args.num_demos,
                                          cache_path=args.pi0_feat_cache, extractor=extractor)
    else:
        seqs = load_or_build_state30(args.hdf5, args.dataset, args.num_demos, args.state30_cache)
```
其余(`stage_entries_aligned`/`build_gc_data`/同源 assert/`train_high_actor`/`save_high_actor`)不变;`save_high_actor` 加可选 `extractor_signature` 透传(与 gc_value 一致)。

- [ ] **Step 4: Run → 通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_cli_wiring.py
git commit -m "feat: high_actor main dispatches state source to pi0_feat extractor"
```

---

## Task 8: 全量回归 + 集成冒烟(真 pi05,opt-in)

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_smoke.py`(默认 skip)

- [ ] **Step 1: 写集成冒烟(env gate,真 pi05)**

```python
import os
import numpy as np
import pytest

RUN = os.environ.get("PI0_FEAT_SMOKE") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="opt-in:需真 pi05 + GPU,设 PI0_FEAT_SMOKE=1")


@pytest.mark.parametrize("dataset", ["dexmg", "libero"])
def test_pi0_feat_trains_a_few_steps_no_nan(dataset):
    """dexmg 与 LIBERO 各取几帧 → extractor → 微型 gc_value 训几步,断言 V 有限。
    路径/键由环境变量给:PI0_CKPT / PI0_FEAT_HDF5_<DS> / PI0_FEAT_IMGKEYS_<DS> / PI0_FEAT_PROPRIO_<DS>。"""
    from resfit.rl_finetuning.chunk_residual.pi0_feature import Pi0FeatureExtractor
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    ds = dataset.upper()
    ex = Pi0FeatureExtractor.from_checkpoint(
        os.environ["PI0_CKPT"], device="cuda",
        image_keys=os.environ[f"PI0_FEAT_IMGKEYS_{ds}"].split(","),
        proprio_key=os.environ[f"PI0_FEAT_PROPRIO_{ds}"])
    seqs, _, stats = read_per_demo_states(
        os.environ[f"PI0_FEAT_HDF5_{ds}"], "smoke", "pi0_feat", num_demos=2, extractor=ex)
    data = build_gc_data(seqs, [np.empty(0, np.int64) for _ in seqs])
    model, v_stats = train_gc_value(data, steps=5)
    assert np.isfinite(v_stats["mean"]) and np.isfinite(v_stats["min"])
```

- [ ] **Step 2: Run(默认 skip)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_smoke.py -v`
Expected: 2 SKIPPED(未设 `PI0_FEAT_SMOKE`)。手动 opt-in 跑见 docstring。

- [ ] **Step 3: 全量回归(确认默认 eef/eef_piece 零影响)**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全绿(既有用例 + 本计划新增)。**若任何既有用例变红 → 停下修到逐位等价**。

- [ ] **Step 4: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/tests/test_pi0_feat_smoke.py
git commit -m "test: opt-in pi0_feat integration smoke (dexmg + LIBERO) + full regression green"
```

---

## Self-Review(写完计划自查)

**Spec 覆盖**:§3.1 Extractor→Task1/2;§3.2 缓存→Task3;§3.3 read 分支+归一化→Task4;§3.4 接线/守卫/save 签名→Task5/6/7;§4 测试(stub 单测 + opt-in 冒烟 + 回归)→各 Task Step1 + Task8;§6 前置(spike 先行)→Task1。全覆盖。

**占位符扫描**:无 TBD/TODO;唯一"按 spike 跑通后补全"的是 `_model_prefix_pool`/`build_observation`——这是 Task1 spike 的**显式产出**(真实 obs 构造无法离线臆测),已在 Task1 Step2 明确"把可跑函数补进脚本",非占位逃避。单测不依赖它(走 stub)。

**类型/命名一致**:`Pi0FeatureExtractor(image_keys/proprio_key/pooling/prompt/ckpt_id/embed_demo/from_checkpoint)`、`pi0_feat_signature(...)`、`save_pi0_feat_cache/pi0_feat_cache_reuse`、`read_per_demo_states(..., extractor=)`、`validate_pi0_feat_cfg`、`save_gc_value(..., extractor_signature=)` 在各 Task 间一致。
