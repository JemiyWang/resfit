"""act_feat 嵌入缓存(纯 numpy,仿 state30_cache)。签名全等才命中。"""
import json
import os

import numpy as np


def _norm_sig(sig):
    """JSON round-trip 归一化(tuple->list 等),保证存/比两侧类型一致。"""
    return json.loads(json.dumps(sig, sort_keys=True))


def save_act_feat_cache(path, seqs, emb_stats, *, signature, fp16=False):
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


def load_act_feat_cache(path):
    """读缓存 → (seqs[float32], (mean,std), signature_dict)。无签名校验,仅读取。"""
    return _load(path)


def act_feat_cache_reuse(path, *, signature, num_demos):
    """签名全等(含 num_demos)才命中,否则 None。"""
    if not (path and os.path.exists(path)):
        return None
    seqs, stats, sig = _load(path)
    want = _norm_sig(signature)
    want["num_demos"] = num_demos
    if sig != want:
        return None
    return seqs, stats
