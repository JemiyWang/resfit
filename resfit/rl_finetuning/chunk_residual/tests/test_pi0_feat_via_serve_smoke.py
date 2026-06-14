"""Opt-in 端到端集成 smoke:真 serve_with_feat → build_pi0_feat_cache → train_gc_value。

默认 skip(需设环境变量 PI0_FEAT_VIA_SERVE_SMOKE=1 才运行)。
还需以下环境变量:
  PI0_SERVE_HOST      serve 主机(默认 localhost)
  PI0_SERVE_PORT      serve 端口
  PI0_FEAT_HDF5       源 hdf5 路径
  PI0_FEAT_IMGKEYS    逗号分隔图像 key(如 agentview_image)
  PI0_FEAT_PROPRIO    proprio key(如 state18)
  PI0_FEAT_PROMPT     (可选)prompt 字符串,默认空
"""
import os

import numpy as np
import pytest

RUN = os.environ.get("PI0_FEAT_VIA_SERVE_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN,
    reason="opt-in: 需真 pi05-with-feat serve + hdf5,设 PI0_FEAT_VIA_SERVE_SMOKE=1",
)


def test_end_to_end_serve_to_gc_value(tmp_path):
    """起真 serve_with_feat 后,build 几帧缓存 → 训几步 gc_value,断言 V 有限。"""
    from resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve import (
        _connect,
        build_main,
    )
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value

    host = os.environ.get("PI0_SERVE_HOST", "localhost")
    port = int(os.environ["PI0_SERVE_PORT"])
    hdf5 = os.environ["PI0_FEAT_HDF5"]
    image_keys = os.environ["PI0_FEAT_IMGKEYS"].split(",")
    proprio_key = os.environ["PI0_FEAT_PROPRIO"]
    prompt = os.environ.get("PI0_FEAT_PROMPT", "")

    cache = str(tmp_path / "pi0_feat_smoke.npz")

    # Step 1: 连接 serve 并 build 缓存(只取前 2 条 demo)
    client = _connect(host, port)
    build_main(
        client,
        hdf5=hdf5,
        dataset_id="smoke",
        image_keys=image_keys,
        proprio_key=proprio_key,
        prompt=prompt,
        pooling="last",
        serve_ckpt_id="smoke",
        out_cache=cache,
        num_demos=2,
    )

    # Step 2: 读缓存 -> (seqs, stats, signature)
    seqs, _stats, _sig = load_pi0_feat_cache(cache)
    assert len(seqs) > 0, "build 后缓存应有至少 1 条 demo"

    # Step 3: 构建 gc_data(空 stage_entries,用 demo 末帧作目标)
    stage_entries = [np.empty(0, dtype=np.int64) for _ in seqs]
    data = build_gc_data(seqs, stage_entries)

    # Step 4: 训几步 gc_value,断言 V 统计有限
    _model, v_stats = train_gc_value(data, steps=5)
    assert np.isfinite(v_stats["mean"]), f"V mean 非有限: {v_stats['mean']}"
    assert np.isfinite(v_stats["min"]), f"V min 非有限: {v_stats['min']}"
