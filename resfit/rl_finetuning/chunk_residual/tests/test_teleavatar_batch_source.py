"""Task 16:teleavatar 批量帧读取器(torchcodec 整段解码,替逐帧 ds[i] 随机访问,~17x)。

路径解析是纯逻辑,合成测。批量解码需真 mp4,用本机 block_fail 数据,缺则 skip。
"""
import os

import numpy as np
import pytest

from resfit.rl_finetuning.chunk_residual.teleavatar_batch_source import (
    list_teleavatar_episodes, teleavatar_parquet_path, teleavatar_video_path,
)

BLOCK_FAIL = "/mnt/mnt/data/domains_rise/block/block_fail"


# ---- 路径解析(纯逻辑)----

def test_parquet_path_chunk_layout():
    p = teleavatar_parquet_path("/root", 5)
    assert p == "/root/data/chunk-000/episode_000005.parquet"


def test_parquet_path_crosses_chunk_boundary():
    p = teleavatar_parquet_path("/root", 1003)   # chunks_size=1000 → chunk-001
    assert p == "/root/data/chunk-001/episode_001003.parquet"


def test_video_path_uses_full_camera_key_as_subdir():
    p = teleavatar_video_path("/root", 7, "observation.images.top_head")
    assert p == "/root/videos/chunk-000/observation.images.top_head/episode_000007.mp4"


def test_list_episodes_parses_indices_from_parquet_names(tmp_path):
    d = tmp_path / "data" / "chunk-000"
    d.mkdir(parents=True)
    for idx in (2, 0, 10):                        # 乱序写入,验证排序
        (d / f"episode_{idx:06d}.parquet").write_bytes(b"")
    assert list_teleavatar_episodes(str(tmp_path)) == [0, 2, 10]


# ---- 批量解码(真数据)----

_HAS_DATA = os.path.isdir(os.path.join(BLOCK_FAIL, "meta"))


@pytest.mark.skipif(not _HAS_DATA, reason="block_fail 数据不在本机")
def test_batched_read_shapes_and_alignment():
    from resfit.rl_finetuning.chunk_residual.teleavatar_batch_source import (
        read_teleavatar_episode_batched)
    cams = ["observation.images.top_head", "observation.images.hand_left",
            "observation.images.hand_right"]
    ep = list_teleavatar_episodes(BLOCK_FAIL)[0]
    out = read_teleavatar_episode_batched(BLOCK_FAIL, ep, cams)
    T = out["state"].shape[0]
    assert out["state"].shape[1] == 16
    assert out["state"].dtype == np.float32
    for c in cams:
        assert out["images"][c].shape[0] == T          # 帧数与 parquet 行数对齐
        assert out["images"][c].shape[1] == 3          # CHW
        assert str(out["images"][c].dtype) == "torch.uint8"


@pytest.mark.skipif(not _HAS_DATA, reason="block_fail 数据不在本机")
def test_batched_read_is_much_faster_than_perframe():
    """整段批量解码应远快于逐帧随机访问(此处只验证 <50ms/帧,逐帧是 ~220ms)。"""
    import time
    from resfit.rl_finetuning.chunk_residual.teleavatar_batch_source import (
        read_teleavatar_episode_batched)
    cams = ["observation.images.top_head"]
    ep = list_teleavatar_episodes(BLOCK_FAIL)[0]
    t = time.time()
    out = read_teleavatar_episode_batched(BLOCK_FAIL, ep, cams)
    per = (time.time() - t) / out["state"].shape[0]
    assert per < 0.05, f"{per*1000:.0f} ms/帧,批量解码应 <50ms"
