"""teleavatar/block 的批量帧读取:torchcodec 整段解码,替 LeRobot 逐帧 ds[i] 随机访问。

逐帧 ds[i](每帧 seek+decode)约 220 ms/帧;整段 get_frames_in_range 约 13 ms/帧(~17x)。
建 ~37 万帧 ψ 缓存时,这把解码从瓶颈变为非瓶颈(变 serve-bound)。

不经 LeRobotDataset(它会因纯本地 repo_id 联网查 hub);直接按 info.json 的路径模板
读 parquet(state/action)+ mp4(视频),自解析 chunk 布局。
"""
from __future__ import annotations

import glob
import os

import numpy as np

CHUNKS_SIZE = 1000   # block info.json 的 chunks_size;跨此值进下一个 chunk 目录


def _chunk(ep: int, chunks_size: int = CHUNKS_SIZE) -> int:
    return ep // chunks_size


def teleavatar_parquet_path(root, ep, chunks_size: int = CHUNKS_SIZE) -> str:
    return os.path.join(
        root, f"data/chunk-{_chunk(ep, chunks_size):03d}/episode_{ep:06d}.parquet")


def teleavatar_video_path(root, ep, cam, chunks_size: int = CHUNKS_SIZE) -> str:
    return os.path.join(
        root, f"videos/chunk-{_chunk(ep, chunks_size):03d}/{cam}/episode_{ep:06d}.mp4")


def list_teleavatar_episodes(root) -> list:
    """从 data/chunk-*/episode_*.parquet 解析出所有 episode_index,升序。"""
    files = glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet"))
    eps = []
    for f in files:
        stem = os.path.basename(f)                       # episode_000015.parquet
        eps.append(int(stem.split("_")[1].split(".")[0]))
    return sorted(eps)


def read_teleavatar_episode_batched(root, ep, cameras,
                                    proprio_key: str = "observation.state",
                                    chunks_size: int = CHUNKS_SIZE) -> dict:
    """整段批量解码一个 episode → {"images": {cam: (T,3,H,W) uint8 tensor}, "state": (T,Dp) f32}。

    cameras 是 LeRobot 键(observation.images.<cam>),即视频子目录名。帧数与 parquet 行数对齐。
    """
    import pandas as pd
    from torchcodec.decoders import VideoDecoder

    df = pd.read_parquet(teleavatar_parquet_path(root, ep, chunks_size))
    T = len(df)
    state = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)
                      for x in df[proprio_key].values]).astype(np.float32)

    images = {}
    for cam in cameras:
        dec = VideoDecoder(teleavatar_video_path(root, ep, cam, chunks_size))
        n = int(dec.metadata.num_frames)
        m = min(n, T)                                    # 容忍视频/parquet 末尾差 1 帧
        images[cam] = dec.get_frames_in_range(start=0, stop=m).data   # (m,3,H,W) uint8
    # 对齐到公共长度(取 min),防某相机少 1 帧
    T_common = min([state.shape[0]] + [images[c].shape[0] for c in cameras])
    if T_common != state.shape[0]:
        state = state[:T_common]
    for c in cameras:
        if images[c].shape[0] != T_common:
            images[c] = images[c][:T_common]
    return {"images": images, "state": state}
