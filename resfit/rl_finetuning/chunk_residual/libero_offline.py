"""LIBERO offline 锚 buffer:从 LeRobot physical-intelligence/libero 直读当前任务的 demo,
产出与在线 add_chunk_transition 同构的 transition。命门:数据集图已预翻正(只 resize 不翻);
reward 末帧+1(不用事件 reward npy);归一化与在线同源。"""
from __future__ import annotations

import glob
import io
import json
import os

import numpy as np

AGENTVIEW_KEY = "observation.images.agentview"
WRIST_KEY = "observation.images.robot0_eye_in_hand"


def libero_task_language(suite: str, task_id: int) -> str:
    """benchmark 任务语言串(demo 按它匹配 LeRobot 数据集)。"""
    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()[suite]()
    return task_suite.get_task(int(task_id)).language.strip()


def find_demo_episodes(lerobot_root: str, language: str) -> list[str]:
    """读 meta/episodes.jsonl,返回 tasks[0]==language 的 episode parquet 路径(按 episode_index 升序)。"""
    ep_path = os.path.join(lerobot_root, "meta", "episodes.jsonl")
    matched = []
    with open(ep_path) as f:
        for line in f:
            rec = json.loads(line)
            tasks = rec.get("tasks", [])
            if tasks and tasks[0].strip() == language.strip():
                matched.append(int(rec["episode_index"]))
    if not matched:
        raise ValueError(f"find_demo_episodes: 数据集 {lerobot_root} 无任务语言 {language!r} 的 episode")
    matched.sort()
    return [os.path.join(lerobot_root, "data", f"chunk-{ei // 1000:03d}",
                         f"episode_{ei:06d}.parquet") for ei in matched]


def _decode_img_col(col) -> np.ndarray:
    """LeRobot v2.0 image 列(每帧 dict{bytes,path} 的编码图,PNG/JPEG 皆可,PIL 自动识别)→ (T,H,W,3) uint8。"""
    from PIL import Image
    out = []
    for cell in col:
        b = cell["bytes"] if isinstance(cell, dict) else cell
        if b is None:
            raise ValueError("_decode_img_col: image 单元 bytes 为 None(数据集可能未完整下载,只有 path)")
        out.append(np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8))
    return np.stack(out, axis=0)


def read_libero_demo(parquet_path: str) -> dict:
    """一条 episode parquet → {state(T,8) f32, action(T,7) f32, agentview(T,256,256,3) u8, wrist(...) u8}。"""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    state = np.stack([np.asarray(x, np.float32) for x in df["state"]], axis=0)
    action = np.stack([np.asarray(x, np.float32) for x in df["actions"]], axis=0)
    return {"state": state, "action": action,
            "agentview": _decode_img_col(df["image"]), "wrist": _decode_img_col(df["wrist_image"])}
