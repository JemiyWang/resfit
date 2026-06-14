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
