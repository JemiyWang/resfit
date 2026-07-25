"""想象起点采样(正式版)：从 Teleavatar 真实 episode 采 4 帧窗口 + 真 proprio。

替代 S1 冒烟里的 _OneEpisodeSampler(占位 proprio)。产 InitState,直接喂 ImaginationVecEnv。
起点池 = 成功集 ∪ 失败集(残差主战场是基座跑偏态,失败集正是)。caption 钉死常量。

reset 频率不高(每 2 chunk 一次),cv2 顺序读 4 帧 ~0.1s,可接受;proprio 从 parquet 读。
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import pandas as pd

from resfit.rl_finetuning.wm_bridge.init_states import BLOCK_CAPTION, InitState
from resfit.rl_finetuning.wm_bridge.wm_driver import ACTION_DIM, CAMERA_KEYS, N_PREVIOUS

_CHUNKS = 1000


def _bare(cam):                      # observation.images.top_head → top_head
    return cam.split(".")[-1]


def _paths(root, ep):
    ch = ep // _CHUNKS
    pq = os.path.join(root, f"data/chunk-{ch:03d}/episode_{ep:06d}.parquet")
    vids = {c: os.path.join(root, f"videos/chunk-{ch:03d}/{c}/episode_{ep:06d}.mp4")
            for c in CAMERA_KEYS}
    return pq, vids


class TeleavatarStartSampler:
    """dataset_roots: Teleavatar 成功/失败数据集的完整路径列表。"""

    def __init__(self, dataset_roots, rng=None, caption=BLOCK_CAPTION):
        assert dataset_roots, "须给至少一个 Teleavatar 数据集路径(成功/失败集)"
        self.rng = rng if rng is not None else np.random.default_rng()
        self.caption = str(caption)
        self.eps = []                # [(root, ep, n_frames)]
        for root in dataset_roots:
            for pq in sorted(glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet"))):
                ep = int(os.path.basename(pq).split("_")[1].split(".")[0])
                n = len(pd.read_parquet(pq, columns=["frame_index"]))
                if n >= N_PREVIOUS:
                    self.eps.append((root, ep, n))
        assert self.eps, f"没从 {dataset_roots} 读到任何合格 episode"

    def sample(self) -> InitState:
        root, ep, n = self.eps[self.rng.integers(len(self.eps))]
        start = int(self.rng.integers(0, n - N_PREVIOUS + 1))    # 4 帧窗口不越界
        pq, vids = _paths(root, ep)
        # 3 视角各读 start..start+3 共 4 帧,resize (256,192)WH,归一化 [-1,1],CHW
        views = []
        for cam in CAMERA_KEYS:
            cap = cv2.VideoCapture(vids[cam])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(N_PREVIOUS):
                ok, img = cap.read()
                assert ok, f"读帧失败 {vids[cam]} @ {start}"
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 192)).astype(np.float32) / 255.0 * 2.0 - 1.0
                frames.append(np.transpose(img, (2, 0, 1)))       # CHW
            cap.release()
            views.append(np.stack(frames, axis=1))                # (C,T,H,W)
        obs_window = np.stack(views, axis=0)                      # (V,C,4,H,W)
        # proprio = 窗口末帧的真实 state(与 init_states 用末帧 proprio 一致)
        df = pd.read_parquet(pq, columns=["observation.state"])
        proprio = np.asarray(df["observation.state"].iloc[start + N_PREVIOUS - 1],
                             dtype=np.float32).reshape(-1)[:ACTION_DIM]
        return InitState(obs_window=obs_window, proprio=proprio, caption=self.caption)
