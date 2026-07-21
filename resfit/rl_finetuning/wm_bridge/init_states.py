"""想象段起点采样。

★ caption 钉死为常量,禁止从数据集读。WM 微调时 block 域 caption 已统一为 "build block"
(RISE_Hi/temp/三域数据构建方案.md §3.4),而本机数据集里是 "build blocks"(复数)。
WM 靠 T5 编 caption 做条件,喂错一个字符即偏离训练分布。

★ expert ∪ rollout 混采。残差的主战场是"基座跑偏"的状态,只用专家集会让它从没见过
需要救场的局面;rollout 集是基座实跑轨迹,与部署时的状态分布天然对齐。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resfit.rl_finetuning.wm_bridge.wm_driver import ACTION_DIM, N_PREVIOUS

BLOCK_CAPTION = "build block"

# ★ 不写死路径。用户实际使用的 block 数据不在本机,且成功集与失败集完全分开。
#   由 --init_state_dataset(可重复)在运行时传入;缺失即硬失败,不用任何默认路径猜测。
BLOCK_DATASETS = ()


@dataclass
class InitState:
    obs_window: np.ndarray      # (3, 3, 4, 192, 256)
    proprio: np.ndarray         # (16,)
    caption: str


class InitStateSampler:
    def __init__(self, episode_sources, rng=None):
        assert len(episode_sources) > 0, "episode_sources 不能为空"
        for src in episode_sources:
            assert src.n_frames >= N_PREVIOUS, \
                f"每集至少需 {N_PREVIOUS} 帧才能凑满历史窗口,got {src.n_frames}"
        self.sources = list(episode_sources)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self) -> InitState:
        src = self.sources[self.rng.integers(len(self.sources))]
        # 末帧索引须 >= N_PREVIOUS-1,否则历史窗口越界
        end = int(self.rng.integers(N_PREVIOUS - 1, src.n_frames))
        frames = []
        proprio = None
        for t in range(end - N_PREVIOUS + 1, end + 1):
            f, p = src.read(t)
            frames.append(np.asarray(f, dtype=np.float32))
            proprio = p
        window = np.stack(frames, axis=2)          # (V,C,4,H,W)
        proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)[:ACTION_DIM]
        assert proprio.shape == (ACTION_DIM,), \
            f"proprio 须 ({ACTION_DIM},),got {proprio.shape}"
        return InitState(obs_window=window, proprio=proprio, caption=BLOCK_CAPTION)
