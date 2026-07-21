"""想象空间的 proprio 外推。

WM 两头都没有机器人状态(输入无 state 键、输出无 state head),所以 proprio 必须由本模块
自己维护。做法沿用 RISE:下一段的关节位置 = 本段动作 chunk 的最后一个动作——成立前提是
action_type=absolute + action_space=joint。

★ 全程在物理空间。RISE 原实现的 next_state 取自归一化空间而喂 WM 的 token 取自物理空间,
混用会静默错位;本模块只在打包 act_tokens 时才归一化。
"""
from __future__ import annotations

import numpy as np

ACTION_DIM = 16


class ProprioTracker:
    def __init__(self, init_proprio, action_dim: int = ACTION_DIM):
        arr = np.asarray(init_proprio, dtype=np.float32).reshape(-1)
        assert arr.shape == (action_dim,), \
            f"init_proprio 须 ({action_dim},),got {arr.shape}"
        self.action_dim = action_dim
        self._p = arr.copy()

    @property
    def proprio(self) -> np.ndarray:
        return self._p.copy()

    def advance(self, action_chunk_physical) -> np.ndarray:
        """action_chunk_physical: (L, action_dim) 物理空间绝对关节动作。"""
        arr = np.asarray(action_chunk_physical, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == self.action_dim, \
            f"chunk 须 (L,{self.action_dim}),got {arr.shape}"
        assert arr.shape[0] > 0, "chunk 不能为空"
        self._p = arr[-1].copy()
        return self.proprio
