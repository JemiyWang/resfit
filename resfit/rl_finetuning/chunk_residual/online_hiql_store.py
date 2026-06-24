"""HIQL 在线联合微调(Phase 3)的在线轨迹 store + 批采样器。

随 episode 增长的 build_gc_data 结构;goal 用 geometric 采样(只需 last_idx,
无需 stage 检测)。与离线 store 同形,采样器对两者通用。
设计见 docs/superpowers/specs/2026-06-24-hiql-online-joint-finetune-design.md。
"""
from collections import deque

import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, sample_gc_goals


class OnlineHiqlStore:
    """完成的 online episode 累积成 build_gc_data 结构;FIFO 控容量,脏标记按需重建。"""

    def __init__(self, max_transitions=50_000):
        self.seqs = deque()
        self.max_transitions = int(max_transitions)
        self._n = 0
        self._data = None

    def add_episode(self, states):
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or len(states) < 2:
            return                       # T<2 无 transition(与 build_gc_data 一致跳过)
        self.seqs.append(states)
        self._n += len(states) - 1       # transition 数 = T-1
        while self._n > self.max_transitions and len(self.seqs) > 1:
            old = self.seqs.popleft()
            self._n -= len(old) - 1
        self._data = None                # 脏

    def __len__(self):
        return self._n

    def ready(self, min_transitions):
        return self._n >= int(min_transitions) and len(self.seqs) >= 1

    def data(self):
        if self._data is None:
            empty = [np.empty(0, dtype=np.int64) for _ in self.seqs]
            self._data = build_gc_data(list(self.seqs), empty)
        return self._data


def _sample_indices(data, bs, rng):
    n = len(data["s_idx"])
    b = rng.integers(0, n, size=bs)
    return b


def sample_value_batch(data, bs, rng, *, future_mode, gamma):
    """从 build_gc_data 形状的 data 采一个 value 训练 batch。返回 torch tensor。"""
    states = data["states"]
    b = _sample_indices(data, bs, rng)
    si, sni, tj = data["s_idx"][b], data["sn_idx"][b], data["traj_id"][b]
    gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"], rng,
                         n_total=len(states), future_mode=future_mode, discount=gamma)
    s = states[si]
    s_next = states[sni]
    g = states[gi]
    success = torch.tensor(si == gi, dtype=torch.float32)
    done = data["done"][b]
    return s, s_next, g, success, done


def sample_high_actor_batch(data, bs, rng, *, way_steps, future_mode, gamma):
    """从 build_gc_data 形状的 data 采一个 high_actor(fixed_waypoint)训练 batch。"""
    states = data["states"]
    b = _sample_indices(data, bs, rng)
    si, tj = data["s_idx"][b], data["traj_id"][b]
    last_arr = np.array([data["last_idx_of"][int(d)] for d in tj], dtype=np.int64)
    wi = np.minimum(si + way_steps, last_arr)
    gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"], rng,
                         n_total=len(states), future_mode=future_mode, discount=gamma)
    return states[si], states[wi], states[gi]
