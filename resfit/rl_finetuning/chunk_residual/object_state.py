"""物体感知 state 的 sim 特权读取 + rel_piece 纯逻辑(③a' object-aware)。

设计见 docs/superpowers/specs/2026-06-07-hiql-value-object-aware-design.md。
纯逻辑(eef_rel_piece / rel_piece_stats)全可单测;sim 读取(read_piece_positions)是薄集成层。
online(dexmg._process_obs)与 offline(offline_stage_replay)共用 compute_eef_rel_piece 保证同源。
"""
import numpy as np

# TwoArmThreePieceAssembly 的两个 piece root body(2026-06-07 sim 探针验证)
PIECE_ROOT_BODIES = ("piece_1_root", "piece_2_root")


def eef_rel_piece(eef_pos_by_arm, piece_pos_by_idx):
    """纯逻辑:双臂 eef 相对各 piece 的世界系位置差。

    eef_pos_by_arm:   长度 2,每个 (3,) — [robot0_eef_pos, robot1_eef_pos]
    piece_pos_by_idx: 长度 2,每个 (3,) — [piece_1_pos, piece_2_pos]
    返回 (12,) float32:[eef0-p1, eef0-p2, eef1-p1, eef1-p2] 展平。
    """
    eefs = [np.asarray(e, dtype=np.float32).reshape(3) for e in eef_pos_by_arm]
    pieces = [np.asarray(p, dtype=np.float32).reshape(3) for p in piece_pos_by_idx]
    parts = [e - p for e in eefs for p in pieces]
    return np.concatenate(parts).astype(np.float32)


def read_piece_positions(sim, piece_root_bodies=PIECE_ROOT_BODIES):
    """集成层:从 mujoco sim 读各 piece root body 世界位置。返回 list[(3,) float32]。"""
    return [np.asarray(sim.data.get_body_xpos(b), dtype=np.float32).copy()
            for b in piece_root_bodies]


def compute_eef_rel_piece(sim, eef_pos_by_arm, piece_root_bodies=PIECE_ROOT_BODIES):
    """组合:读 sim piece pose + 算 eef_rel_piece(12,)。online/offline 共用,保证同源。"""
    pieces = read_piece_positions(sim, piece_root_bodies)
    return eef_rel_piece(eef_pos_by_arm, pieces)


def rel_piece_stats(rel_piece_array):
    """一批 rel_piece (N,12) → (mean(12,), std(12,)) float32。std 下限 1e-6 防除零。"""
    a = np.asarray(rel_piece_array, dtype=np.float32)
    mean = a.mean(axis=0).astype(np.float32)
    std = np.maximum(a.std(axis=0), 1e-6).astype(np.float32)
    return mean, std
