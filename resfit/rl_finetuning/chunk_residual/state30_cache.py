"""复用回放出的 30 维 eef_piece state 缓存,避免每次训练都 MuJoCo 回放全 demo(数小时)。

缓存格式:npz,n=demo数(int64),s{i}=第 i 条 demo 的 (T_i,30) float32(与现有
outputs_chunk/three_piece_state30.npz 对齐;demo 顺序 = read_per_demo_states 的 sorted_demo_keys)。
"""
import os

import numpy as np


def save_state30_cache(path, seqs, rel_stats=None, dataset_id=None):
    """存 30 维 state 序列到 npz。

    rel_stats=(mean(12,),std(12,)) 给了则写 v2(额外 rel_mean/rel_std/dataset_id);
    不给则 v1(仅 n+s{i},向后兼容)。
    """
    payload = {"n": np.int64(len(seqs))}
    for i, s in enumerate(seqs):
        payload[f"s{i}"] = np.asarray(s, dtype=np.float32)
    if rel_stats is not None:
        mean, std = rel_stats
        payload["rel_mean"] = np.asarray(mean, dtype=np.float32)
        payload["rel_std"] = np.asarray(std, dtype=np.float32)
        if dataset_id is not None:
            payload["dataset_id"] = np.asarray(str(dataset_id))
    np.savez_compressed(path, **payload)


def load_state30_cache(path):
    """读 npz,返回 30 维 state 序列列表 [(T_i,30) float32]。v2 额外键被忽略。"""
    with np.load(path) as z:
        n = int(z["n"])
        return [z[f"s{i}"] for i in range(n)]


def load_state30_cache_v2(path):
    """读 npz → (seqs, rel_stats 或 None, dataset_id 或 None)。

    缺 rel_mean 键(旧 v1 格式)→ rel_stats=None, dataset_id=None。
    """
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        seqs = [z[f"s{i}"] for i in range(n)]
        if "rel_mean" in z.files:
            rel_stats = (z["rel_mean"], z["rel_std"])
            ds = str(z["dataset_id"]) if "dataset_id" in z.files else None
        else:
            rel_stats, ds = None, None
    return seqs, rel_stats, ds


def load_or_build_state30(hdf5_path, dataset_id, num_demos, cache_path):
    """有缓存读缓存(秒级,按 num_demos 截前 N);否则 read_per_demo_states 回放 eef_piece 并存缓存。

    缓存以 sorted_demo_keys 全量顺序建;num_demos 不为 None 时截前 N(与 read_per_demo_states 同序一致)。
    返回 30 维 seqs 列表。
    """
    if cache_path and os.path.exists(cache_path):
        seqs = load_state30_cache(cache_path)
        return seqs[:num_demos] if num_demos is not None else seqs
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
    seqs, _, _ = read_per_demo_states(hdf5_path, dataset_id, "eef_piece", num_demos=num_demos)
    # 注意:num_demos 非 None 时这里写的是"部分缓存"(只前 N 条)。若之后用更大的 num_demos
    # 复用,会命中此缓存却件数不足而静默少返回。实运用应以 num_demos=None(全量)建一次缓存。
    if cache_path:
        save_state30_cache(cache_path, seqs)
    return seqs
