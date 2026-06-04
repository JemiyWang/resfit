"""从源 dexmimicgen HDF5 构建残差 RL 的 offline demo buffer（方案 A）。

为什么从 HDF5 而非 LeRobot：LeRobot 那份 observation.state 只有 18 维本体感觉、没有
物体 pose，无法算 stage；且有 4 个 AV1 视频损坏。源 HDF5
(MimicGen/dexmimicgen_datasets) 逐帧存 uint8 图像 + 18 维所需 proprio + actions +
完整 mujoco states，自给自足：proprio/图/action 直接读，stage 在 set_state replay
里用训练同款检测器算，按 (demo, frame) 1:1 拼，彻底绕开对齐与坏视频问题。

本模块只放纯逻辑（可单测）；sim-replay 取 stage 的集成部分单独 smoke 验证。
"""
from __future__ import annotations

import numpy as np

from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import shaping_reward

# 18 维 observation.state 的构成与顺序，严格对齐 LeRobot ankile/dexmg-... 的 names：
# 每臂 eef_pos(3) + eef_quat(4) + gripper_qpos(2)，robot0 在前 robot1 在后。
STATE18_KEYS = [
    ("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_gripper_qpos", 2),
    ("robot1_eef_pos", 3), ("robot1_eef_quat", 4), ("robot1_gripper_qpos", 2),
]


def concat_mixed_batch(online_batch, offline_batch):
    """拼在线 + offline batch 做 RLPD 混采。

    只保留两者**共有**的顶层 key 再 cat:`--stage_balanced` 的在线 batch 走自定义
    rb[idx] 取样、没有 `_weight`,而 offline 走 .sample() 有 `_weight`,直接 torch.cat
    会因 key 不一致报 KeyError。`_weight` 在 alpha=0 时是均匀权重、stage_balanced 在线
    路径本就没有它,agent.update 不依赖,丢弃安全。
    """
    import torch
    common = list(set(online_batch.keys()) & set(offline_batch.keys()))
    return torch.cat([online_batch.select(*common), offline_batch.select(*common)], dim=0)


def save_stage_cache(path: str, stages_by_demo: dict) -> None:
    """把逐 demo 瞬时 stage 数组存成 npz(很小,几百 KB)。键 = demo 名。

    贵的是 sim replay 取 stage;存下来后,后续训练直接读 obs/图/action + 本缓存拼 buffer,
    无需再起 env replay。
    """
    np.savez_compressed(
        path, **{k: np.asarray(v, dtype=np.int8) for k, v in stages_by_demo.items()})


def load_stage_cache(path: str) -> dict:
    """读回 save_stage_cache 存的逐 demo 瞬时 stage(dict: demo 名 -> int8 数组)。"""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def sorted_demo_keys(keys) -> list:
    """HDF5 demo 键按数字后缀排序(demo_2 在 demo_10 前),保证灌装顺序确定。"""
    return sorted(keys, key=lambda d: int(d.split("_")[-1]))


def assemble_state18(obs) -> np.ndarray:
    """从 HDF5 obs（key -> (T, d) 数组）按 LeRobot 顺序拼出 (T, 18) 的 observation.state。"""
    return np.concatenate(
        [np.asarray(obs[k])[:, :d] for k, d in STATE18_KEYS], axis=1
    )


def latch_from_instant(instant) -> np.ndarray:
    """瞬时 stage 序列 → 闩锁序列（episode 内单调不降，= 前缀最大值）。

    demo 是成功轨迹，瞬时阶段可能回退（抓起又掉），闩锁取 running-max，
    与线上 wrapper 的 `self._stage = max(self._stage, s)` 语义一致。
    """
    return np.maximum.accumulate(np.asarray(instant))


def transition_fields(instant_stages, *, bonus: float, mode: str,
                      gamma: float, success: bool = True) -> dict:
    """一条 demo 的 T 帧瞬时 stage → T-1 个 transition 的全部 stage/reward/done 字段。

    汇总索引约定(与线上 cl=1 同构),供灌装层 zip obs/action/图:
      reward        : 闩锁整形 + 成功 base(见 transition_rewards)
      done          : 仅末步 True(成功 demo)
      stage_id      : 当前 obs 瞬时 = instant[:-1](解耦)
      next_stage_id : next obs 瞬时 = instant[1:]
      max_stage     : after-step 闩锁 = latch[1:](供 stage-balanced 采样器读顶层列)
    """
    instant = np.asarray(instant_stages)
    latch = latch_from_instant(instant)
    T = len(instant)
    done = np.zeros(T - 1, dtype=bool)
    if success and T >= 2:
        done[-1] = True
    return {
        "reward": transition_rewards(instant, bonus=bonus, mode=mode,
                                     gamma=gamma, success=success),
        "done": done,
        "stage_id": transition_stage_ids(instant),
        "next_stage_id": instant[1:],
        "max_stage": latch[1:],
    }


def transition_stage_ids(instant_stages) -> np.ndarray:
    """每个 transition 的 obs.stage_id = 当前帧**瞬时**阶段(解耦,handoff §2)。

    T 帧 → T-1 个 transition,当前 obs 取 instant[:-1]。注意:与 reward 用闩锁不同,
    这里用瞬时,避免 stage3 桶被"抓起又掉"的退化样本污染(76%)。
    """
    return np.asarray(instant_stages)[:-1]


def transition_rewards(instant_stages, *, bonus: float, mode: str,
                       gamma: float, success: bool = True) -> np.ndarray:
    """一条 demo 的 T 帧瞬时 stage → T-1 个 transition 的总 reward。

    与线上 cl=1 一致:每步 reward = base 稀疏 + shaping_reward(闩锁[t], 闩锁[t+1])。
    demo 为成功轨迹时,最后一个 transition 终止且成功(base +1)。
    """
    latch = latch_from_instant(instant_stages)
    T = len(latch)
    rewards = np.empty(T - 1, dtype=np.float32)
    for t in range(T - 1):
        done = success and (t == T - 2)
        base = float(done)
        shaped = shaping_reward(int(latch[t]), int(latch[t + 1]),
                                mode=mode, bonus=bonus, gamma=gamma, done=done)
        rewards[t] = base + shaped
    return rewards
