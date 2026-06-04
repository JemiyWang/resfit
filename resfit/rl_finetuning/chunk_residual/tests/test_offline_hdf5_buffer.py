"""从源 dexmimicgen HDF5 构建 offline buffer 的纯逻辑单测（方案 A）。

只测不依赖 robosuite/数据文件的纯函数；sim-replay 取 stage 的集成部分另行 smoke 验证。
"""
import numpy as np

from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    assemble_state18,
    concat_mixed_batch,
    latch_from_instant,
    load_stage_cache,
    save_stage_cache,
    sorted_demo_keys,
    transition_fields,
    transition_rewards,
    transition_stage_ids,
)


def test_concat_mixed_batch_reconciles_mismatched_keys():
    # stage_balanced 在线 batch 无 _weight,offline .sample() 有 _weight → 直接 cat 会 KeyError。
    # concat_mixed_batch 取公共 key 后拼接。
    import torch
    from tensordict import TensorDict
    online = TensorDict({"action": torch.zeros(2, 3), "max_stage": torch.zeros(2)}, batch_size=[2])
    offline = TensorDict({"action": torch.ones(3, 3), "max_stage": torch.ones(3),
                          "_weight": torch.ones(3)}, batch_size=[3])
    out = concat_mixed_batch(online, offline)
    assert out.batch_size[0] == 5            # 2 + 3
    assert "_weight" not in out.keys()       # 非公共,被丢
    assert tuple(out["action"].shape) == (5, 3)


def test_stage_cache_roundtrip(tmp_path):
    stages = {"demo_0": np.array([0, 1, 3, 2, 4]), "demo_10": np.array([0, 1, 1])}
    path = str(tmp_path / "stages.npz")
    save_stage_cache(path, stages)
    loaded = load_stage_cache(path)
    assert set(loaded) == {"demo_0", "demo_10"}
    assert loaded["demo_0"].tolist() == [0, 1, 3, 2, 4]
    assert loaded["demo_10"].tolist() == [0, 1, 1]


def test_sorted_demo_keys_orders_by_numeric_suffix():
    # 字典序会把 demo_10 排在 demo_2 前;应按数字
    keys = ["demo_10", "demo_2", "demo_1", "demo_100"]
    assert sorted_demo_keys(keys) == ["demo_1", "demo_2", "demo_10", "demo_100"]


def test_transition_fields_indexing_matches_online_semantics():
    # T=4 帧 → 3 个 transition。汇总所有 per-transition 字段的索引约定。
    instant = np.array([0, 1, 2, 4])  # latch 同(无回退)
    f = transition_fields(instant, bonus=1.0, mode="staged", gamma=0.99, success=True)
    assert f["reward"].tolist() == [1.0, 1.0, 3.0]          # 闩锁 + 成功 base
    assert f["done"].tolist() == [False, False, True]       # 仅末步终止
    assert f["stage_id"].tolist() == [0, 1, 2]              # 当前 obs 瞬时 = instant[:-1]
    assert f["next_stage_id"].tolist() == [1, 2, 4]         # next obs 瞬时 = instant[1:]
    assert f["max_stage"].tolist() == [1, 2, 4]            # 闩锁 after-step = latch[1:](供采样器)


def test_latch_is_running_max_of_instant_stages():
    # 瞬时阶段可回退（抓起又掉），闩锁单调不降 = 前缀最大值
    instant = np.array([0, 1, 1, 0, 2, 1, 4])
    latch = latch_from_instant(instant)
    assert latch.tolist() == [0, 1, 1, 1, 2, 2, 4]


def test_staged_transition_rewards_match_online_convention():
    # T=4 帧 → 3 个 transition。staged = 每步闩锁推进量×bonus，叠成功步 base +1。
    #   t0: 跨 0→1 = +1
    #   t1: 跨 1→2 = +1
    #   t2: 跨 2→4 = +2，且终止成功 base +1 → 3
    instant = np.array([0, 1, 2, 4])
    r = transition_rewards(instant, bonus=1.0, mode="staged", gamma=0.99, success=True)
    assert r.tolist() == [1.0, 1.0, 3.0]


def test_assemble_state18_matches_lerobot_order_and_width():
    # 顺序 = LeRobot observation.state:r0 eef_pos(3)+eef_quat(4)+gripper(2) 再 r1 同构 = 18
    obs = {
        "robot0_eef_pos": np.array([[1.0, 2, 3]]),
        "robot0_eef_quat": np.array([[4.0, 5, 6, 7]]),
        "robot0_gripper_qpos": np.array([[8.0, 9]]),
        "robot1_eef_pos": np.array([[10.0, 11, 12]]),
        "robot1_eef_quat": np.array([[13.0, 14, 15, 16]]),
        "robot1_gripper_qpos": np.array([[17.0, 18]]),
        # 干扰键，必须被忽略
        "robot0_joint_vel": np.array([[99.0] * 7]),
    }
    state = assemble_state18(obs)
    assert state.shape == (1, 18)
    assert state[0].tolist() == [float(i) for i in range(1, 19)]


def test_stage_id_uses_instant_while_reward_uses_latch():
    # 解耦(handoff §2):抓起 piece2(3)又掉回 2，再成功(4)。
    instant = np.array([0, 1, 3, 2, 4])
    # stage_id = 当前帧瞬时(transition 取 instant[:-1])，能体现回退 3→2
    sid = transition_stage_ids(instant)
    assert sid.tolist() == [0, 1, 3, 2]
    # reward 走闩锁 [0,1,3,3,4]：回退那步(t=2)闩锁持平 → 既不罚也不奖(0)
    r = transition_rewards(instant, bonus=1.0, mode="staged", gamma=0.99, success=True)
    assert r.tolist() == [1.0, 2.0, 0.0, 2.0]
