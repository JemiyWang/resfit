import os
import pytest

RUN = os.environ.get("LIBERO_OFFLINE_SMOKE") == "1"

@pytest.mark.skipif(not RUN, reason="需真 serve + GPU;设 LIBERO_OFFLINE_SMOKE=1 且先起 serve 才跑")
def test_build_task8_offline_buffer_base_policy():
    """真 pi0_libero serve 在 8000;建 task8(libero_10/8)offline buffer(base_policy)验 len>0。"""
    import torch
    from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
    from resfit.rl_finetuning.chunk_residual.libero_offline import (
        build_libero_offline_buffer, count_libero_offline_transitions)
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_libero_scalers, build_base_policy
    import argparse
    root = "/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero"
    stats = root + "/meta/stats.json"
    a = argparse.Namespace(env_family="libero", libero_suite="libero_10", libero_task_id=8,
                           base_policy_type="pi05", base_action_mode="queue", chunk_length=1,
                           pi0_host="127.0.0.1", pi0_port=8000, pi0_action_dim=7, pi0_execute_horizon=30,
                           pi0_prompt="", pi0_image_key_map=None, pi0_kai0_path="/mnt/mnt/data/kai0_new4090")
    act_scaler, state_std = build_libero_scalers(stats, "cpu", action_scale=0.05, min_range_per_dim=0.1)
    base = build_base_policy(a, "cpu")
    cap = count_libero_offline_transitions(root, "libero_10", 8, num_demos=2)
    rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=cap, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority", batch_size=4)
    build_libero_offline_buffer(rb, lerobot_root=root, suite="libero_10", task_id=8,
                                action_scaler=act_scaler, state_standardizer=state_std,
                                base_policy=base, base_mode="base_policy", base_device="cpu",
                                image_size=84, num_demos=2)
    assert len(rb) > 0
