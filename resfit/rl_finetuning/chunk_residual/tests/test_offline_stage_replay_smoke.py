"""集成 smoke 测试(方案 A 实证):在源 HDF5 上 set_state replay,验证检测器把
成功 demo 的闩锁推到终段。需要已下载的源 HDF5 + robosuite env(MUJOCO_GL=egl);
缺文件则 skip。慢,默认不在快测里跑,显式运行。
"""
import glob

import h5py
import numpy as np
import pytest

from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    latch_from_instant,
    sorted_demo_keys,
)
from resfit.rl_finetuning.chunk_residual.stage_detectors import (
    NUM_STAGES,
    get_stage_detector,
)

DATASET_GLOB = "deps/dexmimicgen/**/two_arm_three_piece_assembly.hdf5"


def _find_dataset():
    hits = glob.glob(DATASET_GLOB, recursive=True)
    return hits[0] if hits else None


class _Identity:
    def scale(self, x):
        return x

    def standardize(self, x):
        return x


@pytest.mark.skipif(_find_dataset() is None, reason="源 HDF5 未下载")
def test_build_offline_buffer_two_demos_fields_and_count():
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
        build_offline_buffer,
    )

    path = _find_dataset()
    image_keys = [
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot1_eye_in_hand",
    ]
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=2000, device="cpu"), batch_size=4)
    n = build_offline_buffer(
        rb, path, action_scaler=_Identity(), state_standardizer=_Identity(),
        image_keys=image_keys, bonus=1.0, mode="staged", gamma=0.99, num_demos=2)
    assert n == len(rb) and n > 0
    # 与 add_chunk_transition 同构:每条 transition 带 unsqueeze(0) 的前导单维,故只校尾部维度
    td = rb[0]
    obs = td["obs"]
    assert obs["observation.state"].shape[-1] == 18
    assert obs["observation.stage_id"].numel() == 1
    assert obs["observation.base_action"].shape[-1] == 14
    img = obs["observation.images.agentview"]
    assert img.dtype.is_floating_point is False              # uint8
    assert tuple(img.shape[-3:]) == (3, 84, 84)             # CHW
    assert td["action"].shape[-1] == 14
    assert "max_stage" in td.keys()
    # 末条 transition 应终止且 reward>0(成功 demo 跨入终段 + base)
    last = rb[n - 1]
    assert bool(last["next", "done"].reshape(-1)[0]) is True
    assert float(last["next", "reward"].reshape(-1)[0]) > 0.0


@pytest.mark.skipif(_find_dataset() is None, reason="源 HDF5 未下载")
def test_count_offline_transitions_matches_build():
    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
        count_offline_transitions,
    )
    # 预数 = sum(len-1);2 demo(229,238)→ 228+237 = 465,与 build_offline_buffer 输出一致
    assert count_offline_transitions(_find_dataset(), num_demos=2) == 465


@pytest.mark.skipif(_find_dataset() is None, reason="源 HDF5 未下载")
def test_precompute_cache_then_build_uses_it(tmp_path):
    import os

    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
        build_offline_buffer,
        precompute_stage_cache,
    )

    path = _find_dataset()
    cache = str(tmp_path / "stages.npz")
    n_demos = precompute_stage_cache(path, cache, num_demos=2)   # 只 replay 一次,落盘
    assert n_demos == 2 and os.path.exists(cache)

    image_keys = [
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot1_eye_in_hand",
    ]
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=2000, device="cpu"), batch_size=4)
    n = build_offline_buffer(
        rb, path, action_scaler=_Identity(), state_standardizer=_Identity(),
        image_keys=image_keys, bonus=1.0, mode="staged", gamma=0.99,
        num_demos=2, stage_cache=cache)
    # 用缓存拼出的条数应与 replay 路径一致(2 demo → 465)
    assert n == 465 and n == len(rb)


@pytest.mark.skipif(_find_dataset() is None, reason="源 HDF5 未下载")
def test_smoke_two_demos_reach_success_stage():
    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
        make_replay_env,
        replay_instant_stages,
    )

    path = _find_dataset()
    env, env_name = make_replay_env(path)
    detector = get_stage_detector(env_name)
    top = NUM_STAGES[env_name] - 1
    try:
        with h5py.File(path, "r") as f:
            for ep in sorted_demo_keys(list(f["data"].keys()))[:2]:
                grp = f[f"data/{ep}"]
                states = grp["states"][()]
                stages = replay_instant_stages(
                    env, states,
                    model_file=grp.attrs["model_file"],
                    detector=detector,
                    ep_meta=grp.attrs.get("ep_meta"),
                )
                assert len(stages) == len(states)            # 一帧一标
                assert int(stages.min()) >= 0
                assert int(stages.max()) <= top
                # 成功 demo:闩锁必须推到终段(检测器在源数据上判得对 = method A 成立)
                assert int(latch_from_instant(stages)[-1]) == top
    finally:
        env.close()
