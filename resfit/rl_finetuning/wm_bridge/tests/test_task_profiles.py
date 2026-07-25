import numpy as np
import pandas as pd
import pytest

from resfit.rl_finetuning.wm_bridge.task_profiles import get_task_profile
from resfit.rl_finetuning.wm_bridge.teleavatar_policy_state import (
    map_teleavatar_policy_state,
)
from resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler import (
    TeleavatarStartSampler,
)


def test_builtin_task_profiles_are_exact():
    block = get_task_profile("block")
    cup = get_task_profile("cup")
    assert (block.prompt, block.success_dataset, block.action_dim) == (
        "build block", "block_success", 16)
    assert (cup.prompt, cup.success_dataset, cup.action_dim) == (
        "pick cup", "cup_success", 16)


def test_paper_profile_is_exact():
    paper = get_task_profile("paper")
    assert (
        paper.prompt,
        paper.success_dataset,
        paper.action_dim,
        paper.policy_state_dim,
        paper.dataset_fps,
    ) == (
        "put the paper roll on the holder",
        "paper_success",
        16,
        14,
        20.0,
    )


def test_policy_state_14_drops_only_grippers():
    state = np.arange(16, dtype=np.float32)
    np.testing.assert_array_equal(
        map_teleavatar_policy_state(state, 14),
        np.concatenate([state[:7], state[8:15]]),
    )


def test_policy_state_16_is_identity():
    state = np.arange(16, dtype=np.float32)
    np.testing.assert_array_equal(
        map_teleavatar_policy_state(state, 16),
        state,
    )


@pytest.mark.parametrize("target_dim", [13, 15, 17])
def test_invalid_policy_state_dim_fails(target_dim):
    with pytest.raises(ValueError, match="14 or 16"):
        map_teleavatar_policy_state(np.zeros(16), target_dim)


def test_unknown_task_profile_fails():
    with pytest.raises(ValueError, match="unknown task profile"):
        get_task_profile("nonexistent")


def test_start_sampler_uses_injected_caption(monkeypatch):
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.glob.glob",
        lambda pattern: [
            "/data/cup_success/data/chunk-000/episode_000000.parquet"
        ],
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.pd.read_parquet",
        lambda path, columns: (
            pd.DataFrame({"frame_index": range(4)})
            if columns == ["frame_index"]
            else pd.DataFrame({
                "observation.state": [
                    np.zeros(16, dtype=np.float32) for _ in range(4)
                ]
            })
        ),
    )

    class Capture:
        def set(self, *args):
            return None

        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            return None

    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.cv2.VideoCapture",
        lambda path: Capture(),
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.cv2.cvtColor",
        lambda image, code: image,
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler.cv2.resize",
        lambda image, size: np.zeros((192, 256, 3), dtype=np.uint8),
    )
    sampler = TeleavatarStartSampler(
        ["/data/cup_success"],
        rng=np.random.default_rng(0),
        caption="pick cup",
    )
    assert sampler.sample().caption == "pick cup"
