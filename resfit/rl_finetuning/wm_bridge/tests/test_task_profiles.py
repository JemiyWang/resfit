import numpy as np
import pandas as pd
import pytest

from resfit.rl_finetuning.wm_bridge.task_profiles import get_task_profile
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


def test_unknown_task_profile_fails():
    with pytest.raises(ValueError, match="unknown task profile"):
        get_task_profile("paper")


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
