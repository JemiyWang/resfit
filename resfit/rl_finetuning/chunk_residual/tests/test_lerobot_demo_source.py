import torch
from resfit.rl_finetuning.chunk_residual.lerobot_demo_source import (
    lerobot_episode_frames, lerobot_episode_count, count_lerobot_transitions, to_chw01,
)


class _StubDS:
    def __init__(self):
        self.episode_data_index = {"from": torch.tensor([0, 3]), "to": torch.tensor([3, 5])}
    def __getitem__(self, i):
        return {"observation.images.agentview": torch.full((3, 4, 4), float(i)),
                "observation.state": torch.tensor([float(i)] * 18),
                "action": torch.tensor([float(i)] * 7)}


def test_to_chw01_variants():
    assert to_chw01(torch.zeros(3, 4, 4)).shape == (3, 4, 4)            # CHW 原样
    hwc = torch.zeros(4, 4, 3); assert to_chw01(hwc).shape == (3, 4, 4)  # HWC->CHW
    u8 = torch.full((3, 4, 4), 255.0); assert float(to_chw01(u8).max()) <= 1.0  # uint8 尺度->[0,1]


def test_episode_count_and_frames():
    ds = _StubDS()
    assert lerobot_episode_count(ds) == 2
    assert count_lerobot_transitions(ds) == 3                    # (3-1)+(2-1)
    assert count_lerobot_transitions(ds, num_demos=1) == 2       # 仅第 0 集
    fr = lerobot_episode_frames(ds, 0, ["observation.images.agentview"], "observation.state", "action")
    assert fr["images"]["observation.images.agentview"].shape == (3, 3, 4, 4)   # (T=3,C,H,W)
    assert fr["state"].shape == (3, 18) and fr["actions"].shape == (3, 7)
    # 第 1 集(帧 3..5) → T=2
    fr1 = lerobot_episode_frames(ds, 1, ["observation.images.agentview"], "observation.state", "action")
    assert fr1["state"].shape == (2, 18)
