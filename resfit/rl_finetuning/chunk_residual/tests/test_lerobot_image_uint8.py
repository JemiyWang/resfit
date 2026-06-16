"""lerobot 数据源 offline buffer:图像应以 uint8 存(对齐 hdf5 路,offcache 省 4x)。

用最小 fake LeRobot ds 注入(_lerobot_ds),不依赖真实数据集 / torchcodec / robosuite,
故可进快测。lerobot_episode_frames 只用 ds.episode_data_index + ds[i],替身够用。
"""
import torch


class _Identity:
    def scale(self, x):
        return x

    def standardize(self, x):
        return x


class _FakeLeRobotDS:
    """最小 LeRobotDataset 替身:1 集 T 帧,图为 float[0,1](模拟 torchcodec 解码输出)。"""

    def __init__(self, T, image_keys, state_dim=18, action_dim=14):
        self.episode_data_index = {"from": [0], "to": [T]}
        g = torch.Generator().manual_seed(0)
        self._frames = [
            {**{k: torch.rand(3, 84, 84, generator=g) for k in image_keys},
             "observation.state": torch.rand(state_dim, generator=g),
             "action": torch.rand(action_dim, generator=g)}
            for _ in range(T)
        ]

    def __getitem__(self, i):
        return self._frames[i]


IMAGE_KEYS = [
    "observation.images.agentview",
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot1_eye_in_hand",
]


def test_lerobot_offline_buffer_stores_images_as_uint8():
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
        build_offline_buffer,
    )

    T = 6
    fake_ds = _FakeLeRobotDS(T=T, image_keys=IMAGE_KEYS)
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=100, device="cpu"), batch_size=2)
    n = build_offline_buffer(
        rb, None, action_scaler=_Identity(), state_standardizer=_Identity(),
        image_keys=IMAGE_KEYS, bonus=1.0, mode="staged", gamma=0.99,
        data_source="lerobot", _lerobot_ds=fake_ds, base_mode="gt")

    assert n == len(rb) and n == T - 1            # 1 集 T 帧 → T-1 条 transition

    for key in IMAGE_KEYS:
        img = rb[0]["obs"][key]
        assert img.dtype == torch.uint8, f"{key} 应存 uint8,实为 {img.dtype}"
        assert tuple(img.shape[-3:]) == (3, 84, 84)            # CHW
        assert int(img.min()) >= 0 and int(img.max()) <= 255   # 值域 [0,255]
    # next.obs 的图同样 uint8
    assert rb[0]["next"]["obs"][IMAGE_KEYS[0]].dtype == torch.uint8
