import numpy as np
import pytest

from resfit.rl_finetuning.wm_bridge.init_states import (
    BLOCK_CAPTION, InitStateSampler,
)


class _StubEpisode:
    """每帧的像素值 = 帧号,便于断言取到的是哪 4 帧。"""

    def __init__(self, n_frames, tag=0.0):
        self.n_frames = n_frames
        self.tag = tag

    def read(self, frame_idx):
        frame = np.full((3, 3, 192, 256), float(frame_idx), np.float32)
        proprio = np.full(16, self.tag, np.float32)
        return frame, proprio


def test_caption_is_pinned_constant_not_from_data():
    s = InitStateSampler([_StubEpisode(100)], rng=np.random.default_rng(0))
    assert s.sample().caption == BLOCK_CAPTION == "build block"


def test_obs_window_has_four_consecutive_frames():
    s = InitStateSampler([_StubEpisode(100)], rng=np.random.default_rng(0))
    st = s.sample()
    assert st.obs_window.shape == (3, 3, 4, 192, 256)
    # 时间维上 4 帧应是连续递增的帧号
    vals = [st.obs_window[0, 0, t, 0, 0] for t in range(4)]
    assert vals == [vals[0] + i for i in range(4)]


def test_never_samples_before_frame_index_three():
    """前 3 帧凑不满 4 帧历史窗口,必须被排除。"""
    s = InitStateSampler([_StubEpisode(4)], rng=np.random.default_rng(0))
    for _ in range(20):
        st = s.sample()
        assert st.obs_window[0, 0, 0, 0, 0] >= 0.0   # 起始帧号 >= 0
        assert st.obs_window[0, 0, 3, 0, 0] <= 3.0   # 末帧号 <= n_frames-1


def test_samples_from_all_sources():
    """expert 与 rollout 混采:两个源都必须被抽到。"""
    eps = [_StubEpisode(50, tag=1.0), _StubEpisode(50, tag=2.0)]
    s = InitStateSampler(eps, rng=np.random.default_rng(0))
    tags = {float(s.sample().proprio[0]) for _ in range(60)}
    assert tags == {1.0, 2.0}


def test_proprio_shape():
    s = InitStateSampler([_StubEpisode(50)], rng=np.random.default_rng(0))
    assert s.sample().proprio.shape == (16,)


def test_rejects_episode_too_short():
    with pytest.raises(AssertionError):
        InitStateSampler([_StubEpisode(3)], rng=np.random.default_rng(0))
