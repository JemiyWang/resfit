import numpy as np
import pytest
import torch
from types import SimpleNamespace

import resfit.rl_finetuning.wm_bridge.block_offline_chunk as block
import resfit.rl_finetuning.wm_bridge.verify_block_sparse_decode as verify

from resfit.rl_finetuning.wm_bridge.verify_block_sparse_decode import (
    check_report,
    measure_reader_peak_mib,
    verify_episode,
)


def test_check_report_accepts_exact_fast_bounded_result():
    report = {
        "pixel_equal": True,
        "legacy_seconds": 12.0,
        "sparse_seconds": 2.0,
        "peak_extra_mib": 120.0,
    }
    assert check_report(report)["speedup"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"pixel_equal": False}, "pixel equality"),
        ({"sparse_seconds": 5.0}, "3.0x"),
        ({"peak_extra_mib": 513.0}, "512 MiB"),
    ],
)
def test_check_report_rejects_failed_gate(change, message):
    report = {
        "pixel_equal": True,
        "legacy_seconds": 12.0,
        "sparse_seconds": 2.0,
        "peak_extra_mib": 120.0,
    }
    report.update(change)
    with pytest.raises(RuntimeError, match=message):
        check_report(report)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"legacy_seconds": 0.0}, "legacy_seconds"),
        ({"legacy_seconds": np.nan}, "legacy_seconds"),
        ({"sparse_seconds": -1.0}, "sparse_seconds"),
        ({"sparse_seconds": np.inf}, "sparse_seconds"),
        ({"peak_extra_mib": -1.0}, "peak_extra_mib"),
        ({"peak_extra_mib": 0.0}, "peak_extra_mib"),
        ({"peak_extra_mib": np.nan}, "peak_extra_mib"),
    ],
)
def test_check_report_rejects_non_finite_or_out_of_range_metrics(
    change, message,
):
    report = {
        "pixel_equal": True,
        "legacy_seconds": 12.0,
        "sparse_seconds": 2.0,
        "peak_extra_mib": 120.0,
    }
    report.update(change)
    with pytest.raises(RuntimeError, match=message):
        check_report(report)


def test_check_report_rejects_non_boolean_pixel_result():
    report = {
        "pixel_equal": np.nan,
        "legacy_seconds": 12.0,
        "sparse_seconds": 2.0,
        "peak_extra_mib": 120.0,
    }
    with pytest.raises(RuntimeError, match="pixel_equal"):
        check_report(report)


def test_verify_episode_warms_both_then_times_sparse_before_legacy():
    calls = []
    values = {
        camera: np.full((2, 3, 2, 2), index, dtype=np.uint8)
        for index, camera in enumerate(block.CAMERA_KEYS)
    }

    def legacy_decoder(episode, indices):
        calls.append(("legacy", tuple(indices)))
        return {camera: value.copy() for camera, value in values.items()}

    def sparse_decoder(episode, indices):
        calls.append(("sparse", tuple(indices)))
        return {
            camera: torch.from_numpy(value.copy())
            for camera, value in values.items()
        }

    ticks = iter([10.0, 12.0, 20.0, 26.0])
    episode = SimpleNamespace(episode_id=7, num_frames=51)
    report = verify_episode(
        episode,
        legacy_decoder=legacy_decoder,
        sparse_decoder=sparse_decoder,
        peak_measure=lambda value: 123.0,
        clock=lambda: next(ticks),
    )

    assert calls == [
        ("legacy", (0, 50)),
        ("sparse", (0, 50)),
        ("sparse", (0, 50)),
        ("legacy", (0, 50)),
    ]
    assert report["pixel_equal"] is True
    assert report["sparse_seconds"] == pytest.approx(2.0)
    assert report["legacy_seconds"] == pytest.approx(6.0)
    assert report["peak_extra_mib"] == pytest.approx(123.0)


def test_reader_peak_measure_uses_spawn_and_full_reader_path(monkeypatch):
    events = []
    messages = []

    class FakeReader:
        def __init__(self, episode):
            events.append(("init", episode.episode_id))

        def prepare_native_frames(self, indices):
            events.append(("prepare", tuple(indices)))

        def native_frames(self, index):
            events.append(("access", index))

    class Receiver:
        def poll(self):
            return bool(messages)

        def recv(self):
            return messages.pop(0)

        def close(self):
            pass

    class Sender:
        def send(self, value):
            messages.append(value)

        def close(self):
            pass

    class FakeProcess:
        exitcode = 0

        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

        def join(self):
            pass

        def close(self):
            pass

    class Context:
        @staticmethod
        def Pipe(duplex=False):
            assert duplex is False
            return Receiver(), Sender()

        @staticmethod
        def Process(target, args):
            return FakeProcess(target, args)

    monkeypatch.setattr(
        verify.multiprocessing,
        "get_context",
        lambda method: events.append(("context", method)) or Context(),
    )
    monkeypatch.setattr(verify, "BlockEpisodeReader", FakeReader)
    monkeypatch.setattr(verify, "_current_rss_mib", lambda: 40.0)
    monkeypatch.setattr(verify, "_max_rss_mib", lambda: 175.5)

    episode = SimpleNamespace(episode_id=9, num_frames=51)
    assert measure_reader_peak_mib(episode) == pytest.approx(135.5)
    assert events == [
        ("context", "spawn"),
        ("init", 9),
        ("prepare", (0, 50)),
        ("access", 0),
        ("access", 50),
    ]


class _FakeCapture:
    instances = []

    def __init__(self, path):
        self.path = path
        self.frame_index = -1
        self.released = False
        type(self).instances.append(self)

    def grab(self):
        self.frame_index += 1
        return self.frame_index <= 4

    def retrieve(self):
        value = self.frame_index
        bgr = np.empty((2, 3, 3), dtype=np.uint8)
        bgr[..., 0] = value
        bgr[..., 1] = value + 1
        bgr[..., 2] = value + 2
        return True, bgr

    def release(self):
        self.released = True


def test_sequential_opencv_opens_once_per_camera(monkeypatch):
    _FakeCapture.instances = []
    monkeypatch.setattr(block.cv2, "VideoCapture", _FakeCapture)
    cameras = ["cam_a", "cam_b", "cam_c"]

    images = block.read_teleavatar_episode_frames_opencv_sequential(
        "/dataset", 7, cameras, [0, 2, 4])

    assert set(images) == set(cameras)
    assert len(_FakeCapture.instances) == 3
    assert all(capture.released for capture in _FakeCapture.instances)
    assert all(tuple(value.shape) == (3, 3, 2, 3)
               for value in images.values())
    assert all(value.dtype == torch.uint8 for value in images.values())
    for value in images.values():
        torch.testing.assert_close(
            value[1, :, 0, 0],
            torch.tensor([4, 3, 2], dtype=torch.uint8),
        )


@pytest.mark.parametrize(
    "indices, message",
    [
        ([], "non-empty"),
        ([0, 0], "sorted unique"),
        ([2, 0], "sorted unique"),
        ([-1, 0], "non-negative"),
        ([False, 1], "integers"),
    ],
)
def test_sequential_opencv_rejects_invalid_indices(indices, message):
    with pytest.raises(ValueError, match=message):
        block.read_teleavatar_episode_frames_opencv_sequential(
            "/dataset", 0, ["cam"], indices)
