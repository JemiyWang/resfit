import numpy as np
import pandas as pd
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    concat_mixed_batch,
)
from resfit.rl_finetuning.wm_bridge.block_offline_chunk import (
    EpisodeRef,
    build_block_offline_buffer,
    catalog_episodes,
    collect_episode_endpoints,
    count_block_chunk_transitions,
    plan_episode_chunks,
)
from resfit.rl_finetuning.wm_bridge.block_offline_cache import EndpointRecord
from resfit.rl_finetuning.wm_bridge.wm_driver import CAMERA_KEYS


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    [
        (50, ()),
        (51, ((0, 50, True),)),
        (101, ((0, 50, False), (50, 100, True))),
        (120, ((0, 50, False), (50, 100, False), (69, 119, True))),
    ],
)
def test_plan_episode_chunks(num_frames, expected):
    got = tuple(
        (x.start, x.end, x.terminal) for x in plan_episode_chunks(num_frames)
    )
    assert got == expected


def test_chunk_plan_has_no_duplicate_aligned_terminal():
    got = plan_episode_chunks(151)
    assert [x.start for x in got] == [0, 50, 100]
    assert sum(x.terminal for x in got) == 1


def _write_episode_parquet(root, episode_id, num_frames):
    chunk = episode_id // 1000
    path = (
        root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_id:06d}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame_index": range(num_frames)}).to_parquet(path)


def test_catalog_and_count_use_sorted_episode_ids(tmp_path):
    _write_episode_parquet(tmp_path, episode_id=2, num_frames=120)
    _write_episode_parquet(tmp_path, episode_id=0, num_frames=51)
    episodes = catalog_episodes(str(tmp_path))
    assert [x.episode_id for x in episodes] == [0, 2]
    assert count_block_chunk_transitions(str(tmp_path)) == 4


def test_num_demos_limits_sorted_prefix(tmp_path):
    _write_episode_parquet(tmp_path, episode_id=0, num_frames=51)
    _write_episode_parquet(tmp_path, episode_id=1, num_frames=101)
    assert count_block_chunk_transitions(str(tmp_path), num_demos=1) == 1


def test_catalog_sorts_numeric_episode_ids_across_chunk_boundary(tmp_path):
    _write_episode_parquet(tmp_path, episode_id=1_000_000, num_frames=51)
    _write_episode_parquet(tmp_path, episode_id=999_000, num_frames=51)

    episodes = catalog_episodes(str(tmp_path))

    assert [episode.episode_id for episode in episodes] == [999_000, 1_000_000]
    assert [episode.episode_id for episode in catalog_episodes(str(tmp_path), 1)] == [
        999_000
    ]


def synthetic_episode(root, num_frames=51, episode_id=0):
    return EpisodeRef(
        root=str(root),
        episode_id=episode_id,
        parquet_path=str(root / f"episode_{episode_id:06d}.parquet"),
        video_paths={
            key: str(root / f"{episode_id}-{key}.mp4") for key in CAMERA_KEYS
        },
        num_frames=num_frames,
    )


class FakeReader:
    def __init__(self, episode=None, num_frames=None):
        n = num_frames if num_frames is not None else episode.num_frames
        self.actions = np.stack([
            np.full(16, i, dtype=np.float32) for i in range(n)
        ])
        self.states = np.stack([
            np.full(16, i, dtype=np.float32) for i in range(n)
        ])

    def native_frames(self, frame_index):
        return np.zeros(
            (len(CAMERA_KEYS), 3, 192, 256), dtype=np.float32)


class FakeBase:
    def __init__(self):
        self.tokens = []

    def query(self, obs):
        token = obs["_wm_window_token"]
        self.tokens.append(token)
        value = float(token.rsplit(":", 1)[-1])
        return (
            np.full((50, 16), value, np.float32),
            np.array([value, value + 1], np.float32),
        )


class MemoryEndpointStore:
    def __init__(self):
        self.records = {}

    def load_episode(self, episode_id):
        return self.records.get(episode_id)

    def save_episode(self, episode_id, record):
        self.records[episode_id] = record


class IdentityActionScaler:
    def scale(self, value):
        return value.float()


class IdentityStateStandardizer:
    def standardize(self, value):
        return value.float()


class SumFeatureScorer:
    def phi(self, feature, proprio):
        return float(np.asarray(feature, dtype=np.float32).sum())

    @staticmethod
    def phi_for_frame(frame_index):
        return 2.0 * frame_index + 1.0


class ListReplay:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)


def test_collect_episode_endpoints_queries_each_unique_frame_once(
    tmp_path, capsys,
):
    slices = plan_episode_chunks(120)
    reader = FakeReader(num_frames=120)
    base = FakeBase()
    record = collect_episode_endpoints(
        reader, episode_id=3, slices=slices, base_policy=base)
    assert record.frame_indices.tolist() == [0, 50, 69, 100, 119]
    assert len(base.tokens) == 5
    assert record.base_actions.shape == (5, 50, 16)
    assert record.proprio.shape == (5, 16)
    output = capsys.readouterr().out
    assert "episode=3 frame=0" in output
    assert "episode=3 frame=119" in output


def test_build_transition_matches_online_schema_and_exact_pbrs(tmp_path):
    rb = ListReplay()
    stats = build_block_offline_buffer(
        rb,
        str(tmp_path),
        action_scaler=IdentityActionScaler(),
        state_standardizer=IdentityStateStandardizer(),
        image_keys=list(CAMERA_KEYS),
        gamma=0.5,
        num_demos=None,
        base_policy=FakeBase(),
        scorer=SumFeatureScorer(),
        endpoint_store=MemoryEndpointStore(),
        episodes=(synthetic_episode(tmp_path, num_frames=51),),
        reader_factory=FakeReader,
    )
    item = rb.items[0][0]
    assert tuple(item["action"].shape) == (800,)
    assert tuple(item["obs"]["observation.base_action"].shape) == (800,)
    assert tuple(item["next"]["obs"]["observation.base_action"].shape) == (800,)
    assert item["next"]["done"].item() is True
    assert item["next"]["reward"].item() == pytest.approx(
        0.5 * SumFeatureScorer.phi_for_frame(50)
        - SumFeatureScorer.phi_for_frame(0)
    )
    assert stats.transitions == 1
    assert stats.expert_norm_mean > 0.0
    assert stats.base_norm_mean == 0.0
    assert stats.residual_norm_mean > 0.0
    for key in CAMERA_KEYS:
        assert item["obs"][key].dtype == torch.uint8
        assert tuple(item["obs"][key].shape) == (3, 84, 84)
        assert item["next"]["obs"][key].dtype == torch.uint8
    assert tuple(item["obs"]["observation.state"].shape) == (16,)
    assert tuple(item["next"]["obs"]["observation.state"].shape) == (16,)
    assert tuple(item["obs"]["observation.stage_id"].shape) == (1,)
    assert tuple(item["next"]["obs"]["observation.stage_id"].shape) == (1,)
    assert item["obs"]["observation.stage_id"].item() == 0.0
    assert item["next"]["obs"]["observation.stage_id"].item() == 0.0
    assert item["max_stage"].item() == 0.0
    assert item["_priority"].item() == 10.0


def _run_fake_build(tmp_path, num_frames, store, base):
    rb = ListReplay()
    build_block_offline_buffer(
        rb,
        str(tmp_path),
        action_scaler=IdentityActionScaler(),
        state_standardizer=IdentityStateStandardizer(),
        image_keys=list(CAMERA_KEYS),
        gamma=0.5,
        num_demos=None,
        base_policy=base,
        scorer=SumFeatureScorer(),
        endpoint_store=store,
        episodes=(synthetic_episode(tmp_path, num_frames=num_frames),),
        reader_factory=FakeReader,
    )
    return rb


def test_second_build_reuses_endpoint_cache_without_base_queries(tmp_path):
    store = MemoryEndpointStore()
    _run_fake_build(tmp_path, 120, store, FakeBase())
    second_base = FakeBase()
    _run_fake_build(tmp_path, 120, store, second_base)
    assert second_base.tokens == []


@pytest.mark.parametrize("indices", ([0], [0, 0, 50]))
def test_build_rebuilds_incomplete_or_duplicate_endpoint_cache(
    tmp_path, indices,
):
    store = MemoryEndpointStore()
    n = len(indices)
    store.records[0] = EndpointRecord(
        frame_indices=np.asarray(indices, dtype=np.int64),
        base_actions=np.zeros((n, 50, 16), dtype=np.float32),
        prefix_features=np.zeros((n, 2), dtype=np.float32),
        proprio=np.zeros((n, 16), dtype=np.float32),
    )
    base = FakeBase()

    stats = build_block_offline_buffer(
        ListReplay(),
        str(tmp_path),
        action_scaler=IdentityActionScaler(),
        state_standardizer=IdentityStateStandardizer(),
        image_keys=list(CAMERA_KEYS),
        gamma=0.5,
        num_demos=None,
        base_policy=base,
        scorer=SumFeatureScorer(),
        endpoint_store=store,
        episodes=(synthetic_episode(tmp_path, num_frames=51),),
        reader_factory=FakeReader,
    )

    assert stats.transitions == 1
    assert stats.endpoint_hits == 0
    assert stats.endpoint_misses == 1
    assert base.tokens == ["offline:0:0", "offline:0:50"]


def test_build_rebuilds_damaged_endpoint_cache(tmp_path):
    class DamagedStore(MemoryEndpointStore):
        def load_episode(self, episode_id):
            raise ValueError("damaged endpoint cache")

    base = FakeBase()
    stats = build_block_offline_buffer(
        ListReplay(),
        str(tmp_path),
        action_scaler=IdentityActionScaler(),
        state_standardizer=IdentityStateStandardizer(),
        image_keys=list(CAMERA_KEYS),
        gamma=0.5,
        num_demos=None,
        base_policy=base,
        scorer=SumFeatureScorer(),
        endpoint_store=DamagedStore(),
        episodes=(synthetic_episode(tmp_path, num_frames=51),),
        reader_factory=FakeReader,
    )

    assert stats.transitions == 1
    assert stats.endpoint_hits == 0
    assert stats.endpoint_misses == 1


def test_unaligned_terminal_order_done_and_residual_target(tmp_path):
    rb = _run_fake_build(tmp_path, 120, MemoryEndpointStore(), FakeBase())
    starts = [int(batch[0]["action"][0].item()) for batch in rb.items]
    dones = [bool(batch[0]["next"]["done"].item()) for batch in rb.items]
    assert starts == [0, 50, 69]
    assert dones == [False, False, True]
    terminal = rb.items[-1][0]
    residual = terminal["action"] - terminal["obs"]["observation.base_action"]
    expected = torch.arange(50, dtype=torch.float32).repeat_interleave(16)
    torch.testing.assert_close(residual, expected)


def test_offline_transition_concatenates_with_online_stage_shape(tmp_path):
    rb = _run_fake_build(tmp_path, 51, MemoryEndpointStore(), FakeBase())
    offline_batch = rb.items[0]
    online_batch = offline_batch.clone()
    online_batch["obs", "observation.stage_id"] = torch.zeros((1, 1))
    online_batch["next", "obs", "observation.stage_id"] = torch.zeros((1, 1))

    mixed = concat_mixed_batch(online_batch, offline_batch)

    assert tuple(mixed["obs", "observation.stage_id"].shape) == (2, 1)
    assert tuple(mixed["next", "obs", "observation.stage_id"].shape) == (2, 1)


def test_build_rejects_image_keys_that_do_not_match_fixed_cameras(tmp_path):
    with pytest.raises(ValueError, match="image_keys"):
        build_block_offline_buffer(
            ListReplay(),
            str(tmp_path),
            action_scaler=IdentityActionScaler(),
            state_standardizer=IdentityStateStandardizer(),
            image_keys=list(CAMERA_KEYS[:-1]),
            gamma=0.5,
            num_demos=None,
            base_policy=FakeBase(),
            scorer=SumFeatureScorer(),
            endpoint_store=MemoryEndpointStore(),
            episodes=(synthetic_episode(tmp_path, num_frames=51),),
            reader_factory=FakeReader,
        )


class Float32OverflowScorer:
    def phi(self, feature, proprio):
        return 0.0 if float(np.asarray(feature)[0]) == 0.0 else 1e40


def test_build_rejects_reward_that_overflows_float32(tmp_path):
    with pytest.raises(ValueError, match="reward must be finite float32"):
        build_block_offline_buffer(
            ListReplay(),
            str(tmp_path),
            action_scaler=IdentityActionScaler(),
            state_standardizer=IdentityStateStandardizer(),
            image_keys=list(CAMERA_KEYS),
            gamma=1.0,
            num_demos=None,
            base_policy=FakeBase(),
            scorer=Float32OverflowScorer(),
            endpoint_store=MemoryEndpointStore(),
            episodes=(synthetic_episode(tmp_path, num_frames=51),),
            reader_factory=FakeReader,
        )
