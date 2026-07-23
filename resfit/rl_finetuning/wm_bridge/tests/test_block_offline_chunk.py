import pandas as pd
import pytest

from resfit.rl_finetuning.wm_bridge.block_offline_chunk import (
    catalog_episodes,
    count_block_chunk_transitions,
    plan_episode_chunks,
)


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
