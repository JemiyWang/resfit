import json as _json

import pytest

from resfit.rl_finetuning.chunk_residual.libero_offline import libero_task_language


def test_libero_task_language_moka():
    pytest.importorskip("libero")   # 仅此测试需真 libero;residual env / 无 PYTHONPATH 时优雅跳过
    lang = libero_task_language("libero_10", 8)
    assert lang == "put both moka pots on the stove"


def _make_stub_root(tmp_path, episodes):
    """episodes: list[(episode_index, language, length)] → 建 meta/episodes.jsonl + 空 parquet 占位。"""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for ei, lang, length in episodes:
            f.write(_json.dumps({"episode_index": ei, "tasks": [lang], "length": length}) + "\n")
    for ei, _, _ in episodes:
        d = root / "data" / f"chunk-{ei // 1000:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"episode_{ei:06d}.parquet").write_text("stub")
    return str(root)


def test_find_demo_episodes_filters_by_language(tmp_path):
    from resfit.rl_finetuning.chunk_residual.libero_offline import find_demo_episodes
    root = _make_stub_root(tmp_path, [(0, "task A", 5), (1, "task B", 7), (1001, "task A", 9)])
    paths = find_demo_episodes(root, "task A")
    assert len(paths) == 2
    assert paths[0].endswith("data/chunk-000/episode_000000.parquet")
    assert paths[1].endswith("data/chunk-001/episode_001001.parquet")


def test_find_demo_episodes_no_match_raises(tmp_path):
    from resfit.rl_finetuning.chunk_residual.libero_offline import find_demo_episodes
    import pytest
    root = _make_stub_root(tmp_path, [(0, "task A", 5)])
    with pytest.raises(ValueError):
        find_demo_episodes(root, "nonexistent task")
