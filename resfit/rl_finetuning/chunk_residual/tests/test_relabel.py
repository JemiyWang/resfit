from resfit.rl_finetuning.chunk_residual.relabel import (
    productive_prefix_len, RelabelHarvester,
)


def test_prefix_drops_floundering_tail():
    assert productive_prefix_len([0, 1, 1, 2, 2], min_stage=1) == 4


def test_prefix_zero_when_no_advance():
    assert productive_prefix_len([0, 0, 0], min_stage=1) == 0


def test_prefix_full_when_monotone_climb():
    assert productive_prefix_len([1, 2, 3, 4], min_stage=1) == 4


def test_prefix_empty_seq():
    assert productive_prefix_len([], min_stage=1) == 0


def test_prefix_min_stage_filter():
    assert productive_prefix_len([0, 1, 1], min_stage=2) == 0     # max=1 < 2 -> 0


def test_harvester_returns_prefix_and_resets():
    h = RelabelHarvester(min_stage=1)
    for e, s in [("a", 0), ("b", 1), ("c", 1), ("d", 2), ("e", 2)]:
        h.add(e, s)
    assert h.flush() == ["a", "b", "c", "d"]    # 到最后推进(stage2 首次=idx3)含
    assert h.flush() == []                        # 已清空


def test_harvester_no_advance_returns_empty():
    h = RelabelHarvester(min_stage=1)
    h.add("a", 0)
    h.add("b", 0)
    assert h.flush() == []
