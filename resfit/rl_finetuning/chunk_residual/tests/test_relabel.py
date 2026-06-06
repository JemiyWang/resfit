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


import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from resfit.rl_finetuning.chunk_residual.relabel import sample_bc_batch


def _tiny_rb(n):
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=max(n, 1), device="cpu"), batch_size=4)
    for _ in range(n):
        td = TensorDict({
            "obs": TensorDict({"observation.state": torch.randn(3)}, batch_size=[]),
            "action": torch.randn(5),
        }, batch_size=[]).unsqueeze(0)
        rb.add(td)
    return rb


def test_sample_bc_batch_mixes_when_relabel_full():
    bc = sample_bc_batch(_tiny_rb(20), _tiny_rb(20), batch_size=8, device="cpu")
    assert bc.shape[0] == 8                       # half relabel + half demo


def test_sample_bc_batch_falls_back_to_demo_when_relabel_short():
    bc = sample_bc_batch(_tiny_rb(1), _tiny_rb(20), batch_size=8, device="cpu")   # relabel<half(4)
    assert bc.shape[0] == 8                       # 整批来自 demo


def test_sample_bc_batch_none_relabel_uses_demo():
    bc = sample_bc_batch(None, _tiny_rb(20), batch_size=8, device="cpu")
    assert bc.shape[0] == 8


def test_cli_relabel_defaults():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    a = build_parser().parse_args([])
    assert a.relabel is False
    assert a.relabel_buffer_size == 50_000
    assert a.relabel_min_stage == 1


def test_cli_relabel_parses():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    a = build_parser().parse_args(["--relabel", "--relabel_buffer_size", "1000",
                                   "--relabel_min_stage", "2"])
    assert a.relabel is True
    assert a.relabel_buffer_size == 1000
    assert a.relabel_min_stage == 2
