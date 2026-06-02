"""stage-balanced 采样核心:按 stage 配额从 buffer 选索引。

n-step 已在 add 时烤进 transition(MultiStepTransform._inv_call),采样为 identity,
故按任意索引选取是安全的。本单元只负责"选哪些索引"。
"""
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from resfit.rl_finetuning.chunk_residual.stage_replay import (
    sample_stage_balanced,
    stage_balanced_indices,
)


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_returns_exact_batch_size():
    stages = torch.tensor([0] * 90 + [2] * 10)
    out = stage_balanced_indices(stages, batch_size=64, generator=_gen())
    assert out.shape == (64,)


def test_equal_quota_balances_present_stages():
    # 90 个 stage0 + 10 个 stage2,无 quotas → 两个present阶段各占一半
    stages = torch.tensor([0] * 90 + [2] * 10)
    out = stage_balanced_indices(stages, batch_size=100, generator=_gen())
    assert int((stages[out] == 0).sum()) == 50
    assert int((stages[out] == 2).sum()) == 50


def test_custom_quota_proportions_exact():
    stages = torch.tensor([0] * 90 + [2] * 10)
    out = stage_balanced_indices(stages, batch_size=100,
                                 quotas={0: 0.25, 2: 0.75}, generator=_gen())
    assert int((stages[out] == 0).sum()) == 25
    assert int((stages[out] == 2).sum()) == 75


def test_rare_stage_oversampled_with_replacement():
    # stage2 只有 3 个样本,却要 50 个 → 必须有放回(索引会重复)
    stages = torch.tensor([0] * 97 + [2] * 3)
    out = stage_balanced_indices(stages, batch_size=100, generator=_gen())
    s2 = out[stages[out] == 2]
    assert int((stages[out] == 2).sum()) == 50
    assert s2.unique().numel() <= 3            # 只能来自那 3 个样本
    assert set(s2.unique().tolist()) <= {97, 98, 99}


def test_absent_stage_in_quota_is_renormalized():
    # quotas 指定了 stage1,但 buffer 里没有 stage1 → 丢弃并把比例归一到 0/2
    stages = torch.tensor([0] * 50 + [2] * 50)
    out = stage_balanced_indices(stages, batch_size=100,
                                 quotas={0: 0.5, 1: 0.25, 2: 0.25}, generator=_gen())
    assert (stages[out] == 1).sum() == 0
    assert int((stages[out] == 0).sum()) + int((stages[out] == 2).sum()) == 100
    # 0:2 = 0.5:0.25 → 归一后 2:1 → 约 67:33
    assert int((stages[out] == 0).sum()) == 67


def test_single_stage_all_from_it():
    stages = torch.tensor([0] * 40)
    out = stage_balanced_indices(stages, batch_size=32, generator=_gen())
    assert (stages[out] == 0).all()
    assert (out < 40).all()


def test_deterministic_with_seeded_generator():
    stages = torch.tensor([0] * 90 + [2] * 10)
    a = stage_balanced_indices(stages, batch_size=50, generator=_gen(123))
    b = stage_balanced_indices(stages, batch_size=50, generator=_gen(123))
    assert torch.equal(a, b)


def test_indices_in_range():
    stages = torch.tensor([0] * 30 + [1] * 30 + [2] * 30 + [3] * 10)
    out = stage_balanced_indices(stages, batch_size=128, generator=_gen())
    assert (out >= 0).all() and (out < stages.numel()).all()


# ---------- 集成:对真 torchrl buffer 采样 ----------
def _make_buffer_with_stages(counts):
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=1000), batch_size=1)
    for s, n in counts:
        for _ in range(n):
            rb.add(TensorDict({"obs": torch.randn(3),
                               "max_stage": torch.tensor(float(s))}, batch_size=[]))
    return rb


def test_sample_stage_balanced_reads_buffer_and_balances():
    rb = _make_buffer_with_stages([(0, 90), (2, 10)])
    batch = sample_stage_balanced(rb, batch_size=100, generator=_gen())
    ms = batch["max_stage"]
    assert ms.shape == (100,)
    assert int((ms == 0).sum()) == 50
    assert int((ms == 2).sum()) == 50


def test_sample_stage_balanced_custom_quota():
    rb = _make_buffer_with_stages([(0, 90), (2, 10)])
    batch = sample_stage_balanced(rb, batch_size=100, quotas={0: 0.2, 2: 0.8}, generator=_gen())
    ms = batch["max_stage"]
    assert int((ms == 0).sum()) == 20
    assert int((ms == 2).sum()) == 80
