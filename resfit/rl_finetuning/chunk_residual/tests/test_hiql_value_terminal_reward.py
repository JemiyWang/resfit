"""Task 13:HIQL V 的 ±1 终端奖励(成功末 +1、失败末 −1)。

核心红线:reward=None(默认 legacy)时,V 训练必须与既有实现逐位相同——既有仿真 run
(three_piece / pouring / lifttray 那批)的 V 不能被这次改动扰动。
"""
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_value import (
    build_transitions, build_transitions_with_rewards, train_value,
)


def _seqs():
    return [np.arange(8, dtype=np.float32).reshape(4, 2),      # T=4 → 3 transition
            np.arange(6, dtype=np.float32).reshape(3, 2),      # T=3 → 2 transition
            np.zeros((1, 2), dtype=np.float32)]                # T=1 → 应被跳过


def test_done_matches_legacy_build_transitions():
    """★ 回归锁:新函数的 s/s_next/done 必须与既有 build_transitions 逐位相同。"""
    s0, sn0, d0 = build_transitions(_seqs())
    s1, sn1, d1, _ = build_transitions_with_rewards(_seqs(), [True, False, True])
    torch.testing.assert_close(s0, s1)
    torch.testing.assert_close(sn0, sn1)
    torch.testing.assert_close(d0, d1)


def test_short_sequences_skipped_consistently():
    """T<2 的 demo 被跳过时,success_flags 必须跟着跳,不能错位。"""
    _, _, done, reward = build_transitions_with_rewards(
        _seqs(), [True, False, True])
    # seq0(成功)产 3 个 transition,seq1(失败)产 2 个 → 共 5;seq2(T=1)整条被跳
    assert done.shape[0] == 5
    assert reward.shape[0] == 5
    assert float(reward[2]) == pytest.approx(1.0)    # seq0 末 → 成功 +1
    assert float(reward[4]) == pytest.approx(-1.0)   # seq1 末 → 失败 −1


def test_intermediate_rewards_are_zero():
    _, _, _, reward = build_transitions_with_rewards(_seqs(), [True, False, True])
    for i in (0, 1, 3):
        assert float(reward[i]) == 0.0


def test_reward_terminal_aligns_with_done_terminal():
    """非零 reward 只出现在 done==1 的位置(轨迹末端)。"""
    _, _, done, reward = build_transitions_with_rewards(_seqs(), [True, False, True])
    nonzero = (reward != 0).reshape(-1)
    terminal = (done == 1).reshape(-1)
    assert torch.equal(nonzero, terminal)


def test_flags_length_mismatch_raises():
    with pytest.raises(AssertionError):
        build_transitions_with_rewards(_seqs(), [True, False])


def test_reward_shape_is_column():
    _, _, _, reward = build_transitions_with_rewards(_seqs(), [True, False, True])
    assert reward.shape == (5, 1)


def test_train_value_default_is_bitwise_legacy():
    """★ 回归锁:reward=None 时必须与不传 reward 的现状逐位相同。"""
    s, sn, done = build_transitions(_seqs())
    m0, v0 = train_value(s, sn, done, steps=50, seed=0)
    m1, v1 = train_value(s, sn, done, reward=None, steps=50, seed=0)
    for p0, p1 in zip(m0.parameters(), m1.parameters()):
        torch.testing.assert_close(p0, p1)
    assert v0 == v1


def test_failure_terminal_gets_lower_value_than_success_terminal():
    """训完后,失败轨迹末态的 V 应显著低于成功轨迹末态。"""
    succ = [np.linspace(0, 1, 20, dtype=np.float32).reshape(10, 2)]
    fail = [np.linspace(0, -1, 20, dtype=np.float32).reshape(10, 2)]
    s, sn, done, reward = build_transitions_with_rewards(succ + fail, [True, False])
    model, _ = train_value(s, sn, done, reward=reward, steps=3000, seed=0)
    with torch.no_grad():
        v_succ = model(torch.from_numpy(succ[0][-1])).item()
        v_fail = model(torch.from_numpy(fail[0][-1])).item()
    assert v_succ > v_fail
