"""Task 13 Step 5:train_hiql_value.py 的 success_signed CLI + 多数据集特征合并。

act_feat 路:成功集与失败集各自标准化,必须恢复原始特征 → 联合 stats → 重标准化,
使全部 demo 落在同一特征空间(否则 online scorer 用 value.pt 的 mean/std 标准化在线特征时,
一半训练数据处在错误空间)。
"""
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    build_parser, merge_labeled_reads, validate_terminal_reward_cfg,
)


# ---- merge_labeled_reads:多数据集特征合并到同一空间 ----

def _read(raw_seqs):
    """模拟 read_per_demo_states(act_feat) 的返回:用该组自己的 stats 标准化。"""
    allf = np.concatenate(raw_seqs, axis=0)
    mean = allf.mean(axis=0).astype(np.float32)
    std = np.maximum(allf.std(axis=0), 1e-6).astype(np.float32)
    seqs_std = [((s - mean) / std).astype(np.float32) for s in raw_seqs]
    return seqs_std, mean, std


def test_merge_recovers_and_jointly_standardizes():
    """两组不同 stats 的读入合并后,全部输出必须落在同一(联合)标准化空间:
    concat 后逐维 mean≈0、std≈1。"""
    rng = np.random.default_rng(0)
    succ_raw = [rng.normal(5.0, 2.0, (8, 3)).astype(np.float32)]     # 均值/方差与失败组不同
    fail_raw = [rng.normal(-3.0, 0.5, (6, 3)).astype(np.float32)]
    s_seqs, s_mean, s_std = _read(succ_raw)
    f_seqs, f_mean, f_std = _read(fail_raw)

    seqs, flags, jmean, jstd = merge_labeled_reads([
        (s_seqs, s_mean, s_std, True),
        (f_seqs, f_mean, f_std, False),
    ])
    allout = np.concatenate(seqs, axis=0)
    np.testing.assert_allclose(allout.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(allout.std(axis=0), 1.0, atol=1e-4)


def test_merge_flags_align_with_demo_order():
    s_seqs, s_mean, s_std = _read([np.ones((4, 2), np.float32),
                                   np.ones((4, 2), np.float32)])   # 2 成功 demo
    f_seqs, f_mean, f_std = _read([np.zeros((4, 2), np.float32)])  # 1 失败 demo
    seqs, flags, _, _ = merge_labeled_reads([
        (s_seqs, s_mean, s_std, True),
        (f_seqs, f_mean, f_std, False),
    ])
    assert flags == [True, True, False]
    assert len(seqs) == 3


def test_merge_single_read_is_near_identity():
    """只有一组时,恢复原始→重标准化应约等于原输出(联合 stats == 该组 stats)。"""
    raw = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32)]
    seqs_std, mean, std = _read(raw)
    seqs, flags, jmean, jstd = merge_labeled_reads([(seqs_std, mean, std, True)])
    np.testing.assert_allclose(seqs[0], seqs_std[0], atol=1e-5)
    assert flags == [True]


def test_merge_joint_stats_are_saved_shape():
    s_seqs, s_mean, s_std = _read([np.ones((4, 3), np.float32)])
    f_seqs, f_mean, f_std = _read([np.zeros((4, 3), np.float32)])
    _, _, jmean, jstd = merge_labeled_reads([
        (s_seqs, s_mean, s_std, True), (f_seqs, f_mean, f_std, False)])
    assert jmean.shape == (3,) and jstd.shape == (3,)
    assert np.all(jstd > 0)


# ---- CLI ----

def test_terminal_reward_mode_defaults_to_legacy():
    args = build_parser().parse_args(["--hdf5", "x", "--dataset", "y"])
    assert args.terminal_reward_mode == "legacy"


def test_success_and_failure_dataset_are_repeatable():
    args = build_parser().parse_args([
        "--hdf5", "x", "--dataset", "y",
        "--terminal_reward_mode", "success_signed",
        "--success_dataset", "a", "--success_dataset", "b",
        "--failure_dataset", "c"])
    assert args.success_dataset == ["a", "b"]
    assert args.failure_dataset == ["c"]


def test_legacy_hdf5_still_required():
    """legacy 模式沿用既有严格性:缺 --hdf5 报错。"""
    args = build_parser().parse_args(["--dataset", "y"])   # 无 --hdf5
    with pytest.raises(SystemExit):
        validate_terminal_reward_cfg(args)


def test_success_signed_requires_both_success_and_failure():
    args = build_parser().parse_args([
        "--hdf5", "x", "--dataset", "y",
        "--terminal_reward_mode", "success_signed",
        "--success_dataset", "a", "--state_mode", "act_feat"])   # 缺 failure
    with pytest.raises(SystemExit):
        validate_terminal_reward_cfg(args)


def test_success_signed_rejects_non_actfeat_state_mode():
    """eef/eef_piece 的 success_signed 暂不支持,须明确报错而非静默走错路。"""
    args = build_parser().parse_args([
        "--hdf5", "x", "--dataset", "y",
        "--terminal_reward_mode", "success_signed",
        "--success_dataset", "a", "--failure_dataset", "b",
        "--state_mode", "eef"])
    with pytest.raises(SystemExit):
        validate_terminal_reward_cfg(args)


def test_success_signed_actfeat_passes_validation():
    args = build_parser().parse_args([
        "--hdf5", "x", "--dataset", "y",
        "--terminal_reward_mode", "success_signed",
        "--success_dataset", "a", "--failure_dataset", "b",
        "--state_mode", "act_feat"])
    validate_terminal_reward_cfg(args)   # 不抛
