"""train_chunk_residual 金标准默认值断言(2026-06-10)。"""
import types

import pytest

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    _validate_base_action_interface,
    build_parser,
)


def test_value_defaults_aligned_to_golden():
    """3 个带值参数 CLI 默认 = 金标准;旧值可显式回退。
    base_n_action_steps 不纳入改默认(pi05 base 兼容,保留 None)。"""
    a = build_parser().parse_args([])
    assert a.chunk_length == 1
    assert a.actor_lr == 1e-6
    assert a.base_action_mode == "queue"
    assert a.base_n_action_steps is None  # 不纳入改默认(pi05 base 兼容)
    b = build_parser().parse_args(["--chunk_length", "20", "--actor_lr", "5e-6",
                                   "--base_action_mode", "replan"])
    assert b.chunk_length == 20
    assert b.actor_lr == 5e-6
    assert b.base_action_mode == "replan"


def test_stage_balanced_default_on():
    """stage_balanced 默认 True;--no_stage_balanced 可关;--stage_balanced 仍可显式开。"""
    assert build_parser().parse_args([]).stage_balanced is True
    assert build_parser().parse_args(["--no_stage_balanced"]).stage_balanced is False
    assert build_parser().parse_args(["--stage_balanced"]).stage_balanced is True


def test_pi05_replan_accepts_chunk_capable_bridge():
    args = types.SimpleNamespace(
        base_policy_type="pi05", base_action_mode="replan", chunk_length=50)
    base = types.SimpleNamespace(get_action_chunk=lambda obs, length: None)
    _validate_base_action_interface(args, base)


def test_pi05_replan_rejects_step_only_adapter():
    args = types.SimpleNamespace(
        base_policy_type="pi05", base_action_mode="replan", chunk_length=50)
    with pytest.raises(AssertionError, match="get_action_chunk"):
        _validate_base_action_interface(args, types.SimpleNamespace())
