"""train_chunk_residual 金标准默认值断言(2026-06-10)。"""
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def test_value_defaults_aligned_to_golden():
    """4 个带值参数 CLI 默认 = 金标准;旧值可显式回退。"""
    a = build_parser().parse_args([])
    assert a.chunk_length == 1
    assert a.actor_lr == 1e-6
    assert a.base_action_mode == "queue"
    assert a.base_n_action_steps == 10
    b = build_parser().parse_args(["--chunk_length", "20", "--actor_lr", "5e-6",
                                   "--base_action_mode", "replan", "--base_n_action_steps", "5"])
    assert b.chunk_length == 20
    assert b.actor_lr == 5e-6
    assert b.base_action_mode == "replan"
    assert b.base_n_action_steps == 5
