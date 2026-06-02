"""staged_bonus 纯函数:按本 chunk 的 stage 推进量给 additive 加分。"""
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import staged_bonus


def test_staged_bonus_zero_delta():
    assert staged_bonus(2, 2, 1.0) == 0.0


def test_staged_bonus_positive_delta():
    assert staged_bonus(0, 2, 1.0) == 2.0


def test_staged_bonus_uses_magnitude():
    assert staged_bonus(1, 3, 0.5) == 1.0


def test_staged_bonus_negative_delta_clamped():
    assert staged_bonus(3, 1, 1.0) == 0.0


def test_staged_bonus_zero_bonus_disabled():
    assert staged_bonus(0, 4, 0.0) == 0.0
