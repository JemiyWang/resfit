import pytest

from resfit.rl_finetuning.chunk_residual.bc_schedule import linear_bc_coef


def test_start_returns_c0():
    assert linear_bc_coef(0, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.1)


def test_end_returns_c_final():
    assert linear_bc_coef(500_000, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.01)


def test_midpoint_is_average():
    assert linear_bc_coef(250_000, c0=0.1, c_final=0.0, total_steps=500_000) == pytest.approx(0.05)


def test_beyond_end_clipped_to_c_final():
    assert linear_bc_coef(900_000, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.01)


def test_negative_steps_clipped_to_c0():
    assert linear_bc_coef(-100, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.1)


def test_nonpositive_total_steps_returns_c_final():
    # 防除零:total_steps<=0(含负值)退化为终值
    assert linear_bc_coef(0, c0=0.1, c_final=0.01, total_steps=0) == pytest.approx(0.01)
    assert linear_bc_coef(123, c0=0.1, c_final=0.01, total_steps=-1) == pytest.approx(0.01)


def test_flat_when_final_equals_c0():
    for t in (0, 123_456, 500_000):
        assert linear_bc_coef(t, c0=0.1, c_final=0.1, total_steps=500_000) == pytest.approx(0.1)


def test_concrete_quarter_point():
    # c0=0.1, c_final=0.01, t=125000/500000=0.25 -> 0.1 + (0.01-0.1)*0.25 = 0.0775
    assert linear_bc_coef(125_000, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.0775)
