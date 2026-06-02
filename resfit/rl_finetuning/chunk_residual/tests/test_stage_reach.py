"""per-stage 到达率纯聚合:给定每 episode 最高 stage,算各 stage 的到达比例。"""
from resfit.rl_finetuning.chunk_residual.stage_reach import stage_reach_rates


def test_known_distribution():
    # 5 个 episode 的最高 stage = [4,3,3,2,3];num_stages=5
    rates = stage_reach_rates([4, 3, 3, 2, 3], num_stages=5)
    assert rates == {1: 1.0, 2: 1.0, 3: 0.8, 4: 0.2}


def test_reach_rate_monotone_non_increasing():
    # 到达率必随 stage 升高单调不增(到 stage k+1 必先到 k)
    rates = stage_reach_rates([0, 1, 2, 3, 4], num_stages=5)
    vals = [rates[k] for k in sorted(rates)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_all_stage_zero_gives_zero_rates():
    rates = stage_reach_rates([0, 0, 0], num_stages=5)
    assert rates == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


def test_all_success_gives_one_rates():
    rates = stage_reach_rates([4, 4, 4], num_stages=5)
    assert rates == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}


def test_empty_returns_zeros():
    rates = stage_reach_rates([], num_stages=5)
    assert rates == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
