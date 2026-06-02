"""stage-aware 诊断聚合:按 stage 分组求各指标均值 + 计数。"""
import torch

from resfit.rl_finetuning.chunk_residual.stage_diag import (
    flatten_stage_diagnostics,
    stage_diagnostics,
)


def test_counts_per_stage():
    stages = torch.tensor([0, 0, 2, 2, 2])
    out = stage_diagnostics(stages, {})
    assert out["counts"] == {0: 2, 2: 3}


def test_mean_per_stage():
    stages = torch.tensor([0, 0, 2, 2])
    out = stage_diagnostics(stages, {"q": torch.tensor([1.0, 3.0, 10.0, 20.0])})
    assert out["means"]["q"][0] == 2.0
    assert out["means"]["q"][2] == 15.0


def test_multiple_metrics():
    stages = torch.tensor([0, 1])
    out = stage_diagnostics(stages, {"q": torch.tensor([1.0, 2.0]),
                                     "rnorm": torch.tensor([0.5, 1.5])})
    assert out["means"]["q"] == {0: 1.0, 1: 2.0}
    assert out["means"]["rnorm"] == {0: 0.5, 1: 1.5}


def test_float_stage_labels_cast_to_int():
    stages = torch.tensor([0.0, 0.0, 2.0])
    out = stage_diagnostics(stages, {"q": torch.tensor([1.0, 1.0, 5.0])})
    assert set(out["counts"].keys()) == {0, 2}
    assert out["means"]["q"][2] == 5.0


def test_single_stage():
    stages = torch.tensor([3, 3, 3])
    out = stage_diagnostics(stages, {"q": torch.tensor([1.0, 2.0, 3.0])})
    assert out["counts"] == {3: 3}
    assert abs(out["means"]["q"][3] - 2.0) < 1e-6


def test_empty_values_only_counts():
    stages = torch.tensor([0, 1, 1])
    out = stage_diagnostics(stages, {})
    assert out["counts"] == {0: 1, 1: 2}
    assert out["means"] == {}


def test_extra_dims_reduced_per_sample():
    # 指标可能是 [N, D](如逐维残差),应按样本聚合(先对非样本维求范数/均值由调用方决定;
    # 这里约定:传入已是 [N] 标量。多维直接报错以暴露误用)
    stages = torch.tensor([0, 0])
    import pytest
    with pytest.raises(Exception):
        stage_diagnostics(stages, {"bad": torch.zeros(2, 5)})


def test_flatten_for_logging():
    out = {"counts": {0: 2, 2: 3}, "means": {"q": {0: 2.0, 2: 15.0}}}
    flat = flatten_stage_diagnostics(out, prefix="diag")
    assert flat["diag/count/stage0"] == 2
    assert flat["diag/count/stage2"] == 3
    assert flat["diag/q/stage0"] == 2.0
    assert flat["diag/q/stage2"] == 15.0
