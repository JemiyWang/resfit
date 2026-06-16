import torch
from torch import nn

from resfit.rl_finetuning.chunk_residual.act_feature import act_weight_fingerprint


def _tiny():
    torch.manual_seed(0)
    m = nn.Linear(4, 3)
    m.register_buffer("ctr", torch.tensor([7], dtype=torch.int64))  # 整型缓冲
    return m


def test_fingerprint_deterministic():
    m = _tiny()
    assert act_weight_fingerprint(m) == act_weight_fingerprint(m)


def test_fingerprint_sensitive_to_weight_change():
    m = _tiny()
    h0 = act_weight_fingerprint(m)
    with torch.no_grad():
        m.weight.add_(1e-3)
    assert act_weight_fingerprint(m) != h0


def test_fingerprint_handles_int_buffer():
    # 含 int64 buffer 不崩，且改 buffer 值 → hash 变（身份敏感）
    m = _tiny()
    h0 = act_weight_fingerprint(m)
    m.ctr.fill_(9)
    assert act_weight_fingerprint(m) != h0


def test_fingerprint_is_hexdigest_str():
    h = act_weight_fingerprint(_tiny())
    assert isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_fingerprint_differs_for_different_structure():
    torch.manual_seed(0)
    m1 = nn.Linear(4, 3)
    m2 = nn.Linear(3, 4)
    assert act_weight_fingerprint(m1) != act_weight_fingerprint(m2)


import warnings
import pytest

from resfit.rl_finetuning.chunk_residual.act_feature import assert_act_base_samesource


def test_samesource_old_artifact_warns_and_skips():
    # gv_sha=None（旧产物）→ warn，不 raise
    with pytest.warns(UserWarning):
        assert_act_base_samesource(gv_sha=None, cache_sha=None, base_sha="x")


def test_samesource_match_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # 任何 warning 都会失败
        assert_act_base_samesource(gv_sha="a", cache_sha="a", base_sha="a")  # 不抛即通过


def test_samesource_offline_inconsistent_raises():
    # cache 与 gc_value 两边都有指纹且不等 → 离线异源
    with pytest.raises(ValueError):
        assert_act_base_samesource(gv_sha="a", cache_sha="b", base_sha="a")


def test_samesource_online_mismatch_raises_by_default():
    with pytest.raises(ValueError):
        assert_act_base_samesource(gv_sha="a", cache_sha="a", base_sha="b")


def test_samesource_online_mismatch_warns_with_escape():
    with pytest.warns(UserWarning):
        assert_act_base_samesource(gv_sha="a", cache_sha="a", base_sha="b", allow_mismatch=True)
