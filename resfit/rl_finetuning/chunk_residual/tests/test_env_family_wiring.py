import json
import numpy as np
import pytest
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    build_parser, validate_libero_cfg, build_libero_scalers,
)


def test_env_family_default_dexmg():
    a = build_parser().parse_args(["--task", "TwoArmBoxCleanup"])
    assert a.env_family == "dexmg"


def test_libero_args_present():
    a = build_parser().parse_args(["--env_family", "libero", "--libero_suite", "libero_spatial",
                                   "--libero_task_id", "0", "--base_policy_type", "pi05"])
    assert a.libero_suite == "libero_spatial" and a.libero_task_id == 0


def test_validate_libero_cfg_rejects_stage_on():
    a = build_parser().parse_args(["--env_family", "libero", "--base_policy_type", "pi05",
                                   "--libero_stats_json", "/tmp/stats.json", "--stage_conditioned"])
    with pytest.raises(ValueError):
        validate_libero_cfg(a)


def test_validate_libero_cfg_rejects_non_pi05():
    a = build_parser().parse_args(["--env_family", "libero", "--base_policy_type", "act",
                                   "--libero_stats_json", "/tmp/stats.json"])
    with pytest.raises(ValueError):
        validate_libero_cfg(a)


def test_validate_libero_cfg_requires_stats_json():
    a = build_parser().parse_args(["--env_family", "libero", "--base_policy_type", "pi05"])
    with pytest.raises(ValueError):
        validate_libero_cfg(a)


def test_validate_libero_cfg_rejects_staged_reward_alias():
    # --staged_reward 别名也必须被守卫拒(canonical shaping mode)
    a = build_parser().parse_args(["--env_family", "libero", "--base_policy_type", "pi05",
                                   "--libero_stats_json", "/tmp/stats.json", "--staged_reward"])
    with pytest.raises(ValueError):
        validate_libero_cfg(a)


def test_validate_libero_cfg_noop_for_dexmg():
    validate_libero_cfg(build_parser().parse_args(["--task", "TwoArmBoxCleanup"]))  # 不报


def test_build_libero_scalers_from_stats_json(tmp_path):
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({"actions": {"mean": [0.] * 7, "std": [0.5] * 7,
                                         "min": [-2.] * 7, "max": [2.] * 7},
                             "state": {"mean": [0.] * 8, "std": [2.] * 8}}))
    asc, sst = build_libero_scalers(str(p), "cpu")
    # 标准化:state (x-0)/2
    out = sst.standardize(__import__("torch").zeros(1, 8))
    assert out.shape == (1, 8)


def test_build_libero_scalers_uses_real_min_max_not_mean_std(tmp_path):
    # 非平凡真 min/max=±2(mean=0,std=0.5);锁"用 min/max 而非 mean±std"。
    import torch
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({"actions": {"mean": [0.] * 7, "std": [0.5] * 7,
                                         "min": [-2.] * 7, "max": [2.] * 7},
                             "state": {"mean": [0.] * 8, "std": [2.] * 8}}))
    # action_scale=0 → limits 就是真 min/max(不外扩),便于直接比对
    asc, _ = build_libero_scalers(str(p), "cpu", action_scale=0.0)
    # limits 来自真 min/max(±2),而非 mean±std(那会是 ±0.5)
    assert torch.allclose(asc.limits.min, torch.full((7,), -2.0), atol=1e-5)
    assert torch.allclose(asc.limits.max, torch.full((7,), 2.0), atol=1e-5)
    # round-trip:范围内动作 unscale(scale(x)) ≈ x(x=1.0 落在 ±2 内,无 clamp)
    x = torch.ones(1, 7)
    assert torch.allclose(asc.unscale(asc.scale(x)), x, atol=1e-5)
