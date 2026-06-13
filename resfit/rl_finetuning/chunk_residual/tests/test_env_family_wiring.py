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
                                   "--stage_conditioned"])
    with pytest.raises(ValueError):
        validate_libero_cfg(a)


def test_validate_libero_cfg_rejects_non_pi05():
    a = build_parser().parse_args(["--env_family", "libero", "--base_policy_type", "act"])
    with pytest.raises(ValueError):
        validate_libero_cfg(a)


def test_validate_libero_cfg_noop_for_dexmg():
    validate_libero_cfg(build_parser().parse_args(["--task", "TwoArmBoxCleanup"]))  # 不报


def test_build_libero_scalers_from_stats_json(tmp_path):
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({"actions": {"mean": [0.] * 7, "std": [1.] * 7},
                             "state": {"mean": [0.] * 8, "std": [2.] * 8}}))
    asc, sst = build_libero_scalers(str(p), "cpu")
    # 标准化:state (x-0)/2;动作 scale 行为按 ActionScaler 现有语义,这里只验能构造+对 state 标准化
    out = sst.standardize(__import__("torch").zeros(1, 8))
    assert out.shape == (1, 8)
