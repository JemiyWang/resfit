import pytest
from resfit.rl_finetuning.chunk_residual.train_hiql_value import validate_pi0_feat_cfg
from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser as gc_parser


def test_gc_parser_state_mode_default_unchanged():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d"])
    assert a.state_mode == "eef_piece"


def test_pi0_feat_choice_accepted():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--state_mode", "pi0_feat",
                                "--pi0_feat_cache", "c.npz", "--pi0_serve_ckpt_id", "pi05_base",
                                "--pi0_image_keys", "agentview_image", "--pi0_proprio_key", "state18"])
    assert a.state_mode == "pi0_feat" and a.pi0_image_keys == ["agentview_image"]


def test_validate_requires_cache():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--state_mode", "pi0_feat",
                                "--pi0_serve_ckpt_id", "ck", "--pi0_image_keys", "a", "--pi0_proprio_key", "p"])
    with pytest.raises(ValueError):
        validate_pi0_feat_cfg(a)                          # 缺 --pi0_feat_cache(或不存在) → 报错


def test_validate_noop_for_eef_modes():
    a = gc_parser().parse_args(["--hdf5", "x", "--dataset", "d"])
    validate_pi0_feat_cfg(a)                              # eef_piece 不报
