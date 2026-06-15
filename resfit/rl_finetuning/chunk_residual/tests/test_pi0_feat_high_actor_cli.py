"""测试 train_hiql_high_actor.py 的 --state_mode pi0_feat dispatch(CLI parser 层)。"""
import pytest
from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser
from resfit.rl_finetuning.chunk_residual.train_hiql_value import validate_pi0_feat_cfg


def test_high_actor_parser_accepts_pi0_feat():
    a = build_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--gc_value_ckpt", "gc.pt",
                                   "--state_mode", "pi0_feat", "--pi0_feat_cache", "c.npz",
                                   "--pi0_serve_ckpt_id", "pi0_libero", "--pi0_image_keys", "a",
                                   "--pi0_proprio_key", "observation.state"])
    assert a.state_mode == "pi0_feat"


def test_high_actor_parser_default_state_mode_unchanged():
    a = build_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--gc_value_ckpt", "gc.pt"])
    assert a.state_mode == "eef_piece"


def test_high_actor_parser_pi0_feat_args_present():
    """build_parser() 加了 add_pi0_feat_args:pi0_feat_cache/serve_ckpt_id/image_keys/proprio_key 等皆可解析。"""
    a = build_parser().parse_args([
        "--hdf5", "x", "--dataset", "d", "--gc_value_ckpt", "gc.pt",
        "--state_mode", "pi0_feat",
        "--pi0_feat_cache", "feat.npz",
        "--pi0_serve_ckpt_id", "pi0_libero",
        "--pi0_image_keys", "cam1,cam2",
        "--pi0_proprio_key", "observation.state",
        "--pi0_prompt", "do task",
        "--pi0_pooling", "mean",
    ])
    assert a.pi0_feat_cache == "feat.npz"
    assert a.pi0_serve_ckpt_id == "pi0_libero"
    assert a.pi0_image_keys == ["cam1", "cam2"]
    assert a.pi0_proprio_key == "observation.state"
    assert a.pi0_prompt == "do task"
    assert a.pi0_pooling == "mean"


def test_validate_pi0_feat_cfg_noop_for_eef_piece():
    """非 pi0_feat 时 validate 为 noop,不报错。"""
    a = build_parser().parse_args(["--hdf5", "x", "--dataset", "d", "--gc_value_ckpt", "gc.pt"])
    # 默认 eef_piece -> validate 应不 raise
    validate_pi0_feat_cfg(a)


def test_assert_pair_consistent_pi0_feat_signature_mismatch():
    """pi0_feat 时,gc_value 与 high_actor 签名核心字段不一致应 AssertionError;一致则通过。"""
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import assert_act_feat_pair_consistent
    gc_info = {"state_mode": "pi0_feat",
               "pi0_feat_signature": {"serve_ckpt_id": "pi0_libero", "image_keys": ["a"],
                                      "proprio_key": "observation.state", "pooling": "last", "prompt": "x"}}
    ha_sig = dict(gc_info["pi0_feat_signature"], pooling="mean")   # 故意不一致
    with pytest.raises(AssertionError):
        assert_act_feat_pair_consistent(gc_info, None, "pi0_feat", pi0_sig=ha_sig)
    # 一致则不报
    assert_act_feat_pair_consistent(gc_info, None, "pi0_feat", pi0_sig=gc_info["pi0_feat_signature"])
