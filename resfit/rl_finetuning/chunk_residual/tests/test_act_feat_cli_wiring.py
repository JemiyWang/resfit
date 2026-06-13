from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    validate_act_feat_cfg, build_parser,
)


def test_gc_parser_has_state_mode_default_eef_piece():
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser as gc_parser
    args = gc_parser().parse_args(["--hdf5", "h", "--dataset", "d"])
    assert args.state_mode == "eef_piece"           # 默认逐位不变
    # act_feat 是合法值(公开 API 校验,不碰 argparse 私有属性)
    args2 = gc_parser().parse_args(["--hdf5", "h", "--dataset", "d", "--state_mode", "act_feat"])
    assert args2.state_mode == "act_feat"


def test_gc_save_load_act_feat_signature(tmp_path):
    import torch
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        save_gc_value, load_gc_value, GoalConditionedVF)
    p = str(tmp_path / "gc.pt")
    m = GoalConditionedVF(state_dim=22, rep_dim=10, hidden=32)
    sig = {"act_ckpt_id": "ckptA", "image_keys": ["observation.images.agentview"],
           "proprio_key": "observation.state", "pooling": "mean"}
    save_gc_value(p, m, v_stats={"min": 0.0, "max": 1.0, "mean": 0.5},
                  mean=torch.zeros(22), std=torch.ones(22), dataset_id="ds",
                  state_mode="act_feat", rel_piece_stats=None, act_feat_signature=sig)
    _, info = load_gc_value(p)
    assert info["state_mode"] == "act_feat"
    assert info["act_feat_signature"] == sig


def test_guard_act_feat_requires_cache_or_base():
    args = build_parser().parse_args(["--hdf5", "h", "--dataset", "d", "--state_mode", "act_feat"])
    try:
        validate_act_feat_cfg(args)
        assert False, "应报错(act_feat 缺 cache 且缺 base)"
    except ValueError:
        pass


def test_parser_defaults_unchanged():
    args = build_parser().parse_args(["--hdf5", "h", "--dataset", "d"])
    assert args.state_mode == "eef"
    assert args.act_feat_cache is None


def test_setup_act_feat_recovers_image_keys_and_ckpt_from_cache(tmp_path):
    import argparse
    import numpy as np
    from resfit.rl_finetuning.chunk_residual.act_feat_cache import save_act_feat_cache
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import setup_act_feat
    cache = str(tmp_path / "a.npz")
    sig = {"act_ckpt_id": "ckptA", "image_keys": ["observation.images.agentview"],
           "proprio_key": "observation.state", "pooling": "mean",
           "dataset_id": "ds", "num_demos": None}
    save_act_feat_cache(cache, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=sig)
    args = argparse.Namespace(state_mode="act_feat", act_feat_cache=cache,
                              act_base_ckpt=None, act_image_keys=None,
                              act_proprio_key="observation.state", pooling="mean")
    ext, ckpt, image_keys, got_sig = setup_act_feat(args)
    assert ext is None
    assert ckpt == "ckptA"                                       # 从缓存签名取回
    assert image_keys == ["observation.images.agentview"]        # 从缓存签名取回(用户没传 flag)
    assert got_sig["dataset_id"] == "ds"


def test_ha_parser_has_state_mode_default_eef_piece():
    from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser as ha_parser
    args = ha_parser().parse_args(["--hdf5", "h", "--dataset", "d", "--gc_value_ckpt", "g"])
    assert args.state_mode == "eef_piece"
    args2 = ha_parser().parse_args(["--hdf5", "h", "--dataset", "d", "--gc_value_ckpt", "g",
                                    "--state_mode", "act_feat"])
    assert args2.state_mode == "act_feat"


def test_ha_save_load_act_feat_signature(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
        save_high_actor, load_high_actor, HighActor)
    p = str(tmp_path / "ha.pt")
    m = HighActor(state_dim=22, rep_dim=10, hidden=32)
    sig = {"act_ckpt_id": "ckptA", "image_keys": ["observation.images.agentview"],
           "proprio_key": "observation.state", "pooling": "mean"}
    save_high_actor(p, m, gc_value_ckpt="g.pt", way_steps=25, beta=1.0, act_feat_signature=sig)
    _, info = load_high_actor(p)
    assert info.get("act_feat_signature") == sig
