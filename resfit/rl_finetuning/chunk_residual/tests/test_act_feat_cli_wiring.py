from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    validate_act_feat_cfg, build_parser,
)


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
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import _setup_act_feat
    cache = str(tmp_path / "a.npz")
    sig = {"act_ckpt_id": "ckptA", "image_keys": ["observation.images.agentview"],
           "proprio_key": "observation.state", "pooling": "mean",
           "dataset_id": "ds", "num_demos": None}
    save_act_feat_cache(cache, [np.ones((1, 5), np.float32)],
                        (np.zeros(5, np.float32), np.ones(5, np.float32)), signature=sig)
    args = argparse.Namespace(state_mode="act_feat", act_feat_cache=cache,
                              act_base_ckpt=None, act_image_keys=None,
                              act_proprio_key="observation.state", pooling="mean")
    ext, ckpt, image_keys, got_sig = _setup_act_feat(args)
    assert ext is None
    assert ckpt == "ckptA"                                       # 从缓存签名取回
    assert image_keys == ["observation.images.agentview"]        # 从缓存签名取回(用户没传 flag)
    assert got_sig["dataset_id"] == "ds"
