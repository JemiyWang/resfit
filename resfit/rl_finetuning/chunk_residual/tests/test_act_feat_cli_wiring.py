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
