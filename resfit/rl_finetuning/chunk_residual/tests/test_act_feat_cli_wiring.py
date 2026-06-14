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


def test_chunk_residual_parser_has_act_feat_cache():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    args = build_parser().parse_args(["--task", "TwoArmThreePieceAssembly", "--dataset", "d"])
    assert hasattr(args, "act_feat_cache") and args.act_feat_cache is None


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
    save_high_actor(p, m, gc_value_ckpt="g.pt", way_steps=25, beta=1.0, act_feat_signature=sig,
                    state_mode="act_feat")
    _, info = load_high_actor(p)
    assert info.get("act_feat_signature") == sig
    assert info["state_mode"] == "act_feat"


def test_assert_act_feat_pair_consistent():
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import assert_act_feat_pair_consistent
    core = {"act_ckpt_id": "A", "image_keys": ["observation.images.agentview"],
            "proprio_key": "observation.state", "pooling": "mean"}
    gc_info = {"state_mode": "act_feat", "act_feat_signature": dict(core, dataset_id="ds", num_demos=None)}
    # 核心字段一致(即使一侧多了 dataset_id/num_demos)→ 通过
    assert_act_feat_pair_consistent(gc_info, dict(core), "act_feat")
    # state_mode 不一致 → 报错
    try:
        assert_act_feat_pair_consistent({"state_mode": "eef_piece"}, None, "act_feat")
        assert False
    except AssertionError:
        pass
    # 核心签名不一致(不同 ACT ckpt)→ 报错
    try:
        assert_act_feat_pair_consistent(gc_info, dict(core, act_ckpt_id="B"), "act_feat")
        assert False
    except AssertionError:
        pass
    # eef_piece 同模式 → 通过(不校签名)
    assert_act_feat_pair_consistent({"state_mode": "eef_piece"}, None, "eef_piece")


def test_data_source_flags_present_and_default_hdf5():
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import build_parser as gc
    from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import build_parser as ha
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser as tc
    for p in (gc(), ha()):
        actions = [x.dest for x in p._actions]
        argv = ["--hdf5", "h", "--dataset", "d"]
        if "gc_value_ckpt" in actions:
            argv += ["--gc_value_ckpt", "g"]
        a = p.parse_args(argv)
        assert a.data_source == "hdf5" and a.lerobot_root is None
    a = tc().parse_args(["--task", "TwoArmThreePieceAssembly", "--dataset", "d"])
    assert a.data_source == "hdf5" and a.lerobot_root is None


def test_train_chunk_residual_lerobot_guard_injects_state_mode(tmp_path):
    """回归锁:train_chunk_residual 无 --state_mode flag,lerobot 路守卫须注入 state_mode=act_feat +
    强制 subgoal。此前 validate_data_source_cfg 被复用却没端到端走守卫,smoke 才暴露无条件 assert 失败。"""
    import pytest
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
        build_parser, _resolve_data_source_cfg)
    base = ["--task", "TwoArmThreePieceAssembly", "--dataset", "d"]
    # lerobot + subgoal + 存在 root → 注入 act_feat、放行
    a = build_parser().parse_args(base + ["--data_source", "lerobot",
                                          "--lerobot_root", str(tmp_path), "--subgoal_conditioned"])
    _resolve_data_source_cfg(a)
    assert a.state_mode == "act_feat"
    # lerobot 缺 --subgoal_conditioned → AssertionError(本脚本 lerobot 仅经 act_feat 子目标)
    b = build_parser().parse_args(base + ["--data_source", "lerobot", "--lerobot_root", str(tmp_path)])
    with pytest.raises(AssertionError):
        _resolve_data_source_cfg(b)
    # hdf5 默认路 → 放行,不注入 state_mode=act_feat
    c = build_parser().parse_args(base)
    _resolve_data_source_cfg(c)
    assert getattr(c, "state_mode", None) != "act_feat"
