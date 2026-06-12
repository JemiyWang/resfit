# tests/rl_finetuning/test_hiql_stage_cache_optional.py
"""--stage_cache 按需必需,而非无条件 required。

geometric(gc_value)/clamp_to_goal(high_actor)口径不读 stage,允许省略 --stage_cache;
stage_entry(gc_value)/fixed_waypoint(high_actor)口径仍必需。
覆盖:stage_entries_aligned(None) 回退、validate_stage_cache 守卫、两个 parser 可省、
以及 geometric 在"无 stage(空入口)"数据上照常采样(回归锁,保证与喂 dummy cache 等价)。
"""
import numpy as np
import pytest

from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import (
    stage_entries_aligned,
    build_parser as gc_build_parser,
)
from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import (
    build_parser as ha_build_parser,
)
from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, sample_gc_goals


def test_stage_entries_aligned_none_returns_empty_per_demo():
    # stage_cache=None:无需 hdf5/缓存,按 demo 数返回空入口数组(geometric 不读它)
    out = stage_entries_aligned("/nonexistent.hdf5", None, None, [3, 5])
    assert len(out) == 2
    assert all(isinstance(e, np.ndarray) and e.dtype == np.int64 and len(e) == 0
               for e in out)


def test_validate_stage_cache_required_only_when_needs_stage():
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import validate_stage_cache
    # 需要 stage 的口径缺 cache -> 报错
    with pytest.raises(ValueError):
        validate_stage_cache(None, needs_stage=True)
    # 不需要 stage 的口径缺 cache -> 放行(返回 None)
    assert validate_stage_cache(None, needs_stage=False) is None
    # 有 cache -> 任何口径放行
    assert validate_stage_cache("stages.npz", needs_stage=True) == "stages.npz"


def test_gc_value_parser_stage_cache_optional_default_geometric():
    args = gc_build_parser().parse_args(["--hdf5", "x.hdf5", "--dataset", "d"])
    assert args.stage_cache is None
    assert args.goal_future_mode == "geometric"


def test_high_actor_parser_stage_cache_optional_default_clamp():
    args = ha_build_parser().parse_args(
        ["--hdf5", "x.hdf5", "--dataset", "d", "--gc_value_ckpt", "v.pt"])
    assert args.stage_cache is None
    assert args.target_mode == "clamp_to_goal"


def test_geometric_goal_sampling_ignores_empty_stage_entries():
    # 回归锁:geometric 口径在"空入口"数据上正常出 goal,且不依赖 stage_entries
    seqs = [np.arange(6, dtype=np.float32).reshape(6, 1),
            np.arange(4, dtype=np.float32).reshape(4, 1)]
    stage_entries = [np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)]
    data = build_gc_data(seqs, stage_entries)
    rng = np.random.default_rng(0)
    idx = data["s_idx"]
    goals = sample_gc_goals(idx, data["traj_id"], data["last_idx_of"],
                            data["stage_entries_of"], rng,
                            n_total=len(data["states"]), future_mode="geometric")
    assert len(goals) == len(idx)
    assert goals.min() >= 0 and goals.max() < len(data["states"])


# --- 锁 main() 实际用的"模式 -> 是否需要 stage"接线(具名谓词),防打错字面量/接错 attr ---

def test_gc_value_needs_stage_predicate_tracks_goal_future_mode():
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import (
        build_parser, gc_value_needs_stage,
    )
    a_geo = build_parser().parse_args(["--hdf5", "x.hdf5", "--dataset", "d"])
    assert gc_value_needs_stage(a_geo) is False          # 默认 geometric 不需要 stage
    a_se = build_parser().parse_args(
        ["--hdf5", "x.hdf5", "--dataset", "d", "--goal_future_mode", "stage_entry"])
    assert gc_value_needs_stage(a_se) is True             # stage_entry 需要 stage


def test_high_actor_needs_stage_predicate_tracks_target_mode():
    from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import (
        build_parser, high_actor_needs_stage,
    )
    a_clamp = build_parser().parse_args(
        ["--hdf5", "x.hdf5", "--dataset", "d", "--gc_value_ckpt", "v.pt"])
    assert high_actor_needs_stage(a_clamp) is False       # 默认 clamp_to_goal 不需要 stage
    a_fw = build_parser().parse_args(
        ["--hdf5", "x.hdf5", "--dataset", "d", "--gc_value_ckpt", "v.pt",
         "--target_mode", "fixed_waypoint", "--high_p_randomgoal", "0.0"])
    assert high_actor_needs_stage(a_fw) is True            # fixed_waypoint 需要 stage


def test_gc_value_stage_entry_without_cache_is_rejected():
    # 端到端:stage_entry 口径 + 省略 --stage_cache -> 用 main 同款谓词必须报错(而非静默退化)
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import (
        build_parser, gc_value_needs_stage, validate_stage_cache,
    )
    args = build_parser().parse_args(
        ["--hdf5", "x.hdf5", "--dataset", "d", "--goal_future_mode", "stage_entry"])
    assert args.stage_cache is None
    with pytest.raises(ValueError):
        validate_stage_cache(args.stage_cache, needs_stage=gc_value_needs_stage(args))


def test_high_actor_fixed_waypoint_without_cache_is_rejected():
    from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import validate_stage_cache
    from resfit.rl_finetuning.chunk_residual.train_hiql_high_actor import (
        build_parser, high_actor_needs_stage,
    )
    args = build_parser().parse_args(
        ["--hdf5", "x.hdf5", "--dataset", "d", "--gc_value_ckpt", "v.pt",
         "--target_mode", "fixed_waypoint", "--high_p_randomgoal", "0.0"])
    assert args.stage_cache is None
    with pytest.raises(ValueError):
        validate_stage_cache(args.stage_cache, needs_stage=high_actor_needs_stage(args))
