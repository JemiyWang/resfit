import argparse
import pytest
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import validate_libero_cfg


def _libero_args(**over):
    a = argparse.Namespace(
        env_family="libero", libero_stats_json="/x/stats.json", base_policy_type="pi05",
        base_action_mode="queue", chunk_length=1, reward_shaping=None, staged_reward=False,
        offline_fraction=0.0, pi0_action_dim=7, stage_conditioned=False, stage_budget=None,
        subgoal_conditioned=False, pi0_feat_cache=None, potential_source="stage", actor="raw",
        demo_bc_coef=0.0, offline_dataset_path=None)
    a.__dict__.update(over)
    return a


def test_validate_allows_libero_offline_fraction():
    validate_libero_cfg(_libero_args(offline_fraction=0.5))   # 不再抛

def test_validate_libero_offline_requires_actor_raw():
    with pytest.raises(ValueError):
        validate_libero_cfg(_libero_args(offline_fraction=0.5, actor="flow"))

def test_validate_libero_subgoal_with_pi0_feat_cache_ok():
    # 离线 waypoint 子目标已实现(Tasks 1-4):libero + subgoal + pi0_feat_cache → 不抛
    validate_libero_cfg(_libero_args(
        offline_fraction=0.5, subgoal_conditioned=True, pi0_feat_cache="/x/pi0_feat.npz"))


def test_validate_libero_subgoal_without_pi0_feat_cache_raises():
    # subgoal 但缺 pi0_feat_cache → raise(eef_piece/act_feat 子目标在 libero 下未实现)
    with pytest.raises(ValueError) as ei:
        validate_libero_cfg(_libero_args(subgoal_conditioned=True, pi0_feat_cache=None))
    msg = str(ei.value)
    assert "pi0_feat" in msg


def test_validate_libero_still_rejects_stage_conditioned():
    # stage 守卫不放开:libero + stage_conditioned → 仍 raise
    with pytest.raises(ValueError):
        validate_libero_cfg(_libero_args(stage_conditioned=True))
