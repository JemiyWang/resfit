"""Task5:测 compute_online_subgoal helper 按 state_mode 路由。"""
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import compute_online_subgoal


class _Base:
    def last_prefix_feat(self):
        return np.ones(2048, np.float32)


def test_compute_online_subgoal_pi0_feat_pulls_base_prefix_feat():
    class _SG:
        state_mode = "pi0_feat"

        def subgoal_online(self, obs, rel_raw=None, prefix_feat=None):
            assert prefix_feat is not None  # pi0_feat 必须收到 prefix_feat
            return torch.zeros(1, 10)

    z = compute_online_subgoal(_SG(), {"observation.state": np.ones(8, np.float32)}, _Base())
    assert z.shape[-1] == 10


def test_compute_online_subgoal_act_feat_uses_obs_rel():
    class _SG2:
        state_mode = "act_feat"

        def subgoal_online(self, obs, rel_raw=None, prefix_feat=None):
            assert prefix_feat is None  # act_feat 不走 prefix_feat
            return torch.zeros(1, 10)

    z = compute_online_subgoal(_SG2(), {"x": 1}, _Base(), cur_rel=None)
    assert z.shape[-1] == 10


def test_compute_online_subgoal_eef_piece_uses_obs_rel():
    """eef_piece 和 act_feat 一样走 subgoal_online(obs, rel_raw),不传 prefix_feat。"""
    class _SG3:
        state_mode = "eef_piece"

        def subgoal_online(self, obs, rel_raw=None, prefix_feat=None):
            assert prefix_feat is None
            return torch.zeros(1, 30)

    z = compute_online_subgoal(_SG3(), {"observation.state": np.zeros(18, np.float32)}, _Base(), cur_rel=None)
    assert z.shape[-1] == 30


def test_libero_offline_signature_includes_subgoal_when_conditioned():
    from types import SimpleNamespace
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _libero_offline_signature
    args = SimpleNamespace(
        libero_stats_json="/x/meta/stats.json", libero_suite="libero_10", libero_task_id=8,
        offline_base_mode="base_policy", base_policy_type="pi05", pi0_host="127.0.0.1",
        pi0_port=8000, pi0_action_dim=7, pi0_execute_horizon=10, action_scale=0.05,
        min_range_per_dim=0.1, gamma=0.99, n_step=3, offline_num_demos=None,
        subgoal_conditioned=True, gc_value_ckpt="gc.pt", high_actor_ckpt="ha.pt", subgoal_way_steps=25)
    sig = _libero_offline_signature(args, ["observation.images.agentview"], 100, 84)
    assert sig["subgoal"] is True
    assert sig["subgoal_way_steps"] == 25
    assert sig["gc_value_ckpt"].endswith("gc.pt") and sig["high_actor_ckpt"].endswith("ha.pt")


def test_libero_offline_signature_omits_subgoal_when_off():
    from types import SimpleNamespace
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _libero_offline_signature
    args = SimpleNamespace(
        libero_stats_json="/x/meta/stats.json", libero_suite="libero_10", libero_task_id=8,
        offline_base_mode="base_policy", base_policy_type="pi05", pi0_host="127.0.0.1",
        pi0_port=8000, pi0_action_dim=7, pi0_execute_horizon=10, action_scale=0.05,
        min_range_per_dim=0.1, gamma=0.99, n_step=3, offline_num_demos=None,
        subgoal_conditioned=False)
    sig = _libero_offline_signature(args, ["observation.images.agentview"], 100, 84)
    assert "subgoal" not in sig
