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
