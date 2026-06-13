import numpy as np
import torch
from torch import nn

from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal


class _StubHighActor(nn.Module):
    state_dim = 530
    rep_dim = 10
    def forward(self, s, g):
        b = s.shape[0]
        class D:
            mean = torch.zeros(b, 10)
        return D()


class _StubExtractor:
    def embed_batch(self, obs):
        b = obs["observation.state"].shape[0]
        return torch.cat([torch.zeros(b, 512),
                          torch.as_tensor(obs["observation.state"], dtype=torch.float32)], -1)


def test_subgoal_online_act_feat_dispatch():
    ha = _StubHighActor()
    goal = np.zeros(530, np.float32)
    sg = HiqlSubgoal(gc_value=None, high_actor=ha, goal=goal, device="cpu", renorm_subgoal=False,
                     state_mode="act_feat", extractor=_StubExtractor(),
                     feat_stats=(np.zeros(530, np.float32), np.ones(530, np.float32)))
    obs = {"observation.images.agentview": torch.zeros(3, 3, 4, 4),
           "observation.state": torch.ones(3, 18)}
    z = sg.subgoal_online(obs)
    assert z.shape == (3, 10)
    assert torch.isfinite(z).all()
