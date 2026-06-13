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


def test_subgoal_act_feat_cuda_tensor_stats():
    # 回归:ckpt map_location=cuda 加载出的 feat_stats/goal 是 cuda 张量时,__init__ 不应再走 np.asarray(cuda) 崩。
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("需 GPU")
    dev = "cuda:0"
    sg = HiqlSubgoal(gc_value=None, high_actor=_StubHighActor(), goal=torch.zeros(530, device=dev),
                     device=dev, renorm_subgoal=False, state_mode="act_feat", extractor=_StubExtractor(),
                     feat_stats=(torch.zeros(530, device=dev), torch.ones(530, device=dev)))
    assert sg.feat_mean.device.type == "cuda" and sg.feat_std.shape == (530,)
