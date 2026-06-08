import numpy as np
import torch


def test_append_subgoal():
    from resfit.rl_finetuning.off_policy.rl.stage_utils import append_subgoal
    prop = torch.zeros(4, 18)
    z = torch.ones(4, 10)
    out = append_subgoal(prop, z)
    assert out.shape == (4, 28)
    assert torch.equal(out[:, 18:], z)
    assert out.device == prop.device


def test_actor_subgoal_conditioned_dim():
    from resfit.rl_finetuning.off_policy.rl.actor import Actor
    from resfit.rl_finetuning.config.rlpd import ActorConfig
    cfg = ActorConfig()
    cfg.spatial_emb = 0
    a = Actor(repr_dim=64, patch_repr_dim=8, prop_dim=18, action_dim=12, cfg=cfg,
              residual_actor=True, subgoal_conditioned=True, subgoal_dim=10)
    assert a.prop_dim == 40  # 18 + 12 (base_action) + 10 (subgoal)
    obs = {
        "feat": torch.zeros(2, 64),
        "observation.state": torch.zeros(2, 18),
        "observation.base_action": torch.zeros(2, 12),
        "observation.subgoal": torch.ones(2, 10),
    }
    dist = a.forward(obs, std=0.1)
    assert dist.mean.shape == (2, 12)
