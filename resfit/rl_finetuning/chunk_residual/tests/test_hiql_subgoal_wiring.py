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


def test_actor_stage_and_subgoal_combined():
    from resfit.rl_finetuning.off_policy.rl.actor import Actor
    from resfit.rl_finetuning.config.rlpd import ActorConfig
    cfg = ActorConfig()
    cfg.spatial_emb = 0
    a = Actor(repr_dim=64, patch_repr_dim=8, prop_dim=18, action_dim=12, cfg=cfg,
              residual_actor=True, stage_conditioned=True, num_stages=5,
              subgoal_conditioned=True, subgoal_dim=10)
    assert a.prop_dim == 18 + 12 + 5 + 10  # state + base_action + stage one-hot + subgoal
    obs = {
        "feat": torch.zeros(2, 64),
        "observation.state": torch.zeros(2, 18),
        "observation.base_action": torch.zeros(2, 12),
        "observation.stage_id": torch.zeros(2, 1),
        "observation.subgoal": torch.ones(2, 10),
    }
    dist = a.forward(obs, std=0.1)
    assert dist.mean.shape == (2, 12)


def test_qagent_critic_prop_subgoal():
    from types import SimpleNamespace
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    fake = SimpleNamespace(stage_conditioned=False, subgoal_conditioned=True, num_stages=0)
    obs = {"observation.state": torch.zeros(3, 18), "observation.subgoal": torch.ones(3, 10)}
    prop = QAgent._critic_prop(fake, obs)
    assert prop.shape == (3, 28)
    assert torch.equal(prop[:, 18:], torch.ones(3, 10))
