import argparse
import torch

from resfit.rl_finetuning.chunk_residual.action_autoencoder import ActionAutoencoder
from resfit.rl_finetuning.chunk_residual.residual_flow_actor import ResidualFlowActor
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _maybe_inject_flow_actor

D, L, LAT = 4, 5, 8
REPR, PATCH, PROP = 32, 16, 3
FLAT = L * D


class _FakeAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = argparse.Namespace(actor=argparse.Namespace(feature_dim=16))
        self.actor = torch.nn.Linear(4, 4)
        self.actor_target = torch.nn.Linear(4, 4)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=1e-4)


def _save_ae(tmp_path):
    ae = ActionAutoencoder(action_dim=D, chunk_length=L, latent_dim=LAT,
                           hidden_dim=16, conv_layers=2)
    p = str(tmp_path / "ae.pt")
    torch.save({"state_dict": ae.state_dict(),
                "ae_config": {"action_dim": D, "chunk_length": L, "latent_dim": LAT,
                              "hidden_dim": 16, "conv_layers": 2}}, p)
    return p


def test_inject_replaces_actor_with_flow(tmp_path):
    agent = _FakeAgent()
    args = argparse.Namespace(actor="flow", ae_ckpt=_save_ae(tmp_path), device="cpu",
                              chunk_length=L, action_scale=0.2, actor_lr=1e-4)
    _maybe_inject_flow_actor(args, agent, REPR, PATCH, PROP, FLAT)
    assert isinstance(agent.actor, ResidualFlowActor)
    assert isinstance(agent.actor_target, ResidualFlowActor)
    assert agent.actor_target is not agent.actor
    assert agent.actor_target.training is True
    # 优化器只含可训练(非冻结 AE)参数
    opt_params = {id(p) for g in agent.actor_opt.param_groups for p in g["params"]}
    ae_params = {id(p) for p in agent.actor.frozen_ae.parameters()}
    assert opt_params.isdisjoint(ae_params)
    assert opt_params == {id(p) for p in agent.actor.trainable_parameters()}


def test_inject_noop_for_raw():
    agent = _FakeAgent()
    before = agent.actor
    args = argparse.Namespace(actor="raw", ae_ckpt=None, device="cpu",
                              chunk_length=L, action_scale=0.2, actor_lr=1e-4)
    _maybe_inject_flow_actor(args, agent, REPR, PATCH, PROP, FLAT)
    assert agent.actor is before
