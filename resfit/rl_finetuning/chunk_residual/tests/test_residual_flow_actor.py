import torch
from resfit.rl_finetuning.chunk_residual.action_autoencoder import ActionAutoencoder
from resfit.rl_finetuning.chunk_residual.residual_flow_actor import ResidualFlowActor

D, L, LAT = 4, 5, 8
FLAT = L * D
REPR, PATCH, PROP = 32, 16, 3
B = 6


def _make_actor():
    ae = ActionAutoencoder(action_dim=D, chunk_length=L, latent_dim=LAT,
                           hidden_dim=32, conv_layers=2)
    return ResidualFlowActor(repr_dim=REPR, patch_repr_dim=PATCH, prop_dim=PROP,
                             action_dim=FLAT, chunk_length=L, action_dim_per_step=D,
                             frozen_ae=ae, feature_dim=16, hidden_dim=32, num_layers=3,
                             latent_delta_scale=0.05, action_delta_clip=0.2)


def _fake_obs():
    return {
        "feat": torch.randn(B, REPR // PATCH, PATCH),     # [B, num_patch, patch_dim]
        "observation.state": torch.randn(B, PROP),
        "observation.base_action": torch.tanh(torch.randn(B, FLAT)),
    }


def test_returns_truncated_normal_with_residual_shape():
    actor = _make_actor().eval()
    dist = actor.forward(_fake_obs(), std=0.0)
    assert dist.mean.shape == (B, FLAT)
    s = dist.sample(clip=0.3)
    assert s.shape == (B, FLAT)


def test_zero_init_residual_is_near_zero():
    actor = _make_actor().eval()
    dist = actor.forward(_fake_obs(), std=0.0)
    # velocity 末层 zero-init → decoded_delta ≈ 0 → 残差均值≈0
    assert dist.mean.abs().max().item() < 1e-4


def test_grad_flows_to_velocity_not_ae():
    actor = _make_actor()
    # zero-init 让 mean≡0、梯度恒 0(空测);先把 velocity 末层扰动成非零,真正激活 grad 路径
    torch.nn.init.normal_(actor.velocity_out.weight, std=0.1)
    obs = _fake_obs()
    dist = actor.forward(obs, std=0.0)
    dist.mean.pow(2).sum().backward()
    # 冻结 AE:零梯度
    ae_grad_norm = sum(p.grad.abs().sum().item()
                       for p in actor.frozen_ae.parameters() if p.grad is not None)
    assert ae_grad_norm == 0.0
    # velocity 末层:经 decode(z_corr) 真实回传,梯度非零
    assert actor.velocity_out.weight.grad is not None
    assert actor.velocity_out.weight.grad.abs().sum().item() > 0.0
