import pytest
import torch
from resfit.rl_finetuning.chunk_residual.action_autoencoder import (
    ActionAutoencoder,
    action_autoencoder_loss,
)

D, L, LAT = 24, 20, 64
FLAT = L * D


def _make_ae():
    return ActionAutoencoder(action_dim=D, chunk_length=L, latent_dim=LAT,
                             hidden_dim=256, conv_layers=2, conv_kernel=3)


def test_encode_decode_shapes():
    ae = _make_ae()
    x = torch.randn(8, FLAT)
    z = ae.encode(x)
    assert z.shape == (8, LAT)
    recon = ae.decode(z)
    assert recon.shape == (8, FLAT)
    recon2, z2 = ae(x, return_latent=True)
    assert recon2.shape == (8, FLAT) and z2.shape == (8, LAT)


def test_loss_is_scalar_and_decreases_on_overfit():
    torch.manual_seed(0)
    ae = _make_ae()
    x = torch.randn(16, FLAT)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss0, m0 = action_autoencoder_loss(ae, x, velocity_loss_weight=0.1)
    assert loss0.ndim == 0 and "recon_l1" in m0 and "velocity_l1" in m0
    for _ in range(300):
        opt.zero_grad()
        loss, _ = action_autoencoder_loss(ae, x, velocity_loss_weight=0.1)
        loss.backward()
        opt.step()
    loss1, _ = action_autoencoder_loss(ae, x, velocity_loss_weight=0.1)
    assert loss1.item() < 0.25 * loss0.item()


def test_batch_size_one():
    ae = _make_ae()
    z = ae.encode(torch.randn(1, FLAT))
    assert z.shape == (1, LAT)
    assert ae.decode(z).shape == (1, FLAT)


def test_wrong_shape_raises():
    ae = _make_ae()
    with pytest.raises(ValueError):
        ae.encode(torch.randn(4, FLAT + 1))
