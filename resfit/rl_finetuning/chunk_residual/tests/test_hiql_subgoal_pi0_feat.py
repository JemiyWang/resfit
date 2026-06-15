import numpy as np
import pytest
import torch
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
    HighActor, save_high_actor, load_high_actor,
)
from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal


def test_high_actor_pi0_feat_signature_roundtrip(tmp_path):
    ha = HighActor(state_dim=2056, rep_dim=10, hidden=64)
    pt = str(tmp_path / "ha.pt")
    sig = {"serve_ckpt_id": "pi0_libero", "pooling": "last", "proprio_key": "observation.state"}
    save_high_actor(pt, ha, gc_value_ckpt="gc.pt", way_steps=5, beta=1.0,
                    state_mode="pi0_feat", pi0_feat_signature=sig)
    model, info = load_high_actor(pt)
    assert info["state_mode"] == "pi0_feat"
    assert info["pi0_feat_signature"]["serve_ckpt_id"] == "pi0_libero"


class _StubHA(torch.nn.Module):
    rep_dim = 10
    state_dim = 2056

    def forward(self, s, g):
        b = s.shape[0]

        class _D:
            mean = torch.zeros(b, 10)
        return _D()


def test_subgoal_online_pi0_feat_outputs_z():
    ha = _StubHA()
    sg = HiqlSubgoal(gc_value=None, high_actor=ha, goal=np.zeros(2056, np.float32), device="cpu",
                     state_mode="pi0_feat",
                     feat_stats=(np.zeros(2056, np.float32), np.ones(2056, np.float32)))
    obs = {"observation.state": np.ones(8, np.float32)}
    z = sg.subgoal_online(obs, prefix_feat=np.ones(2048, np.float32))
    assert z.shape[-1] == 10 and torch.isfinite(z).all()


def test_subgoal_online_pi0_feat_needs_prefix_feat():
    ha = _StubHA()
    sg = HiqlSubgoal(gc_value=None, high_actor=ha, goal=np.zeros(2056, np.float32), device="cpu",
                     state_mode="pi0_feat",
                     feat_stats=(np.zeros(2056, np.float32), np.ones(2056, np.float32)))
    with pytest.raises(AssertionError):
        sg.subgoal_online({"observation.state": np.ones(8, np.float32)}, prefix_feat=None)


def test_subgoal_waypoint_rejects_pi0_feat():
    ha = _StubHA()
    sg = HiqlSubgoal(gc_value=None, high_actor=ha, goal=np.zeros(2056, np.float32), device="cpu",
                     state_mode="pi0_feat",
                     feat_stats=(np.zeros(2056, np.float32), np.ones(2056, np.float32)))
    with pytest.raises(AssertionError):
        sg.subgoal_waypoint(np.zeros(2056, np.float32), np.zeros(2056, np.float32))
