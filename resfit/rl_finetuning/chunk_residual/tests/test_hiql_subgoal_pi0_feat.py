import numpy as np
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
    HighActor, save_high_actor, load_high_actor,
)


def test_high_actor_pi0_feat_signature_roundtrip(tmp_path):
    ha = HighActor(state_dim=2056, rep_dim=10, hidden=64)
    pt = str(tmp_path / "ha.pt")
    sig = {"serve_ckpt_id": "pi0_libero", "pooling": "last", "proprio_key": "observation.state"}
    save_high_actor(pt, ha, gc_value_ckpt="gc.pt", way_steps=5, beta=1.0,
                    state_mode="pi0_feat", pi0_feat_signature=sig)
    model, info = load_high_actor(pt)
    assert info["state_mode"] == "pi0_feat"
    assert info["pi0_feat_signature"]["serve_ckpt_id"] == "pi0_libero"
