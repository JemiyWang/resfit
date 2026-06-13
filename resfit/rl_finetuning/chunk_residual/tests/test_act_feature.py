import torch

from resfit.rl_finetuning.chunk_residual.act_feature import (
    pool_encoder_out, concat_proprio, act_feat_signature,
)


def test_pool_encoder_out_mean():
    enc = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)  # [seq=2,B=3,dim=4]
    out = pool_encoder_out(enc, pooling="mean")
    assert out.shape == (3, 4)
    assert torch.allclose(out, enc.mean(dim=0))


def test_pool_encoder_out_unknown_pooling_raises():
    enc = torch.zeros(2, 3, 4)
    try:
        pool_encoder_out(enc, pooling="weird")
        assert False, "应当报错"
    except ValueError:
        pass


def test_concat_proprio_slice():
    emb = torch.zeros(5, 8)
    proprio = torch.ones(5, 18)
    out = concat_proprio(emb, proprio)
    assert out.shape == (5, 26)
    assert torch.allclose(out[:, 8:], torch.ones(5, 18))   # proprio 拼在尾部
    assert out.dtype == torch.float32


def test_signature_stable_and_sensitive():
    a = act_feat_signature("ckptA", ["observation.images.agentview"], "observation.state", "mean")
    b = act_feat_signature("ckptA", ["observation.images.agentview"], "observation.state", "mean")
    c = act_feat_signature("ckptB", ["observation.images.agentview"], "observation.state", "mean")
    assert a == b and a != c
    assert a["act_ckpt_id"] == "ckptA" and a["pooling"] == "mean"
