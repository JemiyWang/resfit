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


import torch
from torch import nn

from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor


class _StubEncoder(nn.Module):
    def forward(self, tokens, pos_embed=None):
        return tokens  # [seq, B, dim]


class _StubModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = _StubEncoder()
        self.dim = dim

    def forward(self, batch):
        b = batch["observation.images"][0].shape[0]
        seq = 4
        tokens = torch.ones(seq, b, self.dim)   # 确定性
        return self.encoder(tokens)


class _StubCfg:
    def __init__(self, image_keys):
        self.image_features = {k: None for k in image_keys}
        self.dim_model = None  # 见下:extractor 用 dim_model 算 feature_dim


class _StubACT(nn.Module):
    def __init__(self, dim, image_keys):
        super().__init__()
        self.model = _StubModel(dim)
        self.config = _StubCfg(image_keys)
        self.config.dim_model = dim

    def normalize_inputs(self, raw):
        return dict(raw)   # 身份预处理


def _raw(b=3):
    return {
        "observation.images.agentview": torch.zeros(b, 3, 4, 4),
        "observation.state": torch.arange(b * 18, dtype=torch.float32).reshape(b, 18),
    }


def test_extractor_shape_dtype_determinism_and_proprio_slice():
    act = _StubACT(dim=8, image_keys=["observation.images.agentview"])
    ext = ActFeatureExtractor(act, image_keys=["observation.images.agentview"],
                              proprio_key="observation.state", pooling="mean")
    raw = _raw()
    out1 = ext.embed_batch(raw)
    out2 = ext.embed_batch(raw)
    assert out1.shape == (3, 8 + 18)
    assert out1.dtype == torch.float32
    assert torch.allclose(out1, out2)                    # 确定性
    assert torch.allclose(out1[:, 8:], raw["observation.state"])  # 原始 proprio 在尾部
    assert ext.feature_dim == 8 + 18
