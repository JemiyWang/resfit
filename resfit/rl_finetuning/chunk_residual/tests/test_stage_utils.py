import torch
from resfit.rl_finetuning.off_policy.rl.stage_utils import stage_onehot, append_stage


def test_stage_onehot_basic():
    sid = torch.tensor([[0.0], [3.0], [4.0]])      # [B,1] float
    oh = stage_onehot(sid, num_stages=5)
    assert oh.shape == (3, 5)
    assert oh.dtype == torch.float32
    assert torch.equal(oh.argmax(dim=-1), torch.tensor([0, 3, 4]))
    assert torch.equal(oh.sum(dim=-1), torch.ones(3))


def test_stage_onehot_accepts_1d():
    sid = torch.tensor([1.0, 2.0])                 # [B] 也接受
    oh = stage_onehot(sid, num_stages=5)
    assert oh.shape == (2, 5)
    assert torch.equal(oh.argmax(dim=-1), torch.tensor([1, 2]))


def test_stage_onehot_clamps_out_of_range():
    sid = torch.tensor([[-1.0], [9.0]])            # 越界被 clamp 到 [0,4]
    oh = stage_onehot(sid, num_stages=5)
    assert torch.equal(oh.argmax(dim=-1), torch.tensor([0, 4]))


def test_append_stage_widens_prop():
    prop = torch.randn(4, 7)
    sid = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    out = append_stage(prop, sid, num_stages=5)
    assert out.shape == (4, 7 + 5)
    assert torch.equal(out[:, :7], prop)
    assert not torch.equal(out[0, 7:], out[3, 7:])
