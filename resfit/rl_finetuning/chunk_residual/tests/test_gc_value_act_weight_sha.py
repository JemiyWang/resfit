import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
    GoalConditionedVF, save_gc_value, load_gc_value,
)


def _model():
    return GoalConditionedVF(state_dim=6, rep_dim=4, hidden=8)


def test_save_load_act_weight_sha(tmp_path):
    p = str(tmp_path / "gc.pt")
    save_gc_value(p, _model(), v_stats={}, mean=torch.zeros(6), std=torch.ones(6),
                  dataset_id="ds", state_mode="act_feat",
                  act_feat_signature={"act_ckpt_id": "A"}, act_weight_sha="cafef00d")
    _, info = load_gc_value(p)
    assert info["act_weight_sha"] == "cafef00d"


def test_load_old_ckpt_sha_none(tmp_path):
    # 不传 act_weight_sha（旧档）→ load 后为 None
    p = str(tmp_path / "gc.pt")
    save_gc_value(p, _model(), v_stats={}, mean=torch.zeros(6), std=torch.ones(6),
                  dataset_id="ds", state_mode="eef_piece")
    _, info = load_gc_value(p)
    assert info["act_weight_sha"] is None
