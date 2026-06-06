import argparse

import torch
import pytest
from resfit.rl_finetuning.chunk_residual.wandb_logging import (
    build_train_log_dict,
    _parse_purity,
    build_eval_log_dict,
    init_wandb,
)
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def _fake_m_upd():
    return {
        "train/critic_loss": 0.5,
        "train/actor_loss_total": -1.2,
        "train/critic_grad_norm": 3.0,
        "_actions": torch.zeros(4, 12),
        "_target_q": torch.ones(10, 4),
        "_td_errors": torch.zeros(4),
    }


_LRS = {"actor": 1e-6, "critic": 1e-4, "encoder": 2e-4}
_BUF = {"online": 100, "offline": 50}
_STUB = lambda a: ("H", len(a))


def test_train_log_keeps_train_keys_and_drops_underscore():
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, hist_fn=_STUB)
    assert out["train/critic_loss"] == 0.5
    assert out["train/actor_loss_total"] == -1.2
    assert out["train/critic_grad_norm"] == 3.0
    assert "_actions" not in out
    assert "_target_q" not in out
    assert "_td_errors" not in out


def test_train_log_adds_lr_and_buffer():
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, hist_fn=_STUB)
    assert out["lr/actor"] == 1e-6
    assert out["lr/critic"] == 1e-4
    assert out["lr/encoder"] == 2e-4
    assert out["buffer/online_size"] == 100
    assert out["buffer/offline_size"] == 50


def test_train_log_no_histograms_when_disabled():
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, with_histograms=False)
    assert "histograms/actions" not in out
    assert "histograms/critic_qt" not in out


def test_train_log_histograms_use_hist_fn():
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, hist_fn=_STUB)
    assert out["histograms/actions"] == ("H", 48)    # 4*12
    assert out["histograms/critic_qt"] == ("H", 40)  # 10*4


def test_parse_purity_normal():
    s = "regress 11410/29939=38.1%  stage0:0/9289=0%  stage1:2184/5916=37%"
    out = _parse_purity(s)
    assert abs(out["purity/regress_frac"] - 11410 / 29939) < 1e-9
    assert out["purity/stage0"] == 0.0
    assert abs(out["purity/stage1"] - 2184 / 5916) < 1e-9


def test_parse_purity_no_stage_steps():
    assert _parse_purity("no stage steps") == {"purity/raw": "no stage steps"}


def test_parse_purity_unparseable_falls_back():
    assert _parse_purity("garbage data here") == {"purity/raw": "garbage data here"}


def test_eval_log_with_diag():
    em = {"eval/success_rate": 0.12, "eval/other": 1.0}
    diag = {"diag/residual_norm/stage0": 0.13, "diag/target_q/stage0": 0.8}
    out = build_eval_log_dict(em, diag, "regress 1/2=50%  stage0:1/2=50%")
    assert out["eval/success_rate"] == 0.12
    assert out["eval/other"] == 1.0
    assert out["diag/residual_norm/stage0"] == 0.13
    assert out["diag/target_q/stage0"] == 0.8
    assert out["purity/regress_frac"] == 0.5
    assert out["purity/stage0"] == 0.5


def test_eval_log_diag_none():
    out = build_eval_log_dict({"eval/success_rate": 0.3}, None, "no stage steps")
    assert out["eval/success_rate"] == 0.3
    assert not any(k.startswith("diag/") for k in out)
    assert out["purity/raw"] == "no stage steps"


def test_init_wandb_smoke_forces_disabled(monkeypatch):
    captured = {}

    def fake_init(**kw):
        captured.update(kw)
        return "RUN"

    monkeypatch.setattr(
        "resfit.rl_finetuning.chunk_residual.wandb_logging.wandb.init", fake_init)
    args = argparse.Namespace(
        smoke=True, wandb_mode="online", wandb_project="P",
        wandb_entity=None, wandb_name="N", output_dir="outputs_chunk/x")
    run = init_wandb(args)
    assert run == "RUN"
    assert captured["mode"] == "disabled"
    assert captured["project"] == "P"
    assert captured["name"] == "N"


def test_init_wandb_name_falls_back_to_output_dir(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "resfit.rl_finetuning.chunk_residual.wandb_logging.wandb.init",
        lambda **kw: captured.update(kw))
    args = argparse.Namespace(
        smoke=False, wandb_mode="disabled", wandb_project="P",
        wandb_entity=None, wandb_name=None, output_dir="outputs_chunk/cl1_stageON/")
    init_wandb(args)
    assert captured["name"] == "cl1_stageON"
    assert captured["mode"] == "disabled"


def test_parser_has_wandb_defaults():
    args = build_parser().parse_args(["--task", "TwoArmThreePieceAssembly"])
    assert args.wandb_project == "dexmg-chunk-residual"
    assert args.wandb_entity is None
    assert args.wandb_name is None
    assert args.wandb_mode == "online"
    assert args.log_freq == 100


def test_parser_wandb_overrides():
    args = build_parser().parse_args(
        ["--wandb_mode", "disabled", "--wandb_project", "P", "--log_freq", "50"])
    assert args.wandb_mode == "disabled"
    assert args.wandb_project == "P"
    assert args.log_freq == 50


def test_train_log_relabel_size_present():
    """buf_sizes 含 'relabel' 键时，out 中 buffer/relabel_size 应等于该值。"""
    buf_with_relabel = {"online": 100, "offline": 50, "relabel": 30}
    out = build_train_log_dict(_fake_m_upd(), _LRS, buf_with_relabel, hist_fn=_STUB)
    assert out["buffer/relabel_size"] == 30


def test_train_log_relabel_size_missing_defaults_zero():
    """buf_sizes 不含 'relabel' 键时（旧接口 _BUF），buffer/relabel_size 应为 0（向后兼容）。"""
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, hist_fn=_STUB)
    assert out["buffer/relabel_size"] == 0
