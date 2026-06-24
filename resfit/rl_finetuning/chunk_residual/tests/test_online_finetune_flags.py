import pytest

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def _req(extra):
    base = ["--base_wandb_id", "x", "--task", "three_piece", "--dataset", "some/ds"]
    return build_parser().parse_args(base + extra)


def test_online_finetune_flags_default_off():
    a = _req([])
    assert a.online_finetune_value is False
    assert a.online_finetune_high_actor is False
    assert a.online_value_lr == 1e-5
    assert a.online_high_actor_lr == 1e-5
    assert a.online_finetune_offline_fraction == 0.5
    assert a.online_finetune_every == 1
    assert a.online_finetune_future_mode == "geometric"


def test_online_finetune_flags_settable():
    a = _req(["--online_finetune_value", "--online_finetune_high_actor",
              "--online_value_lr", "3e-6", "--online_finetune_every", "4",
              "--online_finetune_future_mode", "stage_entry"])
    assert a.online_finetune_value is True
    assert a.online_finetune_high_actor is True
    assert a.online_value_lr == 3e-6
    assert a.online_finetune_every == 4
    assert a.online_finetune_future_mode == "stage_entry"
