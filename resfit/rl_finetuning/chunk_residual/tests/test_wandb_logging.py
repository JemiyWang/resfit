import torch
import pytest
from resfit.rl_finetuning.chunk_residual.wandb_logging import build_train_log_dict


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
