from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
from resfit.rl_finetuning.config.rlpd import QAgentConfig


def _req(extra):
    base = ["--base_wandb_id", "x", "--task", "three_piece", "--dataset", "some/ds"]
    return build_parser().parse_args(base + extra)


def test_oac_flags_default_off():
    a = _req([])
    assert a.oac_explore is False
    assert a.oac_beta_ub == 4.0
    assert a.oac_delta == 0.5


def test_oac_flags_settable():
    a = _req(["--oac_explore", "--oac_beta_ub", "2.0", "--oac_delta", "0.1"])
    assert a.oac_explore is True
    assert a.oac_beta_ub == 2.0
    assert a.oac_delta == 0.1


def test_qagent_config_has_oac_fields_with_defaults():
    cfg = QAgentConfig()
    assert cfg.oac_explore is False
    assert cfg.oac_beta_ub == 4.0
    assert cfg.oac_delta == 0.5
