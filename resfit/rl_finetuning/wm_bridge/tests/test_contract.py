import types

import pytest

from resfit.rl_finetuning.wm_bridge import contract
from resfit.rl_finetuning.wm_bridge.contract import ContractError
from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer


def _args(**kw):
    d = {"reward_shaping": "none", "potential_source": "stage",
         "chunk_length": 50, "base_action_mode": "replan",
         "n_step": 1, "gamma": 0.995}
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_agent_image_size_contract_holds_on_current_repo():
    contract.check_agent_image_size()          # 当前仓库应通过


def test_upstream_symbols_present_on_current_repo():
    contract.check_upstream_symbols()          # 当前仓库应通过


def test_wrapper_still_loops_per_timestep():
    """攒批机制的前提:wrapper 必须逐时间步调 vec_env.step()。"""
    contract.check_wrapper_step_loop()         # 当前仓库应通过


def test_reward_shaping_must_be_none():
    with pytest.raises(ContractError, match="reward_shaping"):
        contract.check_runtime_args(_args(reward_shaping="staged"))


def test_external_potential_source_is_rejected():
    with pytest.raises(ContractError, match="potential_source"):
        contract.check_runtime_args(_args(potential_source="hiql"))


def test_chunk_length_must_be_fifty():
    with pytest.raises(ContractError, match="chunk_length"):
        contract.check_runtime_args(_args(chunk_length=1))


def test_base_action_mode_must_be_replan():
    with pytest.raises(ContractError, match="base_action_mode"):
        contract.check_runtime_args(_args(base_action_mode="queue"))


def test_n_step_must_be_one_chunk():
    with pytest.raises(ContractError, match="n_step"):
        contract.check_runtime_args(_args(n_step=3))


def test_trainer_gamma_must_match_imagination_gamma():
    with pytest.raises(ContractError, match="gamma"):
        contract.check_runtime_args(_args(gamma=0.99), imagination_gamma=0.995)


def test_valid_args_pass():
    contract.check_runtime_args(_args(), imagination_gamma=0.995)


def test_passthrough_runtime_args_parse_as_one_chunk_transition():
    args = contract.check_passthrough_runtime_args([
        "--reward_shaping", "none",
        "--potential_source", "stage",
        "--chunk_length", "50",
        "--base_action_mode", "replan",
        "--n_step", "1",
        "--gamma", "0.995",
        "--dataset", "ignored-by-contract",
    ], imagination_gamma=0.995)
    assert args.chunk_length == 50
    assert args.base_action_mode == "replan"


def test_dummy_scorer_rejected_without_flag():
    with pytest.raises(ContractError, match="allow_dummy_scorer"):
        contract.check_scorer(DummyScorer(), allow_dummy=False)


def test_dummy_scorer_allowed_with_flag():
    contract.check_scorer(DummyScorer(), allow_dummy=True)


def test_psi_sha_mismatch_raises():
    sc = types.SimpleNamespace(expected_psi_anchor="aaa")
    with pytest.raises(ContractError, match="同源"):
        contract.check_psi_samesource(sc, serve_ckpt_id="bbb")


def test_psi_sha_match_passes():
    sc = types.SimpleNamespace(expected_psi_anchor="aaa")
    contract.check_psi_samesource(sc, serve_ckpt_id="aaa")


def test_psi_sha_missing_warns_but_does_not_raise():
    """指纹缺失=无法验证,不等于已知异源。对齐 act_feature.py:149-153 的先例。"""
    sc = types.SimpleNamespace(expected_psi_anchor=None)
    with pytest.warns(UserWarning, match="无法验证同源"):
        contract.check_psi_samesource(sc, serve_ckpt_id="bbb")


def test_serve_sha_missing_warns_but_does_not_raise():
    sc = types.SimpleNamespace(expected_psi_anchor="aaa")
    with pytest.warns(UserWarning, match="无法验证同源"):
        contract.check_psi_samesource(sc, serve_ckpt_id=None)
