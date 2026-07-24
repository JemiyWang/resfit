from pathlib import Path
import types

import pytest

from resfit.rl_finetuning.wm_bridge import contract
from resfit.rl_finetuning.wm_bridge.contract import ContractError
from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer


def _args(**kw):
    d = {"reward_shaping": "none", "potential_source": "stage",
         "chunk_length": 50, "base_action_mode": "replan",
         "n_step": 1, "gamma": 0.995, "subgoal_conditioned": False}
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_agent_image_size_contract_holds_on_current_repo():
    contract.check_agent_image_size()          # 当前仓库应通过


def test_upstream_symbols_present_on_current_repo():
    contract.check_upstream_symbols()          # 当前仓库应通过


def test_wrapper_still_loops_per_timestep():
    """攒批机制的前提:wrapper 必须逐时间步调 vec_env.step()。"""
    contract.check_wrapper_step_loop()         # 当前仓库应通过


def test_offline_hook_points_still_lazy_import_and_call():
    contract.check_offline_hook_points()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("base_mode=args.offline_base_mode,", ""),
        ("base_mode=args.offline_base_mode,", "compat_mode=args.offline_base_mode,"),
        ("gamma=args.gamma,", ""),
        ("num_demos=args.offline_num_demos,", ""),
        ("action_scaler=action_scaler,", ""),
        ("state_standardizer=state_standardizer,", ""),
        ("image_keys=image_keys,", ""),
    ],
)
def test_offline_hook_contract_rejects_build_keyword_drift(
    monkeypatch,
    old,
    new,
):
    original_read_text = Path.read_text

    def altered_source(path, *args, **kwargs):
        source = original_read_text(path, *args, **kwargs)
        start = source.index("                build_offline_buffer(\n")
        end_marker = "                    base_device=args.device, env_hint=args.task,)"
        end = source.index(end_marker, start) + len(end_marker)
        build_call = source[start:end]
        assert old in build_call
        return source[:start] + build_call.replace(old, new, 1) + source[end:]

    monkeypatch.setattr(Path, "read_text", altered_source)
    with pytest.raises(ContractError, match="build_offline_buffer"):
        contract.check_offline_hook_points()


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


def test_subgoal_conditioned_imagination_is_rejected():
    with pytest.raises(ContractError, match="subgoal_conditioned"):
        contract.check_runtime_args(_args(subgoal_conditioned=True))


def test_passthrough_subgoal_conditioned_imagination_is_rejected():
    argv = [
        "--reward_shaping", "none",
        "--potential_source", "stage",
        "--chunk_length", "50",
        "--base_action_mode", "replan",
        "--n_step", "1",
        "--gamma", "0.995",
        "--subgoal_conditioned",
    ]
    with pytest.raises(ContractError, match="subgoal_conditioned"):
        contract.check_passthrough_runtime_args(
            argv, imagination_gamma=0.995)


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


def _mixed_args(**overrides):
    values = {
        "offline_fraction": 0.5,
        "batch_size": 256,
        "base_policy_type": "pi05",
        "base_action_mode": "replan",
        "chunk_length": 50,
        "n_step": 1,
        "actor": "raw",
        "relabel": False,
        "stage_balanced": False,
        "stage_conditioned": False,
        "subgoal_conditioned": False,
        "online_finetune_value": False,
        "online_finetune_high_actor": False,
        "pi0_prompt": "build block",
        "pi0_action_dim": 16,
        "data_source": "hdf5",
        "dataset": "block_success",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_pure_online_does_not_require_mixed_flags():
    contract.check_mixed_replay_args(
        _mixed_args(offline_fraction=0.0),
        offline_chunk_dataset=None,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("offline_fraction", 0.4),
        ("batch_size", 255),
        ("base_policy_type", "act"),
        ("base_action_mode", "queue"),
        ("chunk_length", 1),
        ("n_step", 3),
        ("actor", "flow"),
        ("relabel", True),
        ("stage_balanced", True),
        ("stage_conditioned", True),
        ("subgoal_conditioned", True),
        ("online_finetune_value", True),
        ("online_finetune_high_actor", True),
        ("pi0_prompt", "assemble the three pieces"),
        ("pi0_action_dim", 14),
        ("data_source", "lerobot"),
        ("dataset", "some_other_dataset"),
    ],
)
def test_mixed_contract_rejects_invalid_configuration(field, value):
    args = _mixed_args(**{field: value})
    with pytest.raises(ContractError, match=field):
        contract.check_mixed_replay_args(
            args, offline_chunk_dataset="/data/block_success")


def test_mixed_contract_rejects_non_success_source():
    with pytest.raises(ContractError, match="block_success"):
        contract.check_mixed_replay_args(
            _mixed_args(), offline_chunk_dataset="/data/block_fail")


def test_mixed_mode_rejects_dummy_scorer_even_with_debug_flag():
    with pytest.raises(ContractError, match="DummyScorer"):
        contract.check_mixed_scorer(DummyScorer(), enabled=True)


def test_pure_online_keeps_existing_dummy_scorer_policy():
    contract.check_mixed_scorer(DummyScorer(), enabled=False)
