# tests/rl_finetuning/test_base_policy_switch.py
from resfit.rl_finetuning.config.residual_td3 import BasePolicyConfig


def test_default_type_is_act():
    cfg = BasePolicyConfig()
    assert cfg.type == "act"


def test_pi05_fields_exist_with_defaults():
    cfg = BasePolicyConfig(type="pi05")
    assert cfg.type == "pi05"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.action_dim == 7
    assert cfg.execute_horizon == 30
    assert cfg.prompt == "pick up the cube"
    assert "observation.images.agentview" in cfg.image_key_map
    assert cfg.kai0_paths == ["/data2/kai0"]
