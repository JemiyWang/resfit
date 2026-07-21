import sys
import types

from resfit.rl_finetuning.wm_bridge.launch_imagination import install_fakes


def test_install_fakes_registers_all_three_modules():
    saved = {k: sys.modules.get(k) for k in (
        "resfit.dexmg.environments.dexmg",
        "resfit.rl_finetuning.utils.evaluate_dexmg",
        "resfit.lerobot.policies.pi05")}
    try:
        install_fakes({
            "create_vectorized_env": lambda **kw: "ENV",
            "run_dexmg_evaluation": lambda **kw: {"eval/success_rate": 0.0},
            "load_pi05_base_policy": lambda *a, **k: "BASE",
        })
        m = sys.modules["resfit.dexmg.environments.dexmg"]
        assert m.create_vectorized_env(num_envs=1, device="cpu") == "ENV"
        assert sys.modules[
            "resfit.rl_finetuning.utils.evaluate_dexmg"].run_dexmg_evaluation()[
            "eval/success_rate"] == 0.0
        assert sys.modules[
            "resfit.lerobot.policies.pi05"].load_pi05_base_policy(None, "cpu") == "BASE"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_parent_package_gets_attribute_set():
    """`from a.b.c import d` 有的形式会走父包属性,不只查 sys.modules。"""
    saved = sys.modules.get("resfit.dexmg.environments.dexmg")
    try:
        install_fakes({"create_vectorized_env": lambda **kw: "ENV"})
        parent = sys.modules["resfit.dexmg.environments"]
        assert getattr(parent, "dexmg").create_vectorized_env(
            num_envs=1, device="cpu") == "ENV"
    finally:
        if saved is None:
            sys.modules.pop("resfit.dexmg.environments.dexmg", None)
        else:
            sys.modules["resfit.dexmg.environments.dexmg"] = saved


def test_fake_module_is_a_real_module_object():
    saved = sys.modules.get("resfit.lerobot.policies.pi05")
    try:
        install_fakes({"load_pi05_base_policy": lambda *a, **k: None})
        assert isinstance(sys.modules["resfit.lerobot.policies.pi05"],
                          types.ModuleType)
    finally:
        if saved is None:
            sys.modules.pop("resfit.lerobot.policies.pi05", None)
        else:
            sys.modules["resfit.lerobot.policies.pi05"] = saved
