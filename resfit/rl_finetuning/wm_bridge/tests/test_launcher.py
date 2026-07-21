import importlib
import sys
import types

from resfit.rl_finetuning.wm_bridge.launch_imagination import install_fakes

_MISSING = object()


def _snapshot_parent_attr(mod_name):
    """在调用 install_fakes 之前记录 (真)父包上 leaf 属性的原始状态。

    父包(namespace package,无 __init__.py)本身 import 很轻,不会碰
    robosuite/dexmimicgen,这里预先 import 一次只是为了拿到同一个模块对象
    去读它的 __dict__,不会产生额外副作用。
    """
    parent_name, _, leaf = mod_name.rpartition(".")
    parent = importlib.import_module(parent_name)
    old = parent.__dict__.get(leaf, _MISSING)
    return parent, leaf, old


def _restore_parent_attr(snapshot):
    parent, leaf, old = snapshot
    if old is _MISSING:
        if leaf in parent.__dict__:
            delattr(parent, leaf)
    else:
        setattr(parent, leaf, old)


def test_install_fakes_registers_all_three_modules():
    leaf_names = (
        "resfit.dexmg.environments.dexmg",
        "resfit.rl_finetuning.utils.evaluate_dexmg",
        "resfit.lerobot.policies.pi05")
    saved = {k: sys.modules.get(k) for k in leaf_names}
    parent_snapshots = [_snapshot_parent_attr(k) for k in leaf_names]
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
        for snap in parent_snapshots:
            _restore_parent_attr(snap)


def test_parent_package_gets_attribute_set():
    """`from a.b.c import d` 有的形式会走父包属性,不只查 sys.modules。"""
    mod_name = "resfit.dexmg.environments.dexmg"
    saved = sys.modules.get(mod_name)
    parent_snapshot = _snapshot_parent_attr(mod_name)
    try:
        install_fakes({"create_vectorized_env": lambda **kw: "ENV"})
        parent = sys.modules["resfit.dexmg.environments"]
        assert getattr(parent, "dexmg").create_vectorized_env(
            num_envs=1, device="cpu") == "ENV"
    finally:
        if saved is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = saved
        _restore_parent_attr(parent_snapshot)


def test_fake_module_is_a_real_module_object():
    mod_name = "resfit.lerobot.policies.pi05"
    saved = sys.modules.get(mod_name)
    parent_snapshot = _snapshot_parent_attr(mod_name)
    try:
        install_fakes({"load_pi05_base_policy": lambda *a, **k: None})
        assert isinstance(sys.modules[mod_name], types.ModuleType)
    finally:
        if saved is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = saved
        _restore_parent_attr(parent_snapshot)
