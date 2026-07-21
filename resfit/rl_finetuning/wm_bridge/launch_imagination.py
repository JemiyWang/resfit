"""零改动入口。

在 sys.modules 预置 3 个假模块拦截符号,再用 runpy 以 __main__ 方式跑原 trainer。
WM / RL 两侧源码 0 行改动。仓库先例:run_td3_meta_only_wrapper.py。

★ 不能 patch build_base_policy —— 它定义在 train_chunk_residual.py 自身,而 runpy 以
   run_name="__main__" 执行会创建新的模块对象,对已 import 版本的 patch 作用不到。
   故往下钻一层,patch 它在 :173 lazy import 的 load_pi05_base_policy。

用法:
  python -m resfit.rl_finetuning.wm_bridge.launch_imagination \
      --reward_shaping none --chunk_length 50 --base_action_mode replan \
      --base_policy_type pi05 --pi0_host <kai0> --pi0_port <port> \
      --value_ckpt <value.pt> --wm_ckpt <D> [--allow_dummy_scorer] <其余原有参数>
"""
from __future__ import annotations

import importlib
import runpy
import sys
import types

_TARGETS = {
    "create_vectorized_env": "resfit.dexmg.environments.dexmg",
    "run_dexmg_evaluation": "resfit.rl_finetuning.utils.evaluate_dexmg",
    "load_pi05_base_policy": "resfit.lerobot.policies.pi05",
}

TRAINER = "resfit.rl_finetuning.chunk_residual.train_chunk_residual"


def install_fakes(factories: dict) -> None:
    """把 {符号名: 可调用} 装进对应的假模块。"""
    for symbol, fn in factories.items():
        mod_name = _TARGETS[symbol]
        mod = sys.modules.get(mod_name)
        if mod is None or not getattr(mod, "_wm_bridge_fake", False):
            mod = types.ModuleType(mod_name)
            mod._wm_bridge_fake = True
            sys.modules[mod_name] = mod
        setattr(mod, symbol, fn)

        # 部分 import 形式会走父包属性而非 sys.modules,两头都设上。
        # 父包(resfit.dexmg.environments / resfit.rl_finetuning.utils /
        # resfit.lerobot.policies)本身是空壳(无 __init__.py,namespace
        # package),import 它不会触发 robosuite/dexmimicgen 等重依赖——重的
        # 只是被我们整体假冒掉的叶子子模块。用真父包而非 __path__-less 假
        # 模块,才能让父包继续满足"是一个包"的语义,避免其余同名兄弟子模块
        # 被误判为 "not a package"。
        parent_name, _, leaf = mod_name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is None:
                parent = importlib.import_module(parent_name)
            setattr(parent, leaf, mod)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 延迟到此处 import:contract 会 import 上游真模块做断言,须在装假模块之前
    from resfit.rl_finetuning.wm_bridge import contract
    contract.check_upstream_symbols()
    contract.check_agent_image_size()

    from resfit.rl_finetuning.wm_bridge.builder import (
        build_imagination_factories, parse_bridge_args,
    )
    bridge_args, passthrough = parse_bridge_args(argv)
    factories = build_imagination_factories(bridge_args)

    install_fakes(factories)
    sys.argv = [TRAINER] + passthrough
    runpy.run_module(TRAINER, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
