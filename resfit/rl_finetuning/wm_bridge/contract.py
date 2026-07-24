"""结构断言 —— 零改动路线唯一的兜底。

monkeypatch 最大的风险是上游改了符号/签名/调用点而我们静默失配,跑出一堆垃圾数据。
本模块在启动时把所有前提检一遍,不符当场硬失败。
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import math
import os
from pathlib import Path


class ContractError(RuntimeError):
    pass


def parse_mixed_passthrough(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--offline_fraction", type=float, default=0.0)
    p.add_argument("--offline_num_demos", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--base_policy_type", default="act")
    p.add_argument("--base_action_mode", default="queue")
    p.add_argument("--chunk_length", type=int, default=1)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--actor", default="raw")
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--relabel", action="store_true")
    p.add_argument(
        "--stage_balanced", dest="stage_balanced",
        action="store_true", default=True)
    p.add_argument(
        "--no_stage_balanced", dest="stage_balanced", action="store_false")
    p.add_argument("--stage_conditioned", action="store_true")
    p.add_argument("--subgoal_conditioned", action="store_true")
    p.add_argument("--online_finetune_value", action="store_true")
    p.add_argument("--online_finetune_high_actor", action="store_true")
    p.add_argument("--pi0_prompt", default="build block")
    p.add_argument("--pi0_action_dim", type=int, default=16)
    p.add_argument("--data_source", default="hdf5")
    p.add_argument("--dataset", default="block_success")
    p.add_argument("--output_dir", default="outputs_chunk")
    args, _ = p.parse_known_args(argv)
    return args


def check_mixed_replay_args(trainer_args, offline_chunk_dataset) -> None:
    if offline_chunk_dataset is None:
        return

    source = os.path.realpath(offline_chunk_dataset)
    expected_dataset = os.path.basename(source)
    checks = (
        (
            "offline_fraction",
            math.isclose(
                float(getattr(trainer_args, "offline_fraction", 0.0)),
                0.5,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "must equal 0.5",
        ),
        (
            "batch_size",
            getattr(trainer_args, "batch_size", None) == 256,
            "must equal 256",
        ),
        (
            "base_policy_type",
            getattr(trainer_args, "base_policy_type", None) == "pi05",
            "must equal pi05",
        ),
        (
            "base_action_mode",
            getattr(trainer_args, "base_action_mode", None) == "replan",
            "must equal replan",
        ),
        (
            "chunk_length",
            getattr(trainer_args, "chunk_length", None) == 50,
            "must equal 50",
        ),
        (
            "n_step",
            getattr(trainer_args, "n_step", None) == 1,
            "must equal 1",
        ),
        (
            "actor",
            getattr(trainer_args, "actor", None) == "raw",
            "must equal raw",
        ),
        (
            "relabel",
            not bool(getattr(trainer_args, "relabel", False)),
            "must be disabled",
        ),
        (
            "stage_balanced",
            not bool(getattr(trainer_args, "stage_balanced", False)),
            "must be disabled",
        ),
        (
            "stage_conditioned",
            not bool(getattr(trainer_args, "stage_conditioned", False)),
            "must be disabled",
        ),
        (
            "subgoal_conditioned",
            not bool(getattr(trainer_args, "subgoal_conditioned", False)),
            "must be disabled",
        ),
        (
            "online_finetune_value",
            not bool(getattr(trainer_args, "online_finetune_value", False)),
            "must be disabled",
        ),
        (
            "online_finetune_high_actor",
            not bool(
                getattr(trainer_args, "online_finetune_high_actor", False)),
            "must be disabled",
        ),
        (
            "pi0_prompt",
            getattr(trainer_args, "pi0_prompt", None) == "build block",
            "must equal 'build block'",
        ),
        (
            "pi0_action_dim",
            getattr(trainer_args, "pi0_action_dim", None) == 16,
            "must equal 16",
        ),
        (
            "data_source",
            getattr(trainer_args, "data_source", None) == "hdf5",
            "must equal hdf5",
        ),
        (
            "dataset",
            getattr(trainer_args, "dataset", None) == expected_dataset,
            f"must equal {expected_dataset!r}",
        ),
    )
    for field, valid, requirement in checks:
        if not valid:
            value = getattr(trainer_args, field, None)
            raise ContractError(
                f"mixed replay {field}={value!r} {requirement}")

    if os.path.basename(source) != "block_success":
        raise ContractError(
            "offline_chunk_dataset must resolve to block_success, "
            f"got {source!r}")


def check_mixed_scorer(scorer, enabled: bool) -> None:
    if not enabled:
        return
    from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer

    if isinstance(scorer, DummyScorer):
        raise ContractError(
            "DummyScorer is forbidden when block mixed replay is enabled")


def check_upstream_symbols() -> None:
    """被 patch 的目标符号必须存在且签名未变。"""
    import importlib

    dexmg = importlib.import_module("resfit.dexmg.environments.dexmg")
    if not hasattr(dexmg, "create_vectorized_env"):
        raise ContractError(
            "resfit.dexmg.environments.dexmg.create_vectorized_env 不存在 —— "
            "注入点 1 失效")
    sig = inspect.signature(dexmg.create_vectorized_env)
    for p in ("num_envs", "device"):
        if p not in sig.parameters:
            raise ContractError(
                f"create_vectorized_env 签名缺参数 {p!r} —— 注入点 1 的假工厂需同签名")

    ev = importlib.import_module("resfit.rl_finetuning.utils.evaluate_dexmg")
    if not hasattr(ev, "run_dexmg_evaluation"):
        raise ContractError(
            "resfit.rl_finetuning.utils.evaluate_dexmg.run_dexmg_evaluation 不存在 —— "
            "注入点 2 失效,checkpoint 将无处存盘")

    pi05 = importlib.import_module("resfit.lerobot.policies.pi05")
    if not hasattr(pi05, "load_pi05_base_policy"):
        raise ContractError(
            "resfit.lerobot.policies.pi05.load_pi05_base_policy 不存在 —— 注入点 3 失效")


def check_agent_image_size() -> None:
    """残差 ViT 的 patch 数写死 81(=84×84),别的尺寸会在加位置编码时崩。"""
    from resfit.rl_finetuning.off_policy.networks.min_vit import PatchEmbed2

    embed = PatchEmbed2(128, use_norm=False)
    if embed.num_patch != 81:
        raise ContractError(
            f"PatchEmbed2.num_patch={embed.num_patch} != 81 —— "
            "84×84 前提已变,wm_driver.AGENT_IMG 需同步调整")


def check_wrapper_step_loop() -> None:
    """攒批机制的前提:ChunkResidualEnvWrapper.step 必须逐时间步调 vec_env.step()。

    若上游改成"一次把整个 chunk 交给 vec_env",我们攒 50 步再点火的时序就全错了,
    而且不会报错——只会把动作错位地喂给 WM。故在此硬检源码结构。
    """
    import inspect

    from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import (
        ChunkResidualEnvWrapper,
    )

    src = inspect.getsource(ChunkResidualEnvWrapper.step)
    if "for t in range(self.chunk_length)" not in src:
        raise ContractError(
            "ChunkResidualEnvWrapper.step 不再逐时间步循环 —— "
            "ImaginationVecEnv 攒 50 步再点火 WM 的时序前提已失效")
    if "self.vec_env.step(env_chunk[:, t])" not in src:
        raise ContractError(
            "ChunkResidualEnvWrapper.step 不再以单步动作调 vec_env.step —— "
            "ImaginationVecEnv.step 的入参约定已失效")


def _attribute_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def check_offline_hook_points() -> None:
    """Statically verify the trainer's lazy block-offline injection points."""
    module_name = (
        "resfit.rl_finetuning.chunk_residual.train_chunk_residual")
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ContractError(f"cannot locate trainer source for {module_name}")
    source = Path(spec.origin).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=spec.origin)
    main_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    if len(main_nodes) != 1:
        raise ContractError("train_chunk_residual.main source is missing or duplicated")
    main_node = main_nodes[0]

    module = (
        "resfit.rl_finetuning.chunk_residual.offline_stage_replay")
    imports = [
        node for node in ast.walk(main_node)
        if isinstance(node, ast.ImportFrom) and node.module == module
    ]
    expected_names = ["build_offline_buffer", "count_offline_transitions"]
    if len(imports) != 1 or [alias.name for alias in imports[0].names] != expected_names:
        raise ContractError(
            "train_chunk_residual.main must lazily import exactly "
            "build_offline_buffer and count_offline_transitions")

    calls = {
        name: [
            node for node in ast.walk(main_node)
            if isinstance(node, ast.Call)
            and _attribute_name(node.func) == name
        ]
        for name in expected_names
    }
    count_calls = calls["count_offline_transitions"]
    count_valid = (
        len(count_calls) == 1
        and len(count_calls[0].args) == 1
        and _attribute_name(count_calls[0].args[0])
        == "args.offline_dataset_path"
        and len(count_calls[0].keywords) == 1
        and count_calls[0].keywords[0].arg == "num_demos"
        and _attribute_name(count_calls[0].keywords[0].value)
        == "args.offline_num_demos"
    )
    if not count_valid:
        raise ContractError(
            "count_offline_transitions call no longer uses "
            "args.offline_dataset_path and args.offline_num_demos")

    build_calls = calls["build_offline_buffer"]
    build_valid = (
        len(build_calls) == 1
        and len(build_calls[0].args) == 2
        and _attribute_name(build_calls[0].args[0]) == "offline_rb"
        and _attribute_name(build_calls[0].args[1])
        == "args.offline_dataset_path"
    )
    if not build_valid:
        raise ContractError(
            "build_offline_buffer call no longer uses offline_rb and "
            "args.offline_dataset_path")


def check_runtime_args(args, imagination_gamma=None) -> None:
    """校验“一次 actor 决策 = 一次 WM 推进 = 一条 replay transition”。"""
    shaping = getattr(args, "reward_shaping", None)
    if shaping != "none":
        raise ContractError(
            f"想象路必须 --reward_shaping none(当前 {shaping!r})—— "
            "否则 chunk_env_wrapper:202-213 会在 PBRS reward 上再叠一层 = double-shaping")

    potential_source = getattr(args, "potential_source", None)
    if potential_source not in (None, "stage"):
        raise ContractError(
            f"想象路 --potential_source 只能是 stage/不传(当前 {potential_source!r}) —— "
            "Φ 由 wm_bridge 自己算,wrapper 必须走 reward_shaping=none 的零路径")

    cl = getattr(args, "chunk_length", None)
    if cl != 50:
        raise ContractError(
            f"想象路必须 --chunk_length 50(当前 {cl!r})—— WM 一次吃 25 token = 50 个动作")

    base_mode = getattr(args, "base_action_mode", None)
    if base_mode != "replan":
        raise ContractError(
            f"想象路必须 --base_action_mode replan(当前 {base_mode!r})—— "
            "actor 要一次修正完整 50 步动作块，不能逐步 queue")

    n_step = getattr(args, "n_step", None)
    if n_step != 1:
        raise ContractError(
            f"想象路必须 --n_step 1(当前 {n_step!r})—— "
            "每次 WM 推进本身就是一条 chunk transition")

    trainer_gamma = getattr(args, "gamma", None)
    if imagination_gamma is not None and not math.isclose(
            float(trainer_gamma), float(imagination_gamma), rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            f"trainer --gamma={trainer_gamma} 必须等于 "
            f"--imagination_gamma={imagination_gamma}，否则 PBRS reward 与 TD target 折扣不一致")

    if bool(getattr(args, "subgoal_conditioned", False)):
        raise ContractError(
            "想象路暂不支持 --subgoal_conditioned：truncation bootstrap 的 "
            "final_observation 尚无独立终点 subgoal 协议")


def check_passthrough_runtime_args(argv, imagination_gamma):
    """只解析 trainer 中影响 WM/chunk 时序的参数，其余参数原样透传。"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--reward_shaping", default=None)
    p.add_argument("--potential_source", default="stage")
    p.add_argument("--chunk_length", type=int, default=1)
    p.add_argument("--base_action_mode", default="queue")
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--subgoal_conditioned", action="store_true")
    args, _ = p.parse_known_args(argv)
    check_runtime_args(args, imagination_gamma=imagination_gamma)
    return args


def check_scorer(scorer, allow_dummy: bool) -> None:
    from resfit.rl_finetuning.wm_bridge.scorers import DummyScorer

    if isinstance(scorer, DummyScorer) and not allow_dummy:
        raise ContractError(
            "DummyScorer 的 reward 恒 0,TD3 会安静跑完全程并产出看起来正常的曲线。"
            "确要如此请显式传 --allow_dummy_scorer")


def check_psi_samesource(scorer, serve_ckpt_id) -> None:
    """训 V 时编 ψ 的 kai0(pi05)权重必须与在线 serve 同源,否则静默给出垃圾势。

    同源锚 = value.pt 的 pi0_feat_signature.serve_ckpt_id(base 统一 kai0,不涉及 ACT)。
    在线 serve_ckpt_id 由 launcher 的 --pi0_serve_ckpt_id 提供,与之比对。

    ★ 锚缺失 = 无法验证,不等于已知异源。对齐仓库既有 warn-not-raise 惯例
      (pi0_feat 的 assert_pi0_caches_samesource / hiql 既有 same-source 检查):
      warn 并要求先过 S1.5 一致性检查,不 raise。只有"两边都在且不等"才 raise。
    """
    import warnings

    expected = getattr(scorer, "expected_psi_anchor", None)
    if expected is None or serve_ckpt_id is None:
        warnings.warn(
            "[wm_bridge] ψ 同源锚缺失(value.pt 未记 pi0_feat_signature.serve_ckpt_id "
            "或未传在线 --pi0_serve_ckpt_id),无法验证同源。异源不会报错,只会静默给出"
            "垃圾势 —— 务必先跑 S1.5 一致性检查再开训。", stacklevel=2)
        return
    if str(expected) != str(serve_ckpt_id):
        raise ContractError(
            f"ψ 不同源:value.pt 记的 serve_ckpt_id={expected},在线是 {serve_ckpt_id}。"
            "Φ 会被喂进它没见过的特征空间,且不会报错,只会静默给出垃圾势")


def check_all(args, scorer, serve_ckpt_id, allow_dummy: bool) -> None:
    check_upstream_symbols()
    check_wrapper_step_loop()
    check_agent_image_size()
    check_runtime_args(args)
    check_scorer(scorer, allow_dummy)
    if not allow_dummy:
        check_psi_samesource(scorer, serve_ckpt_id)
