"""结构断言 —— 零改动路线唯一的兜底。

monkeypatch 最大的风险是上游改了符号/签名/调用点而我们静默失配,跑出一堆垃圾数据。
本模块在启动时把所有前提检一遍,不符当场硬失败。
"""
from __future__ import annotations

import argparse
import inspect
import math


class ContractError(RuntimeError):
    pass


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
