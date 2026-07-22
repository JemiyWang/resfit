"""bridge 参数解析 + 组装 launcher 的 3 个假符号。

parse_bridge_args 摘走 bridge 专属参数(--value_ckpt/--wm_host/... ),其余原样透传给 trainer
(trainer 的 --pi0_host/--pi0_port/--task/--chunk_length 等不认识 bridge 参数,不摘会报错)。

3 个假符号(见 launch_imagination 的 _TARGETS):
  create_vectorized_env → ImaginationVecEnv(wm=D-serve客户端, base=kai0共享实例, scorer=V, sampler)
  run_dexmg_evaluation  → make_imagination_evaluator(存 checkpoint;可选 adv-logging)
  load_pi05_base_policy → Kai0ImaginationBase(kai0 serve 客户端);与 env 共享同一实例→1 serve/chunk

★ 共享实例:trainer 先 build_base_policy(→ load_pi05_base_policy 假物,建 Kai0ImaginationBase 存 state),
   再 create_vectorized_env(用 state["base"] 建 env)。两处同一 base,窗口缓存跨两个调用方生效。
"""
from __future__ import annotations

import argparse

import numpy as np


def parse_bridge_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--value_ckpt", required=True, help="block V (pi0_feat, block_value_pi0feat.pt)")
    p.add_argument("--wm_host", default="127.0.0.1", help="D-serve host(d_serve.py)")
    p.add_argument("--wm_port", type=int, default=9000, help="D-serve port")
    p.add_argument("--init_state_dataset", action="append", default=[],
                   help="想象起点数据集(block_success/fail 完整路径),可重复。成/失败都传")
    p.add_argument("--pi0_serve_ckpt_id", default=None,
                   help="在线 kai0 serve 标签,与 value.pt 的 pi0_feat_signature.serve_ckpt_id 比对")
    p.add_argument("--imagination_gamma", type=float, default=0.995)
    p.add_argument("--max_segments", type=int, default=2)
    p.add_argument("--num_denois_steps", type=int, default=10)
    p.add_argument("--action_norm_json", default=None, help="block 动作 min/max JSON;缺省 ±1")
    p.add_argument("--allow_dummy_scorer", action="store_true")
    # 优势估计器 proxy 监控(spec §6.7):给了 --adv_host 才开;每 eval rollout N 集逐帧打分写 jsonl
    p.add_argument("--adv_host", default=None,
                   help="优势估计器 serve host(adv_serve.py);给了才开 adv proxy 监控")
    p.add_argument("--adv_port", type=int, default=8002)
    p.add_argument("--adv_prompt", default="build block")
    p.add_argument("--n_eval_episodes", type=int, default=10,
                   help="每 eval 点 adv rollout 集数(spec §6.7 默认 10)")
    bridge_args, rest = p.parse_known_args(argv)
    return bridge_args, rest


def _normalizer(path):
    from resfit.rl_finetuning.wm_bridge.wm_driver import ACTION_DIM, ActionNormalizer
    if path is None:
        return ActionNormalizer(-np.ones(ACTION_DIM, np.float32), np.ones(ACTION_DIM, np.float32))
    import json
    d = json.load(open(path))
    return ActionNormalizer(np.asarray(d["min"], np.float32), np.asarray(d["max"], np.float32))


def build_imagination_factories(bridge_args) -> dict:
    from resfit.rl_finetuning.wm_bridge import contract
    from resfit.rl_finetuning.wm_bridge.base_bridge import Kai0ImaginationBase
    from resfit.rl_finetuning.wm_bridge.fake_eval import make_imagination_evaluator
    from resfit.rl_finetuning.wm_bridge.imagination_env import ImaginationVecEnv
    from resfit.rl_finetuning.wm_bridge.scorers import Kai0HiqlScorer
    from resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler import TeleavatarStartSampler
    from resfit.rl_finetuning.wm_bridge.wm_client import WmServeClient

    scorer = Kai0HiqlScorer.from_value_ckpt(bridge_args.value_ckpt)
    contract.check_scorer(scorer, bridge_args.allow_dummy_scorer)
    if not bridge_args.allow_dummy_scorer:
        contract.check_psi_samesource(scorer, bridge_args.pi0_serve_ckpt_id)
    normalizer = _normalizer(bridge_args.action_norm_json)

    # 优势估计器打分器(adv proxy 监控):给了 --adv_host 才建;否则 eval 只存 checkpoint。
    adv_scorer = None
    if bridge_args.adv_host:
        from resfit.rl_finetuning.wm_bridge.adv_client import AdvServeClient
        adv_scorer = AdvServeClient(host=bridge_args.adv_host, port=bridge_args.adv_port,
                                    prompt=bridge_args.adv_prompt)

    state = {"base": None, "env": None, "output_dir": None, "config": None}

    def fake_load_pi05_base_policy(cfg, device, schema="dexmg"):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        # queue 模式 eval 会建第二个 base 实例;想象路只用一个(env+trainer 共享),复用 state["base"]
        if state["base"] is None:
            client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)
            state["base"] = Kai0ImaginationBase(
                client, prompt=getattr(cfg, "prompt", "build block"),
                action_dim=getattr(cfg, "action_dim", 16))
        return state["base"]

    def fake_create_vectorized_env(*, env_name=None, num_envs=1, device="cpu",
                                   state_mode=None, **kw):
        if state["env"] is not None:
            return state["env"]                       # eval_vec 复用同一 env(不另起想象流)
        assert state["base"] is not None, \
            "load_pi05_base_policy(建 base)须在 create_vectorized_env 之前调"
        state["env"] = ImaginationVecEnv(
            wm=WmServeClient(host=bridge_args.wm_host, port=bridge_args.wm_port),
            base=state["base"], scorer=scorer,
            sampler=TeleavatarStartSampler(bridge_args.init_state_dataset),
            normalizer=normalizer, gamma=bridge_args.imagination_gamma,
            max_segments=bridge_args.max_segments,
            num_denois_steps=bridge_args.num_denois_steps, device=device)
        return state["env"]

    def fake_run_dexmg_evaluation(**kw):
        out_dir = kw.get("output_dir") or state["output_dir"] or "outputs_imagination"
        return make_imagination_evaluator(
            out_dir, state["config"], adv_scorer=adv_scorer,
            n_eval_episodes=bridge_args.n_eval_episodes)(**kw)

    return {
        "create_vectorized_env": fake_create_vectorized_env,
        "run_dexmg_evaluation": fake_run_dexmg_evaluation,
        "load_pi05_base_policy": fake_load_pi05_base_policy,
    }
