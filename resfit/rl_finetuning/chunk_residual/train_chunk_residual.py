"""chunk 级残差 RL 训练编排(不改原仓库)。

复用:create_vectorized_env、ACTPolicy 基座、QAgent(action_dim=L*D=480)、torchrl buffer、
run_dexmg_evaluation。只把 env wrapper 换成 ChunkResidualEnvWrapper,RL 决策粒度=chunk。

--actor raw  : 用现成 Actor(M1)
--actor flow : 注入 residual_flow actor(M2,见 _maybe_inject_flow_actor)

用法(从仓库根目录,conda env residual):
  CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual --actor raw

x 轴用"环境步"(每 chunk 计 chunk_length 步),与单步基线可比。
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np

import wandb
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import (
    ChunkResidualEnvWrapper, resolve_shaping_mode)
from resfit.rl_finetuning.chunk_residual.stage_replay import sample_stage_balanced
from resfit.rl_finetuning.chunk_residual.stage_diag import flatten_stage_diagnostics, stage_diagnostics
from resfit.rl_finetuning.chunk_residual.stage_detectors import NUM_STAGES
from resfit.rl_finetuning.utils.checkpoint import save_checkpoint
from resfit.rl_finetuning.chunk_residual.wandb_logging import (
    init_wandb, build_train_log_dict, build_eval_log_dict,
)


def to_uint8(obs: dict, image_keys):
    for k in image_keys:
        v = obs[k]
        if v.dtype != torch.uint8:
            obs[k] = (v.clamp(0, 1) * 255).round().to(torch.uint8)


def add_chunk_transition(*, obs, next_obs, combined_action, reward, done, info,
                         image_keys, lowdim_keys, online_rb):
    """单环境;构造与 train_residual_td3._add_transitions_to_buffer 同构的 TensorDict。

    注:done 的 transition 直接用返回的 next_obs(已 augment);其 Q 被 done 掩掉,
    内容不影响 target,故不特殊处理 final_obs(wrapper 也不再透出)。
    """
    keys = set(image_keys) | set(lowdim_keys)
    curr = {k: obs[k][0].detach().cpu() for k in keys}
    nxt = {k: next_obs[k][0].detach().cpu() for k in keys}
    to_uint8(curr, image_keys)
    to_uint8(nxt, image_keys)
    max_stage = float(info.get("max_stage_in_chunk", 0))
    td = TensorDict({
        "obs": TensorDict(curr, batch_size=[]),
        "next": TensorDict({"obs": TensorDict(nxt, batch_size=[]),
                            "done": done[0].cpu(), "reward": reward[0].cpu()}, batch_size=[]),
        "action": combined_action[0].detach().cpu(),
        "max_stage": torch.tensor(max_stage, dtype=torch.float32),
        "_priority": torch.tensor(10.0, dtype=torch.float32),
    }, batch_size=[]).unsqueeze(0)
    online_rb.add(td)


def make_bc_entry(obs, action, image_keys, lowdim_keys):
    """构造 relabel 用的 bc 条目 td:{obs:{图+lowdim}, action}(与 offline_rb 同构,供混采 bc_batch)。"""
    keys = set(image_keys) | set(lowdim_keys)
    curr = {k: obs[k][0].detach().cpu() for k in keys}
    to_uint8(curr, image_keys)
    return TensorDict({"obs": TensorDict(curr, batch_size=[]),
                       "action": action[0].detach().cpu()}, batch_size=[]).unsqueeze(0)


def _maybe_inject_flow_actor(args, agent, repr_dim, patch_repr_dim, prop_dim, action_dim):
    """M2:把 agent.actor 替换为 residual_flow actor(用 M0 冻结 AE),并重建 actor_opt。
    --actor raw 时为 no-op。纯注入,不改 QAgent 源码。"""
    if args.actor != "flow":
        return
    assert args.ae_ckpt is not None, "--actor flow 需要 --ae_ckpt 指向 M0 训好的 AE"
    from resfit.rl_finetuning.chunk_residual.action_autoencoder import ActionAutoencoder
    from resfit.rl_finetuning.chunk_residual.residual_flow_actor import ResidualFlowActor

    ckpt = torch.load(args.ae_ckpt, map_location=args.device, weights_only=True)
    ae_cfg = ckpt["ae_config"]
    ae = ActionAutoencoder(**ae_cfg)
    ae.load_state_dict(ckpt["state_dict"])
    ae = ae.to(args.device).eval()

    flow_actor = ResidualFlowActor(
        repr_dim=repr_dim, patch_repr_dim=patch_repr_dim, prop_dim=prop_dim,
        action_dim=action_dim, chunk_length=args.chunk_length,
        action_dim_per_step=ae_cfg["action_dim"], frozen_ae=ae,
        feature_dim=agent.cfg.actor.feature_dim, hidden_dim=512, num_layers=3,
        latent_delta_scale=0.05, action_delta_clip=args.action_scale,  # 对齐 chunk-raw 残差幅度
    ).to(args.device)

    agent.actor = flow_actor
    agent.actor_target = copy.deepcopy(flow_actor)
    agent.actor_target.train(True)   # 显式置 train(deepcopy 已继承 train,QAgent.update 断言其为 True)
    agent.actor_opt = torch.optim.AdamW(flow_actor.trainable_parameters(), lr=args.actor_lr)
    print("[inject] residual_flow actor 已注入,actor_opt 仅含 velocity 网络参数")


# three-piece(TwoArmThreePieceAssembly)默认相机映射:env obs 图像键 -> server 端 DexmgInputs
# 期望的相机名。注意:dexmg serve 的 DexmgInputs.EXPECTED_CAMERAS 就是原始相机名
# (agentview/robot0_eye_in_hand/robot1_eye_in_hand),它自己再 rename 到 base_0_rgb 等 pi0 槽位。
# 所以这里的"目标名"必须保持原始相机名,不能改成 base/left_wrist/right_wrist。
PIECE_IMAGE_KEY_MAP = {
    "observation.images.agentview": "agentview",
    "observation.images.robot0_eye_in_hand": "robot0_eye_in_hand",
    "observation.images.robot1_eye_in_hand": "robot1_eye_in_hand",
}


def build_base_policy(args, device: str, wt_type: str = "best", wt_version: str = "latest"):
    """加载冻结基座(eval 模式),按 args.base_policy_type 分发。

    - act (默认,行为不变):ACT(PyTorch lerobot)。args.base_wandb_id 若是本地目录则直接 load
      (自动取其 policy/ 子目录或直接 policy 目录),否则按 wandb artifact 拉。
    - pi05:pi0/pi05(openpi,JAX)。经 openpi-client websocket 连 serve 进程,用
      Pi05PolicyAdapter 包成 step 级基座(select_action / reset / config.image_features),
      内部自带 action queue(execute_horizon)。仅 queue 模式(chunk_length==1)用。
    """
    if getattr(args, "base_policy_type", "act") == "pi05":
        from resfit.rl_finetuning.config.residual_td3 import BasePolicyConfig
        from resfit.lerobot.policies.pi05 import load_pi05_base_policy
        image_key_map = (json.loads(args.pi0_image_key_map)
                         if args.pi0_image_key_map else dict(PIECE_IMAGE_KEY_MAP))
        bp = BasePolicyConfig(
            type="pi05", host=args.pi0_host, port=args.pi0_port,
            prompt=args.pi0_prompt, action_dim=args.pi0_action_dim,
            execute_horizon=args.pi0_execute_horizon,
            image_key_map=image_key_map, kai0_paths=[args.pi0_kai0_path],
        )
        # adapter 已按 device 构造(连 serve);不需再 .to/.eval(它是远端推理的薄封装)
        return load_pi05_base_policy(bp, device)

    wandb_id = args.base_wandb_id
    if os.path.isdir(wandb_id):
        from pathlib import Path
        cand = Path(wandb_id) / "policy"
        policy_dir = cand if cand.is_dir() else Path(wandb_id)
    else:
        policy_dir, _ = download_policy_from_wandb(wandb_id, step=wt_type, artifact_version=wt_version)
    base_policy = load_policy(policy_dir)
    base_policy.to(device)
    base_policy.eval()
    return base_policy


def _offline_buffer_signature(args, image_keys, offline_cap, shaping_mode, potential=None):
    """决定 offline buffer 内容的全部参数;任一变化都意味着旧缓存失效需重建。

    内容依赖:动作/base_action 用 action_scaler(action_scale+min_range);state 标准化(来自
    dataset stats);reward/done/stage 由 (reward_shaping mode, bonus, gamma) 烤入;n-step 由
    MultiStepTransform(n_step, gamma) 在 add 时合并进存储;图像由 image_keys;stage 由 stage_cache。

    ③b:potential_source=hiql 时 Φ=V(state)*scale 被烤进 offline reward,故还依赖 value ckpt /
    phi_scale / scale / value 的 state_mode(18 eef vs 30 eef_piece);任一变(含换 value.pt)都
    必须重建,否则会错误复用旧 V 算的 reward。stage 源不加这些键 → 现有 stage 缓存向后兼容。
    """
    sig = {
        "dataset": args.dataset,
        "offline_dataset_path": os.path.abspath(args.offline_dataset_path),
        "num_demos": args.offline_num_demos,
        "offline_cap": int(offline_cap),
        "action_scale": float(args.action_scale),
        "min_range_per_dim": float(args.min_range_per_dim),
        "reward_shaping": shaping_mode,
        "stage_reward_bonus": float(args.stage_reward_bonus),
        "gamma": float(args.gamma),
        "n_step": int(args.n_step),
        "image_keys": sorted(image_keys),
        "task": args.task,
        "stage_cache": (os.path.abspath(args.offline_stage_cache)
                        if args.offline_stage_cache else None),
    }
    if args.potential_source == "hiql":
        sig["potential_source"] = "hiql"
        sig["hiql_value_ckpt"] = (os.path.abspath(args.hiql_value_ckpt)
                                  if args.hiql_value_ckpt else None)
        sig["phi_scale"] = float(args.phi_scale)
        sig["potential_scale"] = (round(float(potential.scale), 8)
                                  if potential is not None else None)
        sig["value_state_mode"] = (getattr(potential, "state_mode", None)
                                   if potential is not None else None)
    if args.subgoal_conditioned:
        sig["subgoal"] = True
        sig["gc_value_ckpt"] = os.path.abspath(args.gc_value_ckpt)
        sig["high_actor_ckpt"] = os.path.abspath(args.high_actor_ckpt)
        sig["subgoal_way_steps"] = int(args.subgoal_way_steps)
    if args.offline_base_mode != "gt":
        sig["offline_base_mode"] = args.offline_base_mode
        sig["base_policy_type"] = args.base_policy_type
        sig["base_n_action_steps"] = args.base_n_action_steps   # ACT 队列重规划步幅,改变 base_action
        sig["base_wandb_id"] = (os.path.abspath(args.base_wandb_id)
                                if args.base_wandb_id and os.path.isdir(args.base_wandb_id)
                                else args.base_wandb_id)
        if args.base_policy_type == "pi05":
            # pi05 身份由 serve 连接参数决定(base_wandb_id 对它无意义),换 serve 必须重建
            sig["pi0_host"] = args.pi0_host
            sig["pi0_port"] = args.pi0_port
            sig["pi0_prompt"] = args.pi0_prompt
            sig["pi0_execute_horizon"] = args.pi0_execute_horizon   # pi05 队列执行步幅,改变 base_action
    return sig


def _validate_offline_base_mode(args):
    """base_policy 模式需 queue(chunk_length==1 且 base_action_mode=="queue");gt 模式跳过。"""
    if args.offline_base_mode != "base_policy":
        return
    assert args.base_action_mode == "queue" and args.chunk_length == 1, \
        ("--offline_base_mode base_policy 需 queue 模式 "
         "(--base_action_mode queue --chunk_length 1);当前 "
         f"base_action_mode={args.base_action_mode!r} chunk_length={args.chunk_length}")
    if args.base_policy_type == "pi05":
        print("[offline-base] WARN: pi05 base 走 websocket 逐帧推理(~23.8 万次),"
              "首次建 buffer 很慢;--offline_buffer_cache 缓存后秒读")


def _offline_cache_valid(cache_dir, sig):
    """缓存目录存在、meta 完整且签名完全匹配才算命中。"""
    meta = os.path.join(cache_dir, "buffer_meta.json")
    storage = os.path.join(cache_dir, "storage")
    if not (os.path.isfile(meta) and os.path.isdir(storage)):
        return False
    try:
        with open(meta) as fp:
            saved = json.load(fp)
    except Exception:
        return False
    if saved.get("signature") != sig:
        print(f"[offline] 缓存 {cache_dir} 签名与当前配置不符 → 重建")
        return False
    return True


def _save_offline_buffer(offline_rb, cache_dir, sig, n_off):
    """把建好的 offline buffer(已经过 MultiStepTransform 合并)的存储落成 memmap + 写签名。

    存的是合并后的 transition;loads 时用不带 MultiStepTransform 的 buffer 注入(勿二次合并)。
    """
    os.makedirs(cache_dir, exist_ok=True)
    data = offline_rb[:len(offline_rb)]
    for k in ("index", "_weight"):     # 采样期/读取期附加键,不入存储
        if k in data.keys():
            data = data.exclude(k)
    data.memmap(os.path.join(cache_dir, "storage"))
    with open(os.path.join(cache_dir, "buffer_meta.json"), "w") as fp:
        json.dump({"signature": sig, "n_transitions": int(n_off)}, fp, indent=2)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", choices=["raw", "flow"], default="raw")
    p.add_argument("--task", default="TwoArmBoxCleanup")
    p.add_argument("--base_wandb_id", default="dexmg-boxcleanup-bc/d59wny58")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-box-cleanup")
    p.add_argument("--chunk_length", type=int, default=1)
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--total_env_steps", type=int, default=500_000)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--eval_every_env_steps", type=int, default=10_000)
    p.add_argument("--utd", type=int, default=4)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer_size", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--actor_lr", type=float, default=1e-6)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--stddev", type=float, default=0.05)
    p.add_argument("--ae_ckpt", default=None)   # M2 flow 用
    p.add_argument("--eval_num_envs", type=int, default=8)
    p.add_argument("--eval_num_episodes", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="少量步数冒烟")
    p.add_argument("--stage_balanced", action="store_true", help="按 stage 配额采样(stage-balanced replay)")
    p.add_argument("--stage_conditioned", action="store_true",
                   help="把 stage_id one-hot 喂进 actor/critic（§22 分段修正；默认关=baseline）")
    p.add_argument("--subgoal_conditioned", action="store_true",
                   help="把 HIQL 潜子目标 z(10维)喂进 actor/critic(分层路 Phase 3;默认关=baseline)")
    p.add_argument("--gc_value_ckpt", default=None, help="Phase 1 gc_value.pt(--subgoal_conditioned 需)")
    p.add_argument("--high_actor_ckpt", default=None, help="Phase 2 high_actor.pt(在线提 z)")
    p.add_argument("--subgoal_way_steps", type=int, default=25, help="offline 真航点 z 的 k 步")
    p.add_argument("--subgoal_state30_cache", default=None,
                   help="state30 缓存 npz(算 goal30,避免回放;建议设 outputs_chunk/three_piece_state30.npz)")
    p.add_argument("--renorm_subgoal", dest="renorm_subgoal", action="store_true", default=True,
                   help="online z 投回半径 sqrt(rep_dim) 的球面,对齐 offline φ 与 HIQL eval(evaluation.py:113-114);"
                        "只动模长不动方向。默认开=对齐 HIQL(实测偏差仅 1-2%,clamp 版尾部 ±10-20%)")
    p.add_argument("--no_renorm_subgoal", dest="renorm_subgoal", action="store_false",
                   help="关闭 online z 投球面(回到旧版逐位行为)")
    p.add_argument("--stage_budget", default=None,
                   help="逐阶段残差幅度乘子,逗号分隔,长度=num_stages(如 '1,1,1,0.3,0.1');不传=关(§18.3)")
    p.add_argument("--offline_base_mode", choices=["gt", "base_policy"], default="gt",
                   help="离线 buffer 的 base_action 来源:gt(默认,逐位等价=GT-as-base,残差目标0)|"
                        "base_policy(冻结 base 现算 base_action,action 仍存 GT;bc_target=GT-base,"
                        "锚向专家 + critic offline 锚对齐在线流形)。base_policy 需 queue 模式。")
    p.add_argument("--demo_bc_coef", type=float, default=0.0,
                   help="残差 actor 的 demo-BC 权重(模块②a);0=关(逐位等价 baseline)。"
                        ">0 需 offline_fraction>0(bc_batch 取自 offline_rb)、--actor raw")
    p.add_argument("--relabel", action="store_true",
                   help="在线 relay relabeling(模块②b):harvest 产出前缀段进 relabel buffer,"
                        "bc_batch 改 relabel+demo 50/50 混采。默认关=逐位等价 ②a。"
                        "需 --demo_bc_coef>0 + --offline_fraction>0 + --actor raw")
    p.add_argument("--relabel_buffer_size", type=int, default=50_000,
                   help="relabel buffer 容量(FIFO,CPU storage)")
    p.add_argument("--relabel_min_stage", type=int, default=1,
                   help="只 harvest 走到 stage>=该值的 episode(默认 1=只要推进过)")
    p.add_argument("--reward_shaping", choices=["none", "staged", "potential"], default=None,
                   help="奖励整形模式(canonical):none|staged(净加)|potential(PBS,不改最优策略)")
    p.add_argument("--staged_reward", action="store_true",
                   help="[别名] 等价 --reward_shaping staged;canonical flag 优先")
    p.add_argument("--stage_reward_bonus", type=float, default=1.0,
                   help="stage 整形幅度旋钮(staged/potential 共用;仅在 shaping≠none 时生效)")
    p.add_argument("--potential_source", choices=["stage", "hiql"], default="stage",
                   help="PBS 势函数 Φ 来源(③b):stage=整数 stage(默认,逐位等价);hiql=V(state)*scale")
    p.add_argument("--hiql_value_ckpt", default=None,
                   help="--potential_source hiql 时 ③a 产出的 value.pt 路径")
    p.add_argument("--phi_scale", type=float, default=1.0,
                   help="hiql Φ 的额外缩放乘子(在 auto_scale 之上;默认 1.0)")
    p.add_argument("--output_dir", default="outputs_chunk",
                   help="ckpt / eval 产物目录(并行 run 用不同目录避免抢 best.pt)")
    p.add_argument("--base_action_mode", choices=["replan", "queue"], default="queue",
                   help="基座动作来源:replan(每边界重跑模型取前chunk步)|queue(ACT原生action queue,仅cl=1,复刻原版step级)")
    # --- 基座类型开关(act 默认行为不变;pi05=pi0/pi05 经 websocket 连 openpi serve)---
    p.add_argument("--base_policy_type", choices=["act", "pi05"], default="act",
                   help="基座类型:act(PyTorch lerobot,默认)|pi05(pi0/pi05,经 openpi-client websocket 连 serve)")
    p.add_argument("--pi0_host", default="127.0.0.1", help="pi0/pi05 serve websocket host")
    p.add_argument("--pi0_port", type=int, default=8000, help="pi0/pi05 serve websocket port")
    p.add_argument("--pi0_prompt", default="assemble the three pieces",
                   help="pi0/pi05 prompt;必须与微调/serve 的 default-prompt 完全一致")
    p.add_argument("--pi0_action_dim", type=int, default=14,
                   help="pi0/pi05 基座输出 action 维度(three-piece=14)")
    p.add_argument("--pi0_execute_horizon", type=int, default=30,
                   help="pi0/pi05 每次推理实际执行的步数(adapter 内部 action queue 长度)")
    p.add_argument("--pi0_kai0_path", default="/mnt/mnt/data/kai0_new4090",
                   help="kai0 仓路径(供 import resfit_pi05.pi05_policy_adapter;本机=kai0_new4090)")
    p.add_argument("--pi0_image_key_map", default=None,
                   help="JSON: env obs 图像键->pi0 槽位(base/left_wrist/right_wrist);缺省=three-piece 默认映射")
    p.add_argument("--base_n_action_steps", type=int, default=None,
                   help="覆盖基座 ACT 的 n_action_steps(每多少步重规划;默认用 checkpoint 的 20)。"
                        "≤chunk_size;设 10 即基座预测20步但只执行前10就重推理")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    # --- offline demo buffer(锚;方案 A:从源 dexmimicgen HDF5 灌装,GT-as-base)---
    p.add_argument("--offline_dataset_path", default=None,
                   help="源 dexmimicgen HDF5 路径;设了且 offline_fraction>0 才灌 offline 锚 buffer")
    p.add_argument("--offline_fraction", type=float, default=0.0,
                   help="每个 batch 中 offline demo 占比(RLPD 默认 0.5);0=纯在线(行为同改造前)")
    p.add_argument("--offline_num_demos", type=int, default=None,
                   help="灌装的 demo 条数(默认全部 ~1000)")
    p.add_argument("--offline_stage_cache", default=None,
                   help="stage 缓存 npz 路径;命中则秒级读、不 replay。"
                        "缺失会 replay 并落盘到此路径(见 precompute_stage_cache)")
    p.add_argument("--offline_buffer_cache", default=None,
                   help="offline buffer 落盘目录(建议放 /mnt 大盘)。命中且签名匹配则秒级 loads、"
                        "跳过 ~27min 重建;缺失/签名不符则正常重建并 dump。签名锁 "
                        "dataset/action_scale/min_range/reward_shaping/bonus/gamma/n_step/stage_cache 等")
    p.add_argument("--wandb_project", default="dexmg-chunk-residual",
                   help="wandb project 名")
    p.add_argument("--wandb_entity", default=None, help="wandb entity(默认用账号默认)")
    p.add_argument("--wandb_name", default=None,
                   help="wandb run 名;缺省回落为 output_dir 的 basename")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online", help="wandb 模式;--smoke 时自动 disabled")
    p.add_argument("--log_freq", type=int, default=100,
                   help="训练指标上报间隔(env_steps)")
    return p


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    sample_gen = torch.Generator().manual_seed(args.seed)   # stage-balanced 采样用

    # --- 归一化器(从 dataset stats 建,与 AE / RL 同款)---
    # 只需 dataset 的统计量来建归一化器:用 LeRobotDatasetMetadata(仅拉 meta/ 几个小文件)
    # 而非 LeRobotDataset(会 snapshot 整个 repo,含上百 MB 视频)。.stats 完全一致,且可离线工作,
    # 避免国内直连 HF 下视频频繁超时。
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(args.dataset)
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device=args.device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=args.device)

    # --- 基座 + env ---
    if args.base_policy_type == "pi05":
        assert args.base_action_mode == "queue", \
            "pi05/pi0 基座是 step 级(select_action),只支持 --base_action_mode queue(且 --chunk_length 1)"
    if args.base_action_mode == "queue":
        assert args.chunk_length == 1, "--base_action_mode queue 仅支持 --chunk_length 1"
    _validate_offline_base_mode(args)
    base_policy = build_base_policy(args, args.device)
    shaping_mode = resolve_shaping_mode(args.reward_shaping, args.staged_reward)
    num_stages = NUM_STAGES.get(args.task, 1)   # 无检测器任务退化为 1 段

    # --- ③b: HiqlPotential(potential_source=hiql 时构建;stage 时保持 None 逐位等价)---
    # 必须在 create_vectorized_env 之前:value.pt 的 state_mode 决定训练 env 要不要经 info
    # 透出特权 rel_piece(eef_piece object-aware);observation.state 始终 18 维、actor/critic 不变。
    potential = None
    if args.potential_source == "hiql":
        import os as _os
        assert shaping_mode == "potential", \
            "--potential_source hiql 需 --reward_shaping potential"
        assert args.hiql_value_ckpt and _os.path.exists(args.hiql_value_ckpt), \
            f"--hiql_value_ckpt 不存在: {args.hiql_value_ckpt!r}"
        from resfit.rl_finetuning.chunk_residual.hiql_potential import HiqlPotential
        potential = HiqlPotential.from_ckpt(
            args.hiql_value_ckpt, num_stages=num_stages,
            phi_scale=args.phi_scale, device=args.device)
        print(f"[hiql-phi] potential on; ckpt={args.hiql_value_ckpt} "
              f"state_mode={potential.state_mode} scale={potential.scale:.4f}")
    # value 的 state_mode 决定训练 env 是否经 info 透出 rel_piece(只喂 Φ,不进 observation.state)
    env_state_mode = getattr(potential, "state_mode", "eef") if potential is not None else "eef"
    if args.subgoal_conditioned:
        env_state_mode = "eef_piece"   # 子目标在线需 env 经 info["rel_piece"] 透出 rel

    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device,
                                    state_mode=env_state_mode)
    print(f"[reward-shaping] mode={shaping_mode} bonus={args.stage_reward_bonus} gamma={args.gamma}")
    print(f"[base-action] mode={args.base_action_mode}")
    print(f"[state-mode] env_state_mode={env_state_mode} "
          f"(observation.state 仍 18 维;rel_piece 经 info 只喂 Φ/z)")
    # eval 不加 shaping、不喂 Φ → 保持 eef(省每步 sim rel 开销);
    # FIX A: --subgoal_conditioned 时需 eef_piece 让 eval info 携带 rel_piece 供 z 注入
    eval_state_mode = "eef_piece" if args.subgoal_conditioned else "eef"
    eval_vec = create_vectorized_env(env_name=args.task, num_envs=args.eval_num_envs,
                                     device=args.device, state_mode=eval_state_mode)
    # queue 有状态(per-env action queue):eval(num_envs>1)与训练(num_envs=1)共享同一 base_policy
    # 会互踩 queue。queue 模式给 eval 单独的 base_policy 实例(对齐原版 train_residual_td3 双实例)。
    eval_base_policy = (build_base_policy(args, args.device)
                        if args.base_action_mode == "queue" else base_policy)
    # 可选:覆盖基座 n_action_steps(每多少步重规划)。在任何 reset 前设置,reset() 会按此建队列
    if args.base_n_action_steps is not None:
        assert args.base_policy_type == "act", \
            "--base_n_action_steps 只对 ACT 基座有效;pi0/pi05 用 --pi0_execute_horizon 控制每次推理执行步数"
        assert args.base_n_action_steps <= base_policy.config.chunk_size, \
            f"n_action_steps({args.base_n_action_steps}) 不能超过 chunk_size({base_policy.config.chunk_size})"
        base_policy.config.n_action_steps = args.base_n_action_steps
        eval_base_policy.config.n_action_steps = args.base_n_action_steps
        print(f"[base-action] 覆盖 n_action_steps={args.base_n_action_steps} "
              f"(基座每{args.base_n_action_steps}步重规划;chunk_size={base_policy.config.chunk_size})")
    eval_env = ChunkResidualEnvWrapper(eval_vec, eval_base_policy, action_scaler, state_standardizer,
                                       chunk_length=args.chunk_length,
                                       reward_shaping_mode="none",   # eval 不加 shaping,指标纯净
                                       base_action_mode=args.base_action_mode)

    # --- agent(复用 QAgent,action_dim=480)---
    from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
    cfg = ResidualTD3BoxCleanConfig()
    cfg.agent.actor_lr = args.actor_lr
    cfg.agent.critic_lr = args.critic_lr
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.bc_loss_coef = args.demo_bc_coef
    cfg.agent.bc_loss_dynamic = 0          # 均匀 BC(②a 不开 DAPG 动态)
    if args.demo_bc_coef > 0:
        assert args.actor == "raw", "demo_bc 第一版只支持 --actor raw"
        assert args.offline_fraction > 0, \
            "demo_bc_coef>0 需 offline_fraction>0(bc_batch 取自 offline_rb)"

    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length,
                                  stage_reward_bonus=args.stage_reward_bonus,
                                  reward_shaping_mode=shaping_mode, gamma=args.gamma,
                                  base_action_mode=args.base_action_mode,
                                  potential=potential)

    # --- 维度 ---
    image_keys = list(base_policy.config.image_features.keys())
    lowdim_keys = ["observation.state", "observation.base_action", "observation.stage_id"]
    if args.subgoal_conditioned:
        lowdim_keys.append("observation.subgoal")
    obs0, _ = env.reset()
    img_c, img_h, img_w = obs0[image_keys[0]].shape[1:]
    state_dim = obs0["observation.state"].shape[1]
    if potential is not None:
        # 命门:observation.state(给 actor/critic)绝不含特权 rel_piece;喂 Φ 的 V 输入维度
        # = state_dim(+12 当 eef_piece)。断言 value.pt 与 task / 接线一致(online/offline 同源)。
        exp_v_dim = state_dim + (12 if env_state_mode == "eef_piece" else 0)
        assert potential.model.state_dim == exp_v_dim, (
            f"Φ value 输入维度 {potential.model.state_dim} != observation.state({state_dim})"
            f"+rel({12 if env_state_mode == 'eef_piece' else 0});value.pt 的 state_mode 与 task 不匹配")
    action_dim = env.action_dim * args.chunk_length     # = 480
    if args.stage_conditioned:
        assert args.actor == "raw", "stage-conditioning 第一版只支持 --actor raw（flow 注入未接 stage）"
        assert args.task in NUM_STAGES, f"--stage_conditioned 需要 {args.task} 有 stage 检测器"
    from resfit.rl_finetuning.off_policy.rl.stage_utils import parse_stage_budget
    stage_budget = parse_stage_budget(args.stage_budget, num_stages)
    if stage_budget is not None:
        assert args.actor == "raw", "stage_budget 第一版只支持 --actor raw"
        assert args.task in NUM_STAGES, f"--stage_budget 需要 {args.task} 有 stage 检测器"

    subgoal = None
    if args.subgoal_conditioned:
        assert args.actor == "raw", "subgoal-conditioning 第一版只支持 --actor raw"
        assert env_state_mode == "eef_piece", "--subgoal_conditioned 需 env 透出 rel(eef_piece)"
        assert args.gc_value_ckpt and args.high_actor_ckpt, \
            "--subgoal_conditioned 需 --gc_value_ckpt 与 --high_actor_ckpt"
        # FIX B: guard goal30 source — 至少有一个来源能构建 goal30
        assert args.subgoal_state30_cache or args.offline_dataset_path, \
            "--subgoal_conditioned 需 --subgoal_state30_cache 或 --offline_dataset_path(用于建 goal30)"
        from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal, representative_goal30
        from resfit.rl_finetuning.chunk_residual.state30_cache import load_or_build_state30
        # goal30 = demo 末态的 medoid(离均值最近的真实末态),作为在线高层的固定任务目标。
        # 不用算术均值:均值是 off-manifold 虚构质心、会糊掉散得最厉害的 rel_piece 物体信息,且
        # high_actor 训练时只见过真实态当 goal(2026-06-09 讨论);用 state30 缓存避免回放。
        _seqs30 = load_or_build_state30(args.offline_dataset_path, args.dataset,
                                        args.offline_num_demos, args.subgoal_state30_cache)
        goal30 = representative_goal30(_seqs30)
        # FIX C: assert goal30 is 30-dim (eef_piece state)
        assert goal30.shape[0] == 30, \
            f"goal30 须 30 维(eef_piece),got {goal30.shape[0]};检查 state30 缓存是否来自 eef_piece"
        subgoal = HiqlSubgoal.from_ckpts(args.gc_value_ckpt, args.high_actor_ckpt,
                                         goal30=goal30, device=args.device,
                                         renorm_subgoal=args.renorm_subgoal)
        print(f"[hiql-subgoal] on; rep_dim={subgoal.rep_dim} renorm={args.renorm_subgoal} "
              f"gc={args.gc_value_ckpt} high={args.high_actor_ckpt}")

    agent = QAgent(obs_shape=(img_c, img_h, img_w), prop_shape=(state_dim,),
                   action_dim=action_dim, rl_cameras=image_keys,
                   cfg=cfg.agent, residual_actor=True,
                   stage_conditioned=args.stage_conditioned, num_stages=num_stages,
                   stage_budget=stage_budget,
                   subgoal_conditioned=args.subgoal_conditioned,
                   subgoal_dim=(subgoal.rep_dim if subgoal is not None else 0),)

    # repr/patch 维(供 flow actor 构造,复用 QAgent 的算法)
    enc0 = agent.encoders[0]
    repr_dim = int(enc0.repr_dim) * len(image_keys)
    patch_repr_dim = int(enc0.patch_repr_dim)
    _maybe_inject_flow_actor(args, agent, repr_dim, patch_repr_dim, state_dim, action_dim)

    # --- buffer(复用 torchrl 构造,动作 480)---
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=args.buffer_size, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority",
        transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
        pin_memory=True, prefetch=4, batch_size=args.batch_size)

    # --- offline demo 锚 buffer(方案 A;offline_fraction=0 时整段跳过,行为同改造前)---
    online_batch_size = int(args.batch_size * (1 - args.offline_fraction))
    offline_batch_size = int(args.batch_size * args.offline_fraction)
    offline_rb = None
    if args.offline_fraction > 0.0:
        assert args.offline_dataset_path is not None, \
            "offline_fraction>0 需 --offline_dataset_path 指向源 dexmimicgen HDF5"
        from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import concat_mixed_batch
        from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
            build_offline_buffer, count_offline_transitions)
        # 按 demo transition 数精确定容,否则 LazyTensorStorage 不足会挤掉早期 demo(锚不全)
        offline_cap = count_offline_transitions(args.offline_dataset_path,
                                                num_demos=args.offline_num_demos)
        sig = _offline_buffer_signature(args, image_keys, offline_cap, shaping_mode,
                                        potential=potential)

        def _new_offline_rb(with_transform):
            # 命中缓存走 with_transform=False:存的是已合并 transition,勿让 MultiStepTransform 二次合并
            tf = MultiStepTransform(n_steps=args.n_step, gamma=args.gamma) if with_transform else None
            return TensorDictPrioritizedReplayBuffer(
                storage=LazyTensorStorage(max_size=offline_cap, device="cpu"),
                alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority",
                transform=tf, pin_memory=True, prefetch=4,
                batch_size=max(offline_batch_size, 1))

        cache_dir = args.offline_buffer_cache
        if cache_dir and _offline_cache_valid(cache_dir, sig):
            from tensordict import TensorDict
            offline_rb = _new_offline_rb(with_transform=False)
            data = TensorDict.load_memmap(os.path.join(cache_dir, "storage"))
            offline_rb.extend(data)
            n_off = len(offline_rb)
            print(f"[offline] 命中缓存 {cache_dir}:loads {n_off} 条(跳过重建)")
        else:
            offline_rb = _new_offline_rb(with_transform=True)
            build_offline_buffer(
                offline_rb, args.offline_dataset_path,
                action_scaler=action_scaler, state_standardizer=state_standardizer,
                image_keys=image_keys, bonus=args.stage_reward_bonus,
                mode=shaping_mode, gamma=args.gamma, num_demos=args.offline_num_demos,
                stage_cache=args.offline_stage_cache, potential=potential,
                subgoal=subgoal, way_steps=args.subgoal_way_steps,
                base_policy=base_policy, base_mode=args.offline_base_mode,
                base_device=args.device,)
            n_off = len(offline_rb)
            if cache_dir:
                _save_offline_buffer(offline_rb, cache_dir, sig, n_off)
                print(f"[offline] 已建 {n_off} 条并落盘 {cache_dir}(下次秒级复用)")
            else:
                print(f"[offline] 灌装 {n_off} 条 demo transition(未设 --offline_buffer_cache,不落盘)")
        print(f"[offline] 混采 online_bs={online_batch_size} "
              f"offline_bs={offline_batch_size}(fraction={args.offline_fraction})")

    relabel_rb = None
    harvester = None
    if args.relabel:
        assert args.demo_bc_coef > 0, "--relabel 需 --demo_bc_coef>0(relabel 走 BC 路径)"
        assert args.offline_fraction > 0, "--relabel 需 --offline_fraction>0(demo 半边)"
        assert args.actor == "raw", "--relabel 第一版只支持 --actor raw"
        from resfit.rl_finetuning.chunk_residual.relabel import RelabelHarvester, sample_bc_batch
        relabel_rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=args.relabel_buffer_size, device="cpu"),
            batch_size=max(args.batch_size // 2, 1))
        harvester = RelabelHarvester(min_stage=args.relabel_min_stage)
        print(f"[relabel] on; buffer_size={args.relabel_buffer_size} min_stage={args.relabel_min_stage}")

    # --- 训练循环(x 轴=环境步;每 chunk 计入 chunk_length 步)---
    run = init_wandb(args)

    obs, reset_info = env.reset()
    cur_rel = reset_info.get("rel_piece") if args.subgoal_conditioned else None
    env_steps = 0
    next_eval = 0
    next_log = args.learning_starts
    best_sr = 0.0
    last_diag = None
    total = 2 * args.chunk_length if args.smoke else args.total_env_steps
    while env_steps <= total:
        next_rel = None   # FIX D: silence unbound-var lint; overwritten below when subgoal_conditioned
        if args.subgoal_conditioned:
            obs["observation.subgoal"] = subgoal.subgoal_online(
                obs["observation.state"], cur_rel).to(obs["observation.state"].device)
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, eval_mode=False, stddev=args.stddev, cpu=False)  # [1,480] 残差
        next_obs, reward, terminated, truncated, info = env.step(action)
        if args.subgoal_conditioned:
            next_rel = info.get("rel_piece")
            next_obs["observation.subgoal"] = subgoal.subgoal_online(
                next_obs["observation.state"], next_rel).to(next_obs["observation.state"].device)
        done = terminated | truncated
        add_chunk_transition(obs=obs, next_obs=next_obs, combined_action=info["scaled_action"],
                             reward=reward, done=done, info=info, image_keys=image_keys,
                             lowdim_keys=lowdim_keys, online_rb=online_rb)
        if harvester is not None:
            harvester.add(make_bc_entry(obs, info["scaled_action"], image_keys, lowdim_keys),
                          info.get("max_stage_in_chunk", 0))
            if bool(done.any()):
                for e in harvester.flush():
                    # extend(非 add):e 是 batch=[1] 的条目,extend 存成 [feat] 元素 → sample 出 [N,feat]
                    # (2D),与从 memmap extend 进来的 offline_rb 同构。relabel_rb 无 MultiStepTransform
                    # 收维(online_rb 有),若用 add 会留下前导 [1] 维 → 与 offline 混采 concat 报 3-vs-2。
                    relabel_rb.extend(e)
        obs = next_obs
        if args.subgoal_conditioned:
            cur_rel = next_rel
        env_steps += args.chunk_length

        if env_steps >= args.learning_starts and len(online_rb) > online_batch_size:
            for i in range(args.utd):
                if args.stage_balanced:
                    online_batch = sample_stage_balanced(online_rb, online_batch_size, generator=sample_gen)
                else:
                    online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(args.device, non_blocking=True)  # 喂 GPU 前搬设备
                if offline_rb is not None:                          # RLPD 混采:online + offline demo 锚
                    offline_batch = offline_rb.sample(offline_batch_size).to(args.device, non_blocking=True)
                    batch = concat_mixed_batch(online_batch, offline_batch)   # 取公共 key,容忍 _weight 不一致
                else:
                    batch = online_batch
                update_actor = ((i + 1) % args.utd == 0)
                bc_batch = None
                if args.demo_bc_coef > 0 and update_actor and offline_rb is not None:
                    if args.relabel:
                        bc_batch = sample_bc_batch(relabel_rb, offline_rb, args.batch_size, args.device)
                    else:
                        bc_batch = offline_rb.sample(args.batch_size).to(args.device, non_blocking=True)
                m_upd = agent.update(batch, args.stddev, update_actor,
                                     bc_batch=bc_batch,
                                     ref_agent=(agent if bc_batch is not None else None))
                # stage-aware 诊断:按 stage 看残差幅度/价值(用 update 已暴露的 _actions/_target_q)
                if update_actor and "_actions" in m_upd:
                    st = batch["obs"]["observation.stage_id"].flatten().cpu()
                    vals = {"residual_norm": m_upd["_actions"].norm(dim=-1)}
                    if "_target_q" in m_upd:
                        vals["target_q"] = m_upd["_target_q"]
                    last_diag = flatten_stage_diagnostics(stage_diagnostics(st, vals))

            if env_steps >= next_log:
                lrs = {"actor": agent.actor_opt.param_groups[0]["lr"],
                       "critic": agent.critic_opt.param_groups[0]["lr"],
                       "encoder": agent.encoder_opt.param_groups[0]["lr"]}
                buf_sizes = {"online": len(online_rb),
                             "offline": len(offline_rb) if offline_rb else 0,
                             "relabel": len(relabel_rb) if relabel_rb else 0}
                wandb.log(build_train_log_dict(m_upd, lrs, buf_sizes), step=env_steps)
                next_log += args.log_freq

        if env_steps >= next_eval:
            with torch.no_grad():
                m = run_dexmg_evaluation(env=eval_env, agent=agent,
                                         num_episodes=args.eval_num_episodes, device=args.device,
                                         global_step=env_steps, save_video=False,
                                         save_q_plots=False, run_name=f"chunk_{args.actor}",
                                         output_dir=args.output_dir,
                                         subgoal=(subgoal if args.subgoal_conditioned else None))
            sr = m["eval/success_rate"]
            if sr > best_sr:
                best_sr = sr
                os.makedirs(args.output_dir, exist_ok=True)
                save_checkpoint(agent, os.path.join(args.output_dir, "best.pt"),
                                global_step=env_steps,
                                config=args, success_rate=sr)
            print(f"[env_steps {env_steps}] eval success_rate={sr:.3f} (best {best_sr:.3f})")
            if last_diag is not None:
                print("[stage-diag] " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(last_diag.items())))
            print("[stage-purity] " + env.stage_purity_summary())
            wandb.log(build_eval_log_dict(m, last_diag, env.stage_purity_summary()),
                      step=env_steps)
            next_eval += args.eval_every_env_steps
        if args.smoke:
            break

    print(f"done. best success_rate={best_sr:.3f}")
    wandb.finish()


if __name__ == "__main__":
    main()
