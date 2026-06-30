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
import hashlib
import json
import os

import numpy as np

import wandb
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# NOTE: create_vectorized_env 走 dexmg.py(顶层 import robosuite-1.5 专属 API),
# libero 路(robosuite-1.4 env)不能触发它 —— 故改为 main() 的 dexmg else 分支内
# lazy import(见下方)。libero 分支不 import 它。
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.chunk_residual.bc_schedule import linear_bc_coef
from resfit.rl_finetuning.chunk_residual.libero_obs import load_libero_norm_stats
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
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
from resfit.rl_finetuning.chunk_residual.online_hiql_finetune import OnlineHiqlFinetuner


def maybe_build_finetuner(args, subgoal, finetune_seqs, *, gc_info):
    """开关任一在线微调 flag 时构造 OnlineHiqlFinetuner;否则 None(零回归)。

    value_loss_mode / value_mask_mode 从 gc_value ckpt 的 info 读(同源,不新引口径)。
    finetune_seqs:与 gc_value 同维的离线 demo 标准化 state 序列(eef=30/act_feat=530/pi0=2056)。
    """
    if not (args.online_finetune_value or args.online_finetune_high_actor):
        return None
    assert subgoal is not None, "在线微调需 --subgoal_conditioned"
    return OnlineHiqlFinetuner(
        subgoal, offline_seqs=finetune_seqs, device=args.device,
        value_lr=args.online_value_lr, high_actor_lr=args.online_high_actor_lr,
        way_steps=args.subgoal_way_steps,
        offline_fraction=args.online_finetune_offline_fraction,
        batch_size=args.batch_size, every=args.online_finetune_every,
        gamma=args.gamma, expectile=args.online_finetune_expectile,
        ema=args.online_finetune_ema, future_mode=args.online_finetune_future_mode,
        value_loss_mode=gc_info["value_loss_mode"],
        value_mask_mode=gc_info["value_mask_mode"],
        adv_agg="min",
        finetune_value=args.online_finetune_value,
        finetune_high_actor=args.online_finetune_high_actor,
        max_online_transitions=args.online_finetune_max_trans,
        min_online_transitions=args.online_finetune_min_trans, seed=args.seed)


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
        # LIBERO 路:schema=libero,prompt 用该 task 的 language(覆盖 --pi0_prompt 默认)。
        schema = "libero" if getattr(args, "env_family", "dexmg") == "libero" else "dexmg"
        prompt = args.pi0_prompt
        if schema == "libero":
            prompt = _libero_task_prompt(args.libero_suite, args.libero_task_id)
        image_key_map = (json.loads(args.pi0_image_key_map)
                         if args.pi0_image_key_map else dict(PIECE_IMAGE_KEY_MAP))
        bp = BasePolicyConfig(
            type="pi05", host=args.pi0_host, port=args.pi0_port,
            prompt=prompt, action_dim=args.pi0_action_dim,
            execute_horizon=args.pi0_execute_horizon,
            image_key_map=image_key_map, kai0_paths=[args.pi0_kai0_path],
        )
        # adapter 已按 device 构造(连 serve);不需再 .to/.eval(它是远端推理的薄封装)
        return load_pi05_base_policy(bp, device, schema=schema)

    # ACT 路才需(lazy:其链拉外部 lerobot,libero/py3.8 env 没装也不需要)
    from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
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


def validate_hiql_subgoal_args(args):
    """A2(--potential_source hiql_subgoal)的前置校验。其余 potential_source 不受影响。"""
    if getattr(args, "potential_source", "stage") != "hiql_subgoal":
        return
    assert args.reward_shaping == "potential", \
        "--potential_source hiql_subgoal 需 --reward_shaping potential"
    assert getattr(args, "subgoal_conditioned", False), \
        "--potential_source hiql_subgoal 需 --subgoal_conditioned(提供 gc_value/high_actor/z)"
    assert getattr(args, "renorm_subgoal", False), \
        "--potential_source hiql_subgoal 需 renorm_subgoal=True(z 须在 sqrt(rep_dim) 球面)"
    assert getattr(args, "gc_value_ckpt", None), \
        "--potential_source hiql_subgoal 需 --gc_value_ckpt"


def train_env_shaping_mode(potential_source: str, shaping_mode: str) -> str:
    """Return the reward_shaping_mode for the TRAINING ChunkResidualEnvWrapper.

    For A2 (potential_source="hiql_subgoal") the main loop owns ALL gc shaping; the
    wrapper must contribute zero stage-PBS shaping, so we force "none".
    All other paths (stage / hiql / unknown) pass shaping_mode through unchanged —
    bit-equivalent to previous behaviour.
    """
    return "none" if potential_source == "hiql_subgoal" else shaping_mode


def compute_online_subgoal(subgoal, obs, base_policy, cur_rel=None):
    """按 state_mode 选特征来源出 z。

    pi0_feat: 取 base_policy.last_prefix_feat()(冻结 pi0 prefix 特征)传 subgoal_online。
    act_feat / eef_piece: 走 subgoal_online(obs, cur_rel)(原有路径,不变)。
    """
    if subgoal.state_mode == "pi0_feat":
        return subgoal.subgoal_online(obs, prefix_feat=base_policy.last_prefix_feat())
    return subgoal.subgoal_online(obs, cur_rel)


def _offline_buffer_signature(args, image_keys, offline_cap, shaping_mode, potential=None,
                              act_feat_cache_sig=None, act_feat_cache_sha=None):
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
        # lerobot 路无 hdf5 源(offline_dataset_path 为 None)→ None-safe;hdf5 路逐位等价(始终非 None)
        "offline_dataset_path": (os.path.abspath(args.offline_dataset_path)
                                 if args.offline_dataset_path else None),
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
        sig["value_state_dim"] = (int(potential.model.state_dim)
                                  if potential is not None else None)
        if potential is not None and getattr(potential, "feature_mean", None) is not None:
            sig["value_state_stats_sha"] = _array_sha256(
                np.concatenate([
                    potential.feature_mean.detach().cpu().numpy(),
                    potential.feature_std.detach().cpu().numpy(),
                ])
            )
        if potential is not None and getattr(potential, "state_mode", None) == "act_feat":
            sig["act_feat_cache"] = (os.path.abspath(args.act_feat_cache)
                                     if args.act_feat_cache else None)
            sig["act_feat_cache_signature"] = act_feat_cache_sig
            sig["act_feat_cache_weight_sha"] = act_feat_cache_sha
            sig["hiql_value_act_feat_signature"] = potential.act_feat_signature
            sig["hiql_value_act_weight_sha"] = potential.act_weight_sha
    if args.potential_source == "hiql_subgoal":
        sig["potential_source"] = "hiql_subgoal"
        sig["gc_value_ckpt"] = os.path.abspath(args.gc_value_ckpt) if args.gc_value_ckpt else None
        sig["gc_potential_scale"] = round(float(args.gc_potential_scale), 8)
        sig["subgoal_way_steps"] = int(args.subgoal_way_steps)
    if args.subgoal_conditioned:
        sig["subgoal"] = True
        sig["gc_value_ckpt"] = os.path.abspath(args.gc_value_ckpt)
        sig["high_actor_ckpt"] = os.path.abspath(args.high_actor_ckpt)
        sig["subgoal_way_steps"] = int(args.subgoal_way_steps)
    # 注:--online_finetune_* flag 不纳入本签名 —— 它们只管 V/high_actor 的在线优化,
    # 不改变 offline_rb 的内容(残差 demo 锚),故 offline buffer 缓存无需因之失效。
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
    # data_source 键仅在 lerobot 路加入:同一 --dataset 下 hdf5/lerobot 缓存 key 才不串台;
    # 只在 lerobot 时加 → hdf5 路签名逐位不变(旧 hdf5 缓存仍命中)。
    if getattr(args, "data_source", "hdf5") == "lerobot":
        sig["data_source"] = "lerobot"
        sig["lerobot_root"] = (os.path.abspath(args.lerobot_root)
                               if args.lerobot_root else None)
    return sig


def _array_sha256(x) -> str:
    arr = np.asarray(x, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _env_state_mode_for_training(potential_state_mode, subgoal_state_mode):
    if subgoal_state_mode == "eef_piece":
        return "eef_piece"
    if potential_state_mode == "eef_piece":
        return "eef_piece"
    return "eef"


def _reject_libero_actfeat_potential_offline(potential):
    if potential is not None and getattr(potential, "state_mode", "eef") == "act_feat":
        raise NotImplementedError(
            "LIBERO offline buffer 尚不支持 act_feat HIQL potential reward shaping; "
            "请先设 --offline_fraction 0,或为 libero_offline.py 接入 potential + act_feat cache")


def _prepare_libero_offline(args, image_keys, image_size, potential):
    """LIBERO offline buffer 入口的轻量准备逻辑,便于测试 fail-fast 行为。"""
    _reject_libero_actfeat_potential_offline(potential)
    lerobot_root = os.path.abspath(os.path.join(os.path.dirname(args.libero_stats_json), ".."))
    from resfit.rl_finetuning.chunk_residual.libero_offline import count_libero_offline_transitions
    offline_cap = count_libero_offline_transitions(
        lerobot_root, args.libero_suite, args.libero_task_id, num_demos=args.offline_num_demos)
    sig = _libero_offline_signature(args, image_keys, offline_cap, image_size)
    return lerobot_root, offline_cap, sig


def _libero_offline_signature(args, image_keys, offline_cap, image_size):
    """libero offline buffer 缓存签名(换数据源/base/缩放/图尺寸/subgoal即失效重建)。"""
    sig = {
        "env_family": "libero",
        # 与 build 块的 lerobot_root 推导逐字一致(stats.json 在 <root>/meta/),否则缓存 key 与实际源路径分叉
        "lerobot_root": os.path.abspath(os.path.join(os.path.dirname(args.libero_stats_json), "..")),
        "suite": args.libero_suite, "task_id": int(args.libero_task_id),
        "base_mode": args.offline_base_mode, "base_policy_type": args.base_policy_type,
        "pi0_host": args.pi0_host, "pi0_port": args.pi0_port, "pi0_action_dim": args.pi0_action_dim,
        # execute_horizon 决定 base_policy 在 demo 帧上的重规划频率 → 改变每帧 base_action,必须入签名
        "pi0_execute_horizon": args.pi0_execute_horizon,
        "action_scale": args.action_scale, "min_range_per_dim": args.min_range_per_dim,
        "offline_cap": offline_cap, "image_keys": sorted(image_keys), "image_size": int(image_size),
        "gamma": args.gamma, "n_step": args.n_step, "num_demos": args.offline_num_demos,
    }
    if getattr(args, "subgoal_conditioned", False):
        # subgoal 离线 z 依赖 gc_value/high_actor ckpt 与 way_steps → 改其一缓存须失效重建
        sig["subgoal"] = True
        sig["gc_value_ckpt"] = os.path.abspath(args.gc_value_ckpt)
        sig["high_actor_ckpt"] = os.path.abspath(args.high_actor_ckpt)
        sig["subgoal_way_steps"] = int(args.subgoal_way_steps)
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


def _libero_task_prompt(suite, task_id):
    """取 LIBERO (suite,task_id) 的 task.language 当 pi0 prompt(不建 env,只问 benchmark)。"""
    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()[suite]()
    return task_suite.get_task(int(task_id)).language


def validate_libero_cfg(args):
    """LIBERO 路守卫:强制最小可行配置,任何禁用组合 → 清晰 ValueError。

    dexmg 路(env_family != "libero")直接 return,绝不动既有行为。
    """
    if getattr(args, "env_family", "dexmg") != "libero":
        return
    if not getattr(args, "libero_stats_json", None):
        raise ValueError(
            "--env_family libero 需 --libero_stats_json(LeRobot meta/stats.json 路径)")
    if getattr(args, "base_policy_type", "act") != "pi05":
        raise ValueError(
            "--env_family libero 需 --base_policy_type pi05;当前 "
            f"base_policy_type={getattr(args, 'base_policy_type', None)!r}")
    if getattr(args, "base_action_mode", None) != "queue":
        raise ValueError(
            "--env_family libero 需 --base_action_mode queue(pi05 是 step 级 base);当前 "
            f"base_action_mode={getattr(args, 'base_action_mode', None)!r}")
    if getattr(args, "chunk_length", 1) != 1:
        raise ValueError(
            "--env_family libero 需 --chunk_length 1(queue 模式);当前 "
            f"chunk_length={getattr(args, 'chunk_length', None)!r}")
    shaping_mode = resolve_shaping_mode(
        getattr(args, "reward_shaping", None), getattr(args, "staged_reward", False))
    if shaping_mode != "none":
        raise ValueError(
            "--env_family libero 不支持奖励整形,需 --reward_shaping none(且不传 --staged_reward);"
            f"当前 canonical shaping_mode={shaping_mode!r}")
    if getattr(args, "offline_fraction", 0) > 0:
        if getattr(args, "actor", "raw") != "raw":
            raise ValueError(
                "--env_family libero + offline_fraction>0 需 --actor raw(BC 锚走 raw actor);"
                f"当前 actor={getattr(args, 'actor', None)!r}")
        # 注:subgoal/stage 仍由本函数后面的 stage_conditioned/subgoal_conditioned 检查拦截;
        # offline demo 源是 LeRobot 数据集(自动按任务匹配),无需 --offline_dataset_path。
    if getattr(args, "pi0_action_dim", None) != 7:
        raise ValueError(
            "--env_family libero(单臂)需 --pi0_action_dim 7;当前 "
            f"pi0_action_dim={getattr(args, 'pi0_action_dim', None)!r}")
    if getattr(args, "stage_conditioned", False) or getattr(args, "stage_budget", None):
        raise ValueError(
            "--env_family libero 不支持 stage_conditioned / stage_budget;请关闭这些选项")
    if getattr(args, "subgoal_conditioned", False) and not getattr(args, "pi0_feat_cache", None):
        raise ValueError(
            "--env_family libero 的 subgoal_conditioned 目前仅支持 pi0_feat(需 --pi0_feat_cache);"
            "eef_piece/act_feat 子目标在 libero 下尚未实现")
    if getattr(args, "potential_source", None) == "hiql":
        raise ValueError(
            "--env_family libero 不支持 --potential_source hiql;请用默认 stage")


def build_libero_scalers(stats_json_path, device, action_scale=0.2, min_range_per_dim=0.1):
    """直读 LeRobot meta/stats.json(不调 lerobot)建 (ActionScaler, StateStandardizer)。

    LIBERO 路用此兜底替代 LeRobotDatasetMetadata(lerobot 在 py3.8 目标 env 装不上)。
    ActionScaler 用 action 节点真实 min/max(不再用 mean±std 近似);action_scale /
    min_range_per_dim 透传(与 dexmg 路同款),StateStandardizer 直接吃 mean/std。
    """
    stats = load_libero_norm_stats(stats_json_path)
    action_scaler = ActionScaler.from_dataset_stats(
        {"min": np.asarray(stats["action_min"], np.float32).tolist(),
         "max": np.asarray(stats["action_max"], np.float32).tolist()},
        action_scale=action_scale, min_range_per_dim=min_range_per_dim, device=device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        {"mean": np.asarray(stats["state_mean"], np.float32).tolist(),
         "std": np.asarray(stats["state_std"], np.float32).tolist()},
        device=device)
    return action_scaler, state_standardizer


def _needs_act_feat_cache(potential, subgoal_state_mode) -> bool:
    return ((potential is not None and getattr(potential, "state_mode", "eef") == "act_feat")
            or subgoal_state_mode == "act_feat")


def _validate_actfeat_potential_cache(args, potential, cache_sig, cache_sha, act_feat_seqs,
                                      cache_stats=None) -> None:
    if potential is None or getattr(potential, "state_mode", "act_feat") != "act_feat":
        return
    assert args.act_feat_cache, "act_feat potential 需 --act_feat_cache(离线 act_feat 缓存序列)"
    assert act_feat_seqs is not None and len(act_feat_seqs) > 0, \
        f"act_feat potential 缓存为空: {args.act_feat_cache}"
    first = torch.as_tensor(act_feat_seqs[0], dtype=torch.float32)
    assert first.ndim == 2 and first.shape[0] > 0, \
        "act_feat potential 缓存首条序列须为 [T,D] 且 T>0"
    assert first.shape[1] == potential.model.state_dim, \
        f"act_feat potential 缓存维度 {first.shape[1]} != value state_dim {potential.model.state_dim}"
    pot_sig = getattr(potential, "act_feat_signature", None) or {}
    cache_sig = cache_sig or {}
    if pot_sig:
        for k in ("act_ckpt_id", "image_keys", "proprio_key", "pooling"):
            assert cache_sig.get(k) == pot_sig.get(k), \
                f"act_feat potential cache 与 value 签名不符 [{k}]: {cache_sig.get(k)} vs {pot_sig.get(k)}"
    pot_sha = getattr(potential, "act_weight_sha", None)
    if pot_sha and cache_sha:
        assert cache_sha == pot_sha, \
            f"act_feat potential cache 与 value 权重指纹不符: {cache_sha} vs {pot_sha}"
    if cache_stats is not None and getattr(potential, "feature_mean", None) is not None \
            and getattr(potential, "feature_std", None) is not None:
        cache_mean = np.asarray(cache_stats[0], dtype=np.float32)
        cache_std = np.asarray(cache_stats[1], dtype=np.float32)
        pot_mean = torch.as_tensor(potential.feature_mean).detach().cpu().numpy().astype(np.float32)
        pot_std = torch.as_tensor(potential.feature_std).detach().cpu().numpy().astype(np.float32)
        assert cache_mean.shape == pot_mean.shape and np.allclose(cache_mean, pot_mean), \
            f"act_feat potential cache mean 与 value 不符: {cache_mean.shape} vs {pot_mean.shape}"
        assert cache_std.shape == pot_std.shape and np.allclose(cache_std, pot_std), \
            f"act_feat potential cache std 与 value 不符: {cache_std.shape} vs {pot_std.shape}"


def _load_act_feat_cache_for_training(args, potential, subgoal_state_mode, load_fn=None):
    if not _needs_act_feat_cache(potential, subgoal_state_mode):
        return None, None, None, None
    if load_fn is None:
        from resfit.rl_finetuning.chunk_residual.act_feat_cache import load_act_feat_cache as load_fn
    pot_act_feat = potential is not None and getattr(potential, "state_mode", "eef") == "act_feat"
    if pot_act_feat:
        assert args.act_feat_cache, "act_feat potential 需 --act_feat_cache(离线 act_feat 缓存序列)"
    else:
        assert args.act_feat_cache, "act_feat 子目标需 --act_feat_cache(算 goal530 + 同源)"
    seqs, stats, cache_sig, cache_sha = load_fn(args.act_feat_cache)
    if pot_act_feat:
        _validate_actfeat_potential_cache(
            args, potential, cache_sig, cache_sha, seqs, cache_stats=stats)
    return seqs, stats, cache_sig, cache_sha


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", choices=["raw", "flow"], default="raw")
    p.add_argument("--task", default="TwoArmBoxCleanup")
    # --- env family 开关(dexmg 默认行为不变;libero=LIBERO 单臂 + pi05 base via openpi serve)---
    p.add_argument("--env_family", choices=["dexmg", "libero"], default="dexmg",
                   help="环境族:dexmg(默认,行为不变)|libero(LIBERO 单臂,pi05 base,直读 stats.json)")
    p.add_argument("--libero_suite", default="libero_spatial",
                   help="LIBERO benchmark suite(--env_family libero 用)")
    p.add_argument("--libero_task_id", type=int, default=0,
                   help="LIBERO suite 内的 task id(--env_family libero 用)")
    p.add_argument("--libero_stats_json", default=None,
                   help="LIBERO LeRobot meta/stats.json 路径(直读建 scaler,不调 lerobot;libero 必填)")
    # --- act_feat 数据源开关(dexmg 路:hdf5 默认行为不变;lerobot=从 LeRobot 数据集读 offline 锚,no-stage)---
    p.add_argument("--data_source", choices=["hdf5", "lerobot"], default="hdf5",
                   help="act_feat 数据源:hdf5(默认)|lerobot(no-stage)")
    p.add_argument("--lerobot_root", default=None, help="--data_source lerobot 本地数据根目录")
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
    p.add_argument("--stage_balanced", dest="stage_balanced", action="store_true", default=True,
                   help="按 stage 配额采样(stage-balanced replay;默认开)")
    p.add_argument("--no_stage_balanced", dest="stage_balanced", action="store_false",
                   help="关闭 stage-balanced 采样(回退旧行为)")
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
    p.add_argument("--act_feat_cache", default=None,
                   help="act_feat 子目标:530 序列缓存(算 goal + 同源签名校验);仅 act_feat gc_value 用")
    p.add_argument("--allow_act_base_mismatch", action="store_true",
                   help="act_feat：在线 base 权重指纹 != 离线时，把硬失败降级为 warning 放行")
    p.add_argument("--pi0_feat_cache", default=None,
                   help="pi0_feat 子目标:pi0 prefix 特征序列缓存(算 goal + 同源签名校验);仅 pi0_feat gc_value 用")
    p.add_argument("--stage_budget", default=None,
                   help="逐阶段残差幅度乘子,逗号分隔,长度=num_stages(piece=4 如 '1,1,1,0.3';threading=3 如 '1,1,0.3');不传=关(§18.3)")
    p.add_argument("--offline_base_mode", choices=["gt", "base_policy"], default="base_policy",
                   help="离线 buffer 的 base_action 来源:base_policy(默认,冻结 base 现算 base_action,"
                        "action 仍存 GT;bc_target=GT-base,锚向专家 + critic offline 锚对齐在线流形,"
                        "需 queue 模式)| gt(逐位等价=GT-as-base,残差目标0)")
    p.add_argument("--demo_bc_coef", type=float, default=0.0,
                   help="残差 actor 的 demo-BC 权重(模块②a);0=关(逐位等价 baseline)。"
                        ">0 需 offline_fraction>0(bc_batch 取自 offline_rb)、--actor raw")
    p.add_argument("--bc_coef_final", type=float, default=None,
                   help="demo-BC 系数线性衰减的终值(floor);不传=固定 demo_bc_coef(逐位等价)。"
                        "传值 v 则 bc_loss_coef 从 demo_bc_coef 线性降到 v(区间 0→total_env_steps)。"
                        "需 demo_bc_coef>0 且 0<=v<=demo_bc_coef")
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
    p.add_argument("--potential_source", choices=["stage", "hiql", "hiql_subgoal"], default="stage",
                   help="PBS 势函数 Φ 来源(③b):stage=整数 stage(默认,逐位等价);hiql=V(state)*scale;"
                        "hiql_subgoal=V(s,z)*scale(A2 goal-cond)")
    p.add_argument("--hiql_value_ckpt", default=None,
                   help="--potential_source hiql 时 ③a 产出的 value.pt 路径")
    p.add_argument("--gc_potential_scale", type=float, default=1.0,
                   help="A2(hiql_subgoal)的 phi_scale,乘到 auto_scale 上(默认 1.0)")
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
    p.add_argument("--pi0_execute_horizon", type=int, default=10,
                   help="pi0/pi05 每次推理后开环执行多少步再重规划(adapter 内部 action queue 长度);"
                        "默认 10 对齐 three_piece_aligned 实验的 ACT base_n_action_steps=10")
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
    # --- HIQL 在线联合微调(spec 2026-06-24;默认全关 = 逐位等价)---
    p.add_argument("--online_finetune_value", action="store_true",
                   help="Phase3 在线微调 V(s,g)(冻 φ,只更新 critic head + EMA target);默认关")
    p.add_argument("--online_finetune_high_actor", action="store_true",
                   help="Phase3 在线微调 high_actor(AWR,用当前 V);默认关")
    p.add_argument("--online_value_lr", type=float, default=1e-5,
                   help="在线 V 微调 LR(小;默认 1e-5)")
    p.add_argument("--online_high_actor_lr", type=float, default=1e-5,
                   help="在线 high_actor 微调 LR(小;默认 1e-5)")
    p.add_argument("--online_finetune_offline_fraction", type=float, default=0.5,
                   help="V/high_actor 在线更新 batch 里 offline demo 占比(RLPD 锚;默认 0.5)")
    p.add_argument("--online_finetune_every", type=int, default=1,
                   help="每多少个 update tick 跑一次在线微调(默认 1)")
    p.add_argument("--online_finetune_expectile", type=float, default=0.7,
                   help="在线 V 微调 expectile τ(默认 0.7)")
    p.add_argument("--online_finetune_ema", type=float, default=0.005,
                   help="在线 V target 的 EMA 系数(默认 0.005)")
    p.add_argument("--online_finetune_future_mode", choices=["stage_entry", "geometric"],
                   default="geometric",
                   help="在线 goal 采样 future_mode(默认 geometric,只需 last_idx 无需 stage 检测)")
    p.add_argument("--online_finetune_max_trans", type=int, default=50_000,
                   help="在线 store 容量(transition 数,FIFO)")
    p.add_argument("--online_finetune_min_trans", type=int, default=2_000,
                   help="在线 store 达到此 transition 数才开始微调")
    p.add_argument("--oac_explore", action="store_true",
                   help="开启 OAC 乐观探索(只改探索采样,默认关、关时零回归)")
    p.add_argument("--oac_beta_ub", type=float, default=4.0,
                   help="OAC 乐观上界系数 β_UB(Q_UB = μ_Q + β_UB·σ_Q)")
    p.add_argument("--oac_delta", type=float, default=0.5,
                   help="OAC 均值偏移的 KL 预算 δ(偏移≈√(2δ)·std)")
    return p


def _resolve_data_source_cfg(args):
    """train_chunk_residual 无 --state_mode flag(state_mode 由 gc_value ckpt 隐含);lerobot 数据源
    在本脚本里只经 act_feat 子目标用,故显式标注 state_mode=act_feat + 强制 subgoal,再走共享数据源守卫。
    hdf5 默认路直接放行(不注入 state_mode)。提成函数以便单测(端到端守卫此前漏测,smoke 才暴露)。"""
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import validate_data_source_cfg
    if getattr(args, "data_source", "hdf5") == "lerobot":
        assert args.subgoal_conditioned, \
            "--data_source lerobot 在 train_chunk_residual 仅经 act_feat 子目标用,需 --subgoal_conditioned"
        args.state_mode = "act_feat"
    validate_data_source_cfg(args)


def main():
    args = build_parser().parse_args()
    validate_libero_cfg(args)   # LIBERO 路守卫(dexmg 路 no-op)
    _resolve_data_source_cfg(args)   # --data_source lerobot 守卫(默认 hdf5 直接放行)
    torch.manual_seed(args.seed)
    sample_gen = torch.Generator().manual_seed(args.seed)   # stage-balanced 采样用

    # --- 归一化器(从 dataset stats 建,与 AE / RL 同款)---
    if args.env_family == "libero":
        # LIBERO 路:直读 stats.json(不调 lerobot,目标 env 装不上 LeRobotDatasetMetadata)。
        # libero_stats_json 必填校验已挪进 validate_libero_cfg 守卫。
        action_scaler, state_standardizer = build_libero_scalers(
            args.libero_stats_json, args.device,
            action_scale=args.action_scale, min_range_per_dim=args.min_range_per_dim)
    else:
        # 只需 dataset 的统计量来建归一化器:用 LeRobotDatasetMetadata(仅拉 meta/ 几个小文件)
        # 而非 LeRobotDataset(会 snapshot 整个 repo,含上百 MB 视频)。.stats 完全一致,且可离线工作,
        # 避免国内直连 HF 下视频频繁超时。
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        # lerobot 数据源:标准化器 stats 用本地 root,与 offline act_feat 缓存(read_per_demo_states
        # 同走 root)同源——否则 online proprio 标准化用 HF 默认缓存 stats、offline 用 root,
        # act_feat 在线/离线特征被不同 mean/std 归一 → subgoal 失配;且离线模式无 root 可能找不到该 repo。
        # hdf5 路 root=None,逐位等价原行为。
        meta = LeRobotDatasetMetadata(
            args.dataset,
            root=args.lerobot_root if getattr(args, "data_source", "hdf5") == "lerobot" else None)
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
    image_keys = list(base_policy.config.image_features.keys())
    shaping_mode = resolve_shaping_mode(args.reward_shaping, args.staged_reward)
    num_stages = NUM_STAGES.get(args.task, 1)   # 无检测器任务退化为 1 段

    # --- A2 前置校验(hiql_subgoal 路) ---
    validate_hiql_subgoal_args(args)

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
    # Peek gc_value state_mode BEFORE env construction so env_state_mode can depend on it.
    _subgoal_gc_info = None
    if args.subgoal_conditioned:
        from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value as _lgv
        assert args.gc_value_ckpt, "--subgoal_conditioned 需 --gc_value_ckpt"
        _subgoal_gc_info = _lgv(args.gc_value_ckpt, map_location="cpu")[1]
    _subgoal_sm = _subgoal_gc_info["state_mode"] if _subgoal_gc_info is not None else None
    _offline_act_feat_seqs, _act_feat_cache_stats, _act_feat_cache_sig, _act_feat_cache_sha = \
        _load_act_feat_cache_for_training(args, potential, _subgoal_sm)

    _potential_sm = getattr(potential, "state_mode", None) if potential is not None else None
    # act_feat potential uses images/proprio through PotentialActFeatureEncoder;
    # it does not require env state_mode="act_feat".
    env_state_mode = _env_state_mode_for_training(_potential_sm, _subgoal_sm)

    if args.env_family == "libero":
        from resfit.rl_finetuning.chunk_residual.libero_env import create_libero_vectorized_env
        vec_env = create_libero_vectorized_env(
            args.libero_suite, args.libero_task_id, 1, args.device)
    else:
        # lazy import:dexmg.py 顶层 import robosuite-1.5 专属 API,只在 dexmg 路触发。
        from resfit.dexmg.environments.dexmg import create_vectorized_env
        vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device,
                                        state_mode=env_state_mode)
    print(f"[reward-shaping] mode={shaping_mode} bonus={args.stage_reward_bonus} gamma={args.gamma}")
    print(f"[base-action] mode={args.base_action_mode}")
    print(f"[state-mode] env_state_mode={env_state_mode} "
          f"(observation.state 仍 18 维;rel_piece 经 info 只喂 Φ/z)")
    # eval 不加 shaping、不喂 Φ → 保持 eef(省每步 sim rel 开销);
    # FIX A: eef_piece 子目标时需 eef_piece 让 eval info 携带 rel_piece 供 z 注入;
    # act_feat 子目标用图像特征提取,不需要 rel_piece → eval_state_mode 保持 eef
    eval_state_mode = "eef_piece" if (args.subgoal_conditioned and _subgoal_sm == "eef_piece") else "eef"
    if args.env_family == "libero":
        eval_vec = create_libero_vectorized_env(
            args.libero_suite, args.libero_task_id, args.eval_num_envs, args.device)
    else:
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

    potential_feature_encoder = None
    if potential is not None and potential.state_mode == "act_feat":
        from resfit.rl_finetuning.chunk_residual.act_feature import (
            ActFeatureExtractor, PotentialActFeatureEncoder,
            act_weight_fingerprint, assert_act_base_samesource)

        assert _act_feat_cache_sig is not None, "act_feat potential cache signature missing"
        potential_image_keys = _act_feat_cache_sig.get("image_keys") or image_keys
        potential_proprio_key = _act_feat_cache_sig.get("proprio_key", "observation.state")
        potential_pooling = _act_feat_cache_sig.get("pooling", "mean")
        assert_act_base_samesource(
            gv_sha=potential.act_weight_sha,
            cache_sha=_act_feat_cache_sha,
            base_sha=act_weight_fingerprint(base_policy),
            allow_mismatch=args.allow_act_base_mismatch,
        )
        potential_extractor = ActFeatureExtractor(
            base_policy,
            potential_image_keys,
            proprio_key=potential_proprio_key,
            pooling=potential_pooling,
            proprio_dim=state_standardizer._mean.numel(),
        )
        potential_feature_encoder = PotentialActFeatureEncoder(
            potential_extractor,
            state_standardizer,
            potential,
            proprio_key=potential_proprio_key,
        )

    # --- agent(复用 QAgent,action_dim=480)---
    from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
    cfg = ResidualTD3BoxCleanConfig()
    cfg.agent.actor_lr = args.actor_lr
    cfg.agent.critic_lr = args.critic_lr
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.bc_loss_coef = args.demo_bc_coef
    cfg.agent.bc_loss_dynamic = 0          # 均匀 BC(②a 不开 DAPG 动态)
    cfg.agent.device = args.device         # 让 --device 传到 agent(encoders/nets);否则用 config 默认 cuda
    cfg.agent.oac_explore = args.oac_explore
    cfg.agent.oac_beta_ub = args.oac_beta_ub
    cfg.agent.oac_delta = args.oac_delta
    if args.demo_bc_coef > 0:
        assert args.actor == "raw", "demo_bc 第一版只支持 --actor raw"
        assert args.offline_fraction > 0, \
            "demo_bc_coef>0 需 offline_fraction>0(bc_batch 取自 offline_rb)"
    if args.bc_coef_final is not None:
        assert args.demo_bc_coef > 0, \
            "--bc_coef_final 需 --demo_bc_coef>0(BC 未开则衰减无意义)"
        assert 0.0 <= args.bc_coef_final <= args.demo_bc_coef, \
            f"--bc_coef_final 需在 [0, demo_bc_coef={args.demo_bc_coef}] 内,得到 {args.bc_coef_final}"

    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length,
                                  stage_reward_bonus=args.stage_reward_bonus,
                                  reward_shaping_mode=train_env_shaping_mode(
                                      args.potential_source, shaping_mode),
                                  gamma=args.gamma,
                                  base_action_mode=args.base_action_mode,
                                  potential=potential,
                                  potential_feature_encoder=potential_feature_encoder)

    # --- 维度 ---
    lowdim_keys = ["observation.state", "observation.base_action", "observation.stage_id"]
    if args.subgoal_conditioned:
        lowdim_keys.append("observation.subgoal")
    obs0, _ = env.reset()
    img_c, img_h, img_w = obs0[image_keys[0]].shape[1:]
    state_dim = obs0["observation.state"].shape[1]
    if potential is not None:
        if potential.state_mode == "act_feat":
            assert _offline_act_feat_seqs is not None, "act_feat potential requires loaded act_feat sequences"
            feat_dim = int(torch.as_tensor(_offline_act_feat_seqs[0]).shape[1])
            assert potential.model.state_dim == feat_dim, (
                f"Φ act_feat value 输入维度 {potential.model.state_dim} != act_feat cache dim {feat_dim}")
        else:
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
    _pi0_seqs = None                # pi0_feat:离线 buffer subgoal 复用的 2056 缓存序列
    _finetune_seqs = None           # 在线微调离线序列(三 state_mode 统一接口)
    if args.subgoal_conditioned:
        assert args.actor == "raw", "subgoal-conditioning 第一版只支持 --actor raw"
        assert args.gc_value_ckpt and args.high_actor_ckpt, "需 --gc_value_ckpt 与 --high_actor_ckpt"
        from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal, representative_goal
        _gc_info = _subgoal_gc_info
        _sm = _gc_info["state_mode"]
        if _sm == "act_feat":
            _seqs = _offline_act_feat_seqs
            _stats = _act_feat_cache_stats
            _cache_sig = _act_feat_cache_sig or {}
            _cache_sha = _act_feat_cache_sha
            assert _seqs is not None, "act_feat 子目标需 --act_feat_cache(算 goal530 + 同源)"
            _finetune_seqs = _seqs           # 在线微调同源离线序列(act_feat=530 维)
            assert _gc_info.get("act_feat_signature"), \
                "gc_value 缺 act_feat_signature(须用 --state_mode act_feat 重训该 gc_value)"
            _gv_sig = _gc_info["act_feat_signature"]
            for k in ("act_ckpt_id", "image_keys", "proprio_key", "pooling"):
                assert _cache_sig.get(k) == _gv_sig.get(k), \
                    f"act_feat cache 与 gc_value 签名不符 [{k}]: {_cache_sig.get(k)} vs {_gv_sig.get(k)}"
            from resfit.rl_finetuning.chunk_residual.act_feature import (
                act_weight_fingerprint, assert_act_base_samesource)
            assert_act_base_samesource(
                gv_sha=_gc_info.get("act_weight_sha"),
                cache_sha=_cache_sha,
                base_sha=act_weight_fingerprint(base_policy),
                allow_mismatch=args.allow_act_base_mismatch)
            goal = representative_goal(_seqs)
            assert goal.shape[0] == _gc_info["mean"].shape[0], "goal 维度须 == gc_value state_dim"
            subgoal = HiqlSubgoal.from_ckpts(args.gc_value_ckpt, args.high_actor_ckpt,
                                             goal=goal, device=args.device,
                                             renorm_subgoal=args.renorm_subgoal, base_policy=base_policy)
        elif _sm == "pi0_feat":
            assert args.pi0_feat_cache, "pi0_feat 子目标需 --pi0_feat_cache(算 goal + 同源)"
            from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache
            # _pi0_stats 未用:feat_mean/std 内嵌于 gc_value ckpt(from_ckpts 自取);此处仅用 seqs(算 goal)+ sig(同源校验)
            _pi0_seqs, _pi0_stats, _pi0_cache_sig = load_pi0_feat_cache(args.pi0_feat_cache)
            assert _gc_info.get("pi0_feat_signature"), \
                "gc_value 缺 pi0_feat_signature(须用 --state_mode pi0_feat 重训该 gc_value)"
            _pi0_gv_sig = _gc_info["pi0_feat_signature"]
            for k in ("serve_ckpt_id", "image_keys", "proprio_key", "pooling", "prompt"):
                if _pi0_gv_sig.get(k) is not None:
                    assert _pi0_cache_sig.get(k) == _pi0_gv_sig.get(k), \
                        f"pi0_feat cache 与 gc_value 签名不符 [{k}]: {_pi0_cache_sig.get(k)} vs {_pi0_gv_sig.get(k)}"
            goal = representative_goal(_pi0_seqs)
            assert goal.shape[0] == _gc_info["mean"].shape[0], "goal 维度须 == gc_value state_dim"
            _finetune_seqs = _pi0_seqs       # 在线微调同源离线序列(pi0_feat=2056 维)
            # pi0_feat 在线子目标:base_policy.last_prefix_feat() 在线提特征(无需离线 extractor)
            subgoal = HiqlSubgoal.from_ckpts(args.gc_value_ckpt, args.high_actor_ckpt,
                                             goal=goal, device=args.device,
                                             renorm_subgoal=args.renorm_subgoal, base_policy=None)
        else:  # eef_piece:现状不变
            assert args.subgoal_state30_cache or args.offline_dataset_path, \
                "eef_piece 子目标需 --subgoal_state30_cache 或 --offline_dataset_path"
            from resfit.rl_finetuning.chunk_residual.state30_cache import load_or_build_state30
            _seqs30 = load_or_build_state30(args.offline_dataset_path, args.dataset,
                                            args.offline_num_demos, args.subgoal_state30_cache)
            goal30 = representative_goal(_seqs30)
            assert goal30.shape[0] == 30, f"goal30 须 30 维,got {goal30.shape[0]}"
            _finetune_seqs = _seqs30         # 在线微调同源离线序列(eef_piece=30 维)
            subgoal = HiqlSubgoal.from_ckpts(args.gc_value_ckpt, args.high_actor_ckpt,
                                             goal=goal30, device=args.device,
                                             renorm_subgoal=args.renorm_subgoal)
        print(f"[hiql-subgoal] on; mode={_sm} rep_dim={subgoal.rep_dim} renorm={args.renorm_subgoal}")
    finetuner = maybe_build_finetuner(args, subgoal, _finetune_seqs, gc_info=_subgoal_gc_info) \
        if args.subgoal_conditioned else None
    if finetuner is not None:
        print(f"[online-finetune] on; value={args.online_finetune_value} "
              f"high_actor={args.online_finetune_high_actor} "
              f"offline_frac={args.online_finetune_offline_fraction} "
              f"every={args.online_finetune_every} future={args.online_finetune_future_mode}")

    # --- A2: GcSubgoalPotential(hiql_subgoal 时构建;其余 source 保持 None 逐位等价)---
    # 必须在 subgoal 已构造之后(复用同源 gc_value ckpt);主循环与离线 buffer 共用此对象。
    gc_potential = None
    if args.potential_source == "hiql_subgoal":
        from resfit.rl_finetuning.chunk_residual.hiql_potential import GcSubgoalPotential
        gc_potential = GcSubgoalPotential.from_ckpt(
            args.gc_value_ckpt, num_stages=num_stages,
            phi_scale=args.gc_potential_scale, device=args.device)
        potential = None        # A2 不走 wrapper 的 potential 路径;training env 已由
        # train_env_shaping_mode() 强制 reward_shaping_mode="none",wrapper 贡献零 shaping,
        # 主循环独占全部 gc shaping(online/offline 语义对齐)。
        print(f"[gc-potential] A2 on; ckpt={args.gc_value_ckpt} scale={gc_potential.scale:.4f}")

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
        from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import concat_mixed_batch
        is_libero = getattr(args, "env_family", "dexmg") == "libero"
        if is_libero:
            from resfit.rl_finetuning.chunk_residual.libero_offline import build_libero_offline_buffer
            lerobot_root, offline_cap, sig = _prepare_libero_offline(
                args, image_keys, img_h, potential)
        else:
            from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
                build_offline_buffer, count_offline_transitions)
            if args.data_source == "lerobot":
                # LeRobot 数据源(no-stage, act_feat):不读 hdf5 states 定容,改按集 sum(T-1)。
                # 取集序须与 build_offline_buffer 的 lerobot 路一致(同一 LeRobotDataset)。
                from resfit.rl_finetuning.chunk_residual.lerobot_demo_source import (
                    open_lerobot, count_lerobot_transitions)
                offline_cap = count_lerobot_transitions(
                    open_lerobot(args.dataset, args.lerobot_root), args.offline_num_demos)
            else:
                assert args.offline_dataset_path is not None, \
                    "offline_fraction>0 需 --offline_dataset_path 指向源 dexmimicgen HDF5"
                # 按 demo transition 数精确定容,否则 LazyTensorStorage 不足会挤掉早期 demo(锚不全)
                offline_cap = count_offline_transitions(args.offline_dataset_path,
                                                        num_demos=args.offline_num_demos)
            sig = _offline_buffer_signature(
                args, image_keys, offline_cap, shaping_mode,
                potential=potential,
                act_feat_cache_sig=_act_feat_cache_sig,
                act_feat_cache_sha=_act_feat_cache_sha)

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
            if is_libero:
                build_libero_offline_buffer(
                    offline_rb, lerobot_root=lerobot_root,
                    suite=args.libero_suite, task_id=args.libero_task_id,
                    action_scaler=action_scaler, state_standardizer=state_standardizer,
                    base_policy=base_policy, base_mode=args.offline_base_mode,
                    base_device=args.device, image_size=img_h, num_demos=args.offline_num_demos,
                    subgoal=subgoal, way_steps=args.subgoal_way_steps, feat_seqs=_pi0_seqs)
            else:
                build_offline_buffer(
                    offline_rb, args.offline_dataset_path,
                    action_scaler=action_scaler, state_standardizer=state_standardizer,
                    image_keys=image_keys, bonus=args.stage_reward_bonus,
                    mode=shaping_mode, gamma=args.gamma, num_demos=args.offline_num_demos,
                    stage_cache=args.offline_stage_cache,
                    potential=(gc_potential if args.potential_source == "hiql_subgoal" else potential),
                    subgoal=subgoal, way_steps=args.subgoal_way_steps,
                    act_feat_seqs=_offline_act_feat_seqs,
                    data_source=args.data_source, lerobot_repo_id=args.dataset,
                    lerobot_root=args.lerobot_root,
                    base_policy=base_policy, base_mode=args.offline_base_mode,
                    base_device=args.device, env_hint=args.task,)
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
    last_ft_metrics = None
    total = 2 * args.chunk_length if args.smoke else args.total_env_steps
    while env_steps <= total:
        next_rel = None   # FIX D: silence unbound-var lint; overwritten below when subgoal_conditioned
        if args.subgoal_conditioned:
            obs["observation.subgoal"] = compute_online_subgoal(subgoal, obs, base_policy, cur_rel).to(
                obs["observation.state"].device)
            if finetuner is not None:
                _pf = base_policy.last_prefix_feat() if subgoal.state_mode == "pi0_feat" else None
                finetuner.on_step(subgoal.encode_state(obs, rel_raw=cur_rel, prefix_feat=_pf))
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, eval_mode=False, stddev=args.stddev, cpu=False)  # [1,480] 残差
        next_obs, reward, terminated, truncated, info = env.step(action)
        if args.subgoal_conditioned:
            next_rel = info.get("rel_piece")
            next_obs["observation.subgoal"] = compute_online_subgoal(subgoal, next_obs, base_policy, next_rel).to(
                next_obs["observation.state"].device)
        done = terminated | truncated
        if gc_potential is not None:
            _pf = base_policy.last_prefix_feat() if subgoal.state_mode == "pi0_feat" else None
            _s_start = subgoal.encode_state(obs, rel_raw=cur_rel, prefix_feat=_pf)
            _s_end = subgoal.encode_state(next_obs, rel_raw=next_rel, prefix_feat=_pf)
            _z_start = obs["observation.subgoal"]
            from resfit.rl_finetuning.chunk_residual.hiql_potential import gc_subgoal_shaping
            _shape = gc_subgoal_shaping(gc_potential, _s_start, _s_end, _z_start,
                                        bonus=args.stage_reward_bonus, gamma=args.gamma,
                                        done=bool(done.any()))
            reward = reward + _shape
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
        if finetuner is not None and bool(done.any()):
            finetuner.on_episode_end()
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
                    if args.bc_coef_final is not None:
                        agent.cfg.bc_loss_coef = linear_bc_coef(
                            env_steps, c0=args.demo_bc_coef,
                            c_final=args.bc_coef_final, total_steps=args.total_env_steps)
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

            if finetuner is not None:
                ft_metrics = finetuner.maybe_update()
                if ft_metrics is not None:
                    last_ft_metrics = ft_metrics

            if env_steps >= next_log:
                lrs = {"actor": agent.actor_opt.param_groups[0]["lr"],
                       "critic": agent.critic_opt.param_groups[0]["lr"],
                       "encoder": agent.encoder_opt.param_groups[0]["lr"]}
                buf_sizes = {"online": len(online_rb),
                             "offline": len(offline_rb) if offline_rb else 0,
                             "relabel": len(relabel_rb) if relabel_rb else 0}
                log_dict = build_train_log_dict(m_upd, lrs, buf_sizes)
                log_dict["rft/bc_coef_cur"] = agent.cfg.bc_loss_coef
                if last_ft_metrics is not None:
                    log_dict.update(last_ft_metrics)
                wandb.log(log_dict, step=env_steps)
                next_log += args.log_freq

        if env_steps >= next_eval:
            # libero 与 dexmg 走不同 evaluator:run_dexmg_evaluation 顶层拉 robosuite-1.5 且带
            # dexmg 专属 Q图/视频/subgoal;libero 用 env-无关的 run_libero_evaluation(指标口径一致)。
            # 两路都 lazy import:dexmg 路避免 libero 环境触发 robosuite-1.5。
            if args.env_family == "libero":
                from resfit.rl_finetuning.chunk_residual.libero_eval import run_libero_evaluation
                m = run_libero_evaluation(env=eval_env, agent=agent,
                                          num_episodes=args.eval_num_episodes, device=args.device,
                                          subgoal=(subgoal if args.subgoal_conditioned else None),
                                          base_policy=eval_base_policy)
            else:
                from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
                with torch.no_grad():
                    m = run_dexmg_evaluation(env=eval_env, agent=agent,
                                             num_episodes=args.eval_num_episodes, device=args.device,
                                             global_step=env_steps, save_video=False,
                                             save_q_plots=False, run_name=f"chunk_{args.actor}",
                                             output_dir=args.output_dir,
                                             subgoal=(subgoal if args.subgoal_conditioned else None),
                                             base_policy=base_policy)
            sr = m["eval/success_rate"]
            if sr > best_sr:
                best_sr = sr
                os.makedirs(args.output_dir, exist_ok=True)
                save_checkpoint(agent, os.path.join(args.output_dir, "best.pt"),
                                global_step=env_steps,
                                config=args, success_rate=sr)
            print(f"[env_steps {env_steps}] eval success_rate={sr:.3f} (best {best_sr:.3f})")
            if args.env_family == "libero":
                # libero 无 stage 概念:只 log eval/*(stage purity/diag 不适用)。
                wandb.log({k: v for k, v in m.items() if k.startswith("eval/")}, step=env_steps)
            else:
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
