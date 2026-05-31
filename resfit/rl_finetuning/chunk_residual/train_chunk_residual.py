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
import os

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper


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
    td = TensorDict({
        "obs": TensorDict(curr, batch_size=[]),
        "next": TensorDict({"obs": TensorDict(nxt, batch_size=[]),
                            "done": done[0].cpu(), "reward": reward[0].cpu()}, batch_size=[]),
        "action": combined_action[0].detach().cpu(),
        "_priority": torch.tensor(10.0, dtype=torch.float32),
    }, batch_size=[]).unsqueeze(0)
    online_rb.add(td)


def _maybe_inject_flow_actor(args, agent, repr_dim, patch_repr_dim, prop_dim, action_dim):
    """M1: no-op。M2.2 在此把 agent.actor 替换为 residual_flow actor 并重建 actor_opt。"""
    return


def build_base_policy(wandb_id: str, device: str, wt_type: str = "best", wt_version: str = "latest"):
    """复用 ResFiT 的加载路径,从 wandb artifact 拉冻结 ACT 基座(eval 模式)。"""
    policy_dir, _ = download_policy_from_wandb(wandb_id, step=wt_type, artifact_version=wt_version)
    base_policy = load_policy(policy_dir)
    base_policy.to(device)
    base_policy.eval()
    return base_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", choices=["raw", "flow"], default="raw")
    p.add_argument("--task", default="TwoArmBoxCleanup")
    p.add_argument("--base_wandb_id", default="dexmg-boxcleanup-bc/d59wny58")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-box-cleanup")
    p.add_argument("--chunk_length", type=int, default=20)
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
    p.add_argument("--actor_lr", type=float, default=5e-6)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--stddev", type=float, default=0.05)
    p.add_argument("--ae_ckpt", default=None)   # M2 flow 用
    p.add_argument("--eval_num_envs", type=int, default=8)
    p.add_argument("--eval_num_episodes", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="少量步数冒烟")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    # --- 归一化器(从 dataset stats 建,与 AE / RL 同款)---
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    meta = LeRobotDataset(args.dataset).meta
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device=args.device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=args.device)

    # --- 基座 + env ---
    base_policy = build_base_policy(args.base_wandb_id, args.device)
    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device)
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length)
    eval_vec = create_vectorized_env(env_name=args.task, num_envs=args.eval_num_envs,
                                     device=args.device)
    eval_env = ChunkResidualEnvWrapper(eval_vec, base_policy, action_scaler, state_standardizer,
                                       chunk_length=args.chunk_length)

    # --- 维度 ---
    image_keys = list(base_policy.config.image_features.keys())
    lowdim_keys = ["observation.state", "observation.base_action"]
    obs0, _ = env.reset()
    img_c, img_h, img_w = obs0[image_keys[0]].shape[1:]
    state_dim = obs0["observation.state"].shape[1]
    action_dim = env.action_dim * args.chunk_length     # = 480

    # --- agent(复用 QAgent,action_dim=480)---
    from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
    cfg = ResidualTD3BoxCleanConfig()
    cfg.agent.actor_lr = args.actor_lr
    cfg.agent.critic_lr = args.critic_lr
    cfg.agent.actor.action_scale = args.action_scale
    agent = QAgent(obs_shape=(img_c, img_h, img_w), prop_shape=(state_dim,),
                   action_dim=action_dim, rl_cameras=image_keys,
                   cfg=cfg.agent, residual_actor=True)

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

    # --- 训练循环(x 轴=环境步;每 chunk 计入 chunk_length 步)---
    obs, _ = env.reset()
    env_steps = 0
    next_eval = 0
    best_sr = 0.0
    total = 2 * args.chunk_length if args.smoke else args.total_env_steps
    while env_steps <= total:
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, eval_mode=False, stddev=args.stddev, cpu=False)  # [1,480] 残差
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        add_chunk_transition(obs=obs, next_obs=next_obs, combined_action=info["scaled_action"],
                             reward=reward, done=done, info=info, image_keys=image_keys,
                             lowdim_keys=lowdim_keys, online_rb=online_rb)
        obs = next_obs
        env_steps += args.chunk_length

        if env_steps >= args.learning_starts and len(online_rb) > args.batch_size:
            for i in range(args.utd):
                batch = online_rb.sample()
                update_actor = ((i + 1) % args.utd == 0)
                agent.update(batch, args.stddev, update_actor, bc_batch=None, ref_agent=agent)

        if env_steps >= next_eval:
            with torch.no_grad():
                m = run_dexmg_evaluation(env=eval_env, agent=agent,
                                         num_episodes=args.eval_num_episodes, device=args.device,
                                         global_step=env_steps, save_video=False,
                                         save_q_plots=False, run_name=f"chunk_{args.actor}",
                                         output_dir="outputs_chunk")
            sr = m["eval/success_rate"]
            best_sr = max(best_sr, sr)
            print(f"[env_steps {env_steps}] eval success_rate={sr:.3f} (best {best_sr:.3f})")
            next_eval += args.eval_every_env_steps
        if args.smoke:
            break

    print(f"done. best success_rate={best_sr:.3f}")


if __name__ == "__main__":
    main()
