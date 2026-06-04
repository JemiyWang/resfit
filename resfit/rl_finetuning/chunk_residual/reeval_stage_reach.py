"""重评某 run 的 best.pt:num_envs=1 逐 episode 记最高 stage,产出 <log>_reach.json
(供 plot_stage_diag.py 的 (1,1) 柱状图)。

为何 num_envs=1:ChunkResidualEnvWrapper 的 stage 闩锁是单标量、只读 env0
(chunk_env_wrapper.py:150/170),多环境会互相污染 → 必须单环境才有正确 per-episode 归属。

用法(仓库根目录, conda env residual):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
  python -m resfit.rl_finetuning.chunk_residual.reeval_stage_reach \
    --run_dir outputs_chunk/cl1_queue_potential \
    --log outputs_chunk/cl1_queue_potential.log
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_base_policy
from resfit.rl_finetuning.chunk_residual.stage_reach import stage_reach_rates
from resfit.rl_finetuning.chunk_residual.stage_log_parse import fold_episode_max_stage
from resfit.rl_finetuning.chunk_residual.stage_detectors import NUM_STAGES
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="含 best.pt 的目录")
    p.add_argument("--log", required=True, help="对应 log;sidecar 写到 <log>_reach.json")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=20000, help="安全上限,防卡死")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = torch.load(os.path.join(args.run_dir, "best.pt"),
                      map_location=args.device, weights_only=False)
    cfg_args = ckpt["config"]                       # 训练时存的 argparse Namespace
    step = int(ckpt.get("global_step", -1))
    task = cfg_args.task
    n_stages = NUM_STAGES[task]
    top_stage = n_stages - 1

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    meta = LeRobotDataset(cfg_args.dataset).meta
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=cfg_args.action_scale,
        min_range_per_dim=cfg_args.min_range_per_dim, device=args.device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=args.device)

    base_policy = build_base_policy(cfg_args.base_wandb_id, args.device)
    vec_env = create_vectorized_env(env_name=task, num_envs=1, device=args.device)
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=cfg_args.chunk_length,
                                  reward_shaping_mode="none",
                                  base_action_mode=cfg_args.base_action_mode)

    image_keys = list(base_policy.config.image_features.keys())
    obs, _ = env.reset()
    img_c, img_h, img_w = obs[image_keys[0]].shape[1:]
    state_dim = obs["observation.state"].shape[1]
    action_dim = env.action_dim * cfg_args.chunk_length

    cfg = ResidualTD3BoxCleanConfig()
    cfg.agent.actor.action_scale = cfg_args.action_scale
    agent = QAgent(obs_shape=(img_c, img_h, img_w), prop_shape=(state_dim,),
                   action_dim=action_dim, rl_cameras=image_keys,
                   cfg=cfg.agent, residual_actor=True)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.train(False)

    ep_max_stages: list[int] = []
    ep_max = 0
    for _ in range(args.max_steps):
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
        obs, reward, term, trunc, info = env.step(action)
        ep_max = fold_episode_max_stage(ep_max, info.get("max_stage_in_chunk", 0),
                                        float(reward[0]), top_stage)
        if bool((term | trunc).any()):
            ep_max_stages.append(ep_max)
            ep_max = 0
            obs, _ = env.reset()
            if len(ep_max_stages) >= args.n_episodes:
                break

    rates = stage_reach_rates(ep_max_stages, n_stages)
    out = {"step": step, "n_episodes": len(ep_max_stages),
           "reach": {str(k): rates[k] for k in rates}}
    sidecar = args.log.rsplit(".", 1)[0] + "_reach.json"
    with open(sidecar, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[reeval] task={task} step={step} n={len(ep_max_stages)} reach={rates}")
    print(f"  (自洽: reach[{top_stage}] 应 ≈ 该 run 的 success_rate)")
    print(f"saved -> {sidecar}")
    vec_env.close()


if __name__ == "__main__":
    main()
