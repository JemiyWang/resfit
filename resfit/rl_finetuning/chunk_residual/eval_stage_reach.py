"""per-stage 到达率诊断:base + 受控幅度残差,看残差幅度如何侵蚀各 stage 到达率。

回答核心问题:崩溃是 (a)"够到中间阶段但完不成最终插入"(残差破坏精细装配,中间 stage 仍到)
还是 (b)"基座被毁、连 stage1 都到不了"?

做法:对每个目标残差 L2 范数,跑 N 个 episode(num_envs=1,逐 episode 记最高 stage,
绕开多环境 stage 追踪坑),用 stage_reach_rates 汇总。残差 = 每 chunk 重采样的随机方向、
缩放到目标范数(模拟训练里 residual_norm 长到 ~1.5 的幅度;方向随机,是对"训练出的崩溃残差"
的代理)。若幅度↑时 reach(stage3=success) 先崩、reach(stage1-2) 还在 → 支持 (a);若各 reach
一起崩 → 支持 (b)。

用法(从仓库根目录,conda env residual):
  CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
  python -m resfit.rl_finetuning.chunk_residual.eval_stage_reach \
    --base_dir bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_base_policy
from resfit.rl_finetuning.chunk_residual.stage_reach import stage_reach_rates
from resfit.rl_finetuning.chunk_residual.stage_detectors import NUM_STAGES
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


def run_for_norm(env, flat_dim, target_norm, n_episodes, max_chunks, top_stage, device, gen):
    """跑 n_episodes,每 chunk 注入 L2 范数=target_norm 的随机残差,逐 episode 记最高 stage。"""
    env.reset()
    ep_max_stages: list[int] = []
    ep_max = 0
    for _ in range(max_chunks):
        if target_norm == 0.0:
            res = torch.zeros(1, flat_dim, device=device)
        else:
            r = torch.randn(1, flat_dim, generator=gen)        # CPU 生成,可复现
            res = (r / r.norm() * target_norm).to(device)
        _, reward, term, trunc, info = env.step(res)
        ep_max = max(ep_max, int(info.get("max_stage_in_chunk", 0)))
        if float(reward[0]) >= 1.0:
            ep_max = top_stage                                  # 成功 → 最高 stage
        if bool((term | trunc).any()):
            ep_max_stages.append(ep_max)
            ep_max = 0
            env.reset()
            if len(ep_max_stages) >= n_episodes:
                break
    return ep_max_stages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="TwoArmThreePieceAssembly")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-three-piece-assembly")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--action_scale", type=float, default=0.2)       # 对齐控制组
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--norms", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--max_chunks", type=int, default=600)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    gen = torch.Generator().manual_seed(args.seed)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    meta = LeRobotDataset(args.dataset).meta
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device=args.device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=args.device)

    base_policy = build_base_policy(args.base_dir, args.device)
    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device)
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length)
    flat_dim = env.action_dim * args.chunk_length
    n_stages = NUM_STAGES[args.task]
    top_stage = n_stages - 1

    print(f"\n{'#'*72}")
    print(f"[task] {args.task}  NUM_STAGES={n_stages}  action_scale={args.action_scale}")
    print(f"[每 norm 跑 {args.n_episodes} episode,num_envs=1]\n")
    header = "resid_norm |  n |  " + "  ".join(f"reach{k}" for k in range(1, n_stages))
    print(header)
    print("-" * len(header))
    for norm in args.norms:
        ep_max = run_for_norm(env, flat_dim, norm, args.n_episodes, args.max_chunks,
                              top_stage, args.device, gen)
        rates = stage_reach_rates(ep_max, n_stages)
        cells = "  ".join(f"{rates[k]:.2f} " for k in range(1, n_stages))
        print(f"   {norm:5.2f}   | {len(ep_max):2d} |  {cells}   (reach{top_stage}=success)")
    print(f"\n[读法] norm↑ 时若 reach{top_stage}(success) 先崩、reach1-{top_stage-1} 还在 → "
          f"残差破坏精细装配(a);若各 reach 一起崩 → 基座被毁(b)")
    vec_env.close()


if __name__ == "__main__":
    main()
