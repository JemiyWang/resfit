"""§7.1 奖励稀疏度探针:基座(零残差)在 ThreePiece 上逐 episode 到达的最高 stage 分布。

回答"HER-like stage relabeling 有多少可用信号":
  - 多少比例 episode 到过 stage2 / stage3(部分成功)→ 可被 relabel 成按阶段奖励
  - 若连 stage2 都极少 → 先回头修 chunk 开环太弱,而非做 HER

驱动真实 ChunkResidualEnvWrapper(base + 零残差),num_envs=1 干净逐 episode 归因。
用法:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m resfit.rl_finetuning.chunk_residual.verify_reward_sparsity \
    --base_dir bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_base_policy
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="TwoArmThreePieceAssembly")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-three-piece-assembly")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--n_episodes", type=int, default=25)
    p.add_argument("--max_chunks", type=int, default=700)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

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

    obs, _ = env.reset()
    flat_dim = env.action_dim * args.chunk_length
    zero = torch.zeros(1, flat_dim, device=args.device)

    ep_max_hist = Counter()      # 每 episode 到达的最高 stage
    n_success = 0
    ep_max = 0
    eps_done = 0
    for c in range(args.max_chunks):
        obs, reward, term, trunc, info = env.step(zero)
        ep_max = max(ep_max, int(info.get("max_stage_in_chunk", 0)))
        if float(reward[0]) >= 1.0:
            ep_max = 3
        if bool((term | trunc).any()):
            ep_max_hist[ep_max] += 1
            if ep_max >= 3:
                n_success += 1
            eps_done += 1
            ep_max = 0
            if eps_done >= args.n_episodes:
                break

    print(f"\n{'#'*60}")
    print(f"[完成 episode 数] {eps_done}")
    print(f"[逐 episode 最高 stage 分布] {dict(sorted(ep_max_hist.items()))}")
    if eps_done:
        reach2 = sum(v for k, v in ep_max_hist.items() if k >= 2)
        reach1 = sum(v for k, v in ep_max_hist.items() if k >= 1)
        print(f"[到过 stage>=1] {reach1}/{eps_done} = {reach1/eps_done:.0%}")
        print(f"[到过 stage>=2(部分成功,可 relabel)] {reach2}/{eps_done} = {reach2/eps_done:.0%}")
        print(f"[完整成功 stage3] {n_success}/{eps_done} = {n_success/eps_done:.0%}")
        print("\n[HER 判断] " + (
            "stage>=2 占比可观 → HER relabel 有信号可用,值得做" if reach2 / eps_done >= 0.2
            else "连 stage2 都很少 → 先修 chunk 开环/基座,再谈 HER"))
    vec_env.close()


if __name__ == "__main__":
    main()
