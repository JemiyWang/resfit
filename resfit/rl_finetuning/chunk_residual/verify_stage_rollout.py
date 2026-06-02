"""验证闸:用 62% 的 ThreePiece ACT 基座跑真实 rollout，确认 stage_id 真能 0→1→2→3 跳级。

驱动真实 ChunkResidualEnvWrapper(base + 零残差)，逐 chunk 读 stage:
  - obs["observation.stage_id"]      —— wrapper 闩锁后的 max-so-far(消费端 b)
  - info["max_stage_in_chunk"]       —— 本 chunk 内最高阶段
这同时验证了 worker 侧 info 透出(a) 与 chunk wrapper 消费(b) 的端到端链路。

用法(仓库根目录, conda env residual):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m resfit.rl_finetuning.chunk_residual.verify_stage_rollout \
    --base_dir bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999
退出码: 0=stage 爬到 >0(链路+跳级确认); 1=否。
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
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="TwoArmThreePieceAssembly")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-three-piece-assembly")
    p.add_argument("--base_dir", required=True, help="本地 ACT 基座 step 目录")
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--max_chunks", type=int, default=60)
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
    zero_res = torch.zeros(1, flat_dim, device=args.device)

    traj = []          # (chunk_idx, stage_after, max_in_chunk, reward, done)
    max_stage_seen = 0
    successes = 0
    for c in range(args.max_chunks):
        obs, reward, term, trunc, info = env.step(zero_res)
        stage_after = int(obs["observation.stage_id"][0, 0].item())
        max_in_chunk = int(info.get("max_stage_in_chunk", 0))
        done = bool((term | trunc).any())
        max_stage_seen = max(max_stage_seen, max_in_chunk)
        r = float(reward[0].item())
        if r >= 1.0:
            successes += 1
        traj.append((c, stage_after, max_in_chunk, round(r, 2), done))
        print(f"chunk {c:02d}: stage_after={stage_after} max_in_chunk={max_in_chunk} reward={r:.2f} done={done}")
        if done:
            print(f"  --- episode 结束 (success={r>=1.0}); 续跑新 episode ---")

    print(f"\n{'#'*60}")
    print(f"[max stage_seen] {max_stage_seen}  (0起步/1抓piece1/2 piece1装好/3成功)")
    print(f"[successes] {successes}")
    print(f"[verdict] {'stage 跳级链路确认(>0)' if max_stage_seen > 0 else '未观察到 stage>0(基座可能没抓起，或检测器阈值偏)'}")
    env.close()
    raise SystemExit(0 if max_stage_seen > 0 else 1)


if __name__ == "__main__":
    main()
