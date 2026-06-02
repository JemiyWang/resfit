"""真环境探针:base(零残差)在 ThreePiece 上的 5 段瞬时分布,确认 stage 拆细生效。

回答两件事:
  1. env.piece_2 属性名是否正确 —— stage 3(piece2 抓起)是否真出现。
  2. 5 段 stage 的瞬时出现频次,目测 stage2→3 拆细给瓶颈段带来分辨率。
反证:若见过 stage 4(成功)却从未见 stage 3 → piece_2 属性名几乎必错,需修 detector。

用法:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
  python -m resfit.rl_finetuning.chunk_residual.verify_stage5_instant \
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
from resfit.rl_finetuning.chunk_residual.stage_detectors import NUM_STAGES
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="TwoArmThreePieceAssembly")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-three-piece-assembly")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
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
    env.reset()
    zero = torch.zeros(1, env.action_dim * args.chunk_length, device=args.device)

    seen = Counter()        # 每个 chunk 收尾的 max_in_chunk 计数
    for _ in range(args.max_chunks):
        _, reward, term, trunc, info = env.step(zero)
        seen[int(info.get("max_stage_in_chunk", 0))] += 1
        if bool((term | trunc).any()):
            env.reset()

    print(f"\n{'#'*60}")
    print(f"[NUM_STAGES] {NUM_STAGES[args.task]}(期望 5)")
    print(f"[chunk 级 max_stage 频次] {dict(sorted(seen.items()))}")
    saw3, saw4 = seen.get(3, 0), seen.get(4, 0)
    print(f"[stage 3(piece2 抓起)出现] {saw3} 次")
    print(f"[stage 4(成功)出现] {saw4} 次")
    if saw3 == 0 and saw4 > 0:
        print("[!! 反证] 见过成功却从未见 stage3 → env.piece_2 属性名几乎必错,需修 detector")
    elif saw3 > 0:
        print("[OK] stage3 真触发 → piece_2 属性名正确、stage2→3 拆细生效")
    else:
        print("[?] stage3/4 都没出现(可能 base 太弱/chunk 数太少)→ 加 --max_chunks 再看")
    vec_env.close()


if __name__ == "__main__":
    main()
