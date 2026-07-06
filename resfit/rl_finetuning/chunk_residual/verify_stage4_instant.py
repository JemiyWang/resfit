"""真环境探针:base(零残差)在 ThreePiece 上的 4 段瞬时分布,确认检测器接线正确。

新 4 段(2026-06-29 改):0 起步 / 1 抓起 piece1 和 piece2(两件都被握)/
2 放好 piece1 / 3 放好 piece2(成功)。

回答两件事:
  1. env.piece_1/piece_2 属性名是否正确 —— stage 1 要求两件**都**被握住才触发;
     若见过 stage 2/3 却从未见 stage 1,几乎必是某个 piece 属性名错、_grasped 恒 False。
  2. 4 段 stage 的瞬时出现频次,目测各里程碑是否都被触达。

用法:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
  python -m resfit.rl_finetuning.chunk_residual.verify_stage4_instant \
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
    print(f"[NUM_STAGES] {NUM_STAGES[args.task]}(期望 4)")
    print(f"[chunk 级 max_stage 频次] {dict(sorted(seen.items()))}")
    saw1, saw3 = seen.get(1, 0), seen.get(3, 0)
    print(f"[stage 1(piece1+piece2 都抓起)出现] {saw1} 次")
    print(f"[stage 3(成功)出现] {saw3} 次")
    if saw1 == 0 and (saw3 > 0 or seen.get(2, 0) > 0):
        print("[!! 反证] 见过装配/成功却从未见 stage1(两件都抓)→ 某个 piece 属性名几乎必错,需修 detector")
    elif saw1 > 0:
        print("[OK] stage1 真触发 → piece_1/piece_2 属性名正确、AND 抓取里程碑生效")
    else:
        print("[?] stage1/3 都没出现(可能 base 太弱/chunk 数太少)→ 加 --max_chunks 再看")
    vec_env.close()


if __name__ == "__main__":
    main()
