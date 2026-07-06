"""瞬时探针:逐 env step(不经 chunk 的 max)读 info["stage_id"]，看 stage 1 到底有没有触发。

绕过 ChunkResidualEnvWrapper，直接用基座动作驱动 vec_env,记录出现过的所有阶段值。
回答单测无法回答的问题:真环境里 _grasped 是否真触发(stage 1),还是被 chunk max 掩盖。

用法(仓库根目录, conda env residual):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m resfit.rl_finetuning.chunk_residual.verify_stage_instant \
    --base_dir bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999
退出码: 0=观察到 stage 1(检测器确为 4 阶段); 1=stage 1 从未瞬时出现。
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.chunk_residual.chunk_act_base import get_action_chunk
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_base_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="TwoArmThreePieceAssembly")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=1200)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    base_policy = build_base_policy(args.base_dir, args.device)
    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device)

    raw_obs, _ = vec_env.reset()
    base_policy.reset()
    seen = Counter()
    steps = 0
    while steps < args.max_steps:
        chunk = get_action_chunk(base_policy, raw_obs, args.chunk_length)   # [B,L,D] 原始尺度
        for t in range(args.chunk_length):
            raw_obs, reward, term, trunc, info = vec_env.step(chunk[:, t])
            steps += 1
            if "stage_id" in info:
                seen[int(info["stage_id"][0])] += 1
            if bool((term | trunc).any()):
                base_policy.reset()
                break
            if steps >= args.max_steps:
                break

    print(f"\n{'#'*60}")
    print(f"[stage 瞬时出现次数] {dict(sorted(seen.items()))}  (共 {steps} 步)")
    saw_1 = seen.get(1, 0) > 0
    print(f"[stage 1(抓起piece1和piece2) 是否出现] {saw_1}  次数={seen.get(1,0)}")
    if not saw_1:
        print("  → stage 1 从未瞬时出现:真环境 _grasped 可能没触发(双臂夹爪迭代),或抓取瞬间太短")
    vec_env.close()
    raise SystemExit(0 if saw_1 else 1)


if __name__ == "__main__":
    main()
