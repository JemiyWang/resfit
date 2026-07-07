"""离线 sim-replay 验证 pouring 5 段检测器 + GR1 grasp 可用性 + stages npz 生成。

跑 precompute_stage_cache(与训练 --offline_stage_cache 缺失时同款路径)在前 N 条 demo 上生成瞬时
stage,统计闩锁后分布。回答单测答不了的问题:GR1 双灵巧手上 _grasped 是否触发(stage 1/3 是否出现)。

用法(仓库根, conda env residual, 不需 GPU/EGL,纯物理):
  python -m resfit.rl_finetuning.chunk_residual.verify_pouring_stages --num_demos 20
退出码: 0=五段(0..4)齐现且非退化; 1=有阶段从未出现(grasp 可能没触发→走几何代理退路 Task 3b)。
"""
from __future__ import annotations
import argparse
import os
import tempfile
from collections import Counter

import numpy as np

from resfit.rl_finetuning.chunk_residual.offline_stage_replay import precompute_stage_cache
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import load_stage_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="resfit/dataset/two_arm_pouring.hdf5")
    p.add_argument("--num_demos", type=int, default=20)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "pour_stages_smoke.npz")
        n = precompute_stage_cache(args.dataset, out, num_demos=args.num_demos)
        cache = load_stage_cache(out)

    seen = Counter()
    latched = Counter()
    reached = Counter()
    for _ep, inst in cache.items():
        inst = np.asarray(inst)
        latch = np.maximum.accumulate(inst)
        for v in inst:
            seen[int(v)] += 1
        for v in latch:
            latched[int(v)] += 1
        reached[int(latch[-1])] += 1

    total = sum(latched.values())
    print(f"\n{'#'*60}\n[pouring stages] {n} demos")
    print("  瞬时出现次数:", dict(sorted(seen.items())))
    print("  闩锁后桶占比:", {k: round(latched[k] / total, 3) for k in sorted(latched)})
    print("  每 demo 终到阶段:", dict(sorted(reached.items())))
    all_present = all(seen.get(s, 0) > 0 for s in range(5))
    print(f"  五段(0..4)是否齐现: {all_present}")
    if not all_present:
        missing = [s for s in range(5) if seen.get(s, 0) == 0]
        print(f"  ✗ 缺失阶段 {missing};若缺 1/3(抓取段)→ 大概率 GR1 _grasped 没触发,走 Task 3b 几何代理退路")
    raise SystemExit(0 if all_present else 1)


if __name__ == "__main__":
    main()
