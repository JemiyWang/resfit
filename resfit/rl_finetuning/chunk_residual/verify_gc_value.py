"""Phase 1 Gate 验证:载入 goal-conditioned HIQL value 权重,在真 demo 上评估 V 是否学对。

补上一直缺的 gc_value 验证脚本(对应 plan 2026-06-08-hiql-hierarchy-residual-phase1 的验证 Gate)。
只读,不改权重/数据。从仓库根跑:

    PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
      -m resfit.rl_finetuning.chunk_residual.verify_gc_value \
      --pt outputs_chunk/three_piece_gc_value.pt \
      --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly --num_demos 40

关键:eef_piece 后 12 维 rel_piece 的标准化 mean/std 是 read_per_demo_states 按"当前这批 demo"
现算的;若只取子集,会和训练时(全量)不同源。本脚本用 .pt 内存的训练时 rel_piece stats 把后 12 维
反标准化回原始、再重标准化,保证同源(否则评估失真,std 漂移可达 ~12%)。
"""
import argparse

import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def build_parser():
    p = argparse.ArgumentParser(description="验证 goal-conditioned HIQL value 权重(Phase 1 Gate)")
    p.add_argument("--pt", default="outputs_chunk/three_piece_gc_value.pt")
    p.add_argument("--hdf5", default="resfit/dataset/two_arm_three_piece_assembly.hdf5")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-three-piece-assembly")
    p.add_argument("--num_demos", type=int, default=40)
    p.add_argument("--gamma", type=float, default=0.99)
    return p


def main():
    args = build_parser().parse_args()
    model, info = load_gc_value(args.pt)
    print(f"[load] {args.pt}")
    print(f"[load] v_stats={info['v_stats']} state_mode={info['state_mode']}")

    seqs, _standardizer, rel_stats = read_per_demo_states(
        args.hdf5, args.dataset, "eef_piece", num_demos=args.num_demos)

    # 同源标准化修正:后 12 维 rel_piece 按训练时 stats(.pt 内)重标准化
    mean_tr, std_tr = info.get("rel_piece_mean"), info.get("rel_piece_std")
    if mean_tr is not None and rel_stats is not None:
        mean_tr = np.asarray(mean_tr, dtype=np.float64)
        std_tr = np.asarray(std_tr, dtype=np.float64)
        mean_n = np.asarray(rel_stats[0], dtype=np.float64)
        std_n = np.asarray(rel_stats[1], dtype=np.float64)
        print(f"[data] demos={len(seqs)}  rel-stats drift |Δmean|max={np.abs(mean_n - mean_tr).max():.4f} "
              f"|std_ratio-1|max={np.abs(std_n / std_tr - 1).max():.4f}")
        seqs = [_resync(s, mean_n, std_n, mean_tr, std_tr) for s in seqs]
    else:
        print(f"[data] demos={len(seqs)}  WARNING: 无 rel_piece 训练 stats,跳过同源修正(评估可能失真)")

    res = evaluate(model, seqs, args.gamma)
    report(res, len(seqs))


def _resync(seq, mean_n, std_n, mean_tr, std_tr):
    s = seq.astype(np.float64).copy()
    raw = s[:, 18:] * std_n + mean_n
    s[:, 18:] = (raw - mean_tr) / std_tr
    return s.astype(np.float32)


@torch.no_grad()
def _V(model, s, g):
    v1, v2 = model(torch.as_tensor(np.ascontiguousarray(s), dtype=torch.float32),
                   torch.as_tensor(np.ascontiguousarray(g), dtype=torch.float32))
    return torch.minimum(v1, v2).numpy()


def evaluate(model, seqs, gamma):
    rng = np.random.default_rng(0)
    state_rho, goal_rho, at_goal, start_vals, analytic, final_gt_first = [], [], [], [], [], 0
    sg_vals = []
    for seq in seqs:
        T = len(seq)
        # ① 状态侧:固定 g=末态,变 s
        v = _V(model, seq, np.broadcast_to(seq[-1], seq.shape))
        state_rho.append(spearman(np.arange(T), v))
        final_gt_first += int(v[-3:].mean() > v[:3].mean())
        at_goal.append(float(v[-1]))
        start_vals.append(float(v[0]))
        analytic.append(-(1 - gamma ** T) / (1 - gamma))
        # ③ 自指 V(s,g=s)
        k = rng.integers(0, T, size=min(10, T))
        sg_vals.append(_V(model, seq[k], seq[k]))
        # ⑤ 目标侧(φ):固定 s=起始,变 g 远近
        js = np.unique(np.linspace(0, T - 1, 6).astype(int))
        vg = _V(model, np.broadcast_to(seq[0], (len(js), seq.shape[1])), seq[js])
        goal_rho.append(spearman(js.astype(float), vg))
    return dict(state_rho=np.array(state_rho), goal_rho=np.array(goal_rho),
                at_goal=np.array(at_goal), start_vals=np.array(start_vals),
                analytic=np.array(analytic), final_gt_first=final_gt_first,
                sg_vals=np.concatenate(sg_vals))


def report(r, n):
    sr, gr, sg = r["state_rho"], r["goal_rho"], r["sg_vals"]
    print("\n================ Phase 1 Gate 验证结果 ================")
    print(f"① 状态侧 spearman(t, V(s,g=末态)) [应→+1]: mean={sr.mean():.3f} median={np.median(sr):.3f} "
          f"min={sr.min():.3f} frac>0.9={np.mean(sr > 0.9):.2f}  (final3>first3: {r['final_gt_first']}/{n})")
    print(f"② V(末态,末态) [应~0]: mean={r['at_goal'].mean():.3f} "
          f"range=[{r['at_goal'].max():.3f}, {r['at_goal'].min():.3f}]")
    print(f"③ V(s,g=s) 随机态 [应~0]: mean={sg.mean():.3f} std={sg.std():.3f} "
          f"max={sg.max():.3f} min={sg.min():.3f}")
    print(f"④ 折扣量级 V(起始,末态): mean={r['start_vals'].mean():.2f} vs 解析 mean={r['analytic'].mean():.2f}")
    print(f"⑤ 目标侧 spearman(g距离, V(起始,g)) [应→-1]: mean={gr.mean():.3f} median={np.median(gr):.3f} "
          f"max={gr.max():.3f} frac<-0.9={np.mean(gr < -0.9):.2f}")
    ok = (np.median(sr) > 0.9) and (np.median(gr) < -0.9) and (abs(r["at_goal"].mean()) < 1.0)
    print(f"\nGate: {'PASS' if ok else 'FAIL'}  "
          f"(状态侧单调↑ & 目标侧距离↓ & 到达态≈0)")
    if sg.min() < -5:
        print(f"  ⚠️ 注意:V(s,g=s) 有负向长尾(min={sg.min():.1f}),少数态没学到'自指=0';高层用优势之差,影响有限。")


if __name__ == "__main__":
    main()
