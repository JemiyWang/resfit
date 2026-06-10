"""Phase 2 Gate 验证:载入 HIQL 高层 π^h 权重,在真 demo 上验证它预测的子目标
确实落在当前态"前方约 way_steps 步",而非原地 / 倒退 / 跳到末态。

对应 plan 2026-06-08-hiql-hierarchy-residual-phase2 的验证 Gate 第 3 条。只读,不改权重/数据。
从仓库根跑:

    PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
      -m resfit.rl_finetuning.chunk_residual.verify_high_actor \
      --pt outputs_chunk/three_piece_high_actor.pt \
      --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly --num_demos 40

机制:high_actor(s_t, g=末态) 输出子目标潜表征 z;用它挂的那个**冻结 gc_value 的 φ**,
把 demo 内每个候选态 s_j 编码成 φ(s_t, s_j),找 ‖φ(s_t,s_j) − z‖ 最近的 j*;
高层若学对了"前向 k 步航点",则 j* − t ≈ way_steps。

eef_piece 后 12 维 rel_piece 同源重标准化同 verify_gc_value.py:用 gc_value.pt 内存的训练时
stats 把后 12 维反标准化回原始再重标准化(read_per_demo_states 子集会按当前批现算 stats,不同源)。
high_actor 与它挂的 gc_value 必须同源 eef_piece state,所以用 gc_value.pt 的 stats 修正即可。
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resfit.rl_finetuning.chunk_residual.plot_zh import use_cjk_font
use_cjk_font()  # 让中文标题/轴标签正常渲染(无 CJK 字体时安静退回)

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import load_high_actor
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states


def build_parser():
    p = argparse.ArgumentParser(description="验证 HIQL 高层 π^h 权重(Phase 2 Gate 第 3 条)")
    p.add_argument("--pt", default="outputs_chunk/three_piece_high_actor.pt")
    p.add_argument("--gc_value", default=None,
                   help="冻结 gc_value.pt(取 φ + rel_piece stats);默认用 high_actor.pt 记录的 gc_value_ckpt")
    p.add_argument("--hdf5", default="resfit/dataset/two_arm_three_piece_assembly.hdf5")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-three-piece-assembly")
    p.add_argument("--num_demos", type=int, default=40)
    p.add_argument("--n_per_demo", type=int, default=30, help="每条 demo 采样多少个起点 t")
    p.add_argument("--goal_mode", choices=["final", "near"], default="final",
                   help="final(原 gate,g=末态)| near(clamp-to-goal 近 goal 塌缩诊断)")
    p.add_argument("--out", default="outputs_chunk/high_actor_verify.png")
    return p


def _resync(seq, mean_n, std_n, mean_tr, std_tr):
    s = seq.astype(np.float64).copy()
    raw = s[:, 18:] * std_n + mean_n
    s[:, 18:] = (raw - mean_tr) / std_tr
    return s.astype(np.float32)


@torch.no_grad()
def predict_z(ha, s, g):
    d = ha(torch.as_tensor(np.ascontiguousarray(s), dtype=torch.float32),
           torch.as_tensor(np.ascontiguousarray(g), dtype=torch.float32))
    return d.mean.numpy()


@torch.no_grad()
def phi_of(vf, base, tgt):
    return vf.phi(torch.as_tensor(np.ascontiguousarray(base), dtype=torch.float32),
                  torch.as_tensor(np.ascontiguousarray(tgt), dtype=torch.float32)).numpy()


def evaluate(ha, vf, seqs, way_steps, n_per_demo):
    """对每条 demo 采样若干起点 t(g=末态),求 high_actor 子目标 z 在 demo 内 φ 最近邻 j*。"""
    rows, z_norms = [], []
    for di, seq in enumerate(seqs):
        T = len(seq)
        if T < 4:
            continue
        g = seq[-1]
        ts = np.unique(np.linspace(0, T - 2, min(T - 1, n_per_demo)).astype(int))
        z = predict_z(ha, seq[ts], np.broadcast_to(g, (len(ts), seq.shape[1])))   # (M, rep)
        z_norms.append(np.linalg.norm(z, axis=-1))
        for m, t in enumerate(ts):
            base = np.broadcast_to(seq[t], (T, seq.shape[1]))
            phi = phi_of(vf, base, seq)                 # (T, rep):整条 demo 内找最近邻
            d = np.linalg.norm(phi - z[m], axis=-1)
            jstar = int(d.argmin())
            rows.append(dict(demo=di, t=int(t), T=int(T), jstar=jstar,
                             off=jstar - int(t),
                             exp=int(min(way_steps, T - 1 - int(t))),
                             room=bool(t + way_steps < T)))
    return rows, np.concatenate(z_norms)


def evaluate_near(ha, vf, seqs, way_steps, n_per_demo, rng):
    """近 goal 塌缩诊断:每个起点 t 取距离 dist∈[1,2·way] 的 goal=s_{min(t+dist,末)},
    求子目标 z 的 demo 内 φ 最近邻 j*;clamp-to-goal 期望 off=j*-t ≈ min(way, dist)。"""
    rows = []
    for di, seq in enumerate(seqs):
        T = len(seq)
        if T < 4:
            continue
        ts = np.unique(np.linspace(0, T - 2, min(T - 1, n_per_demo)).astype(int))
        for t0 in ts:
            t = int(t0)
            dist = int(rng.integers(1, 2 * way_steps + 1))
            gj = min(t + dist, T - 1)
            z = predict_z(ha, seq[t:t + 1], seq[gj:gj + 1])[0]
            base = np.broadcast_to(seq[t], (T, seq.shape[1]))
            phi = phi_of(vf, base, seq)
            jstar = int(np.linalg.norm(phi - z, axis=-1).argmin())
            rows.append(dict(demo=di, t=t, T=int(T), gj=int(gj), dist=int(gj - t),
                             jstar=jstar, off=int(jstar - t),
                             expected_off=int(min(way_steps, gj - t)),
                             clamped=bool(t + dist > T - 1)))
    return rows


def report_near(rows, way_steps):
    """clamp-to-goal 近 goal 塌缩诊断。剔除末尾 clamp 到末态的退化样本(非真 near/far),
    near(dist<way)看 |off−dist|≈0(子目标塌到 goal),far(dist>=way)看 off≈way(落 +way 航点)。"""
    clean = [r for r in rows if not r["clamped"]]
    dropped = len(rows) - len(clean)
    near = [r for r in clean if r["dist"] < way_steps]
    far = [r for r in clean if r["dist"] >= way_steps]
    print("\n============ ② clamp-to-goal 近 goal 塌缩诊断 ============")
    print(f"采样 {len(rows)} 个;剔除末尾 clamp 样本 {dropped} 个(goal 被截到末态,非真 near/far);"
          f"剩 near(dist<{way_steps})={len(near)} far(dist>={way_steps})={len(far)}")
    if not near or not far:
        print("WARNING: near 或 far 桶为空,诊断无效(demo 太短或起点太少)")
        return False
    near_err = np.array([abs(r["off"] - r["dist"]) for r in near])
    far_off = np.array([r["off"] for r in far])
    print(f"近 goal: |off − dist| median={np.median(near_err):.1f}(≈0 表示子目标落在 goal 上)")
    print(f"远 goal: off median={np.median(far_off):.1f}(≈way_steps={way_steps} 表示落 +way 航点)")
    ok = bool((np.median(near_err) <= max(2, 0.25 * way_steps)) and
              (abs(np.median(far_off) - way_steps) <= max(3, 0.3 * way_steps)))
    print(f"clamp 诊断: {'PASS' if ok else 'FAIL'}(近 goal 塌到 goal & 远 goal 落 +way)")
    return ok


def report(rows, z_norms, way_steps, rep_dim):
    off = np.array([r["off"] for r in rows])
    room = [r for r in rows if r["room"]]
    off_room = np.array([r["off"] for r in room])
    ratio = np.array([r["off"] / max(r["exp"], 1) for r in rows])
    # 远离末态却塌到末端(t < T-2k 且 j* 落在末 3 帧)
    far = [r for r in room if r["t"] < r["T"] - 2 * way_steps]
    ec = [r for r in far if r["jstar"] >= r["T"] - 3]

    print("\n================ Phase 2 Gate 验证结果(子目标前向性)================")
    print(f"high_actor way_steps={way_steps} rep_dim={rep_dim};采样 {len(rows)} 个 (demo,t),其中 room(t+k<T)={len(room)}")
    print(f"⓪ 子目标 z 的范数 mean={z_norms.mean():.2f} (φ 球面半径 √rep={np.sqrt(rep_dim):.2f};接近=z 落在 φ 流形上)")
    print(f"① 前向步数 j*-t [room, 目标≈{way_steps}]: median={np.median(off_room):.1f} mean={off_room.mean():.1f} "
          f"25/75 分位=[{np.percentile(off_room,25):.0f}, {np.percentile(off_room,75):.0f}]")
    print(f"② 前向占比 j*>t: {np.mean(off_room>0)*100:.1f}%   原地占比 |j*-t|<=2: {np.mean(np.abs(off_room)<=2)*100:.1f}%   "
          f"倒退占比 j*<t: {np.mean(off_room<0)*100:.1f}%")
    print(f"③ 跳末态占比(远离末态 t<T-2k 却 j*→末 3 帧): {len(ec)}/{len(far)} = {100*len(ec)/max(len(far),1):.1f}%")
    print(f"④ off/expected 比值 [全采样, 应≈1]: median={np.median(ratio):.2f} mean={ratio.mean():.2f}")

    ok = (10 <= np.median(off_room) <= 40) and (np.mean(off_room > 0) > 0.7) \
        and (len(ec) / max(len(far), 1) < 0.3)
    print(f"\nGate 第3条: {'PASS' if ok else 'FAIL'}  "
          f"(前向 k 步:median∈[10,40] & 前向>70% & 不塌末态<30%)")
    if np.mean(np.abs(off_room) <= 2) > 0.3:
        print(f"  ⚠️ 原地占比偏高({np.mean(np.abs(off_room)<=2)*100:.0f}%):部分态高层倾向预测'原地',子目标推进力弱。")
    return ok, off, off_room, ratio, room


def make_plot(rows, off, off_room, ratio, room, seqs, way_steps, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) 前向步数直方(room)
    ax = axes[0, 0]
    ax.hist(off_room, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(0, color="k", ls="--", lw=1, label="原地(0)")
    ax.axvline(way_steps, color="tab:green", lw=2, label=f"目标 k={way_steps}")
    ax.axvline(float(np.median(off_room)), color="tab:red", lw=2, ls=":",
               label=f"median={np.median(off_room):.0f}")
    ax.set_title(f"[Gate①] 子目标前向步数 j*-t (room states, n={len(off_room)})")
    ax.set_xlabel("j* - t (帧)"); ax.set_ylabel("# samples"); ax.legend(); ax.grid(alpha=0.3)

    # (0,1) off/expected 比值
    ax = axes[0, 1]
    ax.hist(np.clip(ratio, -1, 3), bins=40, color="tab:purple", alpha=0.8)
    ax.axvline(1.0, color="tab:green", lw=2, label="理想=1")
    ax.axvline(0.0, color="k", ls="--", lw=1, label="原地=0")
    ax.set_title(f"[Gate④] off/expected (clip 到 [-1,3])\nmedian={np.median(ratio):.2f}")
    ax.set_xlabel("(j*-t) / min(k, 末态-t)"); ax.set_ylabel("# samples"); ax.legend(); ax.grid(alpha=0.3)

    # (1,0) 一条代表 demo 的 t -> j* 曲线 + 参考线
    ax = axes[1, 0]
    di_long = max(range(len(seqs)), key=lambda i: len(seqs[i]))
    pts = [r for r in rows if r["demo"] == di_long]
    if pts:
        T = pts[0]["T"]
        ts = np.array([r["t"] for r in pts]); js = np.array([r["jstar"] for r in pts])
        order = np.argsort(ts); ts, js = ts[order], js[order]
        ax.plot(ts, js, "o-", color="tab:blue", ms=4, label="预测 j*")
        ax.plot(ts, ts, color="gray", ls="--", label="原地 y=t")
        ax.plot(ts, np.minimum(ts + way_steps, T - 1), color="tab:green", lw=2,
                label=f"目标 y=min(t+{way_steps},末)")
        ax.axhline(T - 1, color="tab:red", ls=":", label="末态")
        ax.set_title(f"[示例] 最长 demo #{di_long} (T={T}) 的 t→j*")
        ax.set_xlabel("起点 t"); ax.set_ylabel("子目标最近邻 j*"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) 每条 demo 的 room 前向步数中位数
    ax = axes[1, 1]
    per_demo = {}
    for r in room:
        per_demo.setdefault(r["demo"], []).append(r["off"])
    meds = np.array([np.median(v) for v in per_demo.values()])
    ax.hist(meds, bins=20, color="tab:orange", alpha=0.85)
    ax.axvline(way_steps, color="tab:green", lw=2, label=f"目标 k={way_steps}")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_title(f"[逐 demo] room 前向步数中位数 (n={len(meds)} demo)\nmean of medians={meds.mean():.1f}")
    ax.set_xlabel("该 demo 的 median(j*-t)"); ax.set_ylabel("# demos"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Phase 2 高层 π^h 验证:子目标是否落在当前态前方约 k 步", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


def main():
    args = build_parser().parse_args()
    ha, hinfo = load_high_actor(args.pt)
    gc_ckpt = args.gc_value or hinfo["gc_value_ckpt"]
    print(f"[load] high_actor={args.pt}  info={hinfo}")
    print(f"[load] gc_value={gc_ckpt}(取 φ + rel_piece stats)")
    vf, ginfo = load_gc_value(gc_ckpt)
    assert ha.state_dim == vf.state_dim and ha.rep_dim == vf.rep_dim, \
        f"high_actor 与 gc_value 维度不符: {ha.state_dim}/{ha.rep_dim} vs {vf.state_dim}/{vf.rep_dim}"

    seqs, _std, rel_stats = read_per_demo_states(
        args.hdf5, args.dataset, "eef_piece", num_demos=args.num_demos)
    mean_tr, std_tr = ginfo.get("rel_piece_mean"), ginfo.get("rel_piece_std")
    if mean_tr is not None and rel_stats is not None:
        mean_tr = np.asarray(mean_tr, np.float64); std_tr = np.asarray(std_tr, np.float64)
        mean_n = np.asarray(rel_stats[0], np.float64); std_n = np.asarray(rel_stats[1], np.float64)
        print(f"[data] demos={len(seqs)}  rel-stats drift |Δmean|max={np.abs(mean_n-mean_tr).max():.4f} "
              f"|std_ratio-1|max={np.abs(std_n/std_tr-1).max():.4f}")
        seqs = [_resync(s, mean_n, std_n, mean_tr, std_tr) for s in seqs]
    else:
        print(f"[data] demos={len(seqs)}  WARNING: 无 rel_piece 训练 stats,跳过同源修正(评估可能失真)")

    if args.goal_mode == "near":
        rng = np.random.default_rng(0)
        rows = evaluate_near(ha, vf, seqs, hinfo["way_steps"], args.n_per_demo, rng)
        ok = report_near(rows, hinfo["way_steps"])
    else:
        rows, z_norms = evaluate(ha, vf, seqs, hinfo["way_steps"], args.n_per_demo)
        ok, off, off_room, ratio, room = report(rows, z_norms, hinfo["way_steps"], vf.rep_dim)
        make_plot(rows, off, off_room, ratio, room, seqs, hinfo["way_steps"], args.out)


if __name__ == "__main__":
    main()
