"""Phase0 诊断(opt-in,GPU):LeRobot 视频解码图 vs raw-hdf5 无损图,过冻结 ACT 的 530 特征差。

threading 既有 raw hdf5(无损,=在线渲染)又有 LeRobot 视频,直接比同一 demo 帧两路特征。
过(差和 EGL 渲染噪声同量级)→ LeRobot 图能当 act_feat 数据源;不过(差一大截)→ 视频压缩破坏同源,死路。
跑:CUDA_VISIBLE_DEVICES=2 conda run -n residual python -m \
    resfit.rl_finetuning.chunk_residual.verify_lerobot_image_consistency \
    --base resfit/out/threading/best --hdf5 resfit/dataset/two_arm_threading.hdf5 \
    --dataset ankile/dexmg-two-arm-threading --root resfit/dataset/ankile/dexmg-two-arm-threading
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def _to_chw01(v):
    """LeRobot/任意图 tensor -> (1,C,H,W) float[0,1]。兼容 CHW/HWC、uint8/float。"""
    v = torch.as_tensor(v).float()
    if v.ndim == 3 and v.shape[0] in (1, 3):       # CHW
        v = v.unsqueeze(0)
    elif v.ndim == 3:                               # HWC
        v = v.permute(2, 0, 1).unsqueeze(0)
    if float(v.max()) > 1.5:                         # uint8 尺度
        v = v / 255.0
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=None, help="LeRobot 数据本地根目录")
    args = ap.parse_args()
    from resfit.lerobot.utils.load_policy import load_policy
    from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
        STATE18_KEYS, assemble_state18, sorted_demo_keys)
    from resfit.rl_finetuning.utils.normalization import StateStandardizer
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    cand = Path(args.base) / "policy"
    act = load_policy(cand if cand.is_dir() else Path(args.base)); act.eval()
    keys = list(act.config.image_features.keys())
    print("ACT image_features:", keys)
    ext = ActFeatureExtractor(act, image_keys=keys)
    std = StateStandardizer.from_dataset_stats(
        LeRobotDatasetMetadata(args.dataset, root=args.root).stats["observation.state"], device="cpu")

    ds = LeRobotDataset(args.dataset, root=args.root, video_backend="pyav")
    edi = ds.episode_data_index
    print("LeRobot episodes:", len(edi["from"]))
    s0 = ds[0]
    for k in keys:
        if k in s0:
            v = s0[k]
            print(f"  LeRobot {k}: dtype={v.dtype} shape={tuple(v.shape)} "
                  f"range=[{float(v.min()):.3f},{float(v.max()):.3f}]")

    with h5py.File(args.hdf5, "r") as f:
        demos = sorted_demo_keys(list(f["data"].keys()))
        worst = 0.0
        for di in [0, 1, 5]:
            ep = demos[di]
            grp = f[f"data/{ep}"]
            T = grp["obs/agentview_image"].shape[0]
            obs_arrays = {kk: grp[f"obs/{kk}"][()] for kk, _ in STATE18_KEYS}
            hstate = assemble_state18(obs_arrays)                       # (T,18) raw
            lo, hi = int(edi["from"][di]), int(edi["to"][di])
            lr0 = ds[lo]["observation.state"].numpy()
            sd = float(np.abs(hstate[0] - lr0).max())
            print(f"\ndemo {ep}(T={T}) vs LeRobot ep{di}(len={hi-lo}): state[0] max|d|={sd:.4f} "
                  f"{'(对齐✓)' if sd < 1e-2 else '(可能错位!特征比对无意义)'}")
            for t in [0, T // 2, T - 2]:
                rstate = std.standardize(torch.as_tensor(hstate[t:t + 1], dtype=torch.float32))
                # raw hdf5 图
                rimg = {k: _to_chw01(grp[f"obs/{k.replace('observation.images.','')}_image"][t:t + 1][0])
                        for k in keys}
                f_raw = ext.embed_batch({**rimg, "observation.state": rstate}).cpu().numpy()
                # LeRobot 同帧图(同 proprio → 隔离图像差)
                samp = ds[lo + t]
                limg = {k: _to_chw01(samp[k]) for k in keys}
                f_lr = ext.embed_batch({**limg, "observation.state": rstate}).cpu().numpy()
                d = float(np.abs(f_raw - f_lr).max())
                worst = max(worst, d)
                print(f"  frame {t}: 530-feat max|raw-lerobot|={d:.4f}")
        print(f"\n==== 最大特征差(raw 无损图 vs LeRobot 视频图) = {worst:.4f} ====")
        print("参考:上次 EGL 渲染噪声 max|d|≈0.0277。同量级→LeRobot 图可用;>0.1~0.5→视频压缩破坏同源。")


if __name__ == "__main__":
    main()
