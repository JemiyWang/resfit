"""验证 build_chunk_dataset 修复:AE 数据集不再解码视频,可安全读过 drawer 坏 episode。

跑法(仓库根,conda env residual):
  python -m resfit.rl_finetuning.chunk_residual.verify_ae_no_video [repo_id] [bad_ep]
默认验证 drawer(坏 episode 505)。不依赖 GPU。
"""
from __future__ import annotations

import sys

from resfit.rl_finetuning.chunk_residual.train_action_ae import build_chunk_dataset

REPO = sys.argv[1] if len(sys.argv) > 1 else "ankile/dexmg-two-arm-drawer-cleanup"
BAD_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 505
CHUNK = 20

ds = build_chunk_dataset(REPO, CHUNK)
print(f"dataset={REPO}  len={len(ds)}  video_keys={ds.meta.video_keys}")
assert ds.meta.video_keys == [], "视频键未清空,__getitem__ 仍会解码视频"

# 定位坏 episode 的全局帧索引,优先用 episode_data_index,缺失则从 episode_index 列推。
try:
    edi = ds.episode_data_index
    start, end = int(edi["from"][BAD_EP]), int(edi["to"][BAD_EP])
except Exception:
    col = list(ds.hf_dataset["episode_index"])
    idxs_ep = [i for i, e in enumerate(col) if int(e) == BAD_EP]
    start, end = idxs_ep[0], idxs_ep[-1] + 1
print(f"bad episode {BAD_EP}: frames [{start}, {end})  ({end - start} 帧)")

# 读坏 episode 跨度内的样本 + 两端边界,确认不崩、action 形状正确、样本无图像键。
probe = list(range(start, min(start + 40, end))) + [start, end - 1]
D = None
for i in probe:
    s = ds[i]
    a = s["action"]
    assert a.shape[0] == CHUNK, f"idx {i}: action chunk 长度 {a.shape} != {CHUNK}"
    assert not any("image" in k for k in s.keys()), f"idx {i}: 样本仍含图像键 {list(s.keys())}"
    D = a.shape[1]
print(f"OK: 读了 {len(probe)} 个含坏 episode {BAD_EP} 的样本,action shape=({CHUNK}, {D}),无解码崩溃。")
