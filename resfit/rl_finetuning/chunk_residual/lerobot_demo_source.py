"""按集读 LeRobot 数据集的 demo 帧(act_feat 数据源;纯读取,惰性 import LeRobotDataset)。

帧格式与 raw-hdf5 路对齐到 ACT 输入:images=(T,3,84,84) float[0,1]、state=(T,Dp) raw、actions=(T,Da)。
LeRobot 图本就是 CHW float[0,1](见 verify_lerobot_image_consistency.py Phase0);to_chw01 兼容 HWC/uint8 防御。
"""
from __future__ import annotations

import torch


def to_chw01(v) -> torch.Tensor:
    """单帧图 -> (C,H,W) float[0,1]。兼容 CHW/HWC、uint8/float。"""
    v = torch.as_tensor(v).float()
    if v.ndim == 3 and v.shape[0] in (1, 3):        # CHW
        pass
    elif v.ndim == 3:                                # HWC
        v = v.permute(2, 0, 1)
    if float(v.max()) > 1.5:                          # uint8 尺度
        v = v / 255.0
    return v


def open_lerobot(repo_id, root):
    """打开本地 LeRobot 数据集(torchcodec 解码,~8x faster than pyav)。
    需先 `source resfit/lerobot/shell/torchcodec_env.sh` 配 LD_LIBRARY_PATH(指向 nvidia-npp 等
    pip 库),否则 torchcodec 加载 libnppicc.so.12 失败。torchcodec vs pyav 解码实测同帧逐像素
    一致(max abs diff=0),act_feat 离线 cache 同源无忧。"""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id, root=root, video_backend="torchcodec")


def lerobot_episode_count(ds) -> int:
    return int(len(ds.episode_data_index["from"]))


def count_lerobot_transitions(ds, num_demos=None) -> int:
    """sum(T-1) over 前 num_demos 集 → LazyTensorStorage 定容(取集序须与 build 一致)。"""
    edi = ds.episode_data_index
    n = int(len(edi["from"]))
    if num_demos is not None:
        n = min(n, num_demos)
    return int(sum(max(0, int(edi["to"][e]) - int(edi["from"][e]) - 1) for e in range(n)))


def lerobot_episode_frames(ds, ep_idx, image_keys, proprio_key="observation.state",
                           action_key="action") -> dict:
    """读第 ep_idx 集所有帧 -> {images:{k:(T,3,84,84)f01}, state:(T,Dp), actions:(T,Da)}。"""
    lo = int(ds.episode_data_index["from"][ep_idx])
    hi = int(ds.episode_data_index["to"][ep_idx])
    imgs = {k: [] for k in image_keys}
    states, actions = [], []
    for i in range(lo, hi):
        s = ds[i]
        for k in image_keys:
            imgs[k].append(to_chw01(s[k]))
        states.append(torch.as_tensor(s[proprio_key], dtype=torch.float32).reshape(-1))
        actions.append(torch.as_tensor(s[action_key], dtype=torch.float32).reshape(-1))
    return {"images": {k: torch.stack(imgs[k]) for k in image_keys},
            "state": torch.stack(states),
            "actions": torch.stack(actions)}
