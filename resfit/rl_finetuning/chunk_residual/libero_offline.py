"""LIBERO offline 锚 buffer:从 LeRobot physical-intelligence/libero 直读当前任务的 demo,
产出与在线 add_chunk_transition 同构的 transition。命门:数据集图已预翻正(只 resize 不翻);
reward 末帧+1(不用事件 reward npy);归一化与在线同源。"""
from __future__ import annotations

import io
import json
import os

import numpy as np

AGENTVIEW_KEY = "observation.images.agentview"
WRIST_KEY = "observation.images.robot0_eye_in_hand"


def libero_task_language(suite: str, task_id: int) -> str:
    """benchmark 任务语言串(demo 按它匹配 LeRobot 数据集)。"""
    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()[suite]()
    return task_suite.get_task(int(task_id)).language.strip()


def find_demo_episodes(lerobot_root: str, language: str) -> list[str]:
    """读 meta/episodes.jsonl,返回 tasks[0]==language 的 episode parquet 路径(按 episode_index 升序)。"""
    ep_path = os.path.join(lerobot_root, "meta", "episodes.jsonl")
    matched = []
    with open(ep_path) as f:
        for line in f:
            rec = json.loads(line)
            tasks = rec.get("tasks", [])
            if tasks and tasks[0].strip() == language.strip():
                matched.append(int(rec["episode_index"]))
    if not matched:
        raise ValueError(f"find_demo_episodes: 数据集 {lerobot_root} 无任务语言 {language!r} 的 episode")
    matched.sort()
    return [os.path.join(lerobot_root, "data", f"chunk-{ei // 1000:03d}",
                         f"episode_{ei:06d}.parquet") for ei in matched]


def _decode_img_col(col) -> np.ndarray:
    """LeRobot v2.0 image 列(每帧 dict{bytes,path} 的编码图,PNG/JPEG 皆可,PIL 自动识别)→ (T,H,W,3) uint8。"""
    from PIL import Image
    out = []
    for cell in col:
        b = cell["bytes"] if isinstance(cell, dict) else cell
        if b is None:
            raise ValueError("_decode_img_col: image 单元 bytes 为 None(数据集可能未完整下载,只有 path)")
        out.append(np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8))
    return np.stack(out, axis=0)


def read_libero_demo(parquet_path: str) -> dict:
    """一条 episode parquet → {state(T,8) f32, action(T,7) f32, agentview(T,256,256,3) u8, wrist(...) u8}。"""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    state = np.stack([np.asarray(x, np.float32) for x in df["state"]], axis=0)
    action = np.stack([np.asarray(x, np.float32) for x in df["actions"]], axis=0)
    return {"state": state, "action": action,
            "agentview": _decode_img_col(df["image"]), "wrist": _decode_img_col(df["wrist_image"])}


def _img_chw_uint8(hwc_uint8, size):
    """命门①:只 resize_with_pad(不翻转)→ CHW uint8。"""
    from resfit.rl_finetuning.chunk_residual.libero_obs import resize_with_pad
    resized = resize_with_pad(hwc_uint8, size, size)        # HWC uint8,无翻转
    return np.transpose(resized, (2, 0, 1))                 # CHW uint8


def _demo_to_transitions(demo, *, action_scaler, state_standardizer, base_actions,
                         image_size, subgoal_z=None):
    """一条 demo → list[TensorDict],与在线 add_chunk_transition 同构。

    base_actions: (T,7) 已缩放的 base 动作(base_policy 模式),或 None(gt 模式 → base=action)。
    subgoal_z: (T, rep_dim) 子目标表征,非 None 时每帧 obs 含 observation.subgoal=subgoal_z[t](默认 None 不含)。
    命门②:reward 仅末帧 transition=1.0、done 仅末帧 True(demo 是成功轨迹)。
    命门③:state 用 state_standardizer、action 用 action_scaler(与在线同源)。

    注:返回的 td 内含对同一帧张量的视图(相邻 transition 共享边界帧)——只读,调用者勿 in-place
    改其内容;灌进 replay buffer 时 storage 会拷贝,无别名风险(与 offline_stage_replay 一致)。
    """
    import torch
    from tensordict import TensorDict
    state = torch.as_tensor(demo["state"], dtype=torch.float32)
    action_raw = torch.as_tensor(demo["action"], dtype=torch.float32)
    T = state.shape[0]
    if T < 2:
        return []
    state_std = state_standardizer.standardize(state)                  # (T,8)
    action = action_scaler.scale(action_raw)                          # (T,7) 缩放
    base = (torch.as_tensor(base_actions, dtype=torch.float32)        # 容忍 numpy/tensor 入参
            if base_actions is not None else action)                  # (T,7)
    img_av = torch.stack([torch.as_tensor(_img_chw_uint8(demo["agentview"][t], image_size)) for t in range(T)])
    img_wr = torch.stack([torch.as_tensor(_img_chw_uint8(demo["wrist"][t], image_size)) for t in range(T)])

    def _obs(t):
        d = {"observation.state": state_std[t], "observation.base_action": base[t],
             "observation.stage_id": torch.zeros(1, dtype=torch.float32),
             AGENTVIEW_KEY: img_av[t], WRIST_KEY: img_wr[t]}
        if subgoal_z is not None:
            d["observation.subgoal"] = subgoal_z[t]
        return d

    out = []
    for t in range(T - 1):
        last = (t == T - 2)
        td = TensorDict({
            "obs": TensorDict(_obs(t), batch_size=[]),
            "next": TensorDict({"obs": TensorDict(_obs(t + 1), batch_size=[]),
                                "done": torch.tensor(bool(last)),
                                "reward": torch.tensor(1.0 if last else 0.0, dtype=torch.float32)},
                               batch_size=[]),
            "action": action[t],
            "max_stage": torch.tensor(0.0, dtype=torch.float32),
            "_priority": torch.tensor(10.0, dtype=torch.float32),
        }, batch_size=[])
        out.append(td)
    return out


def count_libero_offline_transitions(lerobot_root, suite, task_id, num_demos=None) -> int:
    """从 episodes.jsonl 的 length 求 sum(T-1)(不解 parquet,秒级),给 LazyTensorStorage 精确定容。

    **按 episode_index 升序取 num_demos —— 必须与 find_demo_episodes(build 用)同序**,否则
    num_demos 截断会选到不同子集、storage 定容与实际灌入对不上(overflow/eviction)。
    注:这是 episodes.jsonl 报告的 length;若个别 demo 实际 parquet T<2,build 会跳过它,
    实得 transition 略少于此 cap —— LazyTensorStorage 容量略余是安全的(不会 assert len==cap)。
    """
    language = libero_task_language(suite, task_id)
    ep_path = os.path.join(lerobot_root, "meta", "episodes.jsonl")
    pairs = []  # (episode_index, length)
    with open(ep_path) as f:
        for line in f:
            rec = json.loads(line)
            tasks = rec.get("tasks", [])
            if tasks and tasks[0].strip() == language.strip():
                pairs.append((int(rec["episode_index"]), int(rec["length"])))
    pairs.sort()                       # 按 episode_index 升序,与 find_demo_episodes 一致
    if num_demos is not None:
        pairs = pairs[:num_demos]
    return int(sum(max(0, length - 1) for _, length in pairs))


def build_libero_offline_buffer(offline_rb, *, lerobot_root, suite, task_id,
                                action_scaler, state_standardizer,
                                base_policy, base_mode, base_device="cpu",
                                image_size=84, num_demos=None,
                                subgoal=None, way_steps=25, feat_seqs=None) -> None:
    """灌当前任务的 demo 进 offline_rb。base_mode∈{gt,base_policy}。
    subgoal!=None 时逐 demo 用 feat_seqs[i](pi0_feat 缓存标准化序列)算真航点 z 存 observation.subgoal。"""
    if base_mode not in ("gt", "base_policy"):
        raise ValueError(f"base_mode 必须是 gt|base_policy,得到 {base_mode!r}")
    if base_mode == "base_policy" and base_policy is None:
        raise ValueError("base_mode=base_policy 需传 base_policy")
    if subgoal is not None and feat_seqs is None:
        raise ValueError("subgoal 离线子目标需 feat_seqs(pi0_feat 缓存 per-demo 标准化序列)")
    import torch
    language = libero_task_language(suite, task_id)
    paths = find_demo_episodes(lerobot_root, language)
    if num_demos is not None:
        paths = paths[:num_demos]
    for i, p in enumerate(paths):
        demo = read_libero_demo(p)
        T = demo["state"].shape[0]
        if T < 2:
            print(f"[libero-offline] 跳过 {p}(T<2)")
            continue
        base_actions = (_libero_demo_base_actions(demo, base_policy, action_scaler, image_size, base_device)
                        if base_mode == "base_policy" else None)
        subgoal_z = None
        if subgoal is not None:
            seq = torch.as_tensor(feat_seqs[i], dtype=torch.float32)        # (T, 2056) 已标准化
            assert seq.shape[0] == T, \
                f"pi0_feat 缓存帧数 {seq.shape[0]} != demo {T} (ep#{i}, {p})"
            way = np.minimum(np.arange(T) + way_steps, T - 1)
            subgoal_z = subgoal.subgoal_waypoint(seq, seq[way]).cpu()       # (T, rep_dim)
        for td in _demo_to_transitions(demo, action_scaler=action_scaler,
                                       state_standardizer=state_standardizer,
                                       base_actions=base_actions, image_size=image_size,
                                       subgoal_z=subgoal_z):
            offline_rb.add(td.unsqueeze(0))


def _libero_demo_base_actions(demo, base_policy, action_scaler, image_size, device):
    """逐帧调 base_policy.select_action 出 base 动作(缩放后,(T,7))。

    对齐在线 select_action:喂 raw(未标准化)state(8)+ 84 CHW float[0,1] 图(adapter 内部
    build_libero_serve_obs 再 resize 224,无翻转)。逐 demo 先 reset 清队列、顺序不可批。
    """
    import torch
    state = torch.as_tensor(demo["state"], dtype=torch.float32)
    T = state.shape[0]
    # /255.0:policy 路要 float[0,1](buffer 路 _demo_to_transitions 存的是 uint8,两者用途不同)
    img_av = torch.stack([torch.as_tensor(_img_chw_uint8(demo["agentview"][t], image_size), dtype=torch.float32) / 255.0
                          for t in range(T)])
    img_wr = torch.stack([torch.as_tensor(_img_chw_uint8(demo["wrist"][t], image_size), dtype=torch.float32) / 255.0
                          for t in range(T)])
    base_policy.reset()
    out = []
    for t in range(T):
        raw_obs = {"observation.state": state[t:t + 1].to(device),   # batch=1:adapter 按 shape[0] 取 b,须保持 (1,·)
                   AGENTVIEW_KEY: img_av[t:t + 1].to(device),
                   WRIST_KEY: img_wr[t:t + 1].to(device)}
        out.append(base_policy.select_action(raw_obs).to("cpu"))
    base_raw = torch.cat(out, dim=0)                                  # (T,7) 原始尺度
    assert base_raw.shape == (T, 7), f"base_policy 每帧应返回 (1,7),得到 cat 后 {tuple(base_raw.shape)}"
    return action_scaler.scale(base_raw)                             # (T,7) 缩放
