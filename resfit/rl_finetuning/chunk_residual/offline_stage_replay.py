"""源 dexmimicgen HDF5 的 sim-replay 取 stage（方案 A 集成层）。

逐帧 set_state_from_flattened 后调训练同款检测器拿瞬时 stage。env 起法与 reset_to
忠实复制自 deps/dexmimicgen/scripts/playback_datasets.py（去掉渲染）。需 robosuite +
dexmimicgen 已注册环境,运行时 MUJOCO_GL=egl。

纯逻辑（latch/reward/stage_id/排序）在 offline_hdf5_buffer.py 已单测;本模块是集成层,
靠 tests/test_offline_stage_replay_smoke.py 在真数据上 smoke 验证。
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import robosuite
import torch
from tensordict import TensorDict

import dexmimicgen  # noqa: F401  注册 dexmg 环境(import 副作用必需)

from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    STATE18_KEYS,
    assemble_state18,
    load_stage_cache,
    save_stage_cache,
    sorted_demo_keys,
    transition_fields,
)
from resfit.rl_finetuning.chunk_residual.stage_detectors import get_stage_detector


def reset_to(env, state):
    """复位到指定 sim 状态。精简自 playback_datasets.reset_to（去掉渲染分支）。

    state 含 "model"（mujoco xml,需先重载场景）和/或 "states"（flatten 状态）。
    """
    if "model" in state:
        ep_meta = json.loads(state["ep_meta"]) if state.get("ep_meta") else {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(state["model"])
        else:
            xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def make_replay_env(dataset_path):
    """从 HDF5 的 env_args 起一个不渲染的 robosuite env（仅用于跑检测器）。

    Returns (env, env_name)。
    """
    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs.pop("env_lang", None)
    return robosuite.make(**env_kwargs), env_meta["env_name"]


def replay_instant_stages(env, states, *, model_file, detector, ep_meta=None) -> np.ndarray:
    """逐帧 set_state + 调检测器 → 该 demo 的瞬时 stage 数组(int8, 长度=帧数)。

    先用 model_file 载入本 demo 场景(物体摆放),再逐帧置态。闩锁/解耦由调用方按
    offline_hdf5_buffer 的纯函数处理(瞬时→闩锁=running-max;stage_id 直接用瞬时)。
    """
    states = np.asarray(states)
    reset_to(env, {"model": model_file, "ep_meta": ep_meta, "states": states[0]})
    stages = np.empty(len(states), dtype=np.int8)
    for i in range(len(states)):
        reset_to(env, {"states": states[i]})
        stages[i] = detector(env)
    return stages


def replay_eef_rel_piece(env, states, *, model_file, ep_meta=None) -> np.ndarray:
    """逐帧 set_state + 从 env 读 eef(sim 实时)与 piece(sim) → 该 demo 的 rel_piece (T,12)。

    eef 用 env._eef0/1_xpos(set_state 后立即正确,已验证 == hdf5 eef)、piece 从 sim 读。
    online(dexmg._rel_piece_info → info["rel_piece"])与 offline **共用 compute_eef_rel_piece_from_env** → 严格同源。
    ③a' object-aware。
    """
    from resfit.rl_finetuning.chunk_residual.object_state import compute_eef_rel_piece_from_env
    states = np.asarray(states)
    reset_to(env, {"model": model_file, "ep_meta": ep_meta, "states": states[0]})
    out = np.empty((len(states), 12), dtype=np.float32)
    for i in range(len(states)):
        reset_to(env, {"states": states[i]})
        out[i] = compute_eef_rel_piece_from_env(env)
    return out


def _hdf5_image_key(lerobot_key: str) -> str:
    """LeRobot 图像键 → HDF5 obs 键,如 observation.images.agentview → agentview_image。"""
    return lerobot_key.split("observation.images.")[-1] + "_image"


def count_offline_transitions(dataset_path, num_demos=None) -> int:
    """预数 offline demo 会产生多少 transition(= Σ(每 demo 帧数 − 1)),用于给 offline_rb
    精确定容(否则 LazyTensorStorage 容量不足会挤掉早期 demo,锚不全)。

    只读 states 的 shape(h5py 取 shape 不载数据),很快;不 replay、不读图。
    """
    total = 0
    with h5py.File(dataset_path, "r") as f:
        demos = sorted_demo_keys(list(f["data"].keys()))
        if num_demos is not None:
            demos = demos[:num_demos]
        for ep in demos:
            total += int(f[f"data/{ep}/states"].shape[0]) - 1
    return total


def precompute_stage_cache(dataset_path, out_path, num_demos=None) -> int:
    """只跑 sim replay 取每条 demo 的瞬时 stage 并存 npz(贵的一次性步骤)。返回 demo 数。

    之后训练用 build_offline_buffer(..., stage_cache=out_path) 秒级读缓存,无需再起 env。
    """
    env, env_name = make_replay_env(dataset_path)
    detector = get_stage_detector(env_name)
    stages = {}
    try:
        with h5py.File(dataset_path, "r") as f:
            demos = sorted_demo_keys(list(f["data"].keys()))
            if num_demos is not None:
                demos = demos[:num_demos]
            for ep in demos:
                grp = f[f"data/{ep}"]
                stages[ep] = replay_instant_stages(
                    env, grp["states"][()], model_file=grp.attrs["model_file"],
                    detector=detector, ep_meta=grp.attrs.get("ep_meta"))
    finally:
        env.close()
    save_stage_cache(out_path, stages)
    return len(stages)


def _demo_base_actions(base_policy, grp, image_keys, action_scaler, device):
    """逐帧顺序复现在线 queue 语义,返回 scale 后的 base_action (T, action_dim)。

    与在线 chunk_env_wrapper._base_chunk_flat(queue,:94-97)对齐:每 demo 先 base_policy.reset()
    清 ACT action queue,再按帧顺序调 select_action(内部自管队列,空了才前向、否则弹队列),
    故必须顺序、不可批。raw_obs 严格对齐 env 运行时 _process_obs:observation.state=原始18维
    float32(未标准化);图像 uint8 HWC → float32 CHW /255。action_scaler.scale 把原始尺度 base
    动作转到与 act_n 同一缩放空间。
    """
    state_raw = assemble_state18({k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS})  # (T,18) 原始
    state_t = torch.as_tensor(np.asarray(state_raw), dtype=torch.float32)
    T = state_t.shape[0]
    if T == 0:
        raise ValueError("_demo_base_actions: demo group is empty (T=0)")
    imgs = {}
    for k in image_keys:
        arr = np.asarray(grp[f"obs/{_hdf5_image_key(k)}"][()])          # (T,H,W,3) uint8
        imgs[k] = torch.as_tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # (T,3,H,W)
    base_policy.reset()
    out = []
    for t in range(T):
        raw_obs = {"observation.state": state_t[t:t + 1].to(device)}
        for k in image_keys:
            raw_obs[k] = imgs[k][t:t + 1].to(device)
        a_raw = base_policy.select_action(raw_obs)                      # (1, action_dim) 原始尺度
        out.append(a_raw.to("cpu"))
    base_raw = torch.cat(out, dim=0)                                    # (T, action_dim)
    return action_scaler.scale(base_raw)                               # (T, action_dim)


def build_offline_buffer(rb, dataset_path, *, action_scaler, state_standardizer,
                         image_keys, bonus, mode, gamma, num_demos=None,
                         stage_cache=None, potential=None, subgoal=None, way_steps=25) -> int:
    """从源 HDF5 灌装 offline demo transition 到 rb,返回新增条数。

    GT-as-base:obs.base_action 与 action 都用缩放后的 GT 动作(残差目标 = 0,把 actor
    锚在 demo 流形)。obs.state=18 维标准化;图像 HWC uint8 → CHW;obs.stage_id=瞬时;
    reward/done/max_stage 由 transition_fields 给(reward 闩锁、stage_id 瞬时,见解耦)。
    transition 结构严格对齐 train_chunk_residual.add_chunk_transition。

    stage_cache 命中则读缓存(不起 env、不 replay);缺失的 demo 才懒建 env replay,并把
    新算的补回缓存。
    """
    cached = load_stage_cache(stage_cache) if (stage_cache and os.path.exists(stage_cache)) else None
    env = None
    detector = None
    new_stages = {}
    added = 0
    # ③a' object-aware:eef_piece value 时,每 demo set_state replay 从 sim 算 raw rel_piece,
    # 喂 Φ 重算 reward(observation.state 仍存 18 维)。stage cache 命中也得起 env 算 rel。
    need_rel = (potential is not None and getattr(potential, "state_mode", "eef") == "eef_piece") or (subgoal is not None)
    _sg_rel_mean = subgoal.rel_mean.cpu() if subgoal is not None else None
    _sg_rel_std = subgoal.rel_std.cpu() if subgoal is not None else None
    try:
        with h5py.File(dataset_path, "r") as f:
            demos = sorted_demo_keys(list(f["data"].keys()))
            if num_demos is not None:
                demos = demos[:num_demos]
            for ep in demos:
                grp = f[f"data/{ep}"]
                states = grp["states"][()]
                if cached is not None and ep in cached:
                    instant = np.asarray(cached[ep])
                    if len(instant) != len(states):
                        raise ValueError(
                            f"stage 缓存与数据不符 {ep}: {len(instant)} vs {len(states)}(缓存过期?)")
                else:
                    if env is None:                       # 懒建:仅 cache miss 才起 env replay
                        env, env_name = make_replay_env(dataset_path)
                        detector = get_stage_detector(env_name)
                    instant = replay_instant_stages(
                        env, states, model_file=grp.attrs["model_file"],
                        detector=detector, ep_meta=grp.attrs.get("ep_meta"))
                    new_stages[ep] = instant
                T = len(instant)
                if T < 2:
                    continue
                obs_arrays = {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS}
                state_n = state_standardizer.standardize(
                    torch.as_tensor(assemble_state18(obs_arrays), dtype=torch.float32)).cpu()
                rel_seq = None
                if need_rel:                          # eef_piece:replay 算 raw rel_piece(T,12)
                    if env is None:                   # 懒建(stage cache 命中时 stage 分支没起 env)
                        env, env_name = make_replay_env(dataset_path)
                        detector = get_stage_detector(env_name)
                    rel_seq = replay_eef_rel_piece(
                        env, states, model_file=grp.attrs["model_file"],
                        ep_meta=grp.attrs.get("ep_meta"))
                fld = transition_fields(instant, bonus=bonus, mode=mode,
                                        gamma=gamma, success=True,
                                        potential=potential, state_seq=state_n,
                                        rel_piece_seq=rel_seq)
                act_n = action_scaler.scale(
                    torch.as_tensor(grp["actions"][()], dtype=torch.float32)).cpu()
                imgs = {k: torch.as_tensor(grp[f"obs/{_hdf5_image_key(k)}"][()])
                        .permute(0, 3, 1, 2).contiguous() for k in image_keys}  # (T,3,84,84)
                sid = torch.as_tensor(fld["stage_id"], dtype=torch.float32)
                nsid = torch.as_tensor(fld["next_stage_id"], dtype=torch.float32)

                subgoal_z = None
                if subgoal is not None:
                    if rel_seq is None:
                        raise RuntimeError("subgoal 模式需 eef_piece rel(env replay 应已算出)")
                    rel_n = (torch.as_tensor(rel_seq, dtype=torch.float32) - _sg_rel_mean) \
                        / _sg_rel_std
                    s30 = torch.cat([state_n, rel_n], dim=-1)            # rel_seq 与 state 等长=T → (T,30)
                    way = np.minimum(np.arange(T) + way_steps, T - 1)        # k 步航点(裁到末态)
                    subgoal_z = subgoal.subgoal_waypoint(s30, s30[way]).cpu()  # (T,10)

                for t in range(T - 1):
                    curr = {"observation.state": state_n[t],
                            "observation.base_action": act_n[t],
                            "observation.stage_id": sid[t:t + 1]}
                    nxt = {"observation.state": state_n[t + 1],
                           "observation.base_action": act_n[t + 1],
                           "observation.stage_id": nsid[t:t + 1]}
                    if subgoal_z is not None:
                        curr["observation.subgoal"] = subgoal_z[t]
                        nxt["observation.subgoal"] = subgoal_z[t + 1]
                    for k in image_keys:
                        curr[k] = imgs[k][t]
                        nxt[k] = imgs[k][t + 1]
                    td = TensorDict({
                        "obs": TensorDict(curr, batch_size=[]),
                        "next": TensorDict({
                            "obs": TensorDict(nxt, batch_size=[]),
                            "done": torch.tensor(bool(fld["done"][t])),
                            "reward": torch.tensor(float(fld["reward"][t]), dtype=torch.float32),
                        }, batch_size=[]),
                        "action": act_n[t],
                        "max_stage": torch.tensor(float(fld["max_stage"][t]), dtype=torch.float32),
                        "_priority": torch.tensor(10.0, dtype=torch.float32),
                    }, batch_size=[]).unsqueeze(0)
                    rb.add(td)
                    added += 1
        if stage_cache is not None and new_stages:        # 新算的 stage 补进缓存
            merged = dict(cached) if cached else {}
            merged.update(new_stages)
            save_stage_cache(stage_cache, merged)
    finally:
        if env is not None:
            env.close()
    return added
