"""LIBERO obs/动作纯逻辑(无 libero/openpi_client 依赖,residual 环境可单测)。
语义对齐 chj/openpi/scripts/rollout_libero.py。"""
import json
import math
import numpy as np
from PIL import Image


def quat2axisangle(quat) -> np.ndarray:
    """robosuite (x,y,z,w) → 3 维轴角。对齐 rollout_libero._quat2axisangle。"""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def resize_with_pad(img, h, w) -> np.ndarray:
    """保纵横比缩放 + 居中 pad 到 (h,w);uint8 HWC。"""
    img = np.asarray(img)
    ih, iw = img.shape[:2]
    ratio = min(h / ih, w / iw)
    nh, nw = max(1, int(ih * ratio)), max(1, int(iw * ratio))
    resized = np.asarray(Image.fromarray(img.astype(np.uint8)).resize((nw, nh), Image.BILINEAR))
    if resized.ndim == 2:
        resized = np.stack([resized] * 3, axis=-1)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    top, left = max(0, (h - nh) // 2), max(0, (w - nw) // 2)
    out[top:top + nh, left:left + nw] = resized[:, :, :3]
    return out


def _to_hwc_uint8(img) -> np.ndarray:
    """CHW/HWC → HWC uint8。float 输入按 [0,1] 处理(本管线 env 出的是 CHW float[0,1],
    见 libero_env._chw01),与上游 image_tools 的 [-1,1] 约定不同,故不复用上游。"""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if np.issubdtype(a.dtype, np.floating):
        a = (255.0 * a).clip(0, 255).astype(np.uint8)
    return a.astype(np.uint8)


def flip_resize_image(img) -> np.ndarray:
    """LIBERO 原图 [::-1,::-1] 翻转 + resize_with_pad 224。对齐 rollout_libero:425-430。"""
    a = _to_hwc_uint8(img)
    a = np.ascontiguousarray(a[::-1, ::-1])
    return resize_with_pad(a, 224, 224)


def assemble_libero_state(obs) -> np.ndarray:
    """eef_pos(3)+quat2axisangle(eef_quat)(3)+gripper_qpos(2)=8 维。对齐 rollout_libero:436-440。"""
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], np.float32).reshape(-1)[:3],
        quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"], np.float32).reshape(-1)[:2],
    ]).astype(np.float32)


def adapt_4tuple(obs, reward, done, info):
    """LIBERO 4-tuple → gym 5-tuple。成功(reward==1.0)→terminated;done 非成功→truncated。"""
    success = bool(np.asarray(reward).reshape(-1)[0] == 1.0)
    info = dict(info or {}); info["success"] = success
    return obs, reward, success, (bool(done) and not success), info


def build_libero_serve_obs(raw_obs, *, base_key, wrist_key, state_key, prompt, env_index=0):
    """从(批后)raw_obs 取 env_index,出 pi0_libero serve 扁平 schema。env 侧已翻转,这里只 resize。"""
    def _img(key):
        arr = np.asarray(raw_obs[key])
        arr = arr[env_index] if arr.ndim == 4 else arr
        return resize_with_pad(_to_hwc_uint8(arr), 224, 224)
    st = np.asarray(raw_obs[state_key])
    st = st[env_index] if st.ndim == 2 else st
    return {"observation/image": _img(base_key), "observation/wrist_image": _img(wrist_key),
            "observation/state": st.astype(np.float32).reshape(-1)[:8], "prompt": str(prompt)}


def load_libero_norm_stats(stats_json_path) -> dict:
    """读 LeRobot meta/stats.json → {'action_mean','action_std','state_mean','state_std'} float32。
    键名兜底:action 取 'action'|'actions';state 取 'observation.state'|'state'。"""
    with open(stats_json_path) as f:
        s = json.load(f)
    def pick(*keys):
        for k in keys:
            if k in s:
                return s[k]
        raise KeyError(f"stats.json 缺键,试过 {keys}")
    a = pick("action", "actions"); st = pick("observation.state", "state")
    return {
        "action_mean": np.asarray(a["mean"], np.float32), "action_std": np.asarray(a["std"], np.float32),
        "state_mean": np.asarray(st["mean"], np.float32), "state_std": np.asarray(st["std"], np.float32),
    }
