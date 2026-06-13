# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""robosuite-无关的向量化 env 工具。

从 dexmg.py 原样抽出 cuda_to_egl_device_id / VectorizedEnvWrapper(只依赖
gym/torch/numpy/ctypes),让 libero 路(robosuite-1.4 env)能复用而不触发
dexmg.py 顶层的 robosuite-1.5 import。本模块顶层不得 import
robosuite/dexmg/dexmimicgen。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

# ----------------------------------------------------------------------------
# EGL 渲染设备 != CUDA 计算设备。
# MuJoCo/robosuite 的 EGL 后端用 eglQueryDevicesEXT 的枚举下标来选渲染 GPU,而这个
# 下标顺序和 CUDA / nvidia-smi 的 GPU 编号并不一致(本机上 EGL 下标 0 物理对应
# nvidia-smi gpu3、下标 2 才对应 gpu0)。直接把 CUDA 设备号当 EGL 下标用,会把渲染
# 开到另一张卡上 —— 表现为:训练在指定卡、却莫名其妙在别的卡上占一份 ~400MiB/env
# 的显存且 SM 利用率 0%(看着像残留进程)。
# 下面用 EGL_CUDA_DEVICE_NV 属性把"CUDA 设备号"翻译成"指向同一张物理卡的 EGL 下标",
# 使渲染始终跟随计算卡。查询失败时原样返回(退回旧行为,绝不阻断训练)。
# 全机映射表/诊断脚本见 tools/egl_gpu_map.py。
# ----------------------------------------------------------------------------
_EGL_CUDA_TO_EGL_INDEX: dict[int, int] = {}


def cuda_to_egl_device_id(cuda_device_id: int) -> int:
    """把 CUDA 设备号映射到指向同一物理 GPU 的 EGL 枚举下标(= MUJOCO_EGL_DEVICE_ID)。"""
    if cuda_device_id in _EGL_CUDA_TO_EGL_INDEX:
        return _EGL_CUDA_TO_EGL_INDEX[cuda_device_id]
    egl_index = cuda_device_id  # 兜底:探测失败就退回原值(旧行为)
    try:
        import ctypes  # noqa: PLC0415

        egl = ctypes.CDLL("libEGL.so.1")
        egl.eglGetProcAddress.restype = ctypes.c_void_p
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]

        def _fn(name, restype, argtypes):
            addr = egl.eglGetProcAddress(name.encode())
            return ctypes.CFUNCTYPE(restype, *argtypes)(addr) if addr else None

        Dev = ctypes.c_void_p
        query_devices = _fn(
            "eglQueryDevicesEXT", ctypes.c_uint,
            [ctypes.c_int, ctypes.POINTER(Dev), ctypes.POINTER(ctypes.c_int)])
        query_attrib = _fn(
            "eglQueryDeviceAttribEXT", ctypes.c_uint,
            [Dev, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)])
        if query_devices and query_attrib:
            max_devices = 32
            devices = (Dev * max_devices)()
            num = ctypes.c_int(0)
            query_devices(max_devices, devices, ctypes.byref(num))
            EGL_CUDA_DEVICE_NV = 0x323A
            for i in range(num.value):
                val = ctypes.c_ssize_t(-1)
                if query_attrib(devices[i], EGL_CUDA_DEVICE_NV, ctypes.byref(val)) and val.value == cuda_device_id:
                    egl_index = i
                    break
    except Exception:  # noqa: BLE001  渲染设备探测失败不应阻断训练
        pass
    _EGL_CUDA_TO_EGL_INDEX[cuda_device_id] = egl_index
    return egl_index


class VectorizedEnvWrapper:
    """Simple wrapper around gymnasium vectorized environments to add rendering capability."""

    def __init__(
        self, vec_env: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv, video_key: str, device: str = "cpu"
    ):
        self.vec_env = vec_env
        self.video_key = video_key
        self._last_obs = None
        self.device = device

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        self._last_obs = obs
        obs = self._convert_obs_to_torch(obs, self.device)
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.vec_env.step(actions)
        self._last_obs = obs

        # Convert to torch tensors
        obs = self._convert_obs_to_torch(obs, self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool)

        return obs, rewards, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return RGB frames from all environments (num_envs, H, W, 3, uint8) for video recording."""
        frames: np.ndarray | None = self.vec_env.render()
        if frames is None:
            raise RuntimeError("No frames returned from vectorized environment")
        return frames

    @property
    def fps(self):
        return self.vec_env.metadata["render_fps"]

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)

    def _convert_obs_to_torch(self, obs, device):
        """Convert numpy observations from vectorized env to PyTorch tensors for policy."""
        if isinstance(obs, dict):
            torch_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    torch_obs[key] = torch.from_numpy(value).to(device)
                else:
                    torch_obs[key] = value
            return torch_obs
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).to(device)
        return obs
