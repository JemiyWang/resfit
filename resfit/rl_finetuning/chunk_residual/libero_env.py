"""LIBERO 单臂 env，适配成 dexmg RobosuiteGymWrapper 同款 gym 契约。
纯转换走 libero_obs;OffScreenRenderEnv 惰性 import(单测用 stub,真集成在 live smoke)。"""
import os

import numpy as np
import gymnasium as gym

from resfit.rl_finetuning.chunk_residual.libero_obs import (
    flip_resize_image, assemble_libero_state, adapt_4tuple)

_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)   # 张开夹爪、不动


def _chw01(hwc_uint8):
    return np.transpose(hwc_uint8, (2, 0, 1)).astype(np.float32) / 255.0


class LiberoGymWrapper(gym.Env):
    """单 (suite,task) LIBERO env → resfit 契约。"""

    def __init__(self, libero_env, *, init_states, num_steps_wait=10, init_idx=0):
        self.env = libero_env
        self.init_states = np.asarray(init_states)
        self.num_steps_wait = int(num_steps_wait)
        self.init_idx = int(init_idx)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = gym.spaces.Dict({})

    def _process(self, obs):
        return {
            "observation.state": assemble_libero_state(obs).reshape(1, 8),
            "observation.images.agentview": _chw01(flip_resize_image(obs["agentview_image"])),
            "observation.images.robot0_eye_in_hand": _chw01(flip_resize_image(obs["robot0_eye_in_hand_image"])),
        }

    def reset(self, *, seed=None, options=None):
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[self.init_idx])
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.env.step(_DUMMY_ACTION)
        return self._process(obs), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action, np.float32))
        obs2, r, term, trunc, info = adapt_4tuple(obs, reward, done, info)
        return self._process(obs2), float(r), bool(term), bool(trunc), info

    def render(self, *a, **k):
        return None

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def seed(self, s=None):
        return [s]


def make_libero_env(suite, task_id, *, camera_size=256, render_gpu_device_id=0):
    """惰性构造真 LIBERO env(live smoke / 训练用)。返回 (LiberoGymWrapper, prompt)。"""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_size,
                             camera_widths=camera_size, render_gpu_device_id=render_gpu_device_id)
    init_states = task_suite.get_task_init_states(task_id)
    return LiberoGymWrapper(env, init_states=init_states), task.language


def create_libero_vectorized_env(suite, task_id, num_envs, device="cpu",
                                 video_key="observation.images.agentview", debug=False):
    """镜像 dexmg.create_vectorized_env:AsyncVectorEnv + EGL 逻辑号映射 + VectorizedEnvWrapper。
    注:真 libero,仅新 env / live smoke 跑。dexmg 符号签名已核对:
    - cuda_to_egl_device_id(cuda_device_id: int)：单 int,逻辑号(env_id % num_visible_gpus)在调用侧算好再传。
    - VectorizedEnvWrapper(vec_env, video_key, device)：video_key 是必填位置参,不是只给 device。
    """
    from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper, cuda_to_egl_device_id

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None:
        visible = [int(x) for x in cuda_visible.split(",") if x.strip() != ""]
    else:
        import torch
        visible = list(range(torch.cuda.device_count()))
    num_visible_gpus = len(visible) if visible else 1

    def _factory(env_id):
        # 渲染设备用 CUDA_VISIBLE_DEVICES 掩码后的"逻辑号",再过 cuda_to_egl_device_id
        # 映射到同物理卡的 EGL 下标(否则渲染会漏到别的物理卡,见 dexmg.py 注释)。
        logical_id = env_id % num_visible_gpus
        egl = cuda_to_egl_device_id(logical_id)
        env, _ = make_libero_env(suite, task_id, render_gpu_device_id=egl)
        return env

    env_fns = [lambda i=i: _factory(i) for i in range(num_envs)]
    if debug:
        vec_env = gym.vector.SyncVectorEnv(
            env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns, shared_memory=True, copy=True, context="spawn",
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    return VectorizedEnvWrapper(vec_env, video_key, device)
