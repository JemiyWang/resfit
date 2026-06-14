"""LIBERO 单臂 env，适配成 dexmg RobosuiteGymWrapper 同款 gym 契约。
纯转换走 libero_obs;OffScreenRenderEnv 惰性 import(单测用 stub,真集成在 live smoke)。"""
import os

import numpy as np
import gymnasium as gym

from resfit.rl_finetuning.chunk_residual.libero_obs import (
    flip_resize_image, assemble_libero_state, adapt_4tuple)

_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)   # 张开夹爪、不动

# LIBERO 各 suite 的标准 episode 步数上限(对齐 openpi rollout_libero;均远 < robosuite horizon=1000)。
# 命门:LIBERO 的 bddl_base_domain.step 把 done 覆盖成 _check_success()(只在成功时报 done),
# 超时不报 done;但 robosuite ignore_done=False、horizon=1000,到 1000 步内部置 done、第 1001 步
# 会抛 "executing action in terminated episode"。故必须自己在 max_steps(<1000)处 truncate,
# 让 episode 超时正常结束(autoreset 重置)+ 永不撞 robosuite 守卫。
LIBERO_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}
LIBERO_MAX_STEPS_DEFAULT = 520


def _chw01(hwc_uint8):
    return np.transpose(hwc_uint8, (2, 0, 1)).astype(np.float32) / 255.0


class LiberoGymWrapper(gym.Env):
    """单 (suite,task) LIBERO env → resfit 契约。"""

    def __init__(self, libero_env, *, init_states, num_steps_wait=10, init_idx=0,
                 max_steps=LIBERO_MAX_STEPS_DEFAULT):
        self.env = libero_env
        self.init_states = np.asarray(init_states)
        self.num_steps_wait = int(num_steps_wait)
        self.init_idx = int(init_idx)
        self.max_steps = int(max_steps)   # 超时截断阈值(< robosuite horizon 1000)
        self._step_count = 0
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        # 必须是真 Dict(非空):AsyncVectorEnv 靠 observation_space 组装批后 obs,
        # 空 Dict 会让批后 obs 变成 {} → 丢掉所有键(三键须与 _process 输出一致)。
        self.observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-np.inf, np.inf, (8,), np.float32),
            "observation.images.agentview": gym.spaces.Box(0.0, 1.0, (3, 84, 84), np.float32),
            "observation.images.robot0_eye_in_hand": gym.spaces.Box(0.0, 1.0, (3, 84, 84), np.float32),
        })

    def _process(self, obs):
        return {
            "observation.state": assemble_libero_state(obs).reshape(8),   # per-env 1-D (8,),与 dexmg 对齐(AsyncVectorEnv stack 成 (N,8))
            # agent ViT(min_vit)要 84×84;pi0 serve 用的 224 由 build_libero_serve_obs 再放大
            "observation.images.agentview": _chw01(flip_resize_image(obs["agentview_image"], 84)),
            "observation.images.robot0_eye_in_hand": _chw01(flip_resize_image(obs["robot0_eye_in_hand_image"], 84)),
        }

    def reset(self, *, seed=None, options=None):
        self._step_count = 0                       # 新 episode 步数清零(超时截断用)
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[self.init_idx])
        for _ in range(self.num_steps_wait):       # 落稳的 dummy 步不计入 episode 预算
            obs, _, _, _ = self.env.step(_DUMMY_ACTION)
        return self._process(obs), {}

    def step(self, action):
        # action 可能是 cuda tensor(--device cuda 时);robosuite 要 numpy。对齐 dexmg RobosuiteGymWrapper.step。
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        action = np.asarray(action, np.float32)
        if action.ndim > 1:
            action = action[0]
        obs, reward, done, info = self.env.step(action)
        obs2, r, term, trunc, info = adapt_4tuple(obs, reward, done, info)
        # LIBERO 只在成功时报 done;超时不报 → 自己在 max_steps 处 truncate,
        # 否则会撞 robosuite horizon=1000 的守卫崩(见模块顶部 LIBERO_MAX_STEPS 注释)。
        self._step_count += 1
        if self._step_count >= self.max_steps:
            trunc = True
        return self._process(obs2), float(r), bool(term), bool(trunc), info

    # v1 不录视频(--no-save-videos);需要则用 self.env.sim.render 返帧,live smoke 时再实现。
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
    # torch>=2.6 默认 weights_only=True,拒绝 LIBERO init_states 里的 numpy pickle;
    # 这是 LIBERO 自带的可信文件,临时 shim 成 weights_only=False 加载。
    import torch as _torch
    _orig_load = _torch.load
    _torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    try:
        init_states = task_suite.get_task_init_states(task_id)
    finally:
        _torch.load = _orig_load
    max_steps = LIBERO_MAX_STEPS.get(suite, LIBERO_MAX_STEPS_DEFAULT)   # 按 suite 取标准超时步数
    return LiberoGymWrapper(env, init_states=init_states, max_steps=max_steps), task.language


def create_libero_vectorized_env(suite, task_id, num_envs, device="cpu",
                                 video_key="observation.images.agentview", debug=False):
    """镜像 dexmg.create_vectorized_env:AsyncVectorEnv + EGL 逻辑号映射 + VectorizedEnvWrapper。
    注:真 libero,仅新 env / live smoke 跑。dexmg 符号签名已核对:
    - cuda_to_egl_device_id(cuda_device_id: int)：单 int,逻辑号(env_id % num_visible_gpus)在调用侧算好再传。
    - VectorizedEnvWrapper(vec_env, video_key, device)：video_key 是必填位置参,不是只给 device。
    """
    from resfit.rl_finetuning.chunk_residual.vec_env_util import VectorizedEnvWrapper

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None:
        visible = [int(x) for x in cuda_visible.split(",") if x.strip() != ""]
    else:
        import torch
        visible = list(range(torch.cuda.device_count()))
    num_visible_gpus = len(visible) if visible else 1

    def _factory(env_id):
        # robosuite 1.4.1(LIBERO)的 EGL 约定:MUJOCO_EGL_DEVICE_ID 必须 ∈ CUDA_VISIBLE_DEVICES
        # 的物理卡号(binding_utils 有此断言),与 dexmg 的 robosuite 1.5(EGL 枚举下标,
        # cuda_to_egl_device_id)不同。EGL 按物理卡枚举、不受 CUDA mask 影响,故直接用物理号。
        phys = visible[env_id % num_visible_gpus] if visible else 0
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(phys)
        env, _ = make_libero_env(suite, task_id, render_gpu_device_id=phys)
        return env

    env_fns = [lambda i=i: _factory(i) for i in range(num_envs)]
    if debug:
        vec_env = gym.vector.SyncVectorEnv(
            env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        # shared_memory=False:observation_space 是空 Dict({}),shared_memory 会按声明的(空)键
        # 写回、丢掉真实 3 个 obs 键;False 走 pickle 透传,obs 才完整。
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns, shared_memory=False, copy=True, context="spawn",
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    return VectorizedEnvWrapper(vec_env, video_key, device)
