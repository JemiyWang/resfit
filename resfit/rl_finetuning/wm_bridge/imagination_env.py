"""想象空间的向量化 env,鸭子类型 VectorizedEnvWrapper。

★ 攒批机制:ChunkResidualEnvWrapper(chunk_env_wrapper.py:181-182)是逐时间步循环,
   chunk_length=50 时会调 step() 50 次、每次 1 个动作。而 WM 要 50 个动作一次性打包成
   25 个 token。故本 env 内部攒动作,第 50 次调用才点火 WM。

   前 49 次返回缓存 obs / reward=0 / term=trunc=False 是安全的,因为 wrapper 对中间步的
   返回值本就不消费:raw_obs 与 last_info 每轮被覆盖(只有最后一轮进 :219/:222),
   reward 是累加(加 0 无影响),term/trunc 是 OR 累积(False 无影响)。

★ reward = PBRS 端点差 gamma*Phi(psi_next) - Phi(psi_prev)。不用 RISE Eq.(2) 的
   25 帧均值:均值形式跨 chunk 不 telescoping,正是 potsubgoal 塌方的根因(farming 伪奖励);
   且 prefix_feat 无纯特征端点,每个 psi 要跑一次完整 pi05 扩散推理,均值形式要贵 25 倍。

★ truncated 时绝不把 Phi 置零。仓库既有 potential_shaping(chunk_env_wrapper.py:32)约定
   done 时 Phi(s')=0,那是为真终止态设计的;想象段的 truncated 是人为视界切断。照抄会让
   两段回报退化成常数 -Phi_0,学习信号全丢。见 test_phi_is_not_zeroed_at_truncation。
"""
from __future__ import annotations

import numpy as np
import torch

from resfit.rl_finetuning.wm_bridge.state_tracker import ProprioTracker
from resfit.rl_finetuning.wm_bridge.wm_driver import (
    ACTION_DIM, CHUNK_LENGTH, N_PREVIOUS, WM_TOKEN_STEPS,
    build_obs_window, frames_to_obs_images, pack_act_tokens,
    split_predicted_frames,
)


class _ActionSpace:
    def __init__(self, dim):
        self.shape = (dim,)


class ImaginationVecEnv:
    def __init__(self, *, wm, base, scorer, sampler, normalizer,
                 gamma: float = 0.995, max_segments: int = 2,
                 num_denois_steps: int = 10):
        self.wm = wm
        self.base = base
        self.scorer = scorer
        self.sampler = sampler
        self.normalizer = normalizer
        self.gamma = float(gamma)
        self.max_segments = int(max_segments)
        self.num_denois_steps = int(num_denois_steps)

        self.num_envs = 1
        self.action_space = _ActionSpace(ACTION_DIM)

        self._window = None          # (V,C,4,H,W) [-1,1]
        self._caption = None
        self._tracker = None
        self._phi_prev = None
        self._act_buf = []
        self._seg_step = 0
        self._token = 0
        self._obs_cache = None

    # ---------- obs 构造 ----------

    def _build_obs(self, images: dict) -> dict:
        obs = dict(images)
        obs["observation.state"] = torch.from_numpy(
            self._tracker.proprio).unsqueeze(0)
        obs["_wm_native_frames"] = self._window[:, :, -1].copy()   # (V,C,H,W) 末帧
        obs["_wm_window_token"] = self._token
        return obs

    def _obs_from_window(self) -> dict:
        # 窗口末帧作为当前观测
        images = frames_to_obs_images(torch.from_numpy(self._window),
                                      t_index=N_PREVIOUS - 1)
        return self._build_obs(images)

    # ---------- 生命周期 ----------

    def reset(self, **kwargs):
        st = self.sampler.sample()
        self._window = np.asarray(st.obs_window, dtype=np.float32)
        self._caption = st.caption
        self._tracker = ProprioTracker(st.proprio)
        self._act_buf = []
        self._seg_step = 0
        self._token += 1

        self.base.reset()
        obs = self._obs_from_window()
        _, psi = self.base.query(obs)
        self._phi_prev = self.scorer.phi(psi, self._tracker.proprio)
        self._obs_cache = obs
        return obs, {}

    def step(self, action):
        a = np.asarray(
            action.detach().cpu().numpy() if isinstance(action, torch.Tensor)
            else action, dtype=np.float32).reshape(-1)[:ACTION_DIM]
        assert a.shape == (ACTION_DIM,), \
            f"动作须 ({ACTION_DIM},),got {a.shape}"
        self._act_buf.append(a)

        if len(self._act_buf) < CHUNK_LENGTH:
            return (self._obs_cache,
                    torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                    torch.zeros(1, dtype=torch.bool), {})

        # ---- 第 50 次:点火 ----
        actions = np.stack(self._act_buf, axis=0)      # (50,16) 物理空间
        self._act_buf = []

        act_tokens = pack_act_tokens(actions, self.normalizer)
        obs_in = torch.from_numpy(self._window).to(torch.bfloat16)
        out = self.wm.infer(obs=obs_in, act_tokens=act_tokens,
                            prompt=self._caption,
                            num_denois_steps=self.num_denois_steps)
        # 坑6:custom_pipeline.py:997 退出时会把模块留在 train 模式
        inner = getattr(self.wm, "transformer", None)
        if inner is not None:
            inner.eval()

        video = out["video"]
        if video.ndim == 6:                            # (b,v,c,t,h,w) → 取 b=0
            video = video[0]
        pred = split_predicted_frames(video.float())   # (V,C,25,H,W)

        self._tracker.advance(actions)
        self._window = build_obs_window(
            pred, range(WM_TOKEN_STEPS - N_PREVIOUS, WM_TOKEN_STEPS)).numpy()
        self._token += 1

        obs = self._obs_from_window()
        _, psi = self.base.query(obs)
        phi_next = self.scorer.phi(psi, self._tracker.proprio)

        # ★ PBRS 端点差。truncated 时也不置零 phi_next(见 module docstring)
        reward = self.gamma * phi_next - self._phi_prev
        self._phi_prev = phi_next

        self._seg_step += 1
        terminated = False
        truncated = self._seg_step >= self.max_segments

        self._obs_cache = obs
        return (obs,
                torch.tensor([reward], dtype=torch.float32),
                torch.tensor([terminated], dtype=torch.bool),
                torch.tensor([truncated], dtype=torch.bool),
                {})

    def render(self):
        raise NotImplementedError("想象空间不支持 render")

    def close(self):
        return None
