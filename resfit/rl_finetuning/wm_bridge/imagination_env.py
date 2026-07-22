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
                 num_denois_steps: int = 10, device: str = "cpu"):
        self.wm = wm
        self.base = base
        self.scorer = scorer
        self.sampler = sampler
        self.normalizer = normalizer
        self.gamma = float(gamma)
        self.max_segments = int(max_segments)
        self.num_denois_steps = int(num_denois_steps)
        self.device = device        # obs 张量须与 QAgent encoder 同设备(trainer 传 cuda)

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
        self._pred_frames = None      # 非 None 时收集每 chunk 的 WM 预测帧(adv rollout 用)

    # ---------- obs 构造 ----------

    def _build_obs(self, images: dict) -> dict:
        # 图像/state 张量搬到 device(与 QAgent encoder 同设备);旁路键(native_frames/token)留 CPU
        obs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
               for k, v in images.items()}
        obs["observation.state"] = torch.from_numpy(
            self._tracker.proprio).unsqueeze(0).to(self.device)
        obs["_wm_native_frames"] = self._window[:, :, -1].copy()   # (V,C,H,W) 末帧(numpy,给base_bridge)
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

        if self._pred_frames is not None:              # adv rollout:收 WM 预测帧供逐帧打分
            pf = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
            self._pred_frames.append(pf.astype(np.float32))   # (V,C,25,H,W) ∈ [-1,1]

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

    # ---------- adv rollout 支持(eval 逐帧打分,spec §6.7) ----------

    def start_pred_collection(self):
        self._pred_frames = []

    def collect_pred_frames(self) -> np.ndarray:
        """→ (T,V,C,H,W) ∈ [-1,1],T=本次 rollout 已点火 chunk 数×25;关闭收集。"""
        chunks = self._pred_frames or []
        self._pred_frames = None
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        arr = np.concatenate(chunks, axis=2)            # (V,C,ΣT,H,W)
        return np.transpose(arr, (2, 0, 1, 3, 4)).astype(np.float32)  # (T,V,C,H,W)

    def snapshot_state(self) -> dict:
        """快照可变状态(含 base 缓存)。eval rollout 会 reset/step 本 env(与训练共享同一实例),
        rollout 前 snapshot、后 restore,防污染训练轨迹。"""
        import copy as _copy
        b = self.base
        return {
            "_window": None if self._window is None else self._window.copy(),
            "_caption": self._caption,
            "_tracker": _copy.deepcopy(self._tracker),
            "_phi_prev": self._phi_prev,
            "_act_buf": list(self._act_buf),
            "_seg_step": self._seg_step,
            "_token": self._token,
            "_obs_cache": self._obs_cache,
            "_pred_frames": self._pred_frames,
            "base": {k: getattr(b, k, None) for k in
                     ("_cache_token", "_cache", "_dispense_token", "_dispense_idx")},
        }

    def restore_state(self, snap: dict):
        for k in ("_window", "_caption", "_tracker", "_phi_prev", "_act_buf",
                  "_seg_step", "_token", "_obs_cache", "_pred_frames"):
            setattr(self, k, snap[k])
        for k, v in snap["base"].items():
            setattr(self.base, k, v)

    def render(self):
        raise NotImplementedError("想象空间不支持 render")

    def close(self):
        return None
