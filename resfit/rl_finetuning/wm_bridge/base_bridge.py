"""kai0 serve 的想象空间 shim。

替换 resfit.lerobot.policies.pi05.load_pi05_base_policy 的返回物(不是替换
build_base_policy —— 后者在 trainer 自身,runpy 以 __main__ 跑会造新模块对象,patch 不到)。

★ 按窗口 token 缓存。同一个观测窗口会被请求两次:
   ① ImaginationVecEnv 自己要 ψ 算 reward
   ② wrapper 在 chunk_env_wrapper.py:219 调 _base_chunk_flat 要基座动作
   两次针对同一窗口,缓存后每 chunk 只需 1 次 serve 调用 —— ψ 完全是基座调用的副产物。

★ 用原生分辨率帧,不用 84×84。实机部署时 kai0 吃原生帧,想象空间须与部署一致;
   喂降采样帧会人为削弱基座,反而抬高残差的相对增益,是论文的效度威胁。
"""
from __future__ import annotations

import numpy as np
import torch

from resfit.rl_finetuning.wm_bridge.teleavatar_policy_state import (
    map_teleavatar_policy_state,
)
from resfit.rl_finetuning.wm_bridge.wm_driver import (
    ACTION_DIM, CAMERA_KEYS, CHUNK_LENGTH,
)


class _BaseConfig:
    """鸭子类型 lerobot policy config:trainer 只用 .image_features 的 keys()。"""

    def __init__(self, image_keys):
        self.image_features = {k: None for k in image_keys}


class Kai0ImaginationBase:
    def __init__(
        self,
        client,
        *,
        prompt: str,
        action_dim: int = ACTION_DIM,
        policy_state_dim: int = ACTION_DIM,
    ):
        self.client = client
        self.prompt = prompt
        self.action_dim = action_dim
        self.policy_state_dim = policy_state_dim
        self.config = _BaseConfig(CAMERA_KEYS)
        self.call_count = 0
        self._cache_token = None
        self._cache = None

    def reset(self):
        """想象段之间不需要清缓存(token 单调递增,天然失效)。"""
        return None

    def _serve_obs(self, raw_obs) -> dict:
        # ★ 复用建缓存时已验证的 block kai0 嵌套 schema(build_teleavatar_serve_obs):
        #   {"state", "images":{裸cam键: CHW uint8@224}, "prompt"}。与 pi05_block_awbc
        #   serve 实测匹配(建 ψ 缓存跑通)。base_bridge 早先自拼的扁平 schema 是猜的,已废弃。
        # ★ 值域:_wm_native_frames 是 [-1,1](imagination_env 统一,reset窗与WM输出同),
        #   而 build_teleavatar_serve_obs 的 _to_hwc_uint8 按 [0,1] 处理 float → 先转 [0,1]。
        from resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve import (
            build_teleavatar_serve_obs)
        def _np(x):                                        # obs.state 可能是 cuda tensor
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        native = _np(raw_obs["_wm_native_frames"]).astype(np.float32)
        assert native.shape[0] == len(CAMERA_KEYS), \
            f"原生帧视角数须 {len(CAMERA_KEYS)},got {native.shape[0]}"
        imgs01 = (native + 1.0) * 0.5                       # [-1,1] → [0,1]
        images = {CAMERA_KEYS[i].split(".")[-1]: imgs01[i]
                  for i in range(len(CAMERA_KEYS))}
        state = map_teleavatar_policy_state(
            _np(raw_obs["observation.state"]),
            self.policy_state_dim,
        )
        return build_teleavatar_serve_obs(images, state, self.prompt)

    def query(self, raw_obs):
        """→ (actions_physical (50,16), psi)。同一窗口 token 只打一次 serve。"""
        token = raw_obs.get("_wm_window_token")
        assert token is not None, "raw_obs 缺 _wm_window_token(ImaginationVecEnv 须填)"
        if token == self._cache_token and self._cache is not None:
            return self._cache

        result = self.client.infer(self._serve_obs(raw_obs))
        self.call_count += 1

        psi = result.get("prefix_feat")
        if psi is None:
            raise RuntimeError(
                "kai0 serve 未透出 prefix_feat —— Kai0HiqlScorer 无法工作。"
                "serve 须以透出 prefix_feat 的方式启动。")

        actions = np.asarray(result["actions"], dtype=np.float32)
        assert actions.shape == (CHUNK_LENGTH, self.action_dim), \
            f"serve 返回动作须 ({CHUNK_LENGTH},{self.action_dim}),got {actions.shape}"

        self._cache_token = token
        self._cache = (actions, np.asarray(psi, dtype=np.float32).reshape(-1))
        return self._cache

    def get_action_chunk(self, raw_obs, chunk_length: int) -> torch.Tensor:
        assert chunk_length == CHUNK_LENGTH, \
            f"想象路只支持 chunk_length={CHUNK_LENGTH},got {chunk_length}"
        actions, _ = self.query(raw_obs)
        state = raw_obs.get("observation.state")
        device = state.device if isinstance(state, torch.Tensor) else "cpu"
        return torch.from_numpy(actions).unsqueeze(0).to(device)

    def select_action(self, raw_obs) -> torch.Tensor:
        """兼容 queue 模式(chunk_length=1):从缓存的 50 动作里按窗口逐个分发。

        想象训练使用 get_action_chunk；保留本入口供旧配置/诊断使用。本 shim 用内部
        分发游标模拟：同一窗口(_wm_window_token 不变)内，第 i 次调用返回第 i 个动作，
        50 个动作源自该窗口的 1 次 kai0 调用(query 缓存);窗口推进(点火后)则重查、重置游标。
        返回 (1, action_dim) 物理动作,供 wrapper 的 action_scaler.scale。
        """
        actions, _ = self.query(raw_obs)                   # 同窗口缓存,1 次 kai0/窗口
        token = raw_obs.get("_wm_window_token")
        if token != getattr(self, "_dispense_token", object()):
            self._dispense_token = token
            self._dispense_idx = 0
        i = min(self._dispense_idx, CHUNK_LENGTH - 1)      # 越界兜底(不应发生)
        self._dispense_idx += 1
        # ★ 与 obs 同设备:真实 ACT 基座 .to(device) 后返回 cuda,action_scaler.scale 及下游
        #   agent.act 全在 device 上;本 shim 从 raw_obs 自身推设备(fake base 不经 .to())。
        st = raw_obs.get("observation.state")
        device = st.device if isinstance(st, torch.Tensor) else "cpu"
        return torch.from_numpy(actions[i]).unsqueeze(0).to(device)   # (1, action_dim)
