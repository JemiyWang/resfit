"""chunk 级残差 env wrapper(替代 BasePolicyVecEnvWrapper,不修改原文件)。

约定(与 ResFiT 一致):
- base_action 以归一化 [-1,1] chunk 存于 obs["observation.base_action"](展平 L*D)。
- step 收到的是归一化残差 chunk(展平 L*D);combined = clamp(base+residual,-1,1);
  reshape → unscale → 在底层 env 上开环逐步执行;累积奖励;中途 done 则提前结束。
- info["scaled_action"] = combined(展平),供 buffer 存储(与原 wrapper L145 一致)。
仅支持 num_envs==1(训练端 assert 限制)。
"""
from __future__ import annotations

import torch

from resfit.rl_finetuning.chunk_residual.chunk_act_base import get_action_chunk


class ChunkResidualEnvWrapper:
    def __init__(self, vec_env, base_policy, action_scaler, state_standardizer,
                 chunk_length: int):
        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.chunk_length = chunk_length
        assert chunk_length >= 1, "chunk_length must be >= 1"
        self.action_dim = vec_env.action_space.shape[-1]
        self.num_envs = getattr(vec_env, "num_envs", 1)
        self.flat_dim = chunk_length * self.action_dim
        self._last_base_flat = None
        # 取整段 chunk 的入口(测试可在构造后直接 monkeypatch self._get_chunk)
        self._get_chunk = getattr(base_policy, "get_action_chunk", None)

    # ---- 取整段 chunk 并归一化到 [-1,1] 展平 ----
    def _base_chunk_flat(self, raw_obs):
        if self._get_chunk is not None:                      # fake/可替换路径
            chunk_raw = self._get_chunk(raw_obs, self.chunk_length)
        else:
            chunk_raw = get_action_chunk(self.base_policy, raw_obs, self.chunk_length)
        chunk_n = self.action_scaler.scale(chunk_raw)        # [B,L,D] -> [-1,1]
        return chunk_n.reshape(chunk_n.shape[0], -1)         # [B, L*D]

    def _augment(self, raw_obs, base_flat):
        aug = dict(raw_obs)
        aug["observation.base_action"] = base_flat
        aug["observation.state"] = self.state_standardizer.standardize(raw_obs["observation.state"])
        return aug

    def reset(self, **kwargs):
        raw_obs, info = self.vec_env.reset(**kwargs)
        self.base_policy.reset()
        base_flat = self._base_chunk_flat(raw_obs)
        self._last_base_flat = base_flat
        return self._augment(raw_obs, base_flat), info

    def step(self, residual_flat: torch.Tensor):
        """执行一段 chunk(开环逐步),返回 chunk 级 transition。

        注意:对于 done 的 transition,返回的 next_obs 是 autoreset 后新 episode 的
        (已 augment)起始观测;其 Q 值会被 done 掩掉,内容不影响 target。因此本 wrapper
        不透出底层 env 的 final_obs(那是未 augment 的原始 numpy,直接入 buffer 会踩坑)。
        """
        if self._last_base_flat is None:
            raise RuntimeError("Call reset() before step()")
        combined_flat = torch.clamp(self._last_base_flat + residual_flat, -1.0, 1.0)  # [B,L*D]
        b = combined_flat.shape[0]
        combined_chunk = combined_flat.reshape(b, self.chunk_length, self.action_dim)
        env_chunk = self.action_scaler.unscale(combined_chunk)        # [B,L,D] 原始尺度

        total_reward = torch.zeros(b)
        terminated = torch.zeros(b, dtype=torch.bool)
        truncated = torch.zeros(b, dtype=torch.bool)
        last_info = {}
        raw_obs = None
        for t in range(self.chunk_length):
            raw_obs, reward, term, trunc, info = self.vec_env.step(env_chunk[:, t])
            total_reward = total_reward + reward.float().cpu()
            terminated = terminated | term.bool().cpu()
            truncated = truncated | trunc.bool().cpu()
            last_info = info
            if bool((term | trunc).any()):
                break

        if bool((terminated | truncated).any()):
            self.base_policy.reset()
        base_flat = self._base_chunk_flat(raw_obs)
        self._last_base_flat = base_flat

        aug_obs = self._augment(raw_obs, base_flat)
        info = dict(last_info)
        info["scaled_action"] = combined_flat
        return aug_obs, total_reward, terminated, truncated, info

    def render(self):
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name):
        return getattr(self.vec_env, name)
