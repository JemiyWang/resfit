"""自包含 pi0 base 适配器(LIBERO 单臂):把 pi0_libero serve 当 step 级 base policy。
不子类化 kai0 resfit_pi05(py3.10-only),直接基于 openpi_client.WebsocketClientPolicy。
逻辑参考 kai0 pi05_policy_adapter.py:91-177;obs 转换委托 libero_obs.build_libero_serve_obs。"""
from collections import deque

import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.libero_obs import build_libero_serve_obs


class _Cfg:
    def __init__(self, image_features):
        self.image_features = image_features


class LiberoPi05Adapter:
    BASE_KEY = "observation.images.agentview"
    WRIST_KEY = "observation.images.robot0_eye_in_hand"
    STATE_KEY = "observation.state"

    def __init__(self, policy, *, prompt, action_dim=7, execute_horizon=5, device="cpu"):
        self.policy = policy                 # 须有 .infer(obs)->{"actions": ndarray[horizon,dim]}
        self.prompt = prompt
        self.action_dim = int(action_dim)
        self.execute_horizon = int(execute_horizon)
        self.device = device
        self._queues = []                    # per-env deque
        self.config = _Cfg({self.BASE_KEY: None, self.WRIST_KEY: None})
        self._last_prefix_feat = None        # 最近一次 serve infer 返回的 prefix_feat

    @classmethod
    def from_policy(cls, policy, *, prompt, action_dim=7, device="cpu",
                    execute_horizon=5, image_key_map=None):
        # image_key_map 对 LIBERO 固定单臂无需用(键名固定),收下保持与 load_pi05 调用签名兼容
        return cls(policy, prompt=prompt, action_dim=action_dim,
                   execute_horizon=execute_horizon, device=device)

    def _ensure_queues(self, b):
        while len(self._queues) < b:
            self._queues.append(deque())

    def last_prefix_feat(self):
        """最近一次 serve infer 返回的 prefix_feat(queue 边界更新);serve 未透特征则 None。"""
        return self._last_prefix_feat

    def _infer_chunk(self, obs):
        result = self.policy.infer(obs)
        self._last_prefix_feat = result.get("prefix_feat")   # None 当 serve 不透特征
        if "actions" not in result:
            raise ValueError("policy.infer(...) 必须返回含 'actions' 的 mapping")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < self.action_dim:
            raise ValueError(f"policy actions 形状非法或维度 < {self.action_dim}: {actions.shape}")
        if actions.shape[0] == 0:
            raise ValueError("policy actions chunk 至少要有 1 步")
        sliced = actions[:self.execute_horizon, :self.action_dim]
        return [row.copy() for row in sliced]

    def select_action(self, raw_obs):
        b = int(raw_obs[self.STATE_KEY].shape[0])
        self._ensure_queues(b)
        out = []
        for i in range(b):
            if not self._queues[i]:
                obs = build_libero_serve_obs(
                    raw_obs, base_key=self.BASE_KEY, wrist_key=self.WRIST_KEY,
                    state_key=self.STATE_KEY, prompt=self.prompt, env_index=i)
                self._queues[i].extend(self._infer_chunk(obs))
            out.append(self._queues[i].popleft())
        return torch.as_tensor(np.stack(out), dtype=torch.float32, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            for q in self._queues:
                q.clear()
        else:
            for i in env_ids:
                if 0 <= int(i) < len(self._queues):
                    self._queues[int(i)].clear()

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self
