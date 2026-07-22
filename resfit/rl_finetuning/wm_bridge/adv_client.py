"""AdvServeClient —— fake_eval 用的优势估计器打分器,通过 websocket 调 adv_serve(adv_serve.py)。

value_block(RISE value, ~15GB)在 rise_venv,训练/wm_bridge 在 residual env,一个进程装不下;
故 value 包成 serve,本客户端鸭子类型 `adv_scorer.score_frames(frames) -> np.ndarray (T,)`
(与 fake_eval 里 `adv_scorer.score_frames(frames)` 的调用一致,spec §6.7)。

传输:frames 转 np.float32 发出(msgpack_numpy 不支持 bf16);响应 values 转 np。
"""
from __future__ import annotations

import numpy as np
import torch


class AdvServeClient:
    def __init__(self, host="127.0.0.1", port=8002, prompt="build block"):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        # 关 keepalive ping:首批帧慢(value 大模型),别被心跳超时断连(对齐 wm_client)
        import websockets.sync.client as _wsc
        if not getattr(_wsc.connect, "_no_ping_patched", False):
            _orig = _wsc.connect

            def _no_ping(*a, **k):
                k.setdefault("ping_interval", None)
                return _orig(*a, **k)

            _no_ping._no_ping_patched = True
            _wsc.connect = _no_ping
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.prompt = prompt

    def score_frames(self, frames, prompt=None) -> np.ndarray:
        """frames: (T,V,C,H,W) ∈ [-1,1](torch 或 np);→ (T,) value ∈ [-1,1](≈1=成功)。"""
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().float().numpy()
        frames = np.asarray(frames, dtype=np.float32)
        req = {"frames": frames, "prompt": str(prompt if prompt is not None else self.prompt)}
        resp = self.client.infer(req)
        return np.array(resp["values"], dtype=np.float32, copy=True).reshape(-1)
