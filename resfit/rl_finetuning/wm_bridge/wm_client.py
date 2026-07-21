"""WmServeClient —— ImaginationVecEnv 用的 `wm` 对象,通过 websocket 调 D-serve(d_serve.py)。

D 在 rise_venv(diffusers),训练/wm_bridge 在 residual env,一个进程装不下两套;故 D 包成 serve,
本客户端鸭子类型 `wm.infer(obs=, act_tokens=, prompt=, num_denois_steps=) -> {"video": tensor}`,
与 imagination_env 里对 `self.wm.infer(...)` 的调用一致(Task 6 无需改)。

传输:obs/act_tokens 转 np.float32 发出(msgpack_numpy 不支持 bf16,D-serve 侧转回 bf16);
      响应 video 转 torch。
"""
from __future__ import annotations

import numpy as np
import torch


class WmServeClient:
    def __init__(self, host="127.0.0.1", port=9000):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        # 关 keepalive ping:首帧可能慢(JIT/大张量),别被心跳超时断连(对齐 pi05 adapter)
        import websockets.sync.client as _wsc
        if not getattr(_wsc.connect, "_no_ping_patched", False):
            _orig = _wsc.connect

            def _no_ping(*a, **k):
                k.setdefault("ping_interval", None)
                return _orig(*a, **k)

            _no_ping._no_ping_patched = True
            _wsc.connect = _no_ping
        self.client = WebsocketClientPolicy(host=host, port=port)

    def infer(self, obs=None, act_tokens=None, prompt="", num_denois_steps=10, **kw):
        def _np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float().numpy()
            return np.asarray(x, dtype=np.float32)

        req = {
            "obs": _np(obs),
            "act_tokens": _np(act_tokens),
            "prompt": str(prompt),
            "num_denois_steps": int(num_denois_steps),
        }
        resp = self.client.infer(req)
        # .copy():msgpack 解出的数组只读,torch.from_numpy 会 warn/未定义写入
        video = np.array(resp["video"], dtype=np.float32, copy=True)
        return {"video": torch.from_numpy(video)}
