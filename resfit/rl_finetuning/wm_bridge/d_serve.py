"""RISE dynamics model (D) 的 websocket serve —— 与 kai0 serve 同款,让 residual 环境的
ImaginationVecEnv 通过 websocket 调 D.infer(D 在 rise_venv/diffusers,一个进程装不下两套)。

★ 在 rise_venv 里跑(diffusers),不是 residual env:
  PYTHONPATH 无需设(本文件自带 sys.path);须先补装 accelerate。
  CUDA_VISIBLE_DEVICES=<卡> /mnt/mnt/data/resfit/rise_venv/bin/python \
    -m resfit.rl_finetuning.wm_bridge.d_serve \
    --config <block_infer.yaml> --port 9000

客户端(residual 侧)= wm_client.WmServeClient,鸭子类型 wm.infer(obs, act_tokens, ...)。

请求 obs_dict: {obs: np.float32 (3,3,4,192,256), act_tokens: np.float32 (1,25,30),
               prompt: str, num_denois_steps: int}
响应: {video: np.float32 (3,3,29,192,256)}  (调用方取 [:,:,4:] 为预测段)
"""
from __future__ import annotations

import argparse
import sys

# D 的依赖:dynamics_model 包 + custom_pipeline 的顶层 utils/models + rlinf wrapper。
_RISE = "/mnt/mnt/data/resfit/RISE_Hi"
for _p in (f"{_RISE}/dynamics", f"{_RISE}/dynamics/dynamics_model",
           f"{_RISE}/policy_and_value/policy_online"):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class DPolicy:
    """把加载好的 D 包成 WebsocketPolicyServer 期望的 policy(infer/metadata)。"""

    def __init__(self, config_path, device="cuda"):
        import torch
        from rlinf.models.embodiment.modules.dynamics_model import DynamicsModel
        self.torch = torch
        print(f"[d_serve] 加载 D: {config_path} …", flush=True)
        self.D = DynamicsModel(config_path, device=device)
        print("[d_serve] D 就绪", flush=True)

    @property
    def metadata(self):
        return {"model": "rise_dynamics", "valid_cams": list(self.D.valid_cams)}

    def infer(self, obs: dict) -> dict:
        torch = self.torch
        wm_obs = torch.as_tensor(obs["obs"]).to("cuda", dtype=torch.bfloat16)
        act = torch.as_tensor(obs["act_tokens"]).to("cuda", dtype=torch.bfloat16)
        prompt = obs.get("prompt", "")
        n = int(obs.get("num_denois_steps", 10))
        with torch.no_grad():
            out = self.D.infer(obs=wm_obs, act_tokens=act, prompt=prompt,
                               num_denois_steps=n)
        video = out["video"]
        if video.ndim == 6:                       # (b,v,c,t,h,w) → 取 b=0
            video = video[0]
        return {"video": video.float().cpu().numpy()}

    def reset(self):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="block_infer.yaml(改好路径的)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = DPolicy(args.config, device=args.device)
    print(f"[d_serve] up on {args.host}:{args.port}", flush=True)
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
