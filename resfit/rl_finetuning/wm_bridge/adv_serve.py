"""优势估计器(RISE value_block)的 websocket serve —— 与 d_serve.py 同款跨环境模式。

value 模型(openpi_value 的 PI0Pytorch value head, ~15GB)在 rise_venv 里,residual 侧
ImaginationVecEnv 的 eval 路通过 websocket 调 score_frames 给想象 rollout 逐帧打进度值,
写入 imagined_adv_eval.jsonl(spec §6.7 选项2 监控)。

★ value head 是 Tanh → 输出 ∈ [-1,1],≈1=完成/成功,负=失败进度。不是 [0,1]。
★ 在 rise_venv 跑(openpi_value 需 PYTHONPATH=RISE_Hi/.../src):
    source /mnt/mnt/data/resfit/RISE_Hi/temp/rise_env.sh
    CUDA_VISIBLE_DEVICES=<卡> /mnt/mnt/data/resfit/rise_venv/bin/python \
      -m resfit.rl_finetuning.wm_bridge.adv_serve --port 8002
  (本文件自带 sys.path 兜底,rise_env.sh 没 source 也能 import openpi_value)

客户端(residual 侧)= adv_client.AdvServeClient。
请求 obs: {frames: np.float32 (T,V,C,H,W) ∈ [-1,1], prompt: str}  (V=3: top_head/hand_left/hand_right)
响应:     {values: np.float32 (T,)  ∈ [-1,1]}
"""
# ★★ 必须最先 import datasets:openpi_value.transforms→HF datasets 的 native import 顺序
#    反了会 SIGSEGV(rise_venv 实测确定性复现)。这行不能挪、不能删。
import datasets  # noqa: F401  isort:skip
#   ↑ 必须是文件第一个 import。不加 `from __future__ import annotations`:它按语法须在文件
#     最前,会和"datasets 最先"冲突;本文件不用前向引用类型,3.11 下 `-> dict` 无需它。

import argparse
import os
import sys

import numpy as np

# openpi_value 源码路径兜底(rise_env.sh 已设 PYTHONPATH,这里再保一层)
_VALUE_SRC = "/mnt/mnt/data/resfit/RISE_Hi/policy_and_value/policy_offline_and_value/src"
if _VALUE_SRC not in sys.path:
    sys.path.insert(0, _VALUE_SRC)

# value_block ckpt(用户指认;15GB)。相对 RISE 工程根解析。
_CKPT_DEFAULT = ("/mnt/mnt/data/resfit/RISE_Hi/policy_and_value/policy_offline_and_value/"
                 "checkpoints/value_block/value_block/10000/model.safetensors")
CAMS = ("top_head", "hand_left", "hand_right")   # 与 wm_driver.CAMERA_KEYS 视角一致
STATE_DIM = 32
PROMPT_DEFAULT = "build block"


class ValuePolicy:
    """把 value_block 包成 WebsocketPolicyServer 期望的 policy(infer/metadata)。"""

    def __init__(self, ckpt_path=_CKPT_DEFAULT, config_name="value_block",
                 device="cuda", chunk=32):
        import dataclasses
        import torch
        import safetensors.torch
        from openpi_value.training import config as _config
        from openpi_value import transforms as _transforms
        self.torch = torch
        self.device = device
        self.chunk = int(chunk)

        print(f"[adv_serve] 载 config {config_name} …", flush=True)
        cfg = _config.get_config(config_name)
        # 照 label_frame_value.py:459-466:eval 用 value head、关 TD 目标网络。
        new_model = cfg.model.__class__(**{**cfg.model.__dict__,
                                           "p_mask_ego_state": 1,
                                           "value_TD_learning": False})
        cfg = dataclasses.replace(cfg, model=new_model)

        from openpi_value.models_pytorch.pi0_pytorch import PI0Pytorch
        print("[adv_serve] 构建 PI0Pytorch(约 70s,CPU 随机初始化)…", flush=True)
        model = PI0Pytorch(new_model)
        print(f"[adv_serve] 载权重 {ckpt_path} …", flush=True)
        safetensors.torch.load_model(model, ckpt_path, strict=False)  # missing=0(实测)
        model.eval().to(device)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.model = model

        # 复用 config 的成品 transform,跳过 repack(见配方 §3:repack 带 fut_*/episode_length 会 KeyError)
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        self.input_transform = _transforms.compose([
            *dc.data_transforms.inputs,                                    # CustomAgilexInputs, DeltaActions
            _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,                                   # Inject, Resize224, TokenizePrompt, Pad
        ])
        self._Observation = __import__(
            "openpi_value.models.model", fromlist=["Observation"]).Observation
        print("[adv_serve] value 模型就绪", flush=True)

    @property
    def metadata(self):
        return {"model": "rise_value_block", "cams": list(CAMS), "value_range": [-1.0, 1.0]}

    def _frame_dict(self, frame_views, prompt):
        # frame_views: (V,C,H,W) ∈ [-1,1] → 每 view [0,1] CHW(CustomAgilex 期望 [0,1]/uint8)
        imgs = {}
        for i, cam in enumerate(CAMS):
            chw = np.asarray(frame_views[i], dtype=np.float32)
            imgs[cam] = (chw + 1.0) / 2.0                    # ★[-1,1]→[0,1](配方 §5 关键坑)
        return {"images": imgs,
                "state": np.zeros(STATE_DIM, np.float32),    # 零占位:state 被设计忽略(配方 §3)
                "prompt": prompt}

    def infer(self, obs: dict) -> dict:
        torch = self.torch
        frames = np.asarray(obs["frames"], dtype=np.float32)  # (T,V,C,H,W)
        assert frames.ndim == 5 and frames.shape[1] == len(CAMS), \
            f"frames 须 (T,{len(CAMS)},C,H,W),got {frames.shape}"
        prompt = obs.get("prompt", PROMPT_DEFAULT)
        processed = [self.input_transform(self._frame_dict(frames[t], prompt))
                     for t in range(frames.shape[0])]

        out = []
        for i in range(0, len(processed), self.chunk):
            sub = processed[i:i + self.chunk]
            batch = {
                "image": {c: torch.from_numpy(np.stack([s["image"][c] for s in sub]))
                          for c in sub[0]["image"]},
                "image_mask": {c: torch.from_numpy(np.stack([s["image_mask"][c] for s in sub]))
                               for c in sub[0]["image_mask"]},
                "state": torch.from_numpy(np.stack([s["state"] for s in sub])).float(),
                "tokenized_prompt": torch.from_numpy(
                    np.stack([s["tokenized_prompt"] for s in sub])),
                "tokenized_prompt_mask": torch.from_numpy(
                    np.stack([s["tokenized_prompt_mask"] for s in sub])),
            }
            for c in batch["image"]:
                batch["image"][c] = batch["image"][c].to(self.device)
                batch["image_mask"][c] = batch["image_mask"][c].to(self.device)
            batch["state"] = batch["state"].to(self.device)
            batch["tokenized_prompt"] = batch["tokenized_prompt"].to(self.device)
            batch["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"].to(self.device)
            with torch.no_grad():
                observation = self._Observation.from_dict(batch)
                v = self.model.sample_values(self.device, observation)   # (b,1) ∈ [-1,1]
            out.append(v.reshape(-1).float().cpu().numpy())
        return {"values": np.concatenate(out).astype(np.float32)}         # (T,)

    def reset(self):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=_CKPT_DEFAULT)
    ap.add_argument("--config_name", default="value_block")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    from openpi_value.serving.websocket_policy_server import WebsocketPolicyServer
    policy = ValuePolicy(args.ckpt, config_name=args.config_name,
                         device=args.device, chunk=args.chunk)
    print(f"[adv_serve] up on {args.host}:{args.port}", flush=True)
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
