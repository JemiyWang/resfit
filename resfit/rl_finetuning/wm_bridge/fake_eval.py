"""接管 train_chunk_residual.py 的 eval:存 checkpoint + 记录优势估计器 proxy 值(spec §6.7)。

想象空间没有真环境可 eval。但存盘在 eval 块里,简单关掉 eval 会一个 checkpoint 都不存。
本模块:①每 eval 存 checkpoint(滚动 imagination_last.pt + 按步持久 imagination_step_{step}.pt);
②(adv_scorer 提供时)用冻结策略 rollout N 集想象轨迹,优势估计器逐帧打分,追加写
`imagined_adv_eval.jsonl`(选项2 监控曲线)。

★ 返回恒定 0.0,不返回递增哨兵去骗过 `sr > best_sr` —— 存盘由本模块显式负责。
★ 想象空间不产生任何被报告的指标:adv 值是 imagined proxy(value∈[-1,1],≈1=成功),不是实机成功率。
★ eval 与训练**共享同一个 ImaginationVecEnv 实例**,故 rollout 前 snapshot、后 restore,防污染训练轨迹。
"""
from __future__ import annotations

import json
import os

import numpy as np

from resfit.rl_finetuning.utils.checkpoint import save_checkpoint

CHUNK_LENGTH = 50   # 每 WM 点火产 25 预测帧;queue 模式每 chunk 需 50 内层 step


def _rollout_pred_frames(env, agent, max_segments):
    """用冻结策略在想象里 rollout 一集,收 WM 预测帧。env=wrapped eval env(带 base_action 增广)。
    与训练共享内层 env → snapshot/restore。返回 (T,V,C,H,W) ∈ [-1,1]。"""
    import torch
    from resfit.rl_finetuning.off_policy.common_utils import utils

    inner = getattr(env, "vec_env", env)               # ImaginationVecEnv(带 snapshot/collect)
    snap = inner.snapshot_state()
    try:
        inner.start_pred_collection()
        obs, _ = env.reset()
        guard, cap = 0, max_segments * CHUNK_LENGTH + 20
        with torch.no_grad(), utils.eval_mode(agent):
            done = False
            while not done and guard < cap:
                action = agent.act(obs, eval_mode=True)
                obs, _, term, trunc, _ = env.step(action)
                done = bool((term | trunc).any())
                guard += 1
        frames = inner.collect_pred_frames()           # (T,V,C,H,W) 或 (0,)
    finally:
        inner.restore_state(snap)                       # 无论如何恢复训练轨迹
    return frames


def make_imagination_evaluator(output_dir: str, config, *, adv_scorer=None,
                               n_eval_episodes=10):
    log_path = os.path.join(output_dir, "imagined_adv_eval.jsonl")

    def run_dexmg_evaluation(*, env=None, agent=None, num_episodes=None,
                             device=None, global_step=0, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        # imagination_last.pt:滚动最新(断点/取最新用,覆盖)。
        save_checkpoint(agent, os.path.join(output_dir, "imagination_last.pt"),
                        global_step=global_step, config=config, success_rate=0.0)
        # imagination_step_{step}.pt:按步持久归档,不覆盖 —— 供后续实机逐 checkpoint 调试。
        save_checkpoint(agent,
                        os.path.join(output_dir, f"imagination_step_{int(global_step):07d}.pt"),
                        global_step=global_step, config=config, success_rate=0.0)

        # 选项2:优势估计器 proxy 值 logging(只存原始值,不判成败;value∈[-1,1],≈1=成功)
        if adv_scorer is not None and env is not None and agent is not None:
            inner = getattr(env, "vec_env", env)
            max_segments = int(getattr(inner, "max_segments", 2))
            finals = []
            with open(log_path, "a") as f:
                for ep in range(n_eval_episodes):
                    frames = _rollout_pred_frames(env, agent, max_segments)
                    if getattr(frames, "ndim", 0) != 5 or frames.shape[0] == 0:
                        continue                        # 空 rollout(不应发生),跳过
                    vals = np.asarray(adv_scorer.score_frames(frames), np.float32).reshape(-1)
                    finals.append(float(vals[-1]))
                    f.write(json.dumps({
                        "env_step": int(global_step), "episode_idx": ep,
                        "n_frames": int(vals.shape[0]),
                        "adv_final": float(vals[-1]), "adv_max": float(vals.max()),
                        "adv_mean": float(vals.mean()),
                        "adv_traj": [round(float(v), 4) for v in vals],
                    }) + "\n")
            if finals:
                print(f"[adv-proxy step {global_step}] {len(finals)} 集 "
                      f"adv_final 均值={np.mean(finals):.3f} (≈1=成功) → {log_path}", flush=True)
        return {"eval/success_rate": 0.0}

    return run_dexmg_evaluation
