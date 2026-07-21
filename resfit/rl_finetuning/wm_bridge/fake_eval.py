"""接管 train_chunk_residual.py:1315 那个全脚本唯一的 save_checkpoint。

想象空间没有真环境可 eval。但 :1315 的存盘在 eval 块里,简单关掉 eval(把
--eval_every_env_steps 设大)会让整个 run 一个 checkpoint 都不存,训练白跑。

★ 返回恒定 0.0,不返回递增哨兵去骗过 :1312 的 `sr > best_sr` 比较 —— 那是把存盘寄托在
   一个假指标上。存盘由本模块显式负责,trainer 的 best 逻辑自然失效。

想象空间不产生任何被报告的指标。真实成功率只在实机 eval 测。
"""
from __future__ import annotations

import json
import os

import numpy as np

from resfit.rl_finetuning.utils.checkpoint import save_checkpoint


def make_imagination_evaluator(output_dir: str, config, *, adv_scorer=None,
                               eval_env=None, n_eval_episodes=10):
    log_path = os.path.join(output_dir, "imagined_adv_eval.jsonl")

    def run_dexmg_evaluation(*, env=None, agent=None, num_episodes=None,
                             device=None, global_step=0, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        # imagination_last.pt:滚动最新(断点/取最新用,覆盖)。
        save_checkpoint(agent, os.path.join(output_dir, "imagination_last.pt"),
                        global_step=global_step, config=config, success_rate=0.0)
        # imagination_step_{step}.pt:按步持久归档,不覆盖 —— 供后续实机逐 checkpoint 调试。
        # eval 每 --eval_every_env_steps 触发一次,设 50000 即每 50k 步一个存档。
        save_checkpoint(agent,
                        os.path.join(output_dir, f"imagination_step_{int(global_step):07d}.pt"),
                        global_step=global_step, config=config, success_rate=0.0)
        # 选项2:优势估计器 proxy 值 logging(只存原始值,不判成败)
        if adv_scorer is not None and eval_env is not None:
            with open(log_path, "a") as f:
                for ep in range(n_eval_episodes):
                    frames = eval_env.rollout_frames(agent)      # 冻结策略 rollout 一集
                    vals = np.asarray(adv_scorer.score_frames(frames), np.float32)
                    f.write(json.dumps({
                        "env_step": int(global_step), "episode_idx": ep,
                        "adv_final": float(vals[-1]), "adv_max": float(vals.max()),
                        "adv_mean": float(vals.mean()),
                        "adv_traj": [round(float(v), 5) for v in vals],
                    }) + "\n")
        return {"eval/success_rate": 0.0}

    return run_dexmg_evaluation
