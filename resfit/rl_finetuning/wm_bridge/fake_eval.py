"""接管 train_chunk_residual.py:1315 那个全脚本唯一的 save_checkpoint。

想象空间没有真环境可 eval。但 :1315 的存盘在 eval 块里,简单关掉 eval(把
--eval_every_env_steps 设大)会让整个 run 一个 checkpoint 都不存,训练白跑。

★ 返回恒定 0.0,不返回递增哨兵去骗过 :1312 的 `sr > best_sr` 比较 —— 那是把存盘寄托在
   一个假指标上。存盘由本模块显式负责,trainer 的 best 逻辑自然失效。

想象空间不产生任何被报告的指标。真实成功率只在实机 eval 测。
"""
from __future__ import annotations

import os

from resfit.rl_finetuning.utils.checkpoint import save_checkpoint


def make_imagination_evaluator(output_dir: str, config):
    def run_dexmg_evaluation(*, env=None, agent=None, num_episodes=None,
                             device=None, global_step=0, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        save_checkpoint(agent, os.path.join(output_dir, "imagination_last.pt"),
                        global_step=global_step, config=config,
                        success_rate=0.0)
        print(f"[imagination-eval] saved imagination_last.pt "
              f"@ env_steps={global_step} (想象空间不产出成功率指标)", flush=True)
        return {"eval/success_rate": 0.0}

    return run_dexmg_evaluation
