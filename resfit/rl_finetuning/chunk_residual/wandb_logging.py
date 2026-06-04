"""chunk_residual 训练的 wandb 上报 helper。

train_chunk_residual.py 原本只 print 到 stdout、不接 wandb。这里把指标组装抽成
近乎纯函数(build_train_log_dict / build_eval_log_dict)便于单测;init_wandb 是
wandb.init 的薄封装。详见
docs/superpowers/specs/2026-06-04-chunk-residual-wandb-logging-design.md。
"""
import os

import wandb


def build_train_log_dict(m_upd, lrs, buf_sizes, with_histograms=True,
                         hist_fn=wandb.Histogram):
    """组装训练指标 log_dict。

    m_upd: agent.update() 返回的 dict(含 train/* 与 _ 前缀内部键)。
    lrs: {"actor": float, "critic": float, "encoder": float}。
    buf_sizes: {"online": int, "offline": int}。
    hist_fn: 可注入,单测传 stub 即可绕开 wandb。
    """
    log_dict = {k: v for k, v in m_upd.items() if not k.startswith("_")}
    log_dict["lr/actor"] = lrs["actor"]
    log_dict["lr/critic"] = lrs["critic"]
    log_dict["lr/encoder"] = lrs["encoder"]
    log_dict["buffer/online_size"] = buf_sizes["online"]
    log_dict["buffer/offline_size"] = buf_sizes["offline"]
    if with_histograms:
        if "_actions" in m_upd:
            log_dict["histograms/actions"] = hist_fn(m_upd["_actions"].numpy().reshape(-1))
        if "_target_q" in m_upd:
            log_dict["histograms/critic_qt"] = hist_fn(m_upd["_target_q"].numpy().reshape(-1))
    return log_dict
