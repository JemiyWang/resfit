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
    buf_sizes: {"online": int, "offline": int, "relabel": int (optional, defaults to 0)}。
    hist_fn: 可注入,单测传 stub 即可绕开 wandb。
    """
    log_dict = {k: v for k, v in m_upd.items() if not k.startswith("_")}
    log_dict["lr/actor"] = lrs["actor"]
    log_dict["lr/critic"] = lrs["critic"]
    log_dict["lr/encoder"] = lrs["encoder"]
    log_dict["buffer/online_size"] = buf_sizes["online"]
    log_dict["buffer/offline_size"] = buf_sizes["offline"]
    log_dict["buffer/relabel_size"] = buf_sizes.get("relabel", 0)
    if with_histograms:
        if "_actions" in m_upd:
            log_dict["histograms/actions"] = hist_fn(m_upd["_actions"].numpy().reshape(-1))
        if "_target_q" in m_upd:
            log_dict["histograms/critic_qt"] = hist_fn(m_upd["_target_q"].numpy().reshape(-1))
    return log_dict


def _parse_purity(summary):
    """把 stage_purity_summary() 字符串解析成 {purity/regress_frac, purity/stageN}。
    解析不了就回落 {purity/raw: 原串}(不抛异常)。"""
    if not summary or summary == "no stage steps":
        return {"purity/raw": summary}
    out = {}
    try:
        tokens = summary.split()
        for i, tok in enumerate(tokens):
            if tok == "regress":
                a, b = tokens[i + 1].split("=")[0].split("/")
                out["purity/regress_frac"] = float(a) / float(b)
            elif tok.startswith("stage") and ":" in tok:
                label, rest = tok.split(":", 1)
                stage_idx = int(label[len("stage"):])
                r, n = rest.split("=")[0].split("/")
                out[f"purity/stage{stage_idx}"] = float(r) / float(n)
    except Exception:
        return {"purity/raw": summary}
    if not out:
        return {"purity/raw": summary}
    return out


def build_eval_log_dict(eval_metrics, last_diag, purity_summary):
    """组装 eval + stage 诊断 log_dict。

    eval_metrics: run_dexmg_evaluation 返回的 dict(含 eval/* 键)。
    last_diag: flatten_stage_diagnostics 的扁平输出(键已带 diag/ 前缀),可能为 None。
    purity_summary: env.stage_purity_summary() 字符串。
    """
    log_dict = {k: v for k, v in eval_metrics.items() if k.startswith("eval/")}
    if last_diag:
        log_dict.update(last_diag)
    log_dict.update(_parse_purity(purity_summary))
    return log_dict


def init_wandb(args):
    """按 args 启 wandb run。--smoke 时强制 mode='disabled'(不产生真 run)。

    name 缺省回落为 output_dir 的 basename。返回 wandb run(disabled 下为 no-op run,
    调用方无需判空)。
    """
    name = args.wandb_name or os.path.basename(args.output_dir.rstrip("/"))
    mode = "disabled" if args.smoke else args.wandb_mode
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=name,
        mode=mode,
        config=vars(args),
    )
