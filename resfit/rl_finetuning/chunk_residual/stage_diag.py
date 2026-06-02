"""stage-aware 诊断聚合:把逐样本指标按 stage 分组,给出每阶段均值 + 计数。

用于训练时按 stage 观察 residual_norm / Q(base) / Q(base+residual) / 成功率等,
定位残差在哪个阶段起作用、是否卡在某个 stage。纯逻辑,便于 TDD;指标怎么算由调用方决定。
"""
from __future__ import annotations

import torch


def stage_diagnostics(stages: torch.Tensor, values: dict[str, torch.Tensor]) -> dict:
    """按 stage 分组聚合。

    stages: [N](int 或 float,float 会转 int)。
    values: {metric_name: [N] 标量张量}。多维张量会报错(应由调用方先 reduce 成逐样本标量)。
    返回 {"counts": {stage:int}, "means": {metric: {stage: float}}}。
    """
    stages = stages.flatten().round().long()
    present = sorted(int(s) for s in torch.unique(stages).tolist())

    counts = {s: int((stages == s).sum()) for s in present}

    means: dict[str, dict[int, float]] = {}
    for name, v in values.items():
        v = v.flatten() if v.dim() > 1 and v.shape[0] != stages.numel() else v
        if v.dim() != 1 or v.numel() != stages.numel():
            raise ValueError(
                f"metric '{name}' 必须是 [N] 逐样本标量(N={stages.numel()}),实际 shape={tuple(v.shape)}")
        per_stage = {}
        for s in present:
            mask = stages == s
            per_stage[s] = float(v[mask].float().mean())
        means[name] = per_stage

    return {"counts": counts, "means": means}


def flatten_stage_diagnostics(diag: dict, prefix: str = "diag") -> dict[str, float]:
    """把 stage_diagnostics 输出拍平成 {f"{prefix}/count/stage{s}":..., f"{prefix}/{metric}/stage{s}":...}。"""
    flat: dict[str, float] = {}
    for s, c in diag["counts"].items():
        flat[f"{prefix}/count/stage{s}"] = c
    for metric, per_stage in diag["means"].items():
        for s, m in per_stage.items():
            flat[f"{prefix}/{metric}/stage{s}"] = m
    return flat
