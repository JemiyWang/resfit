"""chunk_residual 训练 log 的纯解析/聚合(无副作用,供 plot_stage_diag 与 reeval 复用)。"""
from __future__ import annotations

import json
import os
import re

_PURITY_RE = re.compile(r"stage(\d+):(\d+)/(\d+)=")


def parse_stage_purity_line(line: str) -> dict[int, float]:
    """从一行 [stage-purity] 解析每 stage 的未回退率 = 1 - regress/total。

    `stageK:a/b=P%` 中 a=回退步数, b=该桶总步数。b==0 的桶跳过(不入结果)。
    非 [stage-purity] 行返回 {}。
    """
    if "[stage-purity]" not in line:
        return {}
    out: dict[int, float] = {}
    for m in _PURITY_RE.finditer(line):
        k, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if b > 0:
            out[k] = 1.0 - a / b
    return out


def load_reach_sidecar(path: str) -> dict | None:
    """读 <log>_reach.json;不存在或内容损坏/不完整时返回 None。

    返回 None 的情况:
      - 文件不存在
      - JSON 解析失败(截断写、损坏)
      - 缺少 "reach" 或 "step" 键
      - 值类型无法转换

    正常返回: {"step": int, "reach": {int: float}}
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        reach = {int(k): float(v) for k, v in data["reach"].items()}
        return {"step": int(data["step"]), "reach": reach}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def fold_episode_max_stage(prev_max: int, max_stage_in_chunk: int,
                           reward: float, top_stage: int) -> int:
    """单步更新 episode 内最高 stage:取最高;若该步成功(reward>=1)→ top_stage。"""
    new = max(int(prev_max), int(max_stage_in_chunk))
    if reward >= 1.0:
        new = max(new, int(top_stage))
    return new
