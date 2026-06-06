"""在线 relay relabeling 生产端(模块②b):产出前缀选段 + episode harvester。

只进 actor 侧 bc_batch(喂 ②a 残差 BC 消费端),不碰 critic 的 reward/done。
"""
from __future__ import annotations


def productive_prefix_len(stage_seq, min_stage: int = 1) -> int:
    """产出前缀长度:起点→最后一次 stage 推进(含),丢掉之后无推进的尾巴。

    stage_seq 是一条 episode 逐 chunk 的 latch(非降,由 chunk_env_wrapper 的 running-max 保证)。
    max < min_stage(没推进够)返回 0(该 episode 不 harvest)。latch 非降 -> max 首次出现 = 最后一次推进。
    """
    if not stage_seq:
        return 0
    top = max(stage_seq)
    if top < min_stage:
        return 0
    return stage_seq.index(top) + 1


class RelabelHarvester:
    """攒当前 episode 的 (entry, max_stage);flush() 按 productive_prefix_len 返回产出前缀 entry 并清空。

    entry 是调用方给的不透明对象(本项目=每 chunk 的 bc td)。min_stage 过滤走得不够远的 episode。
    """

    def __init__(self, min_stage: int = 1):
        self._min_stage = int(min_stage)
        self._entries: list = []
        self._stages: list = []

    def add(self, entry, max_stage) -> None:
        self._entries.append(entry)
        self._stages.append(int(max_stage))

    def flush(self) -> list:
        n = productive_prefix_len(self._stages, self._min_stage)
        out = self._entries[:n]
        self._entries, self._stages = [], []
        return out
