"""per-stage 到达率:给定每个 episode 到达的最高 stage,算各 stage 的到达比例。

reach_rate(k) = 到达过 stage>=k 的 episode 占比。由 stage 单调性,k 越大比例越小;
reach(NUM_STAGES-1) 即最终成功率。用来区分"够到中间阶段但完不成"(reward-hacking 像)
与"基座被毁、连 stage1 都到不了"两种崩溃。
"""
from __future__ import annotations


def stage_reach_rates(episode_max_stages: list[int], num_stages: int) -> dict[int, float]:
    """每个 stage k∈[1, num_stages-1] 的到达率(到过 stage>=k 的 episode 占比)。空输入→全 0。"""
    n = len(episode_max_stages)
    return {
        k: (sum(1 for s in episode_max_stages if s >= k) / n if n else 0.0)
        for k in range(1, num_stages)
    }
