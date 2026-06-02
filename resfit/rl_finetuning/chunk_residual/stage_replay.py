"""stage-balanced replay 采样核心。

stage_balanced_indices: 给定每条已存 transition 的 stage 标签,按 stage 配额选出
batch_size 个索引(后期/稀少阶段不会被早期阶段淹没)。稀少阶段不足配额时有放回采样。

n-step 已在 add 时烤进 transition(rb_transforms.MultiStepTransform 是 inverse transform),
采样为 identity,故按任意索引选取安全。
"""
from __future__ import annotations

import torch


def _allocate_counts(fracs: list[float], batch_size: int) -> list[int]:
    """最大余数法:整数配额,和恰为 batch_size。"""
    raw = [f * batch_size for f in fracs]
    base = [int(x // 1) for x in raw]
    rem = batch_size - sum(base)
    # 按小数余数从大到小分配剩余名额(平局取靠前 stage)
    order = sorted(range(len(fracs)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    return base


def stage_balanced_indices(
    stages: torch.Tensor,
    batch_size: int,
    quotas: dict[int, float] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """返回 [batch_size] 的索引(int64),按 stage 配额分桶采样。

    quotas=None: 在出现过的阶段间等比例分配。
    quotas 指定: 只取出现过的阶段并归一化;缺失阶段的比例被重新分配。
    某阶段样本数 < 其配额时,该桶内有放回采样。
    """
    stages = stages.flatten().long()
    present = sorted(int(s) for s in torch.unique(stages).tolist())

    if quotas is None:
        fracs = [1.0 / len(present)] * len(present)
    else:
        w = [float(quotas.get(s, 0.0)) for s in present]
        total = sum(w)
        if total <= 0:                       # 配额与现存阶段无交集 → 退回等比例
            fracs = [1.0 / len(present)] * len(present)
        else:
            fracs = [x / total for x in w]

    counts = _allocate_counts(fracs, batch_size)

    chunks = []
    for s, k in zip(present, counts):
        if k == 0:
            continue
        pool = torch.nonzero(stages == s, as_tuple=False).flatten()
        n = pool.numel()
        if k <= n:
            sel = torch.randperm(n, generator=generator)[:k]
        else:                                # 配额超过样本数 → 有放回
            sel = torch.randint(n, (k,), generator=generator)
        chunks.append(pool[sel])

    return torch.cat(chunks)


def _stored_stages(rb) -> torch.Tensor:
    """从 torchrl buffer 廉价读取 max_stage 列(不物化 obs/图像)。"""
    n = len(rb)
    td = getattr(rb._storage, "_storage", None)
    if td is not None and "max_stage" in td.keys():
        return td["max_stage"][:n].flatten()
    return rb[:]["max_stage"].flatten()       # 回退(小 buffer / 测试)


def sample_stage_balanced(rb, batch_size, *, quotas=None, generator=None):
    """对 torchrl ReplayBuffer 做 stage-balanced 采样,返回 batch(与 rb.sample 同构)。

    读各 transition 的 max_stage → 按配额选索引 → rb[idx] 取出。n-step 已烤进 transition,
    任意索引取出安全。无 PER 重要性权重(stage 平衡本身即重加权)。
    """
    stages = _stored_stages(rb)
    idx = stage_balanced_indices(stages, batch_size, quotas=quotas, generator=generator)
    return rb[idx]
