"""对比 rb.sample() 与 sample_stage_balanced(rb) 取出的 batch 结构,定位 _encode 崩溃。

按 train 的方式建 prioritized buffer(含 MultiStepTransform),塞类真 transition,
打印两者的 keys / obs 图像 device,dtype,shape / batch_size,看差异。纯 CPU。
"""
from __future__ import annotations

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

from resfit.rl_finetuning.chunk_residual.stage_replay import sample_stage_balanced
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform

IMG = "observation.images.cam"


def _td(stage):
    obs = {IMG: (torch.rand(3, 84, 84) * 255).to(torch.uint8),
           "observation.state": torch.randn(20),
           "observation.base_action": torch.randn(480),
           "observation.stage_id": torch.tensor([float(stage)])}
    nxt = {k: v.clone() for k, v in obs.items()}
    return TensorDict({
        "obs": TensorDict(obs, batch_size=[]),
        "next": TensorDict({"obs": TensorDict(nxt, batch_size=[]),
                            "done": torch.tensor([False]), "reward": torch.tensor([0.0])}, batch_size=[]),
        "action": torch.randn(480),
        "max_stage": torch.tensor(float(stage)),
        "_priority": torch.tensor(10.0),
    }, batch_size=[]).unsqueeze(0)


def _describe(name, b):
    print(f"\n[{name}] type={type(b).__name__} batch_size={list(b.batch_size)}")
    print(f"  top keys: {sorted(b.keys())}")
    try:
        img = b["obs"][IMG]
        print(f"  obs[{IMG}]: shape={tuple(img.shape)} dtype={img.dtype} device={img.device}")
    except Exception as e:
        print(f"  obs image read error: {e}")
    print(f"  has 'gamma'={'gamma' in b.keys()} 'nonterminal'={'nonterminal' in b.keys()} "
          f"'_weight'={'_weight' in b.keys()}")


def main():
    rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=2000, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority",
        transform=MultiStepTransform(n_steps=3, gamma=0.99),
        pin_memory=False, prefetch=0, batch_size=16)
    for i in range(60):
        rb.add(_td(stage=0 if i % 3 else 2))
    print(f"len(rb)={len(rb)}")

    _describe("rb.sample()", rb.sample())
    _describe("sample_stage_balanced", sample_stage_balanced(rb, 16, generator=torch.Generator().manual_seed(0)))


if __name__ == "__main__":
    main()
