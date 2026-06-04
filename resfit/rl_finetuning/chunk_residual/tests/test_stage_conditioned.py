import torch
from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.rl.actor import Actor

REPR, PATCH, PROP, FLAT = 32, 16, 3, 20
B, NUM_STAGES = 6, 5


def _actor(stage_conditioned):
    cfg = ActorConfig()  # spatial_emb=0 默认 → 走 residual 兼容的 else 分支
    return Actor(REPR, PATCH, PROP, FLAT, cfg, residual_actor=True,
                 stage_conditioned=stage_conditioned, num_stages=NUM_STAGES)


def _obs(with_stage, stage_val=0):
    o = {
        "feat": torch.randn(B, REPR // PATCH, PATCH),
        "observation.state": torch.randn(B, PROP),
        "observation.base_action": torch.tanh(torch.randn(B, FLAT)),
    }
    if with_stage:
        o["observation.stage_id"] = torch.full((B, 1), float(stage_val))
    return o


def test_actor_off_does_not_require_stage_id():
    # 关闭时 forward 不读 stage_id（obs 里没有也不报错），输出 shape 不变
    torch.manual_seed(0)
    actor = _actor(stage_conditioned=False).eval()
    dist = actor.forward(_obs(with_stage=False), std=0.0)
    assert dist.mean.shape == (B, FLAT)


def test_actor_on_shape_ok():
    torch.manual_seed(0)
    actor = _actor(stage_conditioned=True).eval()
    dist = actor.forward(_obs(with_stage=True, stage_val=2), std=0.0)
    assert dist.mean.shape == (B, FLAT)


def test_actor_on_is_sensitive_to_stage():
    # 同一 actor、同一非 stage 输入，仅改 stage_id → 输出应不同（证明 stage 真的喂进去了）
    torch.manual_seed(0)
    actor = _actor(stage_conditioned=True).eval()
    base = _obs(with_stage=True, stage_val=0)
    o0 = dict(base); o0["observation.stage_id"] = torch.zeros(B, 1)
    o3 = dict(base); o3["observation.stage_id"] = torch.full((B, 1), 3.0)
    m0 = actor.forward(o0, std=0.0).mean
    m3 = actor.forward(o3, std=0.0).mean
    assert not torch.allclose(m0, m3)
