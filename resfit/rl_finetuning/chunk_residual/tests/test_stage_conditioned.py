import torch
from resfit.rl_finetuning.config.rlpd import ActorConfig, CriticConfig
from resfit.rl_finetuning.off_policy.rl.actor import Actor
from resfit.rl_finetuning.off_policy.rl.critic import Critic
from resfit.rl_finetuning.off_policy.rl.stage_utils import append_stage

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


def _critic(prop_dim):
    cfg = CriticConfig()
    cfg.loss.type = "mse"            # 标量 Q，断言简单
    return Critic(repr_dim=REPR, patch_repr_dim=PATCH, prop_dim=prop_dim,
                  action_dim=FLAT, cfg=cfg).eval()


def test_critic_q_sensitive_to_stage_in_prop():
    torch.manual_seed(0)
    critic = _critic(prop_dim=PROP + NUM_STAGES)   # prop 已加宽
    feat = torch.randn(B, REPR // PATCH, PATCH)
    prop = torch.randn(B, PROP)
    act = torch.tanh(torch.randn(B, FLAT))
    sid0 = torch.zeros(B, 1)
    sid3 = torch.full((B, 1), 3.0)
    # 用 forward：mse 下确定返回 [num_q, B, 1]，shape 可预测
    q0 = critic.forward(feat, append_stage(prop, sid0, NUM_STAGES), act)
    q3 = critic.forward(feat, append_stage(prop, sid3, NUM_STAGES), act)
    assert q0.shape[-2] == B and q0.shape[-1] == 1
    assert not torch.allclose(q0, q3)


# ---------------------------------------------------------------------------
# Task 6: 端到端烟雾（manual — 依赖 VitEncoder，不进默认 CI）
# ---------------------------------------------------------------------------
import math
import pytest
from tensordict import TensorDict
from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


@pytest.mark.manual
def test_qagent_update_smoke_stage_on():
    """QAgent(stage_conditioned=True) の update() 端到端烟雾：
    critic + actor 各跑一步，loss 有限、不 NaN。
    覆盖 QAgent 内部 _critic_prop / _encode / update_critic / update_actor
    的维度对齐（Task 6 盲区验证）。

    VitEncoder 用 embed2: conv(k=8,s=4)+conv(k=3,s=2)，84×84 → 9×9=81 patches。
    """
    torch.manual_seed(0)
    # C,H,W = 3,84,84 ← embed2 公式: floor((84-8)/4+1)=20 → floor((20-3)/2+1)=9 → 81 patches
    C, H, W = 3, 84, 84
    cam = "observation.images.agentview"
    state_dim, flat = 5, 12
    bs, ns = 4, 5

    cfg = QAgentConfig()
    cfg.device = "cpu"
    cfg.critic.loss.type = "mse"
    agent = QAgent(
        obs_shape=(C, H, W),
        prop_shape=(state_dim,),
        action_dim=flat,
        rl_cameras=[cam],
        cfg=cfg,
        residual_actor=True,
        stage_conditioned=True,
        num_stages=ns,
    )
    agent.train(True)
    agent.actor_target.train(True)

    def _obs():
        return {
            cam: torch.rand(bs, C, H, W),
            "observation.state": torch.randn(bs, state_dim),
            "observation.base_action": torch.tanh(torch.randn(bs, flat)),
            "observation.stage_id": torch.randint(0, ns, (bs, 1)).float(),
        }

    batch = TensorDict(
        {
            "obs": TensorDict(_obs(), batch_size=[bs]),
            "action": torch.tanh(torch.randn(bs, flat)),
            ("next", "reward"): torch.zeros(bs),
            "gamma": torch.full((bs,), 0.99),
            "nonterminal": torch.ones(bs),
            ("next", "obs"): TensorDict(_obs(), batch_size=[bs]),
        },
        batch_size=[bs],
    )

    metrics = agent.update(batch, stddev=0.05, update_actor=True)
    assert math.isfinite(metrics["train/critic_loss"]), (
        f"critic_loss is not finite: {metrics['train/critic_loss']}"
    )
    assert math.isfinite(metrics["train/actor_loss_total"]), (
        f"actor_loss_total is not finite: {metrics['train/actor_loss_total']}"
    )


# ---------------------------------------------------------------------------
# Bug 回归：stage one-hot 必须对齐到目标张量的 device。
# 实验在 GPU 崩溃：env wrapper 的 observation.stage_id 用 torch.full 默认在 CPU，
# 而 feat/state/base_action 在 CUDA，cat 跨设备 RuntimeError。stage_onehot 跟随
# stage_id 的 device(CPU)是错的，应跟随它要拼接的目标张量。CPU-only 的 manual
# 烟雾测不出（全 CPU 同设备），故用 CUDA skipif 守住这个盲区。
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device 检查需要 CUDA")
def test_append_stage_aligns_one_hot_to_prop_device():
    prop = torch.randn(3, 7, device="cuda")
    sid_cpu = torch.zeros(3, 1)                       # CPU：模拟 env wrapper 的 torch.full
    out = append_stage(prop, sid_cpu, NUM_STAGES)     # 修复前：RuntimeError(device mismatch)
    assert out.device.type == "cuda"
    assert out.shape == (3, 7 + NUM_STAGES)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device 检查需要 CUDA")
def test_actor_forward_with_stage_id_on_cpu_feat_on_cuda():
    # 复现实验崩溃：feat/state/base 在 CUDA，stage_id 在 CPU
    torch.manual_seed(0)
    actor = _actor(stage_conditioned=True).cuda().eval()
    obs = {
        "feat": torch.randn(B, REPR // PATCH, PATCH, device="cuda"),
        "observation.state": torch.randn(B, PROP, device="cuda"),
        "observation.base_action": torch.tanh(torch.randn(B, FLAT, device="cuda")),
        "observation.stage_id": torch.zeros(B, 1),    # CPU
    }
    dist = actor.forward(obs, std=0.0)                # 修复前：RuntimeError(device mismatch)
    assert dist.mean.shape == (B, FLAT)
    assert dist.mean.device.type == "cuda"
