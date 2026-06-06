import torch

from resfit.rl_finetuning.off_policy.rl.stage_utils import (
    stage_budget_factor, parse_stage_budget,
)
from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.rl.actor import Actor

NUM_STAGES = 5
BUDGET = [1.0, 1.0, 1.0, 0.3, 0.1]


def test_stage_budget_factor_mixed():
    budget = torch.tensor(BUDGET)
    sid = torch.tensor([[0.], [3.], [4.], [1.]])
    f = stage_budget_factor(sid, budget, NUM_STAGES)
    assert f.shape == (4, 1)
    assert torch.allclose(f.reshape(-1), torch.tensor([1.0, 0.3, 0.1, 1.0]))


def test_stage_budget_factor_clamps_out_of_range():
    budget = torch.tensor([1.0, 0.5, 0.2])
    sid = torch.tensor([[5.], [-1.]])          # 越界 -> clamp 到 idx 2 和 0
    f = stage_budget_factor(sid, budget, 3)
    assert torch.allclose(f.reshape(-1), torch.tensor([0.2, 1.0]))


def test_parse_stage_budget_none():
    assert parse_stage_budget(None, NUM_STAGES) is None


def test_parse_stage_budget_ok():
    assert parse_stage_budget("1,1,1,0.3,0.1", NUM_STAGES) == [1.0, 1.0, 1.0, 0.3, 0.1]


def test_parse_stage_budget_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        parse_stage_budget("1,1,0.1", NUM_STAGES)   # 长度 3 != 5


def test_stage_budget_factor_accepts_1d_stage_id():
    budget = torch.tensor(BUDGET)
    sid = torch.tensor([0., 3., 4., 1.])       # 1-D [B],非 [B,1]
    f = stage_budget_factor(sid, budget, NUM_STAGES)
    assert f.shape == (4, 1)
    assert torch.allclose(f.reshape(-1), torch.tensor([1.0, 0.3, 0.1, 1.0]))


REPR, PATCH, PROP, FLAT = 32, 16, 3, 20
B = 6


def _actor(stage_budget):
    cfg = ActorConfig()                       # spatial_emb=0 默认 -> residual 兼容 else 分支
    return Actor(REPR, PATCH, PROP, FLAT, cfg, residual_actor=True,
                 num_stages=NUM_STAGES, stage_budget=stage_budget)


def _obs(stage_vals):                          # stage_vals: 长度 B 的 list
    return {
        "feat": torch.randn(B, REPR // PATCH, PATCH),
        "observation.state": torch.randn(B, PROP),
        "observation.base_action": torch.tanh(torch.randn(B, FLAT)),
        "observation.stage_id": torch.tensor(stage_vals, dtype=torch.float32).reshape(B, 1),
    }


def test_actor_budget_off_scale_uniform():
    # 关:scale 是均匀标量 0.05(TruncatedNormal float -> ones_like*scale),不随 stage 变
    torch.manual_seed(0)
    actor = _actor(stage_budget=None).eval()
    dist = actor.forward(_obs([0, 1, 2, 3, 4, 0]), std=0.05)
    assert torch.allclose(dist.scale, torch.full_like(dist.loc, 0.05))


def test_actor_budget_scales_mean_and_std():
    # 开:同一 actor(同权重),mean 与 std 都被逐样本 factor 缩
    torch.manual_seed(0)
    actor = _actor(stage_budget=BUDGET).eval()
    obs = _obs([0, 3, 4, 1, 2, 3])
    factor = torch.tensor([1.0, 0.3, 0.1, 1.0, 1.0, 0.3]).reshape(B, 1)
    dist_on = actor.forward(obs, std=0.05)
    actor.stage_budget = None                  # 关掉同一 actor 取 baseline
    dist_off = actor.forward(obs, std=0.05)
    assert torch.allclose(dist_on.loc, dist_off.loc * factor, atol=1e-6)
    assert torch.allclose(dist_on.scale, torch.full_like(dist_on.loc, 0.05) * factor, atol=1e-6)


import pytest


def test_cli_stage_budget_default_none():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    args = build_parser().parse_args([])
    assert args.stage_budget is None


def test_cli_stage_budget_parses_string():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    args = build_parser().parse_args(["--stage_budget", "1,1,1,0.3,0.1"])
    assert args.stage_budget == "1,1,1,0.3,0.1"


@pytest.mark.manual
def test_qagent_act_budget_only_scales_output():
    """核心性质:同一 obs 只改 stage_id,残差只被乘 factor、网络计算不变。
    (eval_mode + stddev=0 => 取均值;stage4 budget=0.1 => a4 == a0 * 0.1)
    依赖 VitEncoder,故 manual;CPU 跑。"""
    import math
    from resfit.rl_finetuning.config.rlpd import QAgentConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    torch.manual_seed(0)
    C, H, W = 3, 84, 84
    cam = "observation.images.agentview"
    state_dim, flat, ns = 5, 12, 5
    budget = [1.0, 1.0, 1.0, 1.0, 0.1]                 # 只在 stage4 卡死
    cfg = QAgentConfig(); cfg.device = "cpu"; cfg.critic.loss.type = "mse"
    agent = QAgent(obs_shape=(C, H, W), prop_shape=(state_dim,), action_dim=flat,
                   rl_cameras=[cam], cfg=cfg, residual_actor=True,
                   num_stages=ns, stage_budget=budget)
    agent.train(False)
    base = {cam: torch.rand(8, C, H, W),
            "observation.state": torch.randn(8, state_dim),
            "observation.base_action": torch.tanh(torch.randn(8, flat))}
    o0 = dict(base); o0["observation.stage_id"] = torch.zeros(8, 1)        # budget 1.0
    o4 = dict(base); o4["observation.stage_id"] = torch.full((8, 1), 4.0)  # budget 0.1
    with torch.no_grad():
        a0 = agent.act(o0, eval_mode=True, stddev=0.0, cpu=True)
        a4 = agent.act(o4, eval_mode=True, stddev=0.0, cpu=True)
    assert torch.allclose(a4, a0 * 0.1, atol=1e-5)     # 只缩输出,不改网络计算
    assert math.isfinite(a0.abs().sum().item())
