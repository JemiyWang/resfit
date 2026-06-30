from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
from resfit.rl_finetuning.config.rlpd import QAgentConfig


def _req(extra):
    base = ["--base_wandb_id", "x", "--task", "three_piece", "--dataset", "some/ds"]
    return build_parser().parse_args(base + extra)


def test_oac_flags_default_off():
    a = _req([])
    assert a.oac_explore is False
    assert a.oac_beta_ub == 4.0
    assert a.oac_delta == 0.5


def test_oac_flags_settable():
    a = _req(["--oac_explore", "--oac_beta_ub", "2.0", "--oac_delta", "0.1"])
    assert a.oac_explore is True
    assert a.oac_beta_ub == 2.0
    assert a.oac_delta == 0.1


def test_qagent_config_has_oac_fields_with_defaults():
    cfg = QAgentConfig()
    assert cfg.oac_explore is False
    assert cfg.oac_beta_ub == 4.0
    assert cfg.oac_delta == 0.5


import math
import torch
from resfit.rl_finetuning.off_policy.rl.oac_explore import q_upper_bound, optimistic_mean_shift


def test_q_upper_bound_two_heads_matches_paper():
    # K=2 时 σ = |Q1-Q2|/2,μ = (Q1+Q2)/2
    q = torch.tensor([[[1.0]], [[3.0]]])              # [K=2, B=1, 1]
    out = q_upper_bound(q, beta_ub=2.0)               # μ=2, σ=1 -> 2 + 2*1 = 4
    assert torch.allclose(out, torch.tensor([4.0]))


def test_shift_kl_budget_invariant():
    # 线性 Q_UB: q_ub(a)=sum(w*a) -> grad=w;偏移在 Σ^{-1} 范数下应恰为 sqrt(2δ)
    torch.manual_seed(0)
    B, A = 4, 6
    w = torch.randn(B, A)
    mu_T = torch.randn(B, A)
    std_T = torch.rand(B, A) * 0.1 + 0.01
    delta = 0.3
    mu_E = optimistic_mean_shift(mu_T, std_T, lambda a: (w * a).sum(-1), delta)
    mu_C = mu_E - mu_T
    sigma = std_T ** 2
    kl_norm_sq = (mu_C ** 2 / sigma).sum(-1)          # 应 ≈ 2δ(每行)
    assert torch.allclose(kl_norm_sq, torch.full((B,), 2 * delta), atol=1e-4)
    # 方向:mu_C 与 Σ·w 同向(逐维同号)
    assert torch.all(torch.sign(mu_C) == torch.sign(sigma * w))


def test_shift_delta_zero_is_noop():
    mu_T = torch.randn(3, 5)
    std_T = torch.rand(3, 5) + 0.01
    mu_E = optimistic_mean_shift(mu_T, std_T, lambda a: a.sum(-1), delta=0.0)
    assert torch.allclose(mu_E, mu_T)


def test_shift_rows_independent():
    # 改第 0 行的目标权重不应影响第 1 行的偏移
    mu_T = torch.zeros(2, 4)
    std_T = torch.ones(2, 4) * 0.1
    w_a = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    w_b = torch.tensor([[0, 0, 1.0, 0], [0, 1.0, 0, 0]])    # 第1行与 w_a 相同
    mu_a = optimistic_mean_shift(mu_T, std_T, lambda a: (w_a * a).sum(-1), 0.2)
    mu_b = optimistic_mean_shift(mu_T, std_T, lambda a: (w_b * a).sum(-1), 0.2)
    assert torch.allclose(mu_a[1], mu_b[1])                 # 第1行不受第0行变化影响


def test_shift_zero_grad_no_nan():
    mu_T = torch.randn(2, 3)
    std_T = torch.ones(2, 3) * 0.1
    mu_E = optimistic_mean_shift(mu_T, std_T, lambda a: (a * 0.0).sum(-1), 0.5)
    assert torch.isfinite(mu_E).all()
    assert torch.allclose(mu_E, mu_T, atol=1e-3)           # 梯度为0 -> 几乎不偏移


import pytest


def _build_agent(oac_explore, beta_ub=4.0, delta=0.5):
    from resfit.rl_finetuning.config.rlpd import QAgentConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    torch.manual_seed(0)
    C, H, W = 3, 84, 84
    cfg = QAgentConfig()
    cfg.device = "cpu"
    cfg.critic.loss.type = "mse"
    cfg.oac_explore = oac_explore
    cfg.oac_beta_ub = beta_ub
    cfg.oac_delta = delta
    agent = QAgent(obs_shape=(C, H, W), prop_shape=(5,), action_dim=12,
                   rl_cameras=["observation.images.agentview"], cfg=cfg,
                   residual_actor=True)
    agent.train(False)
    return agent, C, H, W


def _obs(C, H, W, B=8):
    return {"observation.images.agentview": torch.rand(B, C, H, W),
            "observation.state": torch.randn(B, 5),
            "observation.base_action": torch.tanh(torch.randn(B, 12))}


@pytest.mark.manual
def test_act_oac_returns_finite_shape():
    agent, C, H, W = _build_agent(oac_explore=True)
    with torch.no_grad():
        a = agent.act(_obs(C, H, W), eval_mode=False, stddev=0.05, cpu=True)
    assert a.shape == (8, 12)
    assert torch.isfinite(a).all()


@pytest.mark.manual
def test_act_dispatch_default_when_off():
    # oac_explore=False:_act_oac 不应被调用(置爆炸桩仍不报)
    agent, C, H, W = _build_agent(oac_explore=False)
    agent._act_oac = lambda *a, **k: (_ for _ in ()).throw(AssertionError("OAC 不该被调用"))
    with torch.no_grad():
        agent.act(_obs(C, H, W), eval_mode=False, stddev=0.05, cpu=True)   # 不报即对


@pytest.mark.manual
def test_act_dispatch_default_when_eval():
    # oac_explore=True 但 eval_mode=True:走均值,不走 OAC
    agent, C, H, W = _build_agent(oac_explore=True)
    agent._act_oac = lambda *a, **k: (_ for _ in ()).throw(AssertionError("eval 不该走 OAC"))
    with torch.no_grad():
        agent.act(_obs(C, H, W), eval_mode=True, stddev=0.0, cpu=True)      # 不报即对
