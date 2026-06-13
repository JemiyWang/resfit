import numpy as np
import torch


def test_append_subgoal():
    from resfit.rl_finetuning.off_policy.rl.stage_utils import append_subgoal
    prop = torch.zeros(4, 18)
    z = torch.ones(4, 10)
    out = append_subgoal(prop, z)
    assert out.shape == (4, 28)
    assert torch.equal(out[:, 18:], z)
    assert out.device == prop.device


def test_actor_subgoal_conditioned_dim():
    from resfit.rl_finetuning.off_policy.rl.actor import Actor
    from resfit.rl_finetuning.config.rlpd import ActorConfig
    cfg = ActorConfig()
    cfg.spatial_emb = 0
    a = Actor(repr_dim=64, patch_repr_dim=8, prop_dim=18, action_dim=12, cfg=cfg,
              residual_actor=True, subgoal_conditioned=True, subgoal_dim=10)
    assert a.prop_dim == 40  # 18 + 12 (base_action) + 10 (subgoal)
    obs = {
        "feat": torch.zeros(2, 64),
        "observation.state": torch.zeros(2, 18),
        "observation.base_action": torch.zeros(2, 12),
        "observation.subgoal": torch.ones(2, 10),
    }
    dist = a.forward(obs, std=0.1)
    assert dist.mean.shape == (2, 12)


def test_actor_stage_and_subgoal_combined():
    from resfit.rl_finetuning.off_policy.rl.actor import Actor
    from resfit.rl_finetuning.config.rlpd import ActorConfig
    cfg = ActorConfig()
    cfg.spatial_emb = 0
    a = Actor(repr_dim=64, patch_repr_dim=8, prop_dim=18, action_dim=12, cfg=cfg,
              residual_actor=True, stage_conditioned=True, num_stages=5,
              subgoal_conditioned=True, subgoal_dim=10)
    assert a.prop_dim == 18 + 12 + 5 + 10  # state + base_action + stage one-hot + subgoal
    obs = {
        "feat": torch.zeros(2, 64),
        "observation.state": torch.zeros(2, 18),
        "observation.base_action": torch.zeros(2, 12),
        "observation.stage_id": torch.zeros(2, 1),
        "observation.subgoal": torch.ones(2, 10),
    }
    dist = a.forward(obs, std=0.1)
    assert dist.mean.shape == (2, 12)


def test_qagent_critic_prop_subgoal():
    from types import SimpleNamespace
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    fake = SimpleNamespace(stage_conditioned=False, subgoal_conditioned=True, num_stages=0)
    obs = {"observation.state": torch.zeros(3, 18), "observation.subgoal": torch.ones(3, 10)}
    prop = QAgent._critic_prop(fake, obs)
    assert prop.shape == (3, 28)
    assert torch.equal(prop[:, 18:], torch.ones(3, 10))


def test_hiql_subgoal_shapes(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF, save_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor, save_high_actor
    from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal
    gc = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=32)
    gp = str(tmp_path / "gc.pt")
    save_gc_value(gp, gc, v_stats={"min": -1, "max": 0, "mean": -0.5},
                  mean=torch.zeros(18), std=torch.ones(18), dataset_id="ds",
                  state_mode="eef_piece", rel_piece_stats=(np.zeros(12), np.ones(12)))
    ha = HighActor(state_dim=30, rep_dim=10, hidden=32)
    hp = str(tmp_path / "ha.pt")
    save_high_actor(hp, ha, gc_value_ckpt=gp, way_steps=25, beta=1.0)

    goal30 = torch.zeros(30)
    sg = HiqlSubgoal.from_ckpts(gp, hp, goal=goal30, device="cpu")
    assert sg.rep_dim == 10
    s30 = sg.build_state30(torch.zeros(2, 18), np.zeros((2, 12)))
    assert s30.shape == (2, 30)
    z = sg.subgoal_online(torch.zeros(2, 18), np.zeros((2, 12)))
    assert z.shape == (2, 10)
    zw = sg.subgoal_waypoint(torch.zeros(3, 30), torch.ones(3, 30))
    assert zw.shape == (3, 10)
    # 1-D 单步在线调用(典型 rollout 形态)-> (1, rep_dim)
    z1 = sg.subgoal_online(torch.zeros(18), np.zeros(12))
    assert z1.shape == (1, 10)


def test_representative_goal30_is_real_medoid():
    """goal30 应取离均值最近的真实末态(medoid),而非算术均值——保证 on-manifold、
    不糊掉 object(rel_piece)部分。HIQL 用真实 goal 态、从不平均(2026-06-09 讨论)。"""
    from resfit.rl_finetuning.chunk_residual.hiql_subgoal import representative_goal30
    # 三条变长 demo,末态 A=[0,0]、B=[1,1]、C=[5,5];均值=[2,2],离均值最近的真实末态=B
    seqs = [
        np.array([[9.0, 9.0], [0.0, 0.0]], dtype=np.float32),            # 末态 A=[0,0]
        np.array([[8.0, 8.0], [7.0, 7.0], [1.0, 1.0]], dtype=np.float32),  # 末态 B=[1,1]
        np.array([[5.0, 5.0]], dtype=np.float32),                       # 末态 C=[5,5]
    ]
    finals = np.stack([s[-1] for s in seqs])
    g = np.asarray(representative_goal30(seqs))
    assert g.shape == (2,)
    # 必须等于某条真实末态(on-manifold),且就是离均值最近的 B,而不是均值 [2,2]
    assert any(np.allclose(g, f) for f in finals), "goal30 必须是真实末态之一,不能是均值"
    assert np.allclose(g, [1.0, 1.0]), f"应为离均值最近的真实末态 B=[1,1],got {g}"
    assert not np.allclose(g, finals.mean(0)), "不应是算术均值 [2,2]"


def test_subgoal_online_renorm_projects_to_sphere():
    """renorm_subgoal=True:在线子目标 z 投到半径 sqrt(rep_dim)(对齐 HIQL eval);
    默认 False 保持取 .mean 原值(范数一般 != sqrt(rep))。"""
    import numpy as np
    import torch
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
    from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal

    torch.manual_seed(0)
    vf = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=32)
    ha = HighActor(state_dim=30, rep_dim=10, hidden=32)
    # 让 mean 明显偏离球面:放大 mean_net 末层权重
    with torch.no_grad():
        ha.mean_net[-1].weight.mul_(5.0)
    goal30, rel_mean, rel_std = np.zeros(30, np.float32), np.zeros(12, np.float32), np.ones(12, np.float32)

    sg_off = HiqlSubgoal(vf, ha, goal30, state_mode="eef_piece", rel_stats=(rel_mean, rel_std))                       # 默认 False
    sg_on = HiqlSubgoal(vf, ha, goal30, state_mode="eef_piece", rel_stats=(rel_mean, rel_std), renorm_subgoal=True)
    state_std, rel_raw = np.ones((4, 18), np.float32), np.ones((4, 12), np.float32)

    z_on = sg_on.subgoal_online(state_std, rel_raw)
    z_off = sg_off.subgoal_online(state_std, rel_raw)
    sqrt_rep = float(np.sqrt(10))
    assert torch.allclose(z_on.norm(dim=-1), torch.full((4,), sqrt_rep), atol=1e-4)
    # 关掉时范数不被强制贴球面(本构造下明显偏离)
    assert (z_off.norm(dim=-1) - sqrt_rep).abs().max() > 1e-2


def test_cli_renorm_subgoal_default_on():
    """对齐 HIQL eval:不带 flag 时 online z 默认投球面(renorm 默认 True);--no_renorm_subgoal 可关。"""
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    assert build_parser().parse_args([]).renorm_subgoal is True
    assert build_parser().parse_args(["--no_renorm_subgoal"]).renorm_subgoal is False
    assert build_parser().parse_args(["--renorm_subgoal"]).renorm_subgoal is True
