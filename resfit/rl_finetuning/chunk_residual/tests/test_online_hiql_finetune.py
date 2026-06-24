import copy
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal
from resfit.rl_finetuning.chunk_residual.online_hiql_finetune import OnlineHiqlFinetuner


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _make_subgoal(D=6):
    gc = GoalConditionedVF(state_dim=D, rep_dim=4, hidden=16)
    ha = HighActor(state_dim=D, rep_dim=4, hidden=16)
    goal = np.zeros(D, dtype=np.float32)
    # eef_piece 路要 rel_stats;这里直接用 act_feat 风格 feat_stats 太重,改用最简:伪 eef
    return HiqlSubgoal(gc, ha, goal, device="cpu", renorm_subgoal=True,
                       state_mode="eef_piece",
                       rel_stats=(np.zeros(D - 18, np.float32) if D > 18 else np.zeros(0, np.float32),
                                  np.ones(D - 18, np.float32) if D > 18 else np.ones(0, np.float32)))


def _offline_seqs(D=6, n=3):
    return [np.cumsum(np.ones((10, D), np.float32), axis=0) + i for i in range(n)]


def test_finetuner_keeps_phi_frozen():
    sg = _make_subgoal()
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-3, high_actor_lr=1e-3, batch_size=16,
                             offline_fraction=0.5, way_steps=3, every=1, seed=0)
    # φ(goal_encoder)所有参数 requires_grad=False;v1/v2/ha 为 True
    assert all(not p.requires_grad for p in sg.vf.goal_encoder.parameters())
    assert all(p.requires_grad for p in sg.vf.v1.parameters())
    assert all(p.requires_grad for p in sg.vf.v2.parameters())
    assert all(p.requires_grad for p in sg.ha.parameters())


def test_finetuner_maybe_update_noop_until_ready():
    sg = _make_subgoal()
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-3, high_actor_lr=1e-3, batch_size=16,
                             offline_fraction=0.5, way_steps=3, every=1,
                             min_online_transitions=20, seed=0)
    assert ft.maybe_update() is None               # online store 空
    for _ in range(3):
        ft.on_step(torch.randn(1, 6))
    ft.on_episode_end()                            # 才 2 transitions < 20
    assert ft.maybe_update() is None


def test_finetuner_updates_value_heads_and_high_actor_but_not_phi():
    sg = _make_subgoal()
    phi_before = copy.deepcopy(sg.vf.goal_encoder.state_dict())
    v1_before = copy.deepcopy(sg.vf.v1.state_dict())
    ha_before = copy.deepcopy(sg.ha.state_dict())
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-2, high_actor_lr=1e-2, batch_size=16,
                             offline_fraction=0.5, way_steps=3, every=1,
                             min_online_transitions=5, seed=0)
    # 灌几条 online episode
    for ep in range(3):
        for t in range(8):
            ft.on_step(torch.randn(1, 6))
        ft.on_episode_end()
    m = ft.maybe_update()
    assert m is not None
    assert np.isfinite(m["finetune/value_loss"])
    assert np.isfinite(m["finetune/high_actor_loss"])
    # φ 不动
    for k in phi_before:
        assert torch.equal(phi_before[k], sg.vf.goal_encoder.state_dict()[k]), f"φ 被改了: {k}"
    # v1 与 ha 动了
    assert any(not torch.equal(v1_before[k], sg.vf.v1.state_dict()[k]) for k in v1_before)
    assert any(not torch.equal(ha_before[k], sg.ha.state_dict()[k]) for k in ha_before)


def test_finetuner_every_gates_update():
    sg = _make_subgoal()
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-3, high_actor_lr=1e-3, batch_size=16,
                             offline_fraction=0.5, way_steps=3, every=3,
                             min_online_transitions=5, seed=0)
    for ep in range(2):
        for t in range(8):
            ft.on_step(torch.randn(1, 6))
        ft.on_episode_end()
    assert ft.maybe_update() is None        # tick 1
    assert ft.maybe_update() is None        # tick 2
    assert ft.maybe_update() is not None    # tick 3


def test_finetuner_value_only_skips_high_actor():
    sg = _make_subgoal()
    ha_before = copy.deepcopy(sg.ha.state_dict())
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-2, high_actor_lr=1e-2, batch_size=16,
                             offline_fraction=0.0, way_steps=3, every=1,
                             min_online_transitions=5, finetune_high_actor=False, seed=0)
    for t in range(8):
        ft.on_step(torch.randn(1, 6))
    ft.on_episode_end()
    m = ft.maybe_update()
    assert "finetune/high_actor_loss" not in m
    for k in ha_before:
        assert torch.equal(ha_before[k], sg.ha.state_dict()[k]), "high_actor 不该被更新"
