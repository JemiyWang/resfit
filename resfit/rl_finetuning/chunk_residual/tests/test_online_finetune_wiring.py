import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import maybe_build_finetuner


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


class _Args:
    online_finetune_value = False
    online_finetune_high_actor = False
    online_value_lr = 1e-5
    online_high_actor_lr = 1e-5
    online_finetune_offline_fraction = 0.5
    online_finetune_every = 1
    online_finetune_expectile = 0.7
    online_finetune_ema = 0.005
    online_finetune_future_mode = "geometric"
    online_finetune_max_trans = 50_000
    online_finetune_min_trans = 5
    subgoal_way_steps = 3
    gamma = 0.99
    batch_size = 16
    device = "cpu"
    seed = 0


def _make_subgoal():
    gc = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=16)
    ha = HighActor(state_dim=30, rep_dim=10, hidden=16)
    return HiqlSubgoal(gc, ha, np.zeros(30, np.float32), device="cpu",
                       state_mode="eef_piece",
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)))


def test_maybe_build_finetuner_off_returns_none():
    a = _Args()
    ft = maybe_build_finetuner(a, _make_subgoal(),
                               [np.ones((10, 30), np.float32)],
                               gc_info={"value_loss_mode": "shared_min",
                                        "value_mask_mode": "done_aware"})
    assert ft is None


def test_maybe_build_finetuner_on_reads_gc_info_modes():
    a = _Args()
    a.online_finetune_value = True
    a.online_finetune_high_actor = True
    ft = maybe_build_finetuner(a, _make_subgoal(),
                               [np.ones((10, 30), np.float32) + i for i in range(3)],
                               gc_info={"value_loss_mode": "hiql",
                                        "value_mask_mode": "hiql"})
    assert ft is not None
    assert ft.value_loss_mode == "hiql" and ft.value_mask_mode == "hiql"
    # 端到端:灌 episode -> maybe_update 出 metrics
    for ep in range(2):
        for t in range(6):
            ft.on_step(torch.randn(1, 30))
        ft.on_episode_end()
    m = ft.maybe_update()
    assert m is not None and "finetune/value_loss" in m and "finetune/high_actor_loss" in m
