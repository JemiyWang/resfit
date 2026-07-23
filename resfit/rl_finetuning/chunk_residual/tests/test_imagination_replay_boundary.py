import pytest
import torch

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    add_chunk_transition,
    resolve_replay_transition,
)
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform


def _obs(value):
    return {
        "observation.state": torch.full((1, 3), float(value)),
        "observation.base_action": torch.full((1, 4), float(value)),
    }


def test_marked_truncation_ends_control_episode_but_not_replay_bootstrap():
    control_next_obs = _obs(20)
    final_obs = _obs(10)
    terminated = torch.tensor([False])
    truncated = torch.tensor([True])
    info = {
        "bootstrap_on_truncation": True,
        "final_observation": final_obs,
    }

    replay_next_obs, replay_done = resolve_replay_transition(
        control_next_obs, terminated, truncated, info)
    episode_done = terminated | truncated

    assert episode_done.item() is True
    assert replay_done.item() is False
    assert replay_next_obs is final_obs
    assert replay_next_obs["observation.state"][0, 0].item() == 10.0
    assert control_next_obs["observation.state"][0, 0].item() == 20.0


@pytest.mark.parametrize(
    "terminated,truncated,expected_done",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_unmarked_transitions_keep_existing_done_semantics(
        terminated, truncated, expected_done):
    control_next_obs = _obs(20)
    replay_next_obs, replay_done = resolve_replay_transition(
        control_next_obs,
        torch.tensor([terminated]),
        torch.tensor([truncated]),
        {},
    )
    assert replay_next_obs is control_next_obs
    assert replay_done.item() is expected_done


def test_marked_truncation_requires_final_observation():
    with pytest.raises(RuntimeError, match="final_observation"):
        resolve_replay_transition(
            _obs(20), torch.tensor([False]), torch.tensor([True]),
            {"bootstrap_on_truncation": True})


def test_marked_truncation_cannot_be_terminal():
    with pytest.raises(RuntimeError, match="terminated"):
        resolve_replay_transition(
            _obs(20), torch.tensor([True]), torch.tensor([True]), {
                "bootstrap_on_truncation": True,
                "final_observation": _obs(10),
            })


class _TransformCaptureBuffer:
    def __init__(self, gamma):
        self.transform = MultiStepTransform(n_steps=1, gamma=gamma)
        self.stored = None

    def add(self, td):
        self.stored = self.transform._inv_call(td)


def test_second_segment_replay_item_bootstraps_from_s2():
    gamma = 0.995
    terminated = torch.tensor([False])
    truncated = torch.tensor([True])
    info = {
        "bootstrap_on_truncation": True,
        "final_observation": _obs(10),
        "scaled_action": torch.zeros(1, 4),
    }
    replay_next_obs, replay_done = resolve_replay_transition(
        _obs(20), terminated, truncated, info)
    rb = _TransformCaptureBuffer(gamma)

    add_chunk_transition(
        obs=_obs(1),
        next_obs=replay_next_obs,
        combined_action=info["scaled_action"],
        reward=torch.tensor([2.0]),
        done=replay_done,
        info=info,
        image_keys=[],
        lowdim_keys=["observation.state", "observation.base_action"],
        online_rb=rb,
    )

    assert rb.stored["next", "obs", "observation.state"][0].item() == 10.0
    assert rb.stored["nonterminal"].item() is True
    assert rb.stored["gamma"].item() == pytest.approx(gamma)

    target_q_at_s2 = torch.tensor(4.0)
    target = (
        rb.stored["next", "reward"]
        + rb.stored["gamma"]
        * rb.stored["nonterminal"]
        * target_q_at_s2
    )
    assert target.item() == pytest.approx(2.0 + gamma * 4.0)
