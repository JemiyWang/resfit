from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    name: str
    prompt: str
    success_dataset: str
    action_dim: int = 16
    policy_state_dim: int = 16
    dataset_fps: float = 30.0
    feature_pooling: str = "mean"


_PROFILES = {
    "block": TaskProfile("block", "build block", "block_success"),
    "cup": TaskProfile("cup", "pick cup", "cup_success"),
    "paper": TaskProfile(
        "paper",
        "put the paper roll on the holder",
        "paper_success",
        action_dim=16,
        policy_state_dim=14,
        dataset_fps=20.0,
    ),
}


def get_task_profile(name: str) -> TaskProfile:
    try:
        return _PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown task profile {name!r}; expected one of "
            f"{sorted(_PROFILES)}"
        ) from exc
