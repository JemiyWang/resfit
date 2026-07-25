from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    name: str
    prompt: str
    success_dataset: str
    action_dim: int = 16


_PROFILES = {
    "block": TaskProfile("block", "build block", "block_success"),
    "cup": TaskProfile("cup", "pick cup", "cup_success"),
}


def get_task_profile(name: str) -> TaskProfile:
    try:
        return _PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown task profile {name!r}; expected one of "
            f"{sorted(_PROFILES)}"
        ) from exc
