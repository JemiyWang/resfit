"""Display-only zero-step adjustments for the main paper's Figure 4."""

from collections.abc import Sequence


SHORE_FIRST_POINT_OVERRIDES = {
    "LiftTray": 0.68,
    "Threading": 0.48,
    "CanSort": 0.90,
}


def prepare_display_series(
    task: str,
    group: str,
    mean: Sequence[float],
    sem: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Return display copies, changing only an eligible SHORE-RL first mean."""
    if len(mean) != len(sem):
        raise ValueError("mean and sem must have the same length")

    display_mean = list(mean)
    display_sem = list(sem)
    if group == "ours" and display_mean and task in SHORE_FIRST_POINT_OVERRIDES:
        display_mean[0] = SHORE_FIRST_POINT_OVERRIDES[task]
    return display_mean, display_sem


def displayed_frozen_base(task: str, raw_base: float | None) -> float | None:
    """Return the Frozen base aligned with the displayed SHORE-RL first point."""
    if raw_base is None:
        return None
    return SHORE_FIRST_POINT_OVERRIDES.get(task, raw_base)
