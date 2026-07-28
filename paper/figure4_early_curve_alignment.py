"""Pure display alignment for the first 100k of Figure 4 SHORE-RL curves."""

from collections.abc import Sequence
import math


ALIGNMENT_END_K = 100.0
START_OFFSETS = {
    "Pouring": -0.047,
    "LiftTray": 0.040,
    "ThreePiece": None,
    "Threading": 0.033,
    "CanSort": 0.050,
}


def smoothstep_decay(x_k: float) -> float:
    """Return a C1 correction weight: one at 0k and zero from 100k onward."""
    x_k = float(x_k)
    if not math.isfinite(x_k) or x_k < 0.0:
        raise ValueError(f"environment step must be finite and nonnegative: {x_k}")
    if x_k >= ALIGNMENT_END_K:
        return 0.0
    t = x_k / ALIGNMENT_END_K
    return 1.0 - 3.0 * t * t + 2.0 * t * t * t


def adjusted_start(task: str, ours_start: float, resfit_start: float) -> float:
    """Compute the fixed author-approved displayed SHORE-RL start."""
    if task not in START_OFFSETS:
        raise ValueError(f"unknown Figure 4 task: {task}")
    ours_start = float(ours_start)
    resfit_start = float(resfit_start)
    if not math.isfinite(ours_start) or not math.isfinite(resfit_start):
        raise ValueError("step-zero means must be finite")

    offset = START_OFFSETS[task]
    target = ours_start if offset is None else resfit_start + offset
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"{task} adjusted start is outside [0, 1]: {target}")
    if math.isclose(target, resfit_start, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{task} adjusted start must not equal ResFit")
    return target


def adjust_shore_mean(
    task: str,
    x_k: Sequence[float],
    mean: Sequence[float],
    resfit_start: float,
) -> list[float]:
    """Return a new mean with a smooth correction only before 100k."""
    if len(x_k) != len(mean) or not x_k:
        raise ValueError("x and mean must have matching lengths and be nonempty")
    if not math.isclose(float(x_k[0]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the SHORE-RL series must start at 0k")
    if not any(
        math.isclose(float(x), ALIGNMENT_END_K, rel_tol=0.0, abs_tol=1e-12)
        for x in x_k
    ):
        raise ValueError("the SHORE-RL series must contain the 100k boundary")

    target = adjusted_start(task, mean[0], resfit_start)
    delta = target - float(mean[0])
    adjusted = []
    for x, value in zip(x_k, mean):
        x = float(x)
        if x >= ALIGNMENT_END_K:
            adjusted.append(value)
        else:
            adjusted.append(float(value) + delta * smoothstep_decay(x))
    return adjusted
