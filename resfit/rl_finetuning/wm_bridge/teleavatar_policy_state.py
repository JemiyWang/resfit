from __future__ import annotations

import numpy as np


def map_teleavatar_policy_state(state, target_dim: int) -> np.ndarray:
    """Map the 16D TeleAvatar state to the state expected by the base policy."""
    vector = np.asarray(state, dtype=np.float32).reshape(-1)
    if vector.shape != (16,):
        raise ValueError(
            f"TeleAvatar source state must be (16,), got {vector.shape}"
        )
    if target_dim == 16:
        return vector
    if target_dim == 14:
        return np.concatenate([vector[:7], vector[8:15]]).astype(
            np.float32,
            copy=False,
        )
    raise ValueError(f"policy state dim must be 14 or 16, got {target_dim}")
