"""Collision checking utility for gragra gravity kernels.

This module is dependency-light (numpy-only) and shared between Numba and
C++ backend wrappers to prevent transitive compilation-toolchain dependencies.
"""

import numpy as np


def _check_collisions(collision_source_index: np.ndarray, eps: float) -> None:
    if eps == 0.0:
        collided = np.where(collision_source_index >= 0)[0]
        if len(collided) > 0:
            target_idx = collided[0]
            source_idx = collision_source_index[target_idx]
            raise ValueError(
                "Collision detected between target point "
                f"{target_idx} and source point {source_idx} "
                "with zero softening."
            )
