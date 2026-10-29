"""Tests for Plummer softening behavior in kernel contractions."""

import numpy as np
import pytest

from gragra.kernels.inverse_square import (
    acceleration_contract,
    gradient_contract,
    potential_contract,
)

# Source and target placed at the same coordinates to trigger collision detection
_COLLISION_SOURCE = [[1.0, 2.0, 3.0]]
_COLLISION_WEIGHTS = [1.0]
_COLLISION_TARGET = [[1.0, 2.0, 3.0]]

_SOFTENED = 0.5


@pytest.mark.parametrize(
    "contract_fn",
    [potential_contract, acceleration_contract, gradient_contract],
    ids=["potential", "acceleration", "gradient"],
)
def test_collision_zero_softening_raises(contract_fn):
    """Verify that a source-target collision with softening=0 raises ValueError."""
    with pytest.raises(ValueError, match=r"[Cc]ollision|target|source"):
        contract_fn(
            _COLLISION_SOURCE,
            _COLLISION_WEIGHTS,
            _COLLISION_TARGET,
            softening_m=0.0,
        )


@pytest.mark.parametrize(
    "contract_fn",
    [potential_contract, acceleration_contract, gradient_contract],
    ids=["potential", "acceleration", "gradient"],
)
def test_collision_with_softening_is_finite(contract_fn):
    """Verify source-target collision with softening > 0 produces finite output."""
    result = contract_fn(
        _COLLISION_SOURCE,
        _COLLISION_WEIGHTS,
        _COLLISION_TARGET,
        softening_m=_SOFTENED,
    )
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "contract_fn",
    [potential_contract, acceleration_contract, gradient_contract],
    ids=["potential", "acceleration", "gradient"],
)
def test_negative_softening_raises(contract_fn):
    """Verify that a negative softening parameter raises ValueError."""
    with pytest.raises(ValueError, match=r"softening|non-negative|>= 0"):
        contract_fn(
            [[0.0, 0.0, 0.0]],
            [1.0],
            [[1.0, 0.0, 0.0]],
            softening_m=-1.0,
        )
